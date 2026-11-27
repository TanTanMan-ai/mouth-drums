"""입드럼 -> 드럼 변환 엔진 (프로토타입).

파이프라인: 온셋 감지 -> 온셋 직후 짧은 구간의 특징 추출 -> kNN 분류 -> 드럼 합성 / MIDI.
나중에 C++(JUCE)로 옮길 것을 전제로, 실시간에서도 가능한 연산만 쓴다.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 44100
N_FFT = 1024
HOP = 128                 # 약 2.9ms
MIN_GAP_MS = 60           # 온셋 사이 최소 간격
ANALYSIS_MS = 40          # 분류에 쓰는 구간 길이 = 실시간일 때의 최소 지연
N_BANDS = 24
LABELS = ["kick", "snare", "hihat"]
MIDI_NOTES = {"kick": 36, "snare": 38, "hihat": 42}


# ---------- 입출력 ----------

def load_audio(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def save_audio(path: str, x: np.ndarray) -> None:
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 0.99:
        x = x * (0.99 / peak)
    sf.write(path, x, SR)


# ---------- 온셋 감지 ----------

def detect_onsets(x: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
    """온셋 위치(샘플 인덱스) 배열을 돌려준다. sensitivity가 클수록 더 많이 잡는다."""
    if len(x) < N_FFT * 2:
        return np.array([], dtype=int)
    n_frames = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    mag = np.abs(np.fft.rfft(x[idx] * np.hanning(N_FFT), axis=1))
    logmag = np.log1p(100.0 * mag)
    flux = np.maximum(logmag[1:] - logmag[:-1], 0.0).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    if flux.max() <= 0:
        return np.array([], dtype=int)

    # 적응형 문턱값: 주변 평균 + 전체 최대값 대비 바닥
    win = int(0.1 * SR / HOP)
    local_mean = np.convolve(flux, np.ones(win) / win, mode="same")
    thresh = (1.5 * local_mean + 0.12 * flux.max()) / sensitivity

    min_gap = int(MIN_GAP_MS / 1000 * SR)
    onsets: list[int] = []
    for i in range(1, n_frames - 1):
        if flux[i] > thresh[i] and flux[i] >= flux[i - 1] and flux[i] > flux[i + 1]:
            pos = _refine(x, i * HOP + N_FFT // 2)
            if onsets and pos - onsets[-1] < min_gap:
                continue
            onsets.append(pos)
    return np.array(onsets, dtype=int)


def _refine(x: np.ndarray, pos: int) -> int:
    """프레임 단위 위치를 실제 소리가 올라오기 시작하는 샘플로 당긴다."""
    lo, hi = max(0, pos - N_FFT // 2), min(len(x), pos + N_FFT // 2)
    seg = np.abs(x[lo:hi])
    if len(seg) == 0 or seg.max() == 0:
        return pos
    return lo + int(np.argmax(seg > 0.2 * seg.max()))


# ---------- 특징 추출 ----------

_FEAT_FFT = 2048
_freqs = np.fft.rfftfreq(_FEAT_FFT, 1 / SR)
_edges = np.geomspace(50, 16000, N_BANDS + 1)
_band_masks = [(_freqs >= lo) & (_freqs < hi) for lo, hi in zip(_edges[:-1], _edges[1:])]


def extract_features(x: np.ndarray, onset: int, analysis_ms: float = ANALYSIS_MS) -> np.ndarray:
    n = int(analysis_ms / 1000 * SR)
    seg = x[onset:onset + n]
    if len(seg) < n:
        seg = np.pad(seg, (0, n - len(seg)))
    win = np.hanning(n)
    power = np.abs(np.fft.rfft(seg * win, _FEAT_FFT)) ** 2
    # 앞 소리의 꼬리가 겹칠 때(킥 직후 하이햇 등)를 위해 온셋 직전 스펙트럼을 빼준다
    pre_end = onset - int(0.003 * SR)
    if pre_end - n >= 0:
        pre = np.abs(np.fft.rfft(x[pre_end - n:pre_end] * win, _FEAT_FFT)) ** 2
        power = np.maximum(power - pre, 0.01 * power)
    bands = np.log(np.array([power[m].sum() for m in _band_masks]) + 1e-10)
    bands -= bands.mean()                      # 음량에 무관하게
    total = power.sum() + 1e-10
    centroid = np.log((power * _freqs).sum() / total + 1.0)
    zcr = np.mean(np.abs(np.diff(np.sign(seg)))) / 2
    return np.concatenate([bands, [centroid, zcr * 10]])


def peak_level(x: np.ndarray, onset: int) -> float:
    seg = x[onset:onset + int(0.05 * SR)]
    return float(np.max(np.abs(seg))) if len(seg) else 0.0


# ---------- 분류기 (사용자 캘리브레이션 기반 kNN) ----------

@dataclass
class Model:
    X: np.ndarray           # 표준화 전 특징
    y: np.ndarray           # LABELS 인덱스
    analysis_ms: float = ANALYSIS_MS

    def __post_init__(self):
        self.mean = self.X.mean(axis=0)
        self.std = self.X.std(axis=0) + 1e-6
        self.Xn = (self.X - self.mean) / self.std

    def predict(self, feat: np.ndarray, k: int = 3) -> str:
        d = np.linalg.norm(self.Xn - (feat - self.mean) / self.std, axis=1)
        nearest = self.y[np.argsort(d)[:min(k, len(d))]]
        return LABELS[int(np.bincount(nearest, minlength=len(LABELS)).argmax())]

    def save(self, path: str) -> None:
        np.savez(path, X=self.X, y=self.y, analysis_ms=self.analysis_ms)

    @classmethod
    def load(cls, path: str) -> "Model":
        d = np.load(path)
        return cls(d["X"], d["y"], float(d["analysis_ms"]))


def train(examples: dict[str, np.ndarray], analysis_ms: float = ANALYSIS_MS,
          sensitivity: float = 1.0) -> tuple[Model, dict[str, int]]:
    """examples: {"kick": 오디오, "snare": 오디오, "hihat": 오디오}. 각 오디오엔 그 소리만 여러 번."""
    X, y, counts = [], [], {}
    for label, audio in examples.items():
        onsets = detect_onsets(audio, sensitivity)
        counts[label] = len(onsets)
        for o in onsets:
            X.append(extract_features(audio, o, analysis_ms))
            y.append(LABELS.index(label))
    if not X:
        raise ValueError("캘리브레이션 녹음에서 소리를 하나도 찾지 못했습니다.")
    return Model(np.array(X), np.array(y), analysis_ms), counts


# ---------- 변환 ----------

@dataclass
class Hit:
    sample: int
    label: str
    velocity: int           # 1..127

    @property
    def time(self) -> float:
        return self.sample / SR


def transcribe(x: np.ndarray, model: Model, sensitivity: float = 1.0) -> list[Hit]:
    onsets = detect_onsets(x, sensitivity)
    levels = np.array([peak_level(x, o) for o in onsets])
    ref = levels.max() if len(levels) else 1.0
    hits = []
    for o, lv in zip(onsets, levels):
        label = model.predict(extract_features(x, o, model.analysis_ms))
        vel = int(np.clip(40 + 87 * (lv / ref), 1, 127))
        hits.append(Hit(int(o), label, vel))
    return hits


# ---------- 드럼 합성 (샘플 파일 없이 들어볼 수 있게) ----------

def _synth(label: str) -> np.ndarray:
    rng = np.random.default_rng(0)
    if label == "kick":
        t = np.arange(int(0.35 * SR)) / SR
        freq = 45 + 110 * np.exp(-t * 35)
        return np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t * 9)
    if label == "snare":
        t = np.arange(int(0.22 * SR)) / SR
        noise = np.diff(rng.standard_normal(len(t) + 1))        # 고역 강조
        return 0.5 * np.sin(2 * np.pi * 185 * t) * np.exp(-t * 30) + 0.35 * noise * np.exp(-t * 22)
    t = np.arange(int(0.07 * SR)) / SR
    noise = np.diff(rng.standard_normal(len(t) + 2), n=2)
    return 0.15 * noise * np.exp(-t * 70)


def render(hits: list[Hit], length: int) -> np.ndarray:
    sounds = {label: _synth(label) for label in LABELS}
    out = np.zeros(length + SR, dtype=np.float64)
    for h in hits:
        s = sounds[h.label] * (h.velocity / 127)
        end = min(len(out), h.sample + len(s))
        out[h.sample:end] += s[:end - h.sample]
    return out.astype(np.float32)


# ---------- MIDI 내보내기 ----------

def _varlen(n: int) -> bytes:
    out = [n & 0x7F]
    while n := n >> 7:
        out.append((n & 0x7F) | 0x80)
    return bytes(reversed(out))


def write_midi(path: str, hits: list[Hit], bpm: float = 120.0) -> None:
    """GM 드럼(채널 10) 단일 트랙 MIDI 파일. DAW에 끌어다 놓으면 된다."""
    ppq = 480
    ticks_per_sec = ppq * bpm / 60
    events = []
    for h in hits:
        on = int(round(h.time * ticks_per_sec))
        note = MIDI_NOTES[h.label]
        events.append((on, 1, bytes([0x99, note, h.velocity])))
        events.append((on + ppq // 8, 0, bytes([0x89, note, 0])))
    events.sort(key=lambda e: (e[0], e[1]))   # 같은 틱이면 note-off 먼저

    track = b"\x00\xff\x51\x03" + struct.pack(">I", int(60_000_000 / bpm))[1:]
    last = 0
    for tick, _, data in events:
        track += _varlen(tick - last) + data
        last = tick
    track += b"\x00\xff\x2f\x00"
    with open(path, "wb") as f:
        f.write(b"MThd" + struct.pack(">IHHH", 6, 0, 1, ppq))
        f.write(b"MTrk" + struct.pack(">I", len(track)) + track)
