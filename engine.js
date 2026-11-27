// 입드럼 -> 드럼 변환 엔진 (engine.py를 그대로 옮긴 것).
// 온셋 감지 -> 온셋 직후 짧은 구간의 특징 추출 -> kNN 분류 -> 드럼 합성 / MIDI.
const Engine = (() => {
  const N_FFT = 1024;
  const HOP = 128;
  const MIN_GAP_MS = 60;
  const ANALYSIS_MS = 40;
  const N_BANDS = 24;
  const FEAT_FFT = 2048;
  const LABELS = ["kick", "snare", "hihat"];
  const MIDI_NOTES = { kick: 36, snare: 38, hihat: 42 };

  // ---------- FFT ----------
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (-2 * Math.PI) / len;
      const wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const a = i + k, b = a + len / 2;
          const xr = re[b] * cr - im[b] * ci;
          const xi = re[b] * ci + im[b] * cr;
          re[b] = re[a] - xr; im[b] = im[a] - xi;
          re[a] += xr; im[a] += xi;
          const t = cr * wr - ci * wi;
          ci = cr * wi + ci * wr; cr = t;
        }
      }
    }
  }

  function hann(n) {
    const w = new Float64Array(n);
    for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
    return w;
  }

  // x[start .. start+n) 에 창을 씌워 size 길이 FFT의 파워 스펙트럼을 구한다
  function powerSpectrum(x, start, win, size) {
    const re = new Float64Array(size), im = new Float64Array(size);
    for (let i = 0; i < win.length; i++) {
      const v = start + i < x.length ? x[start + i] : 0;
      re[i] = v * win[i];
    }
    fft(re, im);
    const p = new Float64Array(size / 2 + 1);
    for (let i = 0; i < p.length; i++) p[i] = re[i] * re[i] + im[i] * im[i];
    return p;
  }

  // ---------- 온셋 감지 ----------
  function detectOnsets(x, sr, sensitivity = 1) {
    if (x.length < N_FFT * 2) return [];
    const nFrames = 1 + Math.floor((x.length - N_FFT) / HOP);
    const win = hann(N_FFT);
    const flux = new Float64Array(nFrames);
    // 로그 간격 대역으로 묶어서 본다. 주파수 빈을 그대로 더하면 고역 잡음만 크게 잡히고
    // 낮고 둥근 킥 소리("둠")는 놓친다.
    const binHz = sr / N_FFT, ratio = Math.pow(16000 / 40, 1 / N_BANDS);
    const edges = [];
    for (let b = 0; b <= N_BANDS; b++) edges.push(Math.min(N_FFT / 2 + 1, Math.ceil((40 * Math.pow(ratio, b)) / binHz)));
    const amps = [];
    let maxAmp = 0;
    for (let f = 0; f < nFrames; f++) {
      const spec = powerSpectrum(x, f * HOP, win, N_FFT);
      const a = new Float64Array(N_BANDS);
      for (let b = 0; b < N_BANDS; b++) {
        let s = 0;
        for (let i = edges[b]; i < edges[b + 1]; i++) s += spec[i];
        a[b] = Math.sqrt(s);
        if (a[b] > maxAmp) maxAmp = a[b];
      }
      amps.push(a);
    }
    if (maxAmp <= 0) return [];
    const floor = 0.003 * maxAmp;       // 가장 큰 소리보다 50dB 아래는 배경 잡음으로 보고 무시
    let prev = null, maxFlux = 0;
    for (let f = 0; f < nFrames; f++) {
      const p = amps[f];
      for (let b = 0; b < N_BANDS; b++) p[b] = Math.log(p[b] + floor);
      if (prev) {
        let s = 0;
        for (let i = 0; i < p.length; i++) { const d = p[i] - prev[i]; if (d > 0) s += d; }
        flux[f] = s;
        if (s > maxFlux) maxFlux = s;
      }
      prev = p;
    }
    if (maxFlux <= 0) return [];

    // 적응형 문턱값: 주변 평균 + 전체 최대값 대비 바닥
    const w = Math.max(1, Math.floor((0.1 * sr) / HOP));
    const half = Math.floor(w / 2);
    const cum = new Float64Array(nFrames + 1);
    for (let i = 0; i < nFrames; i++) cum[i + 1] = cum[i] + flux[i];
    const minGap = Math.floor((MIN_GAP_MS / 1000) * sr);
    const onsets = [];
    for (let i = 1; i < nFrames - 1; i++) {
      const lo = Math.max(0, i - half), hi = Math.min(nFrames, i - half + w);
      const localMean = (cum[hi] - cum[lo]) / w;
      const thresh = (1.5 * localMean + 0.12 * maxFlux) / sensitivity;
      if (flux[i] > thresh && flux[i] >= flux[i - 1] && flux[i] > flux[i + 1]) {
        const pos = refine(x, i * HOP + N_FFT / 2);
        if (onsets.length && pos - onsets[onsets.length - 1] < minGap) continue;
        onsets.push(pos);
      }
    }
    return onsets;
  }

  // 프레임 단위 위치를 실제 소리가 올라오기 시작하는 샘플로 당긴다
  function refine(x, pos) {
    const lo = Math.max(0, pos - N_FFT / 2), hi = Math.min(x.length, pos + N_FFT / 2);
    let peak = 0;
    for (let i = lo; i < hi; i++) peak = Math.max(peak, Math.abs(x[i]));
    if (peak === 0) return pos;
    for (let i = lo; i < hi; i++) if (Math.abs(x[i]) > 0.2 * peak) return i;
    return pos;
  }

  // ---------- 특징 추출 ----------
  function extractFeatures(x, sr, onset, analysisMs = ANALYSIS_MS) {
    const n = Math.min(FEAT_FFT, Math.round((analysisMs / 1000) * sr));
    const win = hann(n);
    const power = powerSpectrum(x, onset, win, FEAT_FFT);
    // 앞 소리의 꼬리가 겹칠 때(킥 직후 하이햇 등)를 위해 온셋 직전 스펙트럼을 빼준다
    const preEnd = onset - Math.floor(0.003 * sr);
    if (preEnd - n >= 0) {
      const pre = powerSpectrum(x, preEnd - n, win, FEAT_FFT);
      for (let i = 0; i < power.length; i++) power[i] = Math.max(power[i] - pre[i], 0.01 * power[i]);
    }
    const binHz = sr / FEAT_FFT;
    const feat = new Array(N_BANDS + 2);
    const ratio = Math.pow(16000 / 50, 1 / N_BANDS);
    let mean = 0;
    for (let b = 0; b < N_BANDS; b++) {
      const lo = 50 * Math.pow(ratio, b), hi = lo * ratio;
      let s = 0;
      for (let i = Math.ceil(lo / binHz); i * binHz < hi && i < power.length; i++) s += power[i];
      feat[b] = Math.log(s + 1e-10);
      mean += feat[b] / N_BANDS;
    }
    for (let b = 0; b < N_BANDS; b++) feat[b] -= mean;        // 음량에 무관하게
    let total = 1e-10, weighted = 0;
    for (let i = 0; i < power.length; i++) { total += power[i]; weighted += power[i] * i * binHz; }
    feat[N_BANDS] = Math.log(weighted / total + 1);
    let crossings = 0;
    for (let i = onset + 1; i < onset + n && i < x.length; i++) {
      if (Math.sign(x[i]) !== Math.sign(x[i - 1])) crossings++;
    }
    feat[N_BANDS + 1] = (crossings / Math.max(1, n - 1)) * 10;
    return feat;
  }

  function peakLevel(x, sr, onset) {
    const end = Math.min(x.length, onset + Math.floor(0.05 * sr));
    let p = 0;
    for (let i = onset; i < end; i++) p = Math.max(p, Math.abs(x[i]));
    return p;
  }

  // ---------- 분류기 (사용자 캘리브레이션 기반 kNN) ----------
  // model = { sr, analysisMs, examples: { kick: [feat...], snare: [...], hihat: [...] } }
  function newModel(sr) {
    return { sr, analysisMs: ANALYSIS_MS, examples: {} };
  }

  // 한 종류의 소리만 여러 번 담긴 녹음으로 그 종류를 학습. 인식한 개수를 돌려준다.
  // 학습용 온셋: 배경 잡음이나 숨소리가 세어지지 않도록 충분히 큰 소리만 남긴다.
  // 녹음 도중에 반복 호출해서 "지금까지 몇 번 인식했는지"를 보여주는 데도 쓴다.
  // refLevel: 앞 구간에서 이미 들은 가장 큰 소리 크기 (구간을 나눠 볼 때 기준을 이어가기 위함)
  function calibrationOnsets(x, sr, refLevel = 0) {
    const all = detectOnsets(x, sr);
    const levels = all.map((o) => peakLevel(x, sr, o));
    const maxLevel = Math.max(refLevel, ...levels);
    const gate = Math.max(0.02, 0.25 * maxLevel);
    return { onsets: all.filter((_, i) => levels[i] >= gate), maxLevel };
  }

  function learn(model, label, x) {
    const { onsets } = calibrationOnsets(x, model.sr);
    model.examples[label] = onsets.map((o) => extractFeatures(x, model.sr, o, model.analysisMs));
    return onsets;
  }

  function isReady(model) {
    return !!model && LABELS.every((l) => (model.examples[l] || []).length > 0);
  }

  function makeClassifier(model) {
    const X = [], y = [];
    LABELS.forEach((l, li) => (model.examples[l] || []).forEach((f) => { X.push(f); y.push(li); }));
    const dim = X[0].length;
    const mean = new Array(dim).fill(0), std = new Array(dim).fill(0);
    X.forEach((f) => f.forEach((v, i) => (mean[i] += v / X.length)));
    X.forEach((f) => f.forEach((v, i) => (std[i] += (v - mean[i]) ** 2 / X.length)));
    for (let i = 0; i < dim; i++) std[i] = Math.sqrt(std[i]) + 1e-6;
    const norm = (f) => f.map((v, i) => (v - mean[i]) / std[i]);
    const Xn = X.map(norm);
    return (feat, k = 3) => {
      const q = norm(feat);
      const d = Xn.map((f, i) => [f.reduce((s, v, j) => s + (v - q[j]) ** 2, 0), y[i]]);
      d.sort((a, b) => a[0] - b[0]);
      const votes = new Array(LABELS.length).fill(0);
      d.slice(0, k).forEach(([, li]) => votes[li]++);
      return LABELS[votes.indexOf(Math.max(...votes))];
    };
  }

  // ---------- 변환 ----------
  function transcribe(x, model, sensitivity = 1) {
    const sr = model.sr;
    const classify = makeClassifier(model);
    const onsets = detectOnsets(x, sr, sensitivity);
    const levels = onsets.map((o) => peakLevel(x, sr, o));
    const ref = Math.max(1e-9, ...levels);
    return onsets.map((o, i) => ({
      sample: o,
      time: o / sr,
      label: classify(extractFeatures(x, sr, o, model.analysisMs)),
      velocity: Math.max(1, Math.min(127, Math.round(40 + 87 * (levels[i] / ref)))),
    }));
  }

  // ---------- 박자 맞추기 ----------
  // 소리 사이 간격들이 공통으로 나누어떨어지는 가장 긴 단위(격자 한 칸)를 찾고,
  // 각 소리를 그 격자에 붙인다. 박자를 못 찾으면 null.
  function quantize(hits, sr) {
    if (hits.length < 4) return null;
    const gaps = [];
    for (let i = 1; i < hits.length; i++) gaps.push(hits[i].time - hits[i - 1].time);
    const score = (s) => gaps.reduce((a, d) => a + Math.cos((2 * Math.PI * d) / s), 0) / gaps.length;
    const cands = [];
    for (let s = 0.08; s <= 1.0; s *= 1.005) cands.push([s, score(s)]);
    const best = Math.max(...cands.map((c) => c[1]));
    if (best < 0.5) return null;
    // 잘게 쪼갠 격자는 뭐든 잘 맞으므로, 충분히 잘 맞는 것 중 가장 굵은 격자를 고른다
    let k = cands.length - 1;
    while (cands[k][1] < 0.85 * best) k--;
    while (k > 0 && cands[k - 1][1] > cands[k][1]) k--;
    let step = cands[k][0];
    const cells = gaps.map((d) => Math.max(1, Math.round(d / step)));
    step = gaps.reduce((a, d, i) => a + d * cells[i], 0) / cells.reduce((a, n) => a + n * n, 0);

    // 앞 소리와의 간격만 보고 붙이므로, 템포가 조금씩 빨라지거나 느려져도 칸을 잘못 짚지 않는다
    let t = hits[0].time;
    const out = hits.map((h, i) => {
      if (i > 0) t += cells[i - 1] * step;
      return { ...h, time: t, sample: Math.round(t * sr) };
    });
    let bpm = 60 / step;
    for (const div of [1, 2, 4]) {
      if (60 / (step * div) >= 70 && 60 / (step * div) < 140) { bpm = 60 / (step * div); break; }
    }
    return { hits: out, step, bpm };
  }

  // 입으로는 한 번에 한 소리만 낼 수 있으므로, 킥·스네어와 겹치는 하이햇은 자동으로 깔아준다.
  // division: 4(4분음표), 8, 16. 이미 하이햇을 낸 자리는 건너뛴다.
  function addHats(grid, division, sr) {
    const hits = grid.hits, gap = ((60 / grid.bpm) * 4) / division;
    const start = hits[0].time, end = hits[hits.length - 1].time;
    const perBeat = division / 4;
    const out = hits.slice();
    for (let n = 0; start + n * gap <= end + gap * 0.01; n++) {
      const t = start + n * gap;
      if (hits.some((h) => h.label === "hihat" && Math.abs(h.time - t) < gap / 4)) continue;
      out.push({ sample: Math.round(t * sr), time: t, label: "hihat", velocity: n % perBeat === 0 ? 85 : 60, auto: true });
    }
    return out.sort((a, b) => a.time - b.time);
  }

  // ---------- 드럼 합성 ----------
  function synth(label, sr) {
    if (label === "kick") {
      const n = Math.floor(0.35 * sr), out = new Float32Array(n);
      let phase = 0;
      for (let i = 0; i < n; i++) {
        const t = i / sr;
        phase += (2 * Math.PI * (45 + 110 * Math.exp(-t * 35))) / sr;
        out[i] = Math.sin(phase) * Math.exp(-t * 9);
      }
      return out;
    }
    const noise = (n) => { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = Math.random() * 2 - 1; return a; };
    if (label === "snare") {
      const n = Math.floor(0.22 * sr), out = new Float32Array(n), w = noise(n + 1);
      for (let i = 0; i < n; i++) {
        const t = i / sr;
        out[i] = 0.5 * Math.sin(2 * Math.PI * 185 * t) * Math.exp(-t * 30)
               + 0.6 * (w[i + 1] - w[i]) * Math.exp(-t * 22);
      }
      return out;
    }
    const n = Math.floor(0.07 * sr), out = new Float32Array(n), w = noise(n + 2);
    for (let i = 0; i < n; i++) {
      out[i] = 0.25 * (w[i + 2] - 2 * w[i + 1] + w[i]) * Math.exp((-i / sr) * 70);
    }
    return out;
  }

  function render(hits, length, sr) {
    const sounds = {};
    LABELS.forEach((l) => (sounds[l] = synth(l, sr)));
    const out = new Float32Array(length + sr);
    for (const h of hits) {
      const s = sounds[h.label], g = h.velocity / 127;
      for (let i = 0; i < s.length && h.sample + i < out.length; i++) out[h.sample + i] += s[i] * g;
    }
    return out;
  }

  // ---------- 내보내기 ----------
  function encodeWav(x, sr) {
    let peak = 0;
    for (let i = 0; i < x.length; i++) peak = Math.max(peak, Math.abs(x[i]));
    const gain = peak > 0.99 ? 0.99 / peak : 1;
    const buf = new ArrayBuffer(44 + x.length * 2), v = new DataView(buf);
    const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    str(0, "RIFF"); v.setUint32(4, 36 + x.length * 2, true); str(8, "WAVEfmt ");
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true);
    v.setUint16(34, 16, true); str(36, "data"); v.setUint32(40, x.length * 2, true);
    for (let i = 0; i < x.length; i++) {
      v.setInt16(44 + i * 2, Math.round(Math.max(-1, Math.min(1, x[i] * gain)) * 32767), true);
    }
    return buf;
  }

  // GM 드럼(채널 10) 단일 트랙 MIDI 파일
  function encodeMidi(hits, bpm = 120) {
    const ppq = 480, ticksPerSec = (ppq * bpm) / 60;
    const events = [];
    for (const h of hits) {
      const on = Math.round(h.time * ticksPerSec), note = MIDI_NOTES[h.label];
      events.push([on, 1, [0x99, note, h.velocity]]);
      events.push([on + ppq / 8, 0, [0x89, note, 0]]);
    }
    events.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const varlen = (n) => {
      const out = [n & 0x7f];
      while ((n >>= 7)) out.unshift((n & 0x7f) | 0x80);
      return out;
    };
    const tempo = Math.round(60000000 / bpm);
    const track = [0, 0xff, 0x51, 3, (tempo >> 16) & 255, (tempo >> 8) & 255, tempo & 255];
    let last = 0;
    for (const [tick, , data] of events) { track.push(...varlen(tick - last), ...data); last = tick; }
    track.push(0, 0xff, 0x2f, 0);
    const u32 = (n) => [(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255];
    return new Uint8Array([
      0x4d, 0x54, 0x68, 0x64, ...u32(6), 0, 0, 0, 1, ppq >> 8, ppq & 255,
      0x4d, 0x54, 0x72, 0x6b, ...u32(track.length), ...track,
    ]);
  }

  return { LABELS, ANALYSIS_MS, detectOnsets, extractFeatures, newModel, calibrationOnsets, learn, isReady,
           transcribe, quantize, addHats, render, encodeWav, encodeMidi };
})();

if (typeof module !== "undefined") module.exports = Engine;
