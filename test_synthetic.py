"""합성한 가짜 입소리로 파이프라인 전체를 검증한다.

실제 목소리 정확도를 보여주는 건 아니고, 온셋 감지 / 분류 / 렌더 / MIDI가
맞물려 도는지 확인하는 용도다.   실행: .venv/bin/python test_synthetic.py
"""
import os
import tempfile

import numpy as np

import engine
from engine import SR


def mouth_sound(label: str, rng) -> np.ndarray:
    """붐 / 프슈 / 츠 를 흉내낸 소리. 매번 조금씩 다르게."""
    j = lambda lo, hi: rng.uniform(lo, hi)
    if label == "kick":                       # 낮은 유성음
        t = np.arange(int(j(0.10, 0.16) * SR)) / SR
        f = j(90, 130)
        s = np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * 2 * f * t)
        s = s * np.exp(-t * j(18, 28)) + 0.05 * rng.standard_normal(len(t)) * np.exp(-t * 60)
    elif label == "snare":                    # 중역대 잡음
        t = np.arange(int(j(0.10, 0.18) * SR)) / SR
        n = rng.standard_normal(len(t))
        n = np.convolve(n, np.ones(6) / 6, mode="same")      # 고역을 깎음
        s = n * np.exp(-t * j(15, 25)) * 2
    else:                                     # 높은 치찰음
        t = np.arange(int(j(0.03, 0.06) * SR)) / SR
        n = np.diff(rng.standard_normal(len(t) + 2), n=2)
        s = n * np.exp(-t * j(40, 70)) * 0.5
    attack = np.minimum(1, np.arange(len(s)) / (0.002 * SR))
    return (s * attack * j(0.5, 1.0) / np.max(np.abs(s))).astype(np.float32)


def sequence(labels, times, rng, length):
    x = 0.002 * rng.standard_normal(int(length * SR)).astype(np.float32)   # 배경 잡음
    for label, t in zip(labels, times):
        s = mouth_sound(label, rng)
        i = int(t * SR)
        x[i:i + len(s)] += s
    return x


def main():
    rng = np.random.default_rng(1)

    # 캘리브레이션: 종류별로 8번씩
    cal = {lb: sequence([lb] * 8, 0.3 + 0.6 * np.arange(8), rng, 5.5) for lb in engine.LABELS}
    model, counts = engine.train(cal)
    print("캘리브레이션 인식 개수:", counts)
    assert all(n == 8 for n in counts.values()), counts

    # 테스트: 16분음표가 섞인 120bpm 패턴 4마디 (가장 짧은 간격 125ms)
    bar = [("kick", 0.0), ("hihat", 0.25), ("snare", 0.5), ("hihat", 0.75),
           ("kick", 1.0), ("kick", 1.125), ("hihat", 1.25), ("snare", 1.5), ("hihat", 1.75)]
    truth = [(lb, 0.2 + t + 2.0 * b) for b in range(4) for lb, t in bar]
    x = sequence([lb for lb, _ in truth], [t for _, t in truth], rng, 8.6)

    hits = engine.transcribe(x, model)
    matched = correct = 0
    worst = 0.0
    used = set()
    for lb, t in truth:
        cand = [(abs(h.time - t), i) for i, h in enumerate(hits) if i not in used and abs(h.time - t) < 0.03]
        if not cand:
            continue
        err, i = min(cand)
        used.add(i)
        matched += 1
        worst = max(worst, err)
        correct += hits[i].label == lb
    extra = len(hits) - matched
    print(f"정답 {len(truth)}개 중 감지 {matched}개, 분류 정답 {correct}개, 오검출 {extra}개, "
          f"최대 타이밍 오차 {worst * 1000:.1f}ms")

    with tempfile.TemporaryDirectory() as d:
        drums = engine.render(hits, len(x))
        engine.save_audio(os.path.join(d, "o.wav"), drums)
        engine.write_midi(os.path.join(d, "o.mid"), hits)
        data = open(os.path.join(d, "o.mid"), "rb").read()
        assert data[:4] == b"MThd" and data[14:18] == b"MTrk"
        assert int.from_bytes(data[18:22], "big") == len(data) - 22
        assert data.count(bytes([0x99])) >= len(hits)
        assert np.max(np.abs(drums)) > 0.1

    assert matched == len(truth) and extra == 0 and correct == len(truth)
    print("통과")


if __name__ == "__main__":
    main()
