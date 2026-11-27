"""입드럼 변환 프로토타입 CLI.

    python cli.py calibrate              마이크로 킥/스네어/하이햇을 차례로 학습
    python cli.py convert --record 10    10초 녹음해서 드럼으로 변환
    python cli.py convert beat.wav       녹음 파일을 드럼으로 변환
"""
from __future__ import annotations

import argparse
import os
import sys

import engine

MODEL_PATH = "model.npz"
KOREAN = {"kick": "킥 (예: '붐', '둠')", "snare": "스네어 (예: '카', '프슈')", "hihat": "하이햇 (예: '츠', '칫')"}


def record(seconds: float):
    import sounddevice as sd
    audio = sd.rec(int(seconds * engine.SR), samplerate=engine.SR, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


def cmd_calibrate(args):
    examples = {}
    if args.files:
        for label, path in zip(engine.LABELS, args.files):
            examples[label] = engine.load_audio(path)
    else:
        os.makedirs("calibration", exist_ok=True)
        for label in engine.LABELS:
            input(f"\n[{KOREAN[label]}] Enter를 누른 뒤 {args.seconds:.0f}초 동안 "
                  f"이 소리만 8~10번, 간격을 두고 내주세요 > ")
            print("녹음 중...")
            examples[label] = record(args.seconds)
            engine.save_audio(f"calibration/{label}.wav", examples[label])
    model, counts = engine.train(examples, args.analysis_ms, args.sensitivity)
    model.save(MODEL_PATH)
    print()
    for label, n in counts.items():
        warn = "  <- 너무 적습니다. 다시 녹음하는 게 좋습니다" if n < 4 else ""
        print(f"  {label:6s} {n}개 인식{warn}")
    print(f"\n저장: {MODEL_PATH}")


def cmd_convert(args):
    if not os.path.exists(MODEL_PATH):
        sys.exit("먼저 `python cli.py calibrate`로 내 소리를 학습시켜 주세요.")
    model = engine.Model.load(MODEL_PATH)
    if args.record:
        input(f"Enter를 누르면 {args.record:.0f}초 동안 녹음합니다 > ")
        print("녹음 중...")
        x = record(args.record)
        base = args.output or "take"
        engine.save_audio(f"{base}_input.wav", x)
    elif args.input:
        x = engine.load_audio(args.input)
        base = args.output or os.path.splitext(args.input)[0]
    else:
        sys.exit("입력 파일이나 --record 초 를 지정해 주세요.")

    hits = engine.transcribe(x, model, args.sensitivity)
    drums = engine.render(hits, len(x))
    engine.save_audio(f"{base}_drums.wav", drums)
    engine.write_midi(f"{base}.mid", hits, args.bpm)

    for h in hits:
        print(f"  {h.time:6.2f}s  {h.label:6s} vel {h.velocity}")
    print(f"\n{len(hits)}개 인식 -> {base}_drums.wav, {base}.mid")
    if args.play:
        import sounddevice as sd
        sd.play(drums, engine.SR)
        sd.wait()


def main():
    p = argparse.ArgumentParser(description="입드럼을 드럼 사운드로 바꿉니다.")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="내 킥/스네어/하이햇 소리를 학습")
    c.add_argument("--files", nargs=3, metavar=("KICK", "SNARE", "HIHAT"),
                   help="마이크 대신 녹음 파일 3개로 학습")
    c.add_argument("--seconds", type=float, default=8)
    c.add_argument("--analysis-ms", type=float, default=engine.ANALYSIS_MS,
                   help="분류에 쓰는 구간 길이. 줄일수록 실시간 지연이 짧아지지만 정확도가 떨어짐")
    c.add_argument("--sensitivity", type=float, default=1.0)
    c.set_defaults(func=cmd_calibrate)

    v = sub.add_parser("convert", help="입드럼을 드럼 wav + MIDI로 변환")
    v.add_argument("input", nargs="?")
    v.add_argument("--record", type=float, metavar="초")
    v.add_argument("-o", "--output", help="출력 파일 이름(확장자 제외)")
    v.add_argument("--bpm", type=float, default=120, help="MIDI 파일의 템포")
    v.add_argument("--sensitivity", type=float, default=1.0)
    v.add_argument("--play", action="store_true", help="변환 결과를 바로 재생")
    v.set_defaults(func=cmd_convert)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
