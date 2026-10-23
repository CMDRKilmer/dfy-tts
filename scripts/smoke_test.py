#!/usr/bin/env python3
"""End-to-end self check for the Piper training toolchain.

Runs the whole pipeline on synthetic data and fails loudly if any stage is
missing or broken:

  1. monotonic_align numba patch imports and matches brute force
  2. dataset builder turns a folder of clips + transcripts into wavs/ + csv
  3. training entry point completes fast_dev_run on the GPU
  4. ONNX export + piper synthesis produces an audible-length wav

    .venv/Scripts/python.exe scripts/smoke_test.py
    .venv/Scripts/python.exe scripts/smoke_test.py --skip-train   # quick check
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
SCRATCH = ROOT / ".cache" / "smoke_test"


def run(cmd: list[str], label: str) -> None:
    print(f"\n==> {label}")
    print("    " + " ".join(cmd))
    code = subprocess.call(cmd, cwd=ROOT)
    if code != 0:
        print(f"    FAILED ({code})", file=sys.stderr)
        raise SystemExit(code)


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"    ok: {message}")
        return
    print(f"    FAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


def step_alignment() -> None:
    run([PY, "scripts/patch_monotonic_align.py"], "patch monotonic_align")
    run([PY, "scripts/test_monotonic_align.py"], "alignment correctness")


def _stage_clips() -> Path:
    """Turn the synthetic smoke data into a MiniMax-shaped source folder."""
    src = SCRATCH / "source"
    if src.exists():
        shutil.rmtree(src)
    src.mkdir(parents=True)
    audio = ROOT / "data" / "smoke" / "audio"
    texts = [
        "你好，很高兴见到你。",
        "今天天气怎么样？",
        "帮我打开空调。",
        "请为我播放一首音乐。",
    ]
    lines = []
    for i, text in enumerate(texts, start=1):
        name = f"clip{i:03d}.wav"
        shutil.copyfile(audio / f"utt{i:04d}.wav", src / name)
        lines.append(f"{name}|{text}\n")
    (src / "texts.txt").write_text("".join(lines), encoding="utf-8")
    return src


def step_dataset() -> Path:
    src = _stage_clips()
    out = SCRATCH / "dataset"
    if out.exists():
        shutil.rmtree(out)
    run(
        [PY, "scripts/build_dataset.py", "--source", str(src), "--out", str(out)],
        "build dataset",
    )
    rows = (out / "metadata.csv").read_text(encoding="utf-8").strip().splitlines()
    check(len(rows) == 4, f"4 metadata rows written ({len(rows)})")
    check(all("|" in row for row in rows), "rows are pipe separated")
    wavs = sorted((out / "wavs").glob("*.wav"))
    check(len(wavs) == 4, f"4 wav files written ({len(wavs)})")
    return out


def step_train(dataset: Path) -> None:
    run(
        [
            PY, "scripts/train_voice.py",
            "--dataset", str(dataset),
            "--voice-name", "smoke_test",
            "--batch-size", "2",
            "--fast-dev-run",
        ],
        "fast_dev_run on GPU",
    )


def step_export_and_synth(dataset: Path) -> None:
    ckpt = next(
        (dataset / "checkpoints").rglob("*.ckpt"),
        None,
    )
    if ckpt is None:
        # fast_dev_run writes no checkpoints; do a 1-epoch run instead
        run(
            [
                PY, "scripts/train_voice.py",
                "--dataset", str(dataset),
                "--voice-name", "smoke_test",
                "--batch-size", "2",
                "--epochs", "1",
            ],
            "1 epoch to produce a checkpoint",
        )
        ckpt = next((dataset / "checkpoints").rglob("last.ckpt"))
    check(ckpt.exists(), f"checkpoint present ({ckpt.name})")

    export_dir = SCRATCH / "export"
    run(
        [
            PY, "scripts/export_voice.py",
            "--checkpoint", str(ckpt),
            "--config", str(dataset / "config.json"),
            "--out", str(export_dir),
            "--voice-name", "smoke_test",
        ],
        "export ONNX",
    )
    onnx_path = export_dir / "smoke_test.onnx"
    config_path = export_dir / "smoke_test.onnx.json"
    check(onnx_path.exists(), "onnx model written")
    check(config_path.exists(), "onnx config written")

    text_file = SCRATCH / "input.txt"
    text_file.write_text("你好，世界。\n", encoding="utf-8")
    out_wav = SCRATCH / "synth.wav"
    run(
        [
            PY, "-m", "piper",
            "-m", str(onnx_path),
            "-c", str(config_path),
            "-f", str(out_wav),
            "-i", str(text_file),
        ],
        "synthesize with the exported voice",
    )
    with wave.open(str(out_wav), "rb") as wav:
        seconds = wav.getnframes() / wav.getframerate()
        rate = wav.getframerate()
    check(out_wav.stat().st_size > 1000, "synthesis produced audio")
    check(seconds > 0.1, f"synth duration {seconds:.2f}s at {rate} Hz")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true", help="only run the fast checks")
    args = parser.parse_args()

    step_alignment()
    dataset = step_dataset()
    if not args.skip_train:
        step_train(dataset)
        step_export_and_synth(dataset)

    print("\nall smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
