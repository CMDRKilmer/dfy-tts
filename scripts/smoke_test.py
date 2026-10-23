#!/usr/bin/env python3
"""End-to-end self check for the Piper training toolchain.

Runs the whole pipeline on synthetic data and fails loudly if any stage is
missing or broken:

  1. monotonic_align numba patch imports and matches brute force
  2. corpus audit + bilingual phonemizer unit tests
  3. dataset builder turns a folder of clips + transcripts into wavs/ + csv,
     keeping the English and code-switched lines
  4. training entry point completes fast_dev_run on the GPU
  5. ONNX export produces a usable voice
  6. synthesis: Chinese through the stock CLI, Chinese/English/code-switched
     through scripts/say.py

    .venv/Scripts/python.exe scripts/smoke_test.py
    .venv/Scripts/python.exe scripts/smoke_test.py --skip-train   # quick check

``--skip-train`` still runs steps 1-3, i.e. everything that does not need a GPU
or a checkpoint.
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

#: Clips staged for the dataset step: Chinese, English and code-switched.
#: The code-switched line is the one that used to be silently dropped.
STAGED_CLIPS = [
    ("你好，很高兴见到你。", "zh_hello.wav"),
    ("今天天气怎么样？", "zh_weather.wav"),
    ("帮我打开空调。", "zh_ac.wav"),
    ("Good morning, it is seven thirty in the morning.", "en_morning.wav"),
    ("The living room light has been turned on.", "en_light.wav"),
    ("请把卧室的 W I F I 帮我重新打开。", "mix_wifi.wav"),
]

#: Lines fed to the synthesis check, one per language mode.
SYNTH_CASES = {
    "zh": "你好，世界。",
    "en": "Good morning, it is seven thirty.",
    "mix": "打开客厅的 lamp",
}


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


def step_corpus() -> None:
    """Audit the shipped corpus and prove the bilingual phonemizer works."""
    corpus = ROOT / "data" / "text"
    if (corpus / "special.txt").exists():
        run([PY, "scripts/check_corpus.py"], "corpus audit")
    else:
        print("\n==> corpus audit\n    skipped (data/text not generated yet)")

    run([PY, "scripts/test_phonemize_bilingual.py"], "bilingual phonemizer tests")


def _stage_clips() -> Path:
    """Turn the synthetic smoke data into a MiniMax-shaped source folder."""
    src = SCRATCH / "source"
    if src.exists():
        shutil.rmtree(src)
    src.mkdir(parents=True)
    audio = ROOT / "data" / "smoke" / "audio"
    lines = []
    for i, (text, name) in enumerate(STAGED_CLIPS):
        shutil.copyfile(audio / f"utt{(i % 4) + 1:04d}.wav", src / name)
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
    check(len(rows) == len(STAGED_CLIPS),
          f"{len(STAGED_CLIPS)} metadata rows written ({len(rows)})")
    check(all("|" in row for row in rows), "rows are pipe separated")

    texts = [row.split("|", 1)[1] for row in rows]
    has_english = any(not any("\u4e00" <= c <= "\u9fff" for c in t) for t in texts)
    has_mixed = any(
        any("\u4e00" <= c <= "\u9fff" for c in t) and any("A" <= c <= "z" for c in t)
        for t in texts
    )
    check(has_english, "pure-English transcript survived filtering")
    check(has_mixed, "code-switched transcript survived filtering")

    wavs = sorted((out / "wavs").glob("*.wav"))
    check(len(wavs) == len(STAGED_CLIPS), f"{len(STAGED_CLIPS)} wav files written ({len(wavs)})")
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


def _ensure_checkpoint(dataset: Path) -> Path:
    ckpt = next((dataset / "checkpoints").rglob("*.ckpt"), None)
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
    return ckpt


def _wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def step_synth_stock_cli(onnx_path: Path, config_path: Path) -> None:
    """The stock CLI must still work for pure Chinese."""
    text_file = SCRATCH / "input_zh.txt"
    text_file.write_text(SYNTH_CASES["zh"] + "\n", encoding="utf-8")
    out_wav = SCRATCH / "synth_stock.wav"
    run(
        [PY, "-m", "piper",
         "-m", str(onnx_path), "-c", str(config_path),
         "-f", str(out_wav), "-i", str(text_file)],
        "synthesize Chinese with the stock piper CLI",
    )
    check(out_wav.stat().st_size > 1000, "stock CLI produced audio")
    check(_wav_seconds(out_wav) > 0.1, f"stock CLI duration {_wav_seconds(out_wav):.2f}s")


def step_synth_bilingual(onnx_path: Path) -> None:
    """zh, en and code-switched must all come out of scripts/say.py."""
    for tag, text in SYNTH_CASES.items():
        out_wav = SCRATCH / f"synth_{tag}.wav"
        run(
            [PY, "scripts/say.py", str(onnx_path), text, "-f", str(out_wav)],
            f"synthesize {tag!r} with say.py",
        )
        check(out_wav.exists(), f"{tag}: wav written")
        check(out_wav.stat().st_size > 1000, f"{tag}: non-empty audio")
        seconds = _wav_seconds(out_wav)
        check(seconds > 0.1, f"{tag}: duration {seconds:.2f}s")


def step_export_and_synth(dataset: Path) -> None:
    ckpt = _ensure_checkpoint(dataset)

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

    step_synth_stock_cli(onnx_path, config_path)
    step_synth_bilingual(onnx_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true", help="only run the fast checks")
    args = parser.parse_args()

    step_alignment()
    step_corpus()
    dataset = step_dataset()
    if not args.skip_train:
        step_train(dataset)
        step_export_and_synth(dataset)

    print("\nall smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
