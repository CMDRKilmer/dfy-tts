"""Generate a tiny synthetic dataset so the training pipeline can be smoke-tested.

The audio is deliberately *not* speech: this dataset only proves that data
loading, espeak phonemization, the VITS model, the numba monotonic aligner and
the PyTorch Lightning loop all run end to end on this machine. Do not train a
real voice on it.

    .venv/Scripts/python.exe scripts/make_smoke_data.py
    .venv/Scripts/python.exe -m piper.train fit --trainer.fast_dev_run true \
        --data.csv_path data/smoke/metadata.csv \
        --data.audio_dir data/smoke/audio \
        --data.cache_dir data/smoke/cache \
        --data.config_path data/smoke/config.json \
        --data.voice_name smoke \
        --data.espeak_voice en-us \
        --data.trim_silence false \
        --model.sample_rate 22050
"""

from __future__ import annotations

import argparse
import csv
import wave
from pathlib import Path

import numpy as np

TEXTS = [
    "Hello, this is a test of the piper training pipeline.",
    "The quick brown fox jumps over the lazy dog.",
    "Synthetic audio is only used to exercise the code path.",
    "Piper converts text into phonemes with espeak-ng.",
    "Monotonic alignment search runs on the graphics card.",
    "Training should report a loss after the first batch.",
    "This line is short.",
    "A slightly longer sentence keeps the duration distribution varied.",
    "Alignment is monotonic and never runs ahead.",
    "The model learns to map phonemes onto frames.",
]


def write_wav(path: Path, sample_rate: int, seconds: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(sample_rate * seconds)
    t = np.arange(n) / sample_rate
    freq = rng.uniform(120.0, 320.0)
    signal = 0.4 * np.sin(2 * np.pi * freq * t)
    signal += 0.1 * np.sin(2 * np.pi * freq * 2 * t + 0.5)
    signal += 0.02 * rng.normal(size=n)  # keep silero VAD from flattening it
    pcm = (np.clip(signal, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/smoke"))
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--count", type=int, default=len(TEXTS))
    args = parser.parse_args()

    audio_dir = args.out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(args.count):
        name = f"utt{i + 1:04d}.wav"
        seconds = 1.2 + 0.4 * (i % 5)  # 1.2 - 2.8 s
        write_wav(audio_dir / name, args.sample_rate, seconds, seed=i)
        rows.append((name, TEXTS[i % len(TEXTS)]))

    csv_path = args.out_dir / "metadata.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="|", lineterminator="\n")
        writer.writerows(rows)

    print(f"wrote {len(rows)} utterances")
    print(f"csv:   {csv_path}")
    print(f"audio: {audio_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
