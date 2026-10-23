#!/usr/bin/env python3
"""Turn a Qwen-cloned corpus into a MOSS-TTS-Nano fine-tune dataset.

MOSS-TTS-Nano micro-fine-tune consumes a JSONL where every record carries the
target reference audio as the speaker identity:

    {"audio": "<utterance.wav>", "text": "<utterance text>",
     "ref_audio": "<reference_voice.wav>"}

This script produces that from the output of ``qwen_clone_generate.py``:

    data/clone_source/clip0001.wav ...      (the utterances)
    data/clone_source/texts.txt             (name.wav|<utterance>)
    data/target/astrbot_real_voice.mp3      (the reference identity)

It also resamples the reference to 48 kHz (MOSS's audio codec is 48 kHz) into
``moss_dataset/ref_audio.wav``, and links each generated clip (converting to
48 kHz too, since the codec expects it). The JSONL points at the converted wavs.

    .venv-moss/Scripts/python.exe scripts/moss_prepare_data.py

Writes under ``data/moss_dataset/``:

    moss_intl.jsonl      # before codec precompute, per prepared_data.py input
    ref_audio.wav        # 48 kHz mono/mono reference
    wav48/               # converted 48 kHz copies of every utterance
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import soundfile as sf


def to_48k(src: Path, dst: Path, channels: int = 1) -> None:
    """Resample an audio file to 48 kHz (and optionally stereo) via librosa/scipy."""
    import numpy as np
    import librosa

    y, sr = librosa.load(str(src), sr=48000, mono=(channels == 1))
    if channels == 2 and y.ndim == 1:
        y = np.stack([y, y], axis=-1)
    sf.write(str(dst), y, 48000)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", type=Path, default=Path("data/clone_source"))
    ap.add_argument("--target", type=Path, default=Path("data/target/astrbot_real_voice.mp3"))
    ap.add_argument("--out", type=Path, default=Path("data/moss_dataset"))
    ap.add_argument("--speaker-ref", type=Path, default=None,
                    help="use this as the reference; default: --target resampled")
    args = ap.parse_args()

    texts: dict[str, str] = {}
    texts_txt = args.source / "texts.txt"
    if not texts_txt.exists():
        print(f"missing {texts_txt}", file=sys.stderr)
        return 1
    for row in texts_txt.read_text(encoding="utf-8").splitlines():
        if "|" in row:
            name, _, text = row.partition("|")
            texts[name] = text

    wavs = sorted(args.source.glob("*.wav"))
    missing = [w.name for w in wavs if w.name not in texts]
    if missing:
        print(f"{len(missing)} wav(s) have no transcript: {missing[:5]}", file=sys.stderr)
        return 1
    print(f"found {len(wavs)} utterances with transcripts")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "wav48").mkdir(parents=True, exist_ok=True)

    # Reference identity -> 48 kHz
    ref_src = args.speaker_ref or args.target
    ref_out = args.out / "ref_audio.wav"
    to_48k(ref_src, ref_out, channels=1)
    print(f"reference: {ref_src.name} -> {ref_out} (48 kHz mono)")

    # Convert every utterance to 48 kHz and write the JSONL
    records = []
    for w in wavs:
        w48 = args.out / "wav48" / w.name
        to_48k(w, w48, channels=1)
        records.append({
            "audio": str(w48.resolve()),
            "text": texts[w.name],
            "ref_audio": str(ref_out.resolve()),
        })

    records.sort(key=lambda r: Path(r["audio"]).name)
    jsonl = args.out / "moss_intl.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    dur = sum(sf.info(rec["audio"]).duration for rec in records)
    print(f"wrote {len(records)} records -> {jsonl}")
    print(f"total audio: {dur / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())