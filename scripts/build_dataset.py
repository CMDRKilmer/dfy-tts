"""Turn a folder of MiniMax-generated clips into a Piper training dataset.

Piper wants 22050 Hz, 16-bit, mono PCM WAVs plus a ``metadata.csv`` whose
``|``-separated rows are ``<audio file>|<text>``.

Inputs accepted (any format librosa/ffmpeg can decode):
  * ``wavs/``  with audio files, and a ``texts.txt`` / ``*.tsv`` transcript;
  * or a folder where each audio file sits next to a ``.txt`` of the same stem.

Usage
-----
    .venv/Scripts/python.exe scripts/build_dataset.py `
        --source path/to/minimax_output `
        --out data/myvoice --sample-rate 22050

The script writes ``data/myvoice/wavs/*.wav`` and ``data/myvoice/metadata.csv``.
Existing identical files are reused, so it is safe to re-run after adding more
clips.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".wma"}
# Keep a transcript when it carries Chinese or Latin letters; drop only lines that
# are pure digits/symbols, which a voice cannot learn anything from.
# Bilingual corpora need the mixed and pure-English lines kept -- the previous
# "--cjk-only" default silently deleted every one of them.
TEXT_RE = re.compile(r"[A-Za-z\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def ffmpeg_to_wav(src: Path, dst: Path, sample_rate: int) -> bool:
    """Convert without decoders installed: prefer ffmpeg, fall back to librosa."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", str(src),
            "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le",
            str(dst),
        ]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode == 0 and dst.exists():
            return True
        print(f"  ffmpeg failed on {src.name}: {res.stderr.decode(errors='replace').strip()}")
    return librosa_to_wav(src, dst, sample_rate)


def librosa_to_wav(src: Path, dst: Path, sample_rate: int) -> bool:
    try:
        import librosa
        import numpy as np
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        print(f"  no decoder for {src.name}: {exc}")
        return False
    try:
        audio, _ = librosa.load(str(src), sr=sample_rate, mono=True)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        sf.write(str(dst), pcm, sample_rate, subtype="PCM_16")
        return True
    except Exception as exc:  # pragma: no cover
        print(f"  librosa failed on {src.name}: {exc}")
        return False


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
    return frames / float(rate) if rate else 0.0


def read_text_sidecar(audio: Path) -> str | None:
    """``<stem>.txt`` next to the clip, ignoring a ``|`` table row if present."""
    for suffix in (".txt", ".lab", ".transcript"):
        side = audio.with_suffix(suffix)
        if side.exists():
            text = side.read_text(encoding="utf-8-sig").strip()
            if "|" in text and "\n" not in text:
                # looks like a whole metadata row: "name.wav|transcript"
                text = text.split("|", 1)[1].strip()
            return text
    return None


def iter_transcript_rows(source: Path):
    """Yield (stem, transcript) pairs from table-like files next to the clips."""
    candidates = sorted(
        p for p in source.iterdir()
        if p.is_file() and p.suffix.lower() in {".txt", ".tsv", ".csv", ".jsonl"}
    )
    for path in candidates:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            for sep in ("\t", "|", ",", "||"):
                if sep in line:
                    head, _, tail = line.partition(sep)
                    stem = Path(head.strip()).stem
                    transcript = tail.strip().strip('"')
                    if stem and transcript:
                        yield stem, transcript
                    break


def read_transcript_table(source: Path) -> dict[str, str]:
    """Parse ``texts.txt`` / ``*.tsv`` / ``metadata.csv`` into stem -> text."""
    table: dict[str, str] = {}
    for stem, transcript in iter_transcript_rows(source):
        table.setdefault(stem, transcript)
    return table


def normalise_text(text: str) -> str:
    text = " ".join(text.split())
    return text.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="folder with MiniMax clips")
    parser.add_argument("--out", type=Path, required=True, help="dataset folder to create")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--min-seconds", type=float, default=0.4)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument(
        "--keep-non-text", action="store_true",
        help="keep transcripts with neither Chinese nor Latin letters (pure symbols/digits)",
    )
    args = parser.parse_args()

    if not args.source.is_dir():
        print(f"source folder not found: {args.source}", file=sys.stderr)
        return 1

    wavs_dir = args.out / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    table = read_transcript_table(args.source)
    audio_files = sorted(
        p for p in args.source.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )
    if not audio_files:
        print(f"no audio files under {args.source}", file=sys.stderr)
        return 1
    print(f"found {len(audio_files)} clip(s), {len(table)} transcript row(s)")

    rows: list[tuple[str, str]] = []
    skipped = 0
    for audio in audio_files:
        text = table.get(audio.stem) or read_text_sidecar(audio)
        if not text:
            print(f"  skip (no transcript): {audio.name}")
            skipped += 1
            continue
        text = normalise_text(text)
        if not text:
            skipped += 1
            continue
        if not args.keep_non_text and not TEXT_RE.search(text):
            print(f"  skip (no Chinese or English letters): {audio.name} -> {text[:30]}")
            skipped += 1
            continue

        target = wavs_dir / f"{audio.stem}.wav"
        if not target.exists():
            if not ffmpeg_to_wav(audio, target, args.sample_rate):
                skipped += 1
                continue
        duration = wav_duration(target)
        if not (args.min_seconds <= duration <= args.max_seconds):
            print(f"  skip ({duration:.2f}s out of range): {audio.name}")
            target.unlink(missing_ok=True)
            skipped += 1
            continue
        rows.append((target.name, text))

    if not rows:
        print("nothing to write: every clip was skipped", file=sys.stderr)
        return 1

    csv_path = args.out / "metadata.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="|", lineterminator="\n")
        writer.writerows(rows)

    total = sum(wav_duration(wavs_dir / name) for name, _ in rows)
    print(f"\nwrote {len(rows)} utterance(s), skipped {skipped}")
    print(f"total audio: {total / 60:.1f} min  (avg {total / len(rows):.2f}s)")
    print(f"csv:   {csv_path}")
    print(f"wavs:  {wavs_dir}")
    if total / 60 < 5:
        print("hint: Piper fine-tuning usually wants 30+ minutes of audio; more clips help.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
