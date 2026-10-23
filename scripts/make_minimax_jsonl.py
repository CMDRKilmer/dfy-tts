"""Assemble the MiniMax voice-cloning script and its manifest.

MiniMax voice clone accepts one JSONL file per run with one entry per line:

    {"text": "<sentence>", "audio_setting": {"sample_rate": 32000, "bitrate": 128000, "format": "mp3"}}

Entries are kept in category order so the first N lines are a balanced sample
when only a subset can be generated per run:

    special, polyphonic, tone_change, numbers, home, daily, anime, literary,
    reading, mixed

``polyphonic`` and ``tone_change`` carry the pronunciation traps (多音字 and
声调变调), and ``mixed`` is where English lives; all three are placed early so a
partial run still teaches them.

    .venv/Scripts/python.exe scripts/make_corpus.py      # text files
    .venv/Scripts/python.exe scripts/make_minimax_jsonl.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT_DIR = ROOT / "data" / "text"
AUDIO_SETTING = {"sample_rate": 32000, "bitrate": 128000, "format": "mp3"}

#: Order matters: a prefix of the corpus stays balanced (see module docstring).
CATEGORY_ORDER = [
    "special", "polyphonic", "tone_change", "numbers", "home",
    "daily", "anime", "literary", "reading", "mixed",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-dir", type=Path, default=TEXT_DIR)
    parser.add_argument("--out", type=Path, default=TEXT_DIR / "minimax_tts.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 = every line")
    parser.add_argument("--shuffle", action="store_true", help="ignore category order")
    args = parser.parse_args()

    records: list[dict] = []
    missing = [c for c in CATEGORY_ORDER if not (args.text_dir / f"{c}.txt").exists()]
    if missing:
        print(f"missing category files: {missing} (run scripts/make_corpus.py first)")
        return 1

    seen: set[str] = set()
    for cat in CATEGORY_ORDER:
        text = (args.text_dir / f"{cat}.txt").read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line or line in seen:
                continue
            seen.add(line)
            records.append({"text": line, "audio_setting": dict(AUDIO_SETTING)})

    if args.shuffle:
        import random

        random.Random(20261022).shuffle(records)
    if args.limit > 0:
        records = records[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    chars = sum(len(r["text"]) for r in records)
    print(f"wrote {len(records)} entries -> {args.out}")
    print(f"total characters: {chars}  (est. {chars / 4.5 / 60:.1f} min at 4.5 chars/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
