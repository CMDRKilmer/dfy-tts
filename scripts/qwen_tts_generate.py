"""Generate MiniMax-shaped training audio with Qwen3-TTS, fully offline.

Why: the corpus in ``data/text/`` needs real speech before a Piper fine-tune can
produce anything listenable. Qwen3-TTS is multilingual (Chinese + English) and
runs on an 11 GB card, so it can stand in for a cloud voice-cloning service while
the pipeline is being proven.

Output layout is exactly what ``scripts/build_dataset.py`` expects::

    <out>/clip0001.wav ...      # 24 kHz mono, converted later
    <out>/texts.txt             # "clip0001.wav|<the text>"

Usage
-----
    # 3 lines, to check the model loads
    .venv/Scripts/python.exe scripts/qwen_tts_generate.py --limit 3

    # a first real batch: 60 lines of the corpus
    .venv/Scripts/python.exe scripts/qwen_tts_generate.py --limit 60

    # everything (long; ~1380 requests)
    .venv/Scripts/python.exe scripts/qwen_tts_generate.py

Requires the ``qwen-tts`` extra, which is *not* in ``requirements.txt`` because
it is only needed for this bootstrap step:

    uv pip install --python .venv/Scripts/python.exe -U qwen-tts
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT_DIR = ROOT / "data" / "text"
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
#: Ordered the same way as the JSONL: the first N lines are the must-haves.
CATEGORY_ORDER = [
    "special", "polyphonic", "tone_change", "numbers", "home",
    "daily", "anime", "literary", "reading", "mixed",
]


def corpus_lines(limit: int) -> list[str]:
    """Read the corpus in category order, de-duplicated."""
    lines: list[str] = []
    seen: set[str] = set()
    for cat in CATEGORY_ORDER:
        path = TEXT_DIR / f"{cat}.txt"
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            text = raw.strip()
            if text and text not in seen:
                seen.add(text)
                lines.append(text)
    return lines[:limit] if limit > 0 else lines


def language_of(text: str) -> str:
    """Pick the synthesis language for one corpus line."""
    has_cjk = any("\u4e00" <= c <= "\u9fff" for c in text)
    # Code-switched lines lean on the Chinese voice; Qwen reads the Latin run
    # in a Chinese accent, which is exactly what the Piper corpus expects
    # (espeak's cmn voice behaves the same way).
    return "Chinese" if has_cjk else "English"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=0, help="0 = the whole corpus")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "qwen_source")
    ap.add_argument("--speaker", default="Serena", help="Qwen3-TTS built-in voice")
    ap.add_argument("--instruct", default="", help="optional natural-language style hint")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--resume", action="store_true", default=True,
                    help="skip clips that already exist (default on)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    lines = corpus_lines(args.limit)
    if not lines:
        print("corpus is empty; run scripts/make_corpus.py first", file=sys.stderr)
        return 1
    zh = sum(1 for t in lines if language_of(t) == "Chinese")
    print(f"corpus lines: {len(lines)}  (Chinese {zh}, English {len(lines) - zh})")
    print(f"voice: {args.speaker}   model: {args.model}")

    if args.dry_run:
        for text in lines[:5]:
            print(f"  [{language_of(text):7s}] {text}")
        print(f"  ... {len(lines)} total")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "texts.txt"

    existing: dict[str, str] = {}
    if args.resume and manifest_path.exists():
        for row in manifest_path.read_text(encoding="utf-8").splitlines():
            if "|" in row:
                name, _, text = row.partition("|")
                existing[name] = text

    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    print("\nloading model (first run downloads a few GB)...")
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    print(f"model ready in {time.time() - t0:.1f}s")

    rows: dict[str, str] = dict(existing)
    made = skipped = failed = 0
    for i, text in enumerate(lines):
        name = f"clip{i + 1:04d}.wav"
        target = args.out / name
        if args.resume and name in existing and target.exists():
            skipped += 1
            continue
        kwargs = dict(
            text=text,
            language=language_of(text),
            speaker=args.speaker,
        )
        if args.instruct:
            kwargs["instruct"] = args.instruct
        try:
            t0 = time.time()
            wavs, sr = model.generate_custom_voice(**kwargs)
            sf.write(str(target), wavs[0], sr)
        except Exception as exc:  # noqa: BLE001 -- keep going, report at the end
            failed += 1
            print(f"  [{i + 1}/{len(lines)}] FAILED {text[:24]!r}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        rows[name] = text
        made += 1
        # Rewrite the manifest every clip so an interrupted run keeps its work.
        manifest_path.write_text(
            "\n".join(f"{n}|{t}" for n, t in sorted(rows.items())) + "\n",
            encoding="utf-8",
        )
        print(f"  [{i + 1}/{len(lines)}] {time.time() - t0:5.2f}s  {text[:26]}")

    manifest_path.write_text(
        "\n".join(f"{n}|{t}" for n, t in sorted(rows.items())) + "\n",
        encoding="utf-8",
    )
    summary = {
        "model": args.model,
        "speaker": args.speaker,
        "requested": len(lines),
        "made": made,
        "skipped": skipped,
        "failed": failed,
        "clips_present": len(rows),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {made}, skipped {skipped}, failed {failed}")
    print(f"clips:    {args.out}")
    print(f"manifest: {manifest_path}")
    if failed:
        print("some clips failed; re-run the same command to retry only those",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
