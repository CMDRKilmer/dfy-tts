"""Clone a reference voice with Qwen3-TTS and generate the whole training corpus.

Qwen3-TTS Base does zero-shot cloning from a few seconds of audio plus the text
that audio contains. That turns a single reference clip into 30+ minutes of
training data for a Piper fine-tune, without a cloud service.

    .venv/Scripts/python.exe scripts/qwen_clone_generate.py `
        --ref-audio data/target/astrbot_real_voice.mp3 `
        --ref-text "哼！本小姐才不是大肥鱼呢，这是丰满的流线型身材。米饭煮好了哦，快来吃。" `
        --limit 60 --sample

Output is the same shape ``build_dataset.py`` expects: numbered wavs plus
``texts.txt`` (``name.wav|<text>``), with ``--resume`` so an interrupted run
keeps its work.

Reference audio requirements
----------------------------
* 3 seconds or more of clean speech (Qwen's docs call out 3 s as the minimum).
* ``--ref-text`` must be **exactly** what the clip says -- not a paraphrase and
  not a longer script. A mismatch makes the clone reproduce the wrong prosody,
  so the reference is validated first (see ``check_reference``).

Requires the ``qwen-tts`` package, which is not in ``requirements.txt``: it is
only needed for this bootstrap step.

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
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
CATEGORY_ORDER = [
    "special", "polyphonic", "tone_change", "numbers", "home",
    "daily", "anime", "literary", "reading", "mixed",
]


def corpus_lines(limit: int, sample: bool = False,
                 english_share: float = 0.30) -> list[str]:
    """Read the corpus, stratified by language when ``sample`` is set.

    ``mixed`` -- where all the English lives -- is the last category, so a plain
    ``--limit`` never reaches it. ``sample`` keeps a fixed English share so a
    short run still exercises the bilingual path.
    """
    per_cat: dict[str, list[str]] = {}
    seen: set[str] = set()
    for cat in CATEGORY_ORDER:
        path = TEXT_DIR / f"{cat}.txt"
        if not path.exists():
            continue
        bucket: list[str] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            text = raw.strip()
            if text and text not in seen:
                seen.add(text)
                bucket.append(text)
        per_cat[cat] = bucket

    def is_english(text: str) -> bool:
        return not any("\u4e00" <= c <= "\u9fff" for c in text)

    if sample and limit > 0:
        pool = [t for cat in CATEGORY_ORDER for t in per_cat.get(cat, [])]
        english = [t for t in pool if is_english(t)]
        chinese = [t for t in pool if not is_english(t)]
        want_en = min(len(english), max(1, round(limit * english_share)))
        want_zh = min(len(chinese), limit - want_en)
        picked: list[str] = []
        i = j = 0
        while len(picked) < want_en + want_zh:
            if i < want_en:
                picked.append(english[i]); i += 1
            if j < want_zh and len(picked) < want_en + want_zh:
                picked.append(chinese[j]); j += 1
        print(f"sampled {len(picked)} lines: {want_en} English, {want_zh} Chinese")
        return picked

    lines = [t for cat in CATEGORY_ORDER for t in per_cat.get(cat, [])]
    return lines[:limit] if limit > 0 else lines


def language_of(text: str) -> str:
    return "Chinese" if any("\u4e00" <= c <= "\u9fff" for c in text) else "English"


def check_reference(ref_audio: Path, ref_text: str) -> None:
    """Fail early on a reference clip that cloning cannot use.

    A wrong ``ref_text`` is the single most common reason a clone sounds like a
    different speaker, so the script refuses to proceed silently: it measures the
    clip and compares its expected speech length against the transcript.
    """
    import librosa
    import numpy as np

    if not ref_audio.exists():
        raise SystemExit(f"reference audio not found: {ref_audio}")

    y, sr = librosa.load(str(ref_audio), sr=22050, mono=True)
    duration = len(y) / sr
    if duration < 2.5:
        raise SystemExit(
            f"reference audio is only {duration:.2f}s; Qwen3-TTS needs about 3s "
            "of clean speech to clone reliably")

    cjk = sum(1 for c in ref_text if "\u4e00" <= c <= "\u9fff")
    latin_words = len([w for w in ref_text.split() if w.strip()])
    units = cjk if cjk else latin_words
    rate = units / duration
    print(f"reference: {duration:.2f}s, {units} speech units -> {rate:.1f}/s")

    # Mandarin runs about 3.5-6.5 characters per second; anything outside that
    # band means the transcript and the clip do not match.
    if cjk and not 3.0 <= rate <= 7.5:
        raise SystemExit(
            f"ref_text has {cjk} characters over {duration:.2f}s ({rate:.1f}/s), "
            "which is outside the plausible 3.0-7.5 range. The transcript "
            "probably is not the text of this clip -- re-check it.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref-audio", type=Path, required=True)
    ap.add_argument("--ref-text", required=True,
                    help="exactly what the reference clip says")
    ap.add_argument("--limit", type=int, default=0, help="0 = the whole corpus")
    ap.add_argument("--sample", action="store_true",
                    help="stratify by language so English is included")
    ap.add_argument("--english-share", type=float, default=0.30)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "clone_source")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    check_reference(args.ref_audio, args.ref_text)

    lines = corpus_lines(args.limit, sample=args.sample,
                         english_share=args.english_share)
    if not lines:
        print("corpus is empty; run scripts/make_corpus.py first", file=sys.stderr)
        return 1
    zh = sum(1 for t in lines if language_of(t) == "Chinese")
    print(f"corpus lines: {len(lines)}  (Chinese {zh}, English {len(lines) - zh})")

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

    print("\nloading model...")
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        args.model, device_map="cuda:0", dtype=torch.bfloat16)
    print(f"model ready in {time.time() - t0:.1f}s")

    rows: dict[str, str] = dict(existing)
    made = skipped = failed = 0
    for i, text in enumerate(lines):
        name = f"clip{i + 1:04d}.wav"
        target = args.out / name
        if args.resume and name in existing and target.exists():
            skipped += 1
            continue
        try:
            t0 = time.time()
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language_of(text),
                ref_audio=str(args.ref_audio),
                ref_text=args.ref_text,
            )
            sf.write(str(target), wavs[0], sr)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  [{i + 1}/{len(lines)}] FAILED {text[:24]!r}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        rows[name] = text
        made += 1
        manifest_path.write_text(
            "\n".join(f"{n}|{t}" for n, t in sorted(rows.items())) + "\n",
            encoding="utf-8")
        print(f"  [{i + 1}/{len(lines)}] {time.time() - t0:5.2f}s  {text[:26]}")

    manifest_path.write_text(
        "\n".join(f"{n}|{t}" for n, t in sorted(rows.items())) + "\n",
        encoding="utf-8")
    summary = {
        "model": args.model,
        "ref_audio": str(args.ref_audio),
        "ref_text": args.ref_text,
        "requested": len(lines),
        "made": made, "skipped": skipped, "failed": failed,
        "clips_present": len(rows),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {made}, skipped {skipped}, failed {failed}")
    print(f"clips:    {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
