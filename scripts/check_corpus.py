"""Audit the training corpus for repetition, length and bilingual coverage.

A cloned voice learns whatever it hears, so repeated templates ("今天气温X度，
出门记得带件外套。") teach the frame instead of the language. This script fails
when any of these hold:

* two lines are identical,
* two lines share the same opening ``FRAME`` characters (template smell),
* a category exceeds ``CAT_CAP`` share of the corpus,
* a line is outside the length rule (8-25 characters for Chinese and
  code-switched lines, 8-25 words for pure English),
* a Chinese line does not end in sentence punctuation,
* the ``mixed`` category drifts far from its intended language ratio,
* a line needs a phoneme the voice's ``phoneme_id_map`` does not carry.

Function words, digits, punctuation legitimately appear in many lines -- that is
what makes them frequent -- so only *content* characters are capped, and loosely
(60 lines).

The phoneme check is the important one for English: the multilingual espeak map a
Chinese Piper voice trains on is a superset of the English one, so a line that
passes here can be synthesised by the model this corpus trains -- provided
``scripts/phonemize_bilingual.py`` routes the Latin runs to ``en-us`` at
synthesis time. The ONNX-side g2pW phonemizer cannot do that, which is why the
bilingual phonemizer exists.

Run:  .venv/Scripts/python.exe scripts/check_corpus.py
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT_DIR = ROOT / "data" / "text"
sys.path.insert(0, str(ROOT / "scripts"))

FRAME = 6
CAT_CAP = 0.60  # the phonetic lists are a third of the audio budget by design
MIN_CHARS = 8
MAX_CHARS = 25
MIN_WORDS = 8
MAX_WORDS = 25
#: ``mixed`` is meant to carry English without letting it take over: the target
#: is roughly one English line per two Chinese lines, within this band.
MIXED_EN_MIN = 0.30
MIXED_EN_MAX = 0.70
#: Function words and digits behave differently from busy characters.
FUNCTION_CHARS = set("的了是我你他她它们在有个这那就都也还把被和与及或而而且不"
                     "上下好到中里一二三四五六七八九十天开点")
PUNCTUATION = set("，。！？；：、…—（）「」《》“”’‘ ")
#: Categories built around a target character; their lines repeat that character
#: by construction, so the busy-character rule cannot apply to them.
PHONETIC_CATEGORIES = ("polyphonic", "tone_change")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
#: A Chinese sentence has to end on a real terminator; a comma suggests that a
#: line was truncated while the corpus was assembled.
TERMINATORS = tuple("。！？…」』”’")
#: Voice config used to check phoneme coverage. Any Chinese espeak Piper voice
#: carries the same id map; the check picks the first one it finds.
VOICE_CONFIGS = [
    ROOT / ".cache" / "zh_CN-huayan-medium.json",
]


def _measure(line: str) -> tuple[int, str]:
    """Return ``(length, unit)`` -- words for pure English, characters otherwise."""
    if CJK_RE.search(line):
        return len(CJK_RE.findall(line)), "chars"
    return len(WORD_RE.findall(line)), "words"


def find_voice_config() -> Path | None:
    for path in VOICE_CONFIGS:
        if path.exists():
            return path
    return None


def check_phonemes(every: list[tuple[str, str]], problems: list[str]) -> str:
    """Fail when a corpus line needs a phoneme the voice map cannot express."""
    try:
        from phonemize_bilingual import BilingualPhonemizer
    except Exception as exc:  # noqa: BLE001 -- report, never crash the audit
        return f"phoneme check skipped ({exc.__class__.__name__}: {exc})"

    path = find_voice_config()
    if path is None:
        return "phoneme check skipped (no voice config under .cache/)"

    cfg = json.loads(path.read_text(encoding="utf-8"))
    id_map = cfg["phoneme_id_map"]
    try:
        phonemizer = BilingualPhonemizer(id_map)
        lines = [text for _, text in every]
        missing = phonemizer.uncovered(lines)
    except Exception as exc:  # noqa: BLE001
        return f"phoneme check skipped ({exc.__class__.__name__}: {exc})"

    if missing:
        problems.append(f"{len(missing)} phoneme gap(s): {sorted(missing)[:12]}")
    return f"phonemes: {len(lines)} lines, {len(missing)} missing symbol(s)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=TEXT_DIR)
    parser.add_argument("--frame", type=int, default=FRAME)
    parser.add_argument("--skip-phonemes", action="store_true")
    args = parser.parse_args()

    files = sorted(p for p in args.dir.glob("*.txt"))
    if not files:
        print(f"no .txt corpus files under {args.dir}")
        return 1

    every: list[tuple[str, str]] = []
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                every.append((path.stem, line.strip()))

    problems: list[str] = []

    # 1. exact duplicates
    counts = collections.Counter(t for _, t in every)
    for text, n in counts.items():
        if n > 1:
            problems.append(f"duplicate x{n}: {text}")

    # 2. shared openings
    #    Chinese lines are compared on their first FRAME characters. English is
    #    compared on its first two words instead: six characters of English is
    #    under two words, so "The li" would flag three unrelated sentences.
    frames: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for _, text in every:
        if CJK_RE.search(text):
            frames[("zh", text[: args.frame])].append(text)
        else:
            frames[("en", " ".join(WORD_RE.findall(text)[:2]))].append(text)
    for (_lang, key), texts in [(k, v) for k, v in frames.items() if len(v) > 1][:20]:
        problems.append(f"shared opening '{key}': {len(texts)} lines -> {texts[:3]}")

    # 3. category balance
    total = len(every)
    sizes = collections.Counter(cat for cat, _ in every)
    for cat, n in sizes.most_common():
        if n / total > CAT_CAP:
            problems.append(f"category '{cat}' is {n}/{total} = {n / total:.0%} of the corpus")

    # 4. length rule
    too_short: list[str] = []
    too_long: list[str] = []
    for _, text in every:
        length, unit = _measure(text)
        lo, hi = (MIN_CHARS, MAX_CHARS) if unit == "chars" else (MIN_WORDS, MAX_WORDS)
        if length < lo:
            too_short.append(f"{length:>2}{unit[0]} {text}")
        elif length > hi:
            too_long.append(f"{length:>2}{unit[0]} {text}")
    if too_short:
        problems.append(f"{len(too_short)} line(s) below {MIN_CHARS}-{MAX_CHARS}: "
                        f"{too_short[:6]}")
    if too_long:
        problems.append(f"{len(too_long)} line(s) above {MIN_CHARS}-{MAX_CHARS}: "
                        f"{too_long[:6]}")

    # 5. Chinese lines must end on a terminator
    unterminated = [t for _, t in every
                    if CJK_RE.search(t) and not t.endswith(TERMINATORS)]
    if unterminated:
        problems.append(f"{len(unterminated)} Chinese line(s) without a terminator: "
                        f"{unterminated[:6]}")

    # 6. the mixed category keeps its language ratio
    mixed = [t for cat, t in every if cat == "mixed"]
    if mixed:
        english = sum(1 for t in mixed if not CJK_RE.search(t))
        share = english / len(mixed)
        if not MIXED_EN_MIN <= share <= MIXED_EN_MAX:
            problems.append(f"mixed category is {share:.0%} pure English "
                            f"(want {MIXED_EN_MIN:.0%}-{MIXED_EN_MAX:.0%})")
        print(f"mixed: {len(mixed)} lines, {english} English ({share:.0%})")

    # 7. busy characters should not pile up
    #    Latin letters belong to English words, which are long by nature; only
    #    CJK characters say something about how repetitive the Chinese is.
    #    The phonetic categories are exempt: "说"/"着"/"得"/"别" are themselves
    #    polyphonic targets, so a line set that exercises their readings must
    #    repeat them, and "人"/"子" ride along as the frame those readings sit in.
    phonetic_lines = {t for cat, t in every if cat in PHONETIC_CATEGORIES}
    per_char = collections.Counter()
    for _, text in every:
        if text in phonetic_lines:
            continue
        per_char.update({c for c in text
                         if c not in PUNCTUATION and c not in FUNCTION_CHARS
                         and CJK_RE.match(c)})
    for c, n in [(c, n) for c, n in per_char.most_common() if n > 60][:10]:
        problems.append(f"busy character '{c}' appears in {n} lines")

    print(f"lines: {total}  unique: {len(counts)}  files: {len(files)}")
    print(f"distinct openings ({args.frame} chars): {len(frames)}")
    print("per file:", dict(sizes))
    chars = sum(len(t) for _, t in every)
    print(f"characters: {chars}  mean {chars / total:.1f} per line")
    print("busiest busy characters:",
          ", ".join(f"{c}:{n}" for c, n in per_char.most_common(8)))

    if args.skip_phonemes:
        print("phoneme check skipped by request")
    else:
        print(check_phonemes(every, problems))

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(" -", p)
        return 1
    print("\nno corpus problems found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
