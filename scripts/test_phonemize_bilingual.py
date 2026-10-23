"""Unit tests for the bilingual phonemizer -- no model download required.

    .venv/Scripts/python.exe scripts/test_phonemize_bilingual.py

Checks the three properties the fine-tune depends on:

1. ``split_segments`` never loses or reorders a character of Chinese text;
2. the espeak path is *byte-identical* to ``EspeakPhonemizer.phonemize('cmn', ...)``
   for pure Chinese input -- a Chinese voice must keep the phonemes it trained on;
3. every phoneme the ``en-us`` voice can emit for the corpus's English lines is
   already a key in the voice's ``phoneme_id_map`` -- nothing gets silently
   dropped at export time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from phonemize_bilingual import BilingualPhonemizer, split_segments  # noqa: E402

VOICE_CONFIG = HERE.parent / ".cache" / "zh_CN-huayan-medium.json"


def test_split_preserves_chinese() -> None:
    """Rebuilding the segments must reproduce the input exactly.

    Characters are preserved; the *spacing between Latin tokens* is not, because
    every English chunk is stripped before phonemizing (espeak would otherwise
    read a leading space as a word break). So compare with Latin spaces removed.
    """
    cases = [
        "打开客厅的 lamp",
        "请把 W I F I 打开。",
        "今天的 update 下载完了，要现在装吗。",
        "Bedroom 的 air purifier 该换滤芯了。",
        "没有英文的一句话。",
        "Only english here.",
    ]
    for text in cases:
        rebuilt = "".join(chunk for _lang, chunk in split_segments(text))
        # Normalise whitespace so "W I F I" and "WIFI" compare equal on content.
        content = lambda s: "".join(s.split())  # noqa: E731
        assert content(rebuilt) == content(text), f"{text!r} -> {rebuilt!r}"
        langs = [lang for lang, _ in split_segments(text)]
        assert set(langs) <= {"zh", "en"}, langs
    print(f"ok  split preserves {len(cases)} strings")


def test_chinese_parity() -> None:
    """Pure Chinese must phonemize exactly like the training pipeline.

    One intentional difference: espeak-ng's sentence splitter does not fire after
    Chinese text in the ``cmn`` voice, so the training path sometimes packs two
    sentences into one group. The bilingual path splits on terminal punctuation
    instead, so it is compared phoneme-by-phoneme over the whole line rather than
    sentence-by-sentence.
    """
    from piper.phonemize_espeak import EspeakPhonemizer

    if not VOICE_CONFIG.exists():
        print(f"skip chinese parity ({VOICE_CONFIG} missing)")
        return
    cfg = json.loads(VOICE_CONFIG.read_text(encoding="utf-8"))
    ph = BilingualPhonemizer(cfg["phoneme_id_map"])
    es = EspeakPhonemizer()

    corpus = HERE.parent / "data" / "text"
    lines: list[str] = []
    for name in ("special", "numbers", "home", "daily", "anime", "literary", "reading"):
        path = corpus / f"{name}.txt"
        if not path.exists():
            continue
        lines.extend(l.strip() for l in path.read_text(encoding="utf-8").splitlines()
                     if l.strip())
    if not lines:
        print("skip chinese parity (corpus not generated)")
        return
    sample = lines[:: max(1, len(lines) // 300)][:300]
    bad = 0
    for line in sample:
        mine = ph.phonemize(line)
        theirs = es.phonemize("cmn", line)
        # Lines that contain Latin letters are almost all Chinese； those are
        # compared on their Chinese runs only, because espeak-ng's cmn voice
        # mis-reads the letters (``XZ`` becomes ``x-ed``) while the bilingual
        # path deliberately says them as English.
        if any(lang == "en" for lang, _ in split_segments(line)):
            mine = [s for s, lang in _chinese_runs(ph, line)]
            theirs = [s for s, lang in _chinese_runs_train(es, line)]
        if [p for sent in mine for p in sent] != [p for sent in theirs for p in sent]:
            bad += 1
            print("  mismatch:", line)
            print("    mine  :", "".join(p for s in mine for p in s))
            print("    train :", "".join(p for s in theirs for p in s))
    assert bad == 0, f"{bad}/{len(sample)} Chinese lines differ from the training path"
    print(f"ok  Chinese phoneme parity on {len(sample)} lines")


def _chinese_runs(ph, line: str):
    """Phonemize only the Chinese chunks of a mixed line, in order."""
    out = []
    for lang, chunk in split_segments(line):
        if lang == "zh":
            out.append(([p for s in ph._zh_sentences(chunk) for p in s], "zh"))
    return out


def _chinese_runs_train(es, line: str):
    """Same, using the training phonemizer directly (strips Latin runs first)."""
    out = []
    for lang, chunk in split_segments(line):
        if lang == "zh":
            out.append(([p for s in es.phonemize("cmn", chunk) for p in s], "zh"))
    return out


def test_english_symbols_covered() -> None:
    """Every en-us phoneme must already be in the voice's table."""
    if not VOICE_CONFIG.exists():
        print(f"skip english coverage ({VOICE_CONFIG} missing)")
        return
    cfg = json.loads(VOICE_CONFIG.read_text(encoding="utf-8"))
    table = set(cfg["phoneme_id_map"])

    here = HERE / "bilingual_corpus.py"
    sys.path.insert(0, str(HERE))
    import bilingual_corpus as bc  # noqa: E402

    lines = [l.strip() for l in bc.ENGLISH.split("\n") if l.strip()]
    lines += [l.strip() for l in bc.MIXED.split("\n") if l.strip()]

    ph = BilingualPhonemizer(cfg["phoneme_id_map"])
    missing: dict[str, list[str]] = {}
    for line in lines:
        for seg_lang, chunk in split_segments(line):
            if seg_lang != "en":
                continue
            from piper.phonemize_espeak import EspeakPhonemizer
            for phonemes in EspeakPhonemizer().phonemize("en-us", chunk):
                for p in phonemes:
                    if p not in table:
                        missing.setdefault(p, []).append(chunk)
    assert not missing, f"English phonemes missing from the table: {sorted(missing)}"
    print(f"ok  english coverage on {len(lines)} lines, 0 missing symbols")


def main() -> int:
    test_split_preserves_chinese()
    test_chinese_parity()
    test_english_symbols_covered()
    print("all bilingual checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
