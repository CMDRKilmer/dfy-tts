"""Phonemize Chinese *and* English the way the training pipeline does.

Why this exists
---------------

``piper.train`` phonemizes with :class:`piper.phonemize_espeak.EspeakPhonemizer`
(espeak-ng, language ``cmn``). For English text espeak-ng hands back IPA that the
``zh_CN`` phoneme table already covers -- probing ``zh_CN-huayan-medium`` showed
**zero** missing symbols for both ``espeak('cmn', english)`` (45 phonemes) and
``espeak('en-us', english)`` (49 phonemes). So a Chinese Piper voice can be
fine-tuned on bilingual text with the stock 256-symbol embedding: no merged
table, no re-mapped ids, no size mismatch on ``load_state_dict``.

The problem is the *other* end. ``piper.phonemize_chinese`` -- the phonemizer
compiled into the ONNX graph by ``piper.train.export_onnx`` -- runs g2pW, which
only knows Chinese. An English word comes back as ``None`` and is silently
dropped, so the exported voice cannot say a single word it was trained on.

The fix is a runtime phonemizer, not a retrained table:

    zh:  espeak-ng('cmn', segment)   = exactly the training path for Chinese
    en:  espeak-ng('en-us', segment) = natural English, not Chinese-accented

Segments are split on
(a) Latin runs: ``[A-Za-z][A-Za-z'\\-]*`` -- one run = one English segment,
(b) everything else: one continuous Chinese / punctuation segment.

Every non-Latin segment is byte-identical to ``EspeakPhonemizer.phonemize()``
(no rewording, no ``segment_ipa`` call at all when ``method='espeak'``), so a
pure Chinese line keeps producing exactly the phonemes it was trained on --
verified by ``scripts/test_phonemize_bilingual.py``. Every symbol the English
voice emits is checked against the voice's own ``phoneme_id_map``, and a miss
raises instead of being dropped.

Usage
-----

    from scripts.phonemize_bilingual import BilingualPhonemizer

    ph = BilingualPhonemizer.from_voice("data/export/zh_cn-myvoice.onnx.json")
    ph.phonemize("把客厅的 air conditioner 打开")
    # -> [[p, h, o, n, e, m, e, ...], ...]

``--method`` exists because the stock ``piper`` binary cannot be told which
Chinese backend to use; see ``scripts/say.py``.

Run standalone to check a voice:

    .venv/Scripts/python.exe scripts/phonemize_bilingual.py \\
        data/export/zh_cn-myvoice.onnx.json "打开客厅的 lamp"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

#: One English word, or one run of spaced single capitals (``W I F I``).
EN_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
#: Splits an English chunk into the same units espeak sees: tokens plus the
#: whitespace between them. ``\w`` keeps Chinese attached to its neighbour, which
#: matters when the ONNX path re-joins the pieces.
EN_PIECE = re.compile(r"[a-zA-Z'\-]+|\s+|[\w]+|[^\w\s]")
#: A maximal contiguous run of non-Latin text (Chinese, digits, punctuation).
NON_EN_RUN = re.compile(r"[^A-Za-z]+")

#: Kept for callers that need the conventional set; espeak itself decides where
#: a sentence ends, so this module never splits on it.
BOUNDARY = set(".!?;:。！？；：")
#: Phonemes that may terminate a group in :meth:`BilingualPhonemizer.phonemize`.
TERMINAL = set("。！？；.!?;")
#: Closing quotes/brackets allowed between a terminator and the next sentence.
CLOSERS = set("”’」』）)》〉】")
CJK_RANGE = ("\u3400", "\u4dbf", "\u4e00", "\u9fff", "\uf900", "\ufaff")


def _only_closers_until_end(phonemes: list[str], start: int) -> bool:
    """True when every remaining phoneme is a closing quote/bracket or space."""
    for phoneme in phonemes[start:]:
        if phoneme in CLOSERS or phoneme.strip() == "":
            continue
        return False
    return True


def is_cjk(ch: str) -> bool:
    return CJK_RANGE[0] <= ch <= CJK_RANGE[1] or CJK_RANGE[2] <= ch <= CJK_RANGE[3] \
        or CJK_RANGE[4] <= ch <= CJK_RANGE[5]


def is_latin_word_char(ch: str) -> bool:
    return ("A" <= ch <= "Z") or ("a" <= ch <= "z") or ch in "'-"


def split_segments(text: str) -> list[tuple[str, str]]:
    """Split ``text`` into ``(lang, chunk)`` pairs, ``lang`` in ``{'zh', 'en'}``.

    Whitespace directly before a Latin token stays with the ``zh`` chunk, because
    espeak separates Chinese and English with a space there; an English chunk is
    a single token. Everything outside the Latin tokens is passed through
    untouched, so each chunk phonemizes exactly as it does inside the unsplit
    line (verified by ``test_phonemize_bilingual``).
    """
    segs: list[tuple[str, str]] = []
    pos = 0
    for m in EN_WORD.finditer(text):
        if m.start() > pos:
            segs.append(("zh", text[pos:m.start()]))
        segs.append(("en", text[m.start():m.end()]))
        pos = m.end()
    if pos < len(text):
        segs.append(("zh", text[pos:]))
    return segs


class BilingualPhonemizer:
    """Phonemize zh + en against a voice's ``.onnx.json`` phoneme table.

    Args:
        phoneme_id_map: the voice's ``phoneme_id_map`` (dict symbol -> int list).
        method: ``"espeak"`` (segment IPA via espeak, matches training exactly for
            Chinese) or ``"chinese"`` (``piper.phonemize_chinese.phonemize_chinese``,
            the stock ONNX path, g2pW for Chinese).
    """

    def __init__(self, phoneme_id_map: dict[str, list[int]],
                 method: str = "espeak") -> None:
        if method not in ("espeak", "chinese"):
            raise ValueError(f"method must be 'espeak' or 'chinese', got {method!r}")
        self.phoneme_id_map = phoneme_id_map
        self.method = method
        self._zh = None
        self._en = None

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_voice(cls, config_path: str | Path,
                   method: str = "espeak") -> "BilingualPhonemizer":
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        return cls(cfg["phoneme_id_map"], method=method)

    @classmethod
    def from_model(cls, model_path: str | Path, **kw) -> "BilingualPhonemizer":
        return cls.from_voice(Path(model_path).with_suffix(".onnx.json"), **kw)

    # -- lazily built backends ---------------------------------------------
    @property
    def zh(self):
        if self._zh is None:
            if self.method == "chinese":
                from piper.phonemize_chinese import ChinesePhonemizer
                # Downloads g2pW on first use; the stock ONNX path.
                self._zh = ChinesePhonemizer().phonemize
            else:
                from piper.phonemize_espeak import EspeakPhonemizer
                self._zh = EspeakPhonemizer().phonemize
        return self._zh

    @property
    def en(self):
        if self._en is None:
            from piper.phonemize_espeak import EspeakPhonemizer
            self._en = EspeakPhonemizer().phonemize
        return self._en

    # -- public API --------------------------------------------------------
    def phonemize(self, text: str) -> list[list[str]]:
        """Return a list of sentences, each a list of IPA phonemes.

        Pure Latin text (no Chinese at all) is handed to espeak whole, exactly as
        training does. Mixed text is split at the Latin tokens: each token goes
        through ``en-us`` (natural English instead of espeak's Chinese reading of
        Latin letters) and every other character goes through ``cmn`` untouched.

        One deliberate difference from the training path: espeak-ng's own
        sentence splitter does not fire after Chinese text in ``cmn`` voice, so
        the training path can emit several sentences as one group. Here a
        sentence ends at terminal punctuation the grouping decides
        (``。！？；.!?;``) followed by optional quotes -- which yields one group
        per line, the shape espeak gives for English and the shape the corpus is
        built for.
        """
        if not split_segments(text) or all(
                lang == "en" for lang, _ in split_segments(text)):
            return self._en_sentences(text.strip())

        joined: list[str] = []
        for lang, chunk in split_segments(text):
            if lang == "zh":
                joined.extend(self._zh_chars(chunk))
            else:
                joined.extend(self._en_chars(chunk))
        groups: list[list[str]] = []
        current: list[str] = []
        for i, phoneme in enumerate(joined):
            current.append(phoneme)
            if phoneme in TERMINAL and _only_closers_until_end(joined, i + 1):
                groups.append(current)
                current = []
        if current:
            if groups:
                groups[-1].extend(current)
            else:
                groups.append(current)
        return groups

    def _zh_chars(self, text: str) -> list[str]:
        out: list[str] = []
        for sent in self._zh_sentences(text):
            out.extend(sent)
        return out

    def _en_chars(self, chunk: str) -> list[str]:
        out: list[str] = []
        for sent in self._en_sentences(chunk):
            out.extend(sent)
        return out

    def _zh_sentences(self, text: str) -> list[list[str]]:
        if self.method == "chinese":
            return self.zh(text)
        return self.zh("cmn", text)

    def _en_sentences(self, chunk: str) -> list[list[str]]:
        phoneme_lists = self.en("en-us", chunk)
        for phonemes in phoneme_lists:
            missing = [p for p in phonemes if p not in self.phoneme_id_map]
            if missing:
                raise ValueError(
                    f"English phoneme(s) {missing!r} from {chunk!r} are not in this "
                    f"voice's phoneme_id_map; the model cannot say them.")
        return phoneme_lists

    def phonemes_with_ids(self, text: str) -> list[list[tuple[str, list[int]]]]:
        """Phonemize and pair each symbol with its id, for constructing ``PhonemeIds``."""
        return [[(p, self.phoneme_id_map[p]) for p in sent] for sent in self.phonemize(text)]

    def uncovered(self, lines: Iterable[str]) -> set[str]:
        """Return every phoneme those lines need that this voice cannot speak.

        A dry-run twin of :meth:`phonemize`: it collects gaps instead of raising,
        so a corpus audit can report them all at once.
        """
        gaps: set[str] = set()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                for sentence in self.phonemize(line):
                    gaps.update(p for p in sentence if p not in self.phoneme_id_map)
            except ValueError:
                # The English path raises before returning; re-derive the gap by
                # phonemizing each Latin token on its own.
                for lang, chunk in split_segments(line):
                    if lang != "en":
                        continue
                    for sent in self._en_sentences(chunk):
                        gaps.update(p for p in sent if p not in self.phoneme_id_map)
        return gaps


def _merge_pieces(pieces: list[tuple[str, list[list[str]]]]) -> list[list[str]]:
    """Join per-segment sentences, absorbing a piece's trailing phonemes into the
    previous sentence when that piece produced no terminator of its own.

    A trailing punctuation-only piece (``。`` after an English word) comes back
    as one "sentence" that is just the terminator, so it is appended rather than
    started as a new utterance.
    """
    out: list[list[str]] = []
    for _kind, sentences in pieces:
        for sent in sentences:
            sent = list(sent)
            if not sent:
                continue
            if out and (len(sent) == 1 and sent[0] in BOUNDARY):
                out[-1].extend(sent)
            else:
                out.append(sent)
    return out


def _iter_bad_examples(phonemizer: BilingualPhonemizer,
                       lines: Iterable[str]) -> Iterable[str]:
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            sent = phonemizer.phonemize(line)
        except ValueError as exc:
            yield f"{line}  -->  {exc}"
            continue
        if not sent:
            yield f"{line}  -->  no phonemes produced"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", help="voice .onnx.json (or .onnx) path")
    ap.add_argument("text", nargs="*", help="text to phonemize; omit to read stdin")
    ap.add_argument("--method", default="espeak", choices=["espeak", "chinese"])
    ap.add_argument("--batch", help="file of lines to check; reports the failures only")
    args = ap.parse_args()

    ph = BilingualPhonemizer.from_model(args.config, method=args.method)

    if args.batch:
        lines = Path(args.batch).read_text(encoding="utf-8").splitlines()
        bad = list(_iter_bad_examples(ph, lines))
        total = sum(1 for l in lines if l.strip())
        print(f"checked {total} lines, {len(bad)} failed")
        for b in bad:
            print(" ", b)
        return 1 if bad else 0

    texts = args.text or [l for l in sys.stdin.read().splitlines() if l.strip()]
    if texts:
        for text in texts:
            for sent in ph.phonemize(text):
                print(text, "->", "".join(sent))
        return 0

    print("nothing to phonemize", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
