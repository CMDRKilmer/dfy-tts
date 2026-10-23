"""Say bilingual text with a Piper voice (zh + en), including on a Raspberry Pi.

The stock ``python -m piper`` binary phonemizes every line with the Chinese
``g2pW`` model, which silently drops Latin words. This module keeps the same
synthesis core but swaps in :class:`scripts.phonemize_bilingual.BilingualPhonemizer`:
Chinese goes through espeak ``cmn`` (identical to training), English through
espeak ``en-us`` (natural English).

    # one line
    python scripts/say.py my.onnx "打开客厅的 lamp" -f out.wav

    # whole corpus, streamed to stdout as 16-bit PCM at the voice's sample rate
    python scripts/say.py my.onnx --pcm < lines.txt | aplay -r 22050 -f S16_LE

Everything runs on CPU with onnxruntime, so this also works on a Pi 4/5. On a Pi,
build onnxruntime from the system package or wheels that match your OS image --
``pip install onnxruntime`` gets CPU-only wheels for arm64 on Raspberry Pi OS
Bookworm (64-bit).

The stock Chinese path (``--method chinese``) reproduces what the compiled-in
ONNX graph does; use it only to compare.
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np


def synthesize_with_phonemes(voice, phonemize, text: str, syn_config):
    """Yield ``audio`` arrays for a phonemizer that returns a list of sentences."""
    for phonemes in phonemize(text):
        if not phonemes:
            continue
        phoneme_ids = voice.phonemes_to_ids(phonemes)
        audio = voice.phoneme_ids_to_audio(phoneme_ids, syn_config)
        if isinstance(audio, tuple):  # audio + alignments
            audio = audio[0]
        if syn_config.normalize_audio:
            max_val = float(np.max(np.abs(audio)))
            if max_val >= 1e-8:
                audio = (audio / max_val) * 32767.0
        yield audio.astype(np.int16)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", help="voice .onnx (and matching .onnx.json)")
    ap.add_argument("text", nargs="*", help="text to say; omit to read stdin")
    ap.add_argument("-f", "--file", help="write a wav file here instead of stdout PCM")
    ap.add_argument("--pcm", action="store_true", help="stream raw 16-bit PCM to stdout")
    ap.add_argument("--method", default="espeak", choices=["espeak", "chinese"],
                    help="Chinese phonemizer backend (default: espeak, matches training)")
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--noise-scale", type=float, default=0.667)
    ap.add_argument("--noise-w", type=float, default=0.8)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from phonemize_bilingual import BilingualPhonemizer  # noqa: E402
    from piper.config import SynthesisConfig  # noqa: E402
    from piper.voice import PiperVoice  # noqa: E402

    model_path = Path(args.model)
    voice = PiperVoice.load(model_path)
    ph = BilingualPhonemizer.from_model(model_path, method=args.method)
    syn_config = SynthesisConfig(
        length_scale=args.length_scale,
        noise_scale=args.noise_scale,
        noise_w_scale=args.noise_w,
        normalize_audio=True,
    )

    texts = args.text or [l for l in sys.stdin.read().splitlines() if l.strip()]
    if not texts:
        print("nothing to say", file=sys.stderr)
        return 2

    if args.file:
        wav_file = wave.open(args.file, "wb")
        wav_file.setframerate(voice.config.sample_rate)
        wav_file.setsampwidth(2)
        wav_file.setnchannels(1)
    else:
        wav_file = None

    for text in texts:
        for audio in synthesize_with_phonemes(voice, ph.phonemize, text, syn_config):
            data = audio.tobytes()
            if wav_file is not None:
                wav_file.writeframes(data)
            else:
                sys.stdout.buffer.write(data)
        if wav_file is not None:
            continue
        sys.stdout.buffer.flush()

    if wav_file is not None:
        wav_file.close()
        print(f"wrote {args.file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
