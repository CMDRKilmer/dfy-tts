"""Raspberry-Pi side of the split TTS: LAN Qwen first, local Piper as fallback.

Runs on the Pi with **no third-party HTTP client** -- ``urllib`` only, so it works
inside ``/home/pi/tts-venv`` (which has ``piper`` and ``onnxruntime`` but no
``requests``).

Resolution order
----------------
1. **Local cache** -- ``~/.cache/dfy-tts/<sha1>.lan.wav`` (Piper output lives in
   ``<sha1>.piper.wav``, so a fallback never gets mistaken for the real voice). A
   repeated phrase costs a file read, not a network round trip.
2. **LAN Qwen3-TTS** -- ``scripts/tts_server.py`` on the 2080 Ti. This is the
   voice quality the Pi cannot produce locally.
3. **Local Piper** -- ``scripts/say.py`` with the bilingual phonemizer. Lower
   quality, but real-time and completely offline.

The fallback is not a nicety: the desktop sleeps, and a Pi that goes mute because
its GPU host rebooted is worse than one that sounds merely okay.

Usage
-----
    # prepare the cache for phrases the assistant says every day
    python tts_client.py --warm "早上好" "米饭煮好了哦" "该睡觉了"

    # speak (writes cache, then plays)
    python tts_client.py "现在是下午三点"

    # never touch the network -- exercise the Piper path
    python tts_client.py --prefer local "测试本地音色"

    # is the GPU host up?
    python tts_client.py --health

Environment
-----------
``DFY_TTS_SERVER``    server base URL (default ``http://127.0.0.1:8765``, i.e. the
                     local end of the reverse SSH tunnel)
``DFY_TTS_VOICE``     Piper ``.onnx`` path (default ``~/.local/share/dfy-tts/voice.onnx``)
``DFY_TTS_CACHE``     cache directory
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path
from typing import Optional

HOME = Path.home()
# The PC never opens a port (no admin rights on that box): connectivity is the
# SSH reverse tunnel's loopback end, so this is the address that actually works
# out of the box.
DEFAULT_SERVER = os.environ.get("DFY_TTS_SERVER", "http://127.0.0.1:8765")
DEFAULT_VOICE = Path(os.environ.get(
    "DFY_TTS_VOICE", str(HOME / ".local" / "share" / "dfy-tts" / "voice.onnx")))
CACHE_DIR = Path(os.environ.get("DFY_TTS_CACHE", str(HOME / ".cache" / "dfy-tts")))
SCRIPTS_DIR = Path(__file__).resolve().parent

CONNECT_TIMEOUT = 2.0     # a sleeping desktop must not block playback
READ_TIMEOUT = 45.0       # a cold CUDA kernel can take a while, but a stuck
                          # decode is worse than falling back to Piper

#: Set by --no-cache; makes every lookup miss and every write a no-op.
_NO_CACHE = False


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def cache_path(text: str, speaker: str, source: str = "lan") -> Path:
    """Cache slot for one phrase from one backend.

    The two backends get *separate* slots on purpose: a Piper fallback is a
    placeholder, and letting it occupy the main slot would pin the worse voice
    into place long after the GPU host came back.  ``<sha1>.lan.wav`` always wins
    when it exists.
    """
    if _NO_CACHE:
        return Path(os.devnull)
    key = hashlib.sha1(f"{text}\x1f{speaker}".encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{key}.{source}.wav"

def read_cache(path: Path) -> Optional[bytes]:
    if _NO_CACHE:
        return None
    try:
        return path.read_bytes() if path.exists() else None
    except OSError:
        return None


def write_cache(path: Path, payload: bytes) -> None:
    if _NO_CACHE:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(payload)
        tmp.replace(path)
    except OSError as exc:
        print(f"[tts] cache write skipped: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# LAN backend
# ---------------------------------------------------------------------------
def server_health(server: str, timeout: float = CONNECT_TIMEOUT) -> Optional[dict]:
    try:
        with urllib.request.urlopen(f"{server}/health", timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def synthesize_lan(text: str, server: str, speaker: str,
                   language: Optional[str], instruct: Optional[str]) -> bytes:
    body: dict = {"text": text, "speaker": speaker}
    if language:
        body["language"] = language
    if instruct:
        body["instruct"] = instruct
    # Split connect from read: an unreachable desktop must cost ~2 s, not the
    # full synthesis budget, or every fallback feels like a hang.
    url = urllib.parse.urlsplit(f"{server}/tts")
    conn = http.client.HTTPConnection(url.hostname, url.port or 80,
                                      timeout=CONNECT_TIMEOUT)
    try:
        conn.connect()
        conn.sock.settimeout(READ_TIMEOUT)
        conn.request("POST", url.path or "/tts",
                     body=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                     headers={"content-type": "application/json"})
        resp = conn.getresponse()
        payload = resp.read()
        if resp.status != 200:
            raise ValueError(f"server returned HTTP {resp.status}")
    finally:
        conn.close()
    if not payload.startswith(b"RIFF"):
        raise ValueError("server did not return a WAV payload")
    return payload


# ---------------------------------------------------------------------------
# local Piper backend
# ---------------------------------------------------------------------------
def synthesize_piper(text: str, voice: Path) -> bytes:
    """Synthesise through ``scripts/say.py`` -- the bilingual entry point.

    Stock ``python -m piper`` phonemizes with g2pW and silently drops Latin
    words, so the bilingual path is mandatory, not a preference.
    """
    if not voice.exists():
        raise FileNotFoundError(
            f"Piper voice not found: {voice}\n"
            f"copy one over, e.g. scp desktop:tts/data/ref_voice/"
            f"zh_CN-huayan-medium.onnx* {voice.parent}/"
        )
    say = SCRIPTS_DIR / "say.py"
    if not say.exists():
        raise FileNotFoundError(f"scripts/say.py not found next to {__file__}")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        out = Path(handle.name)
    try:
        subprocess.run(
            [sys.executable, str(say), str(voice), text, "-f", str(out)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        return out.read_bytes()
    finally:
        out.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# playback
# ---------------------------------------------------------------------------
def play(payload: bytes) -> None:
    """Play WAV bytes with whatever the image provides."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        handle.write(payload)
        path = Path(handle.name)
    try:
        try:
            import sounddevice  # type: ignore
            import soundfile as sf  # type: ignore

            data, rate = sf.read(str(path), dtype="float32")
            sounddevice.play(data, rate)
            sounddevice.wait()
            return
        except ImportError:
            pass
        for player in (["aplay", "-q"],
                       ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]):
            if shutil.which(player[0]):
                subprocess.run([*player, str(path)], check=True)
                return
        raise RuntimeError("no audio player found (install ffmpeg or alsa-utils)")
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def resolve(text: str, args, speaker: str) -> tuple[bytes, str]:
    lan_slot = cache_path(text, speaker, "lan")
    hit = read_cache(lan_slot)
    if hit is not None:
        return hit, "cache"

    if args.prefer != "local":
        try:
            payload = synthesize_lan(text, args.server, speaker,
                                     args.language, args.instruct)
            write_cache(lan_slot, payload)
            return payload, "lan"
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            if args.prefer == "lan":
                raise SystemExit(f"[tts] LAN synthesis failed: {exc}")
            print(f"[tts] LAN unavailable ({type(exc).__name__}); "
                  f"falling back to Piper", file=sys.stderr)

    piper_slot = cache_path(text, speaker, "piper")
    hit = read_cache(piper_slot)
    if hit is not None:
        return hit, "piper-cache"

    payload = synthesize_piper(text, Path(args.voice))
    write_cache(piper_slot, payload)
    return payload, "piper"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("text", nargs="*", help="text to speak; omit to read stdin")
    ap.add_argument("-f", "--file", help="write WAV here instead of playing")
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--voice", default=str(DEFAULT_VOICE))
    ap.add_argument("--speaker", default="Serena")
    ap.add_argument("--language", default=None, help="zh/en/... (auto if omitted)")
    ap.add_argument("--instruct", default=None, help="style hint, e.g. 温柔一点")
    ap.add_argument("--prefer", default="auto",
                    choices=["auto", "lan", "local"],
                    help="auto: LAN then Piper; lan: LAN only; local: Piper only")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--warm", action="store_true",
                    help="pre-generate every text given (cache only, no playback)")
    ap.add_argument("--health", action="store_true", help="probe the LAN server")
    ap.add_argument("--stats", action="store_true", help="cache size summary")
    args = ap.parse_args()

    if args.stats:
        files = list(CACHE_DIR.glob("*.wav")) if CACHE_DIR.exists() else []
        total = sum(f.stat().st_size for f in files)
        print(f"cache: {len(files)} files, {total / 1e6:.1f} MB at {CACHE_DIR}")
        return 0

    if args.health:
        info = server_health(args.server)
        if info is None:
            print(f"unreachable: {args.server}")
            return 1
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0

    texts = args.text or [line for line in sys.stdin.read().splitlines() if line.strip()]
    if not texts:
        print("nothing to say", file=sys.stderr)
        return 2

    if args.no_cache:
        global _NO_CACHE
        _NO_CACHE = True

    for text in texts:
        payload, source = resolve(text, args, args.speaker)
        print(f"[tts] {source:5s} {len(payload) / 1024:6.1f} KiB  {text[:32]}",
              file=sys.stderr)
        if args.file:
            target = Path(args.file)
            target.write_bytes(payload)
            print(f"[tts] wrote {target}", file=sys.stderr)
            continue
        if args.warm:
            continue
        play(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
