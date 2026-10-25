"""A LAN HTTP front-end for Qwen3-TTS, so the Pi can keep the 0.6B voice quality.

Why this exists
---------------
Qwen3-TTS-12Hz-0.6B is 2.4 GB of weights (1.8 GB LLM + 651 MB speech tokenizer).
On a Pi 4 that is RTF ~3-8 even at Q4, because batch=1 decoding is memory
bandwidth bound (3.65 GB/s measured) and the tokenizer runs at 12 frames/second.
No amount of quantisation gets it under real time -- see ``docs/adr``.

The 2080 Ti already sits on the same LAN (192.168.100.2, ~5 ms RTT). Keeping the
model resident there and streaming WAV bytes to the Pi gives **100% of the Qwen
voice quality** while the Pi does nothing but decode and play audio. When the
desktop is off, the Pi falls back to its local Piper voice (``scripts/say.py``).

Start it
--------
    .venv/Scripts/python.exe scripts/tts_server.py --port 8765

Then:
    curl http://127.0.0.1:8765/health
    curl -X POST http://127.0.0.1:8765/tts \
         -H 'content-type: application/json' \
         -d '{"text":"米饭煮好了哦", "speaker":"Serena"}' -o out.wav

Windows Firewall needs an inbound rule for the port before the Pi can reach it.

Design notes
------------
* One model instance, one ``threading.Lock``. The CUDA context is not a
  thread-safe toy and a synthesis call already saturates the card; queueing is
  simpler and faster than batching two requests.
* An in-process LRU cache keyed by (text, mode, speaker, instruct, language).
  A Pi asking for the same greeting a hundred times should cost one synthesis.
* ``pickle`` never touches the network: responses are plain WAV bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"

#: Synthesis languages understood by the model.
LANGUAGES = {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
             "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
             "es": "Spanish", "it": "Italian"}


class ModelState:
    """The loaded model plus everything that must be serialised around it."""

    def __init__(self, model_id: str, device: str, dtype: str,
                 default_speaker: str, ref_root: Path, cache_size: int,
                 default_ref: Optional[str] = None,
                 default_ref_text: Optional[str] = None) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.default_speaker = default_speaker
        self.ref_root = ref_root
        self.cache_size = cache_size
        # The *Base* checkpoint ships no speaker embeddings, so a ``custom``
        # request has to be served by cloning a reference clip.  Set these and
        # the wire protocol stays identical for the client -- one fixed voice.
        self.default_ref = default_ref
        self.default_ref_text = default_ref_text
        self.model: Any = None
        self.sample_rate: int = 0
        self.lock = threading.Lock()
        self.cache: "OrderedDict[str, bytes]" = OrderedDict()
        self.ready = False
        self.load_seconds: float = 0.0
        self.calls = 0
        self.cache_hits = 0

    # -- cache ----------------------------------------------------------
    def cache_get(self, key: str) -> Optional[bytes]:
        hit = self.cache.get(key)
        if hit is None:
            return None
        self.cache.move_to_end(key)
        self.cache_hits += 1
        return hit

    def cache_put(self, key: str, payload: bytes) -> None:
        self.cache[key] = payload
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)


STATE: Optional[ModelState] = None


# ---------------------------------------------------------------------------
# request models
# ---------------------------------------------------------------------------
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    mode: str = Field("custom", pattern="^(custom|clone|design)$")
    speaker: Optional[str] = None
    language: Optional[str] = None
    instruct: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    length_scale: Optional[float] = Field(None, gt=0.1, le=4.0)


def _resolve_language(text: str, requested: Optional[str]) -> str:
    """Map a hint (``zh`` / ``Chinese``) to the model's language name."""
    if requested:
        lowered = requested.strip().lower()
        for short, full in LANGUAGES.items():
            if lowered in (short, full.lower()):
                return full
        return requested
    return "Chinese" if any("\u4e00" <= c <= "\u9fff" for c in text) else "English"


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _cache_key(req: TTSRequest, language: str) -> str:
    raw = "\x1f".join([
        req.text, req.mode, req.speaker or "", language,
        req.instruct or "", req.ref_audio or "", req.ref_text or "",
        f"{req.length_scale or 0:g}",
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _safe_ref(ref_audio: str, ref_root: Path) -> Path:
    """Only allow reference audio inside the configured directory."""
    candidate = Path(ref_audio)
    if not candidate.is_absolute():
        candidate = (ref_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    try:
        candidate.relative_to(ref_root.resolve())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"ref_audio must live under {ref_root}",
        ) from None
    if not candidate.exists():
        raise HTTPException(status_code=404, detail=f"ref_audio not found: {candidate.name}")
    return candidate


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------
def build_app(state: ModelState) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import torch
        from qwen_tts import Qwen3TTSModel

        started = time.time()
        dtype = getattr(torch, state.dtype)
        state.model = Qwen3TTSModel.from_pretrained(
            state.model_id, device_map=state.device, dtype=dtype,
        )
        state.sample_rate = int(getattr(state.model, "sample_rate", 24000) or 24000)
        state.load_seconds = time.time() - started
        state.ready = True
        print(f"[tts_server] model ready in {state.load_seconds:.1f}s "
              f"({state.model_id}, {state.device}, {state.dtype})", flush=True)
        try:
            yield
        finally:
            state.model = None
            state.ready = False

    app = FastAPI(title="Qwen3-TTS LAN server", version="1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse({
            "ok": state.ready,
            "model": state.model_id,
            "device": state.device,
            "dtype": state.dtype,
            "sample_rate": state.sample_rate,
            "load_seconds": round(state.load_seconds, 2),
            "calls": state.calls,
            "cache_hits": state.cache_hits,
            "cache_entries": len(state.cache),
        })

    @app.get("/voices")
    def voices() -> JSONResponse:
        return JSONResponse({
            "default": state.default_speaker,
            "languages": sorted(LANGUAGES.values()),
            "modes": ["custom", "clone", "design"],
        })

    @app.post("/tts")
    def tts(req: TTSRequest) -> Response:
        if state.model is None or not state.ready:
            raise HTTPException(status_code=503, detail="model not loaded yet")

        # A Base checkpoint cannot honour ``custom`` (no speaker table); rewrite
        # it to a clone of the configured reference clip so one protocol serves
        # both checkpoints.
        if req.mode == "custom" and state.default_ref:
            req = req.model_copy(update={
                "mode": "clone",
                "ref_audio": state.default_ref,
                "ref_text": state.default_ref_text,
            })

        language = _resolve_language(req.text, req.language)
        key = _cache_key(req, language)
        cached = state.cache_get(key)
        if cached is not None:
            return Response(content=cached, media_type="audio/wav",
                            headers={"x-cache": "hit"})

        kwargs: Dict[str, Any] = {"text": req.text, "language": language}
        if req.mode == "custom":
            kwargs["speaker"] = req.speaker or state.default_speaker
            if req.instruct:
                kwargs["instruct"] = req.instruct
            call = state.model.generate_custom_voice
        elif req.mode == "design":
            if not req.instruct:
                raise HTTPException(status_code=400,
                                    detail="mode=design requires instruct")
            kwargs["instruct"] = req.instruct
            call = state.model.generate_voice_design
        else:  # clone
            if not req.ref_audio:
                raise HTTPException(status_code=400,
                                    detail="mode=clone requires ref_audio")
            kwargs["ref_audio"] = str(_safe_ref(req.ref_audio, state.ref_root))
            if req.ref_text:
                kwargs["ref_text"] = req.ref_text
            else:
                kwargs["x_vector_only_mode"] = True
            call = state.model.generate_voice_clone

        if req.length_scale:
            kwargs["length_scale"] = req.length_scale

        # Budget the decode by input length.  The model's own default is 2048
        # frames -- roughly 170 s of audio.  When it fails to emit EOS on a short
        # line it runs that budget to completion, which blocks every later caller
        # behind the lock and looks like a hang on the Pi.  12 frames/s and ~4
        # Chinese chars/s give the estimate; the +2 s pad plus a 96-frame floor
        # leaves room for slow delivery and pauses.
        if "max_new_tokens" not in kwargs:
            estimate = 12.0 * (len(req.text) / 4.0 + 2.0)
            kwargs["max_new_tokens"] = int(min(1024, max(96, estimate)))

        # The client wants one complete WAV, so single-shot decode is strictly
        # better: the streaming path only adds per-chunk host synchronisation.
        kwargs.setdefault("non_streaming_mode", True)

        with state.lock:
            started = time.time()
            try:
                # Autograd bookkeeping costs real time in a 12 Hz autoregressive
                # decode loop; nothing here is trained.
                with torch.inference_mode():
                    wavs, sample_rate = call(**kwargs)
            except HTTPException:
                raise
            except Exception as exc:  # noqa: BLE001 - report, do not leak a traceback
                raise HTTPException(status_code=500,
                                    detail=f"{type(exc).__name__}: {exc}") from exc
            elapsed = time.time() - started
            state.calls += 1

        audio = np.asarray(wavs[0], dtype=np.float32)
        payload = _wav_bytes(audio, int(sample_rate or state.sample_rate))
        state.cache_put(key, payload)
        seconds = len(audio) / float(sample_rate or state.sample_rate)
        rtf = elapsed / seconds if seconds > 0 else 0.0
        print(f"[tts_server] {elapsed:5.2f}s for {seconds:4.2f}s audio "
              f"(RTF {rtf:.2f}) {len(payload) // 1024}KiB  {req.text[:28]!r}",
              flush=True)
        return Response(
            content=payload,
            media_type="audio/wav",
            headers={
                "x-cache": "miss",
                "x-synth-seconds": f"{elapsed:.3f}",
                "x-audio-seconds": f"{seconds:.3f}",
                "x-rtf": f"{rtf:.3f}",
            },
        )

    return app


def main() -> int:
    global STATE

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--speaker", default="Serena", help="default custom voice")
    ap.add_argument("--ref-root", type=Path, default=ROOT / "data" / "ref_voice",
                    help="directory that reference audio must stay inside")
    ap.add_argument("--default-ref", default=None,
                    help="clone this clip for every request (required for the "
                         "Base checkpoint, which has no built-in speakers)")
    ap.add_argument("--default-ref-text", default=None,
                    help="transcript of --default-ref; must match it exactly")
    ap.add_argument("--cache-size", type=int, default=512,
                    help="in-process LRU entries (0 disables caching)")
    ap.add_argument("--reload", action="store_true", help="uvicorn autoreload (dev)")
    args = ap.parse_args()

    args.ref_root.mkdir(parents=True, exist_ok=True)
    STATE = ModelState(
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        default_speaker=args.speaker,
        ref_root=args.ref_root,
        cache_size=max(0, args.cache_size),
        default_ref=args.default_ref,
        default_ref_text=args.default_ref_text,
    )
    app = build_app(STATE)

    import uvicorn
    print(f"[tts_server] listening on http://{args.host}:{args.port} "
          f"(ref-root {args.ref_root})", flush=True)
    uvicorn.run(app if not args.reload else "tts_server:app",
                host=args.host, port=args.port, reload=args.reload,
                workers=1, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
