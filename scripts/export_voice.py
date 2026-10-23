"""Turn a training checkpoint into a speaker-independent synthesizable voice.

``piper.train`` stops at a ``.ckpt``; the runtime (including a Raspberry Pi) needs
``<voice>.onnx`` + ``<voice>.onnx.json``. This wraps ``piper.train.export_onnx``
and copies the config written during training next to the model.

    .venv/Scripts/python.exe scripts/export_voice.py `
        --checkpoint data/myvoice/checkpoints/lightning_logs/version_0/checkpoints/last.ckpt `
        --config data/myvoice/config.json `
        --out data/myvoice-export

Then on the target machine:

    python -m piper -m zh_CN_minimax.onnx -c zh_CN_minimax.onnx.json -f out.wav -i input.txt
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="config.json written by training")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--voice-name", default=None, help="default: checkpoint parent name")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    for path, label in ((args.checkpoint, "checkpoint"), (args.config, "config")):
        if not path.exists():
            print(f"missing {label}: {path}", file=sys.stderr)
            return 1

    args.out.mkdir(parents=True, exist_ok=True)
    voice = args.voice_name or args.checkpoint.stem.replace(".ckpt", "") or "voice"
    onnx_path = args.out / f"{voice}.onnx"
    config_out = args.out / f"{voice}.onnx.json"

    cmd = [
        sys.executable, str(Path(__file__).resolve().parent / "export_onnx_safe.py"),
        "--checkpoint", str(args.checkpoint),
        "--output-file", str(onnx_path),
    ]
    if args.debug:
        cmd.append("--debug")
    print(" ".join(cmd))
    code = subprocess.call(cmd)
    if code != 0:
        print("ONNX export failed", file=sys.stderr)
        return code

    shutil.copyfile(args.config, config_out)
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"\nmodel:  {onnx_path}  ({size_mb:.1f} MB)")
    print(f"config: {config_out}")
    print("\ncopy both files to the Raspberry Pi, then:")
    print(f"  python -m piper -m {onnx_path.name} -c {config_out.name} -f welcome.wav -i input.txt")
    if args.checkpoint.stat().st_size > 0 and args.config.stat().st_size == 0:
        print("warning: config.json is empty; training may not have written it yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
