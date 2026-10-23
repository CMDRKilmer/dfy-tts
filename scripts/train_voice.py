#!/usr/bin/env python3
"""Fine-tune Piper on a MiniMax-generated (or any) Chinese dataset.

Thin wrapper around ``python -m piper.train fit`` that fills in the options this
machine needs: GPU, mixed precision, a sane 2080 Ti batch size, MOS logging off
(no network for the torch.hub checkpoint) and a checkpoint dir under the
dataset folder.

    .venv/Scripts/python.exe scripts/train_voice.py `
        --dataset data/myvoice `
        --voice-name zh_CN_minimax `
        --espeak-voice cmn `
        --epochs 800

Anything not covered by the flags can be passed through after ``--``:

    ... scripts/train_voice.py --dataset data/myvoice -- --trainer.limit_val_batches 8
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="folder with wavs/ + metadata.csv")
    parser.add_argument("--voice-name", required=True, help="name recorded in config.json")
    parser.add_argument("--espeak-voice", default="cmn", help="espeak-ng voice (cmn = Mandarin)")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--accumulate", type=int, default=1)
    # 0 keeps everything in-process: a sandboxed Windows shell denies the pipes
    # DataLoader workers need (_winapi.CreateFile -> WinError 5). Outside a
    # sandbox, 4-8 workers make epoch throughput noticeably better.
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ckpt-path", default=None, help="base checkpoint to fine-tune from")
    parser.add_argument("--no-gpu", action="store_true", help="train on CPU (slow, for debugging)")
    parser.add_argument("--fast-dev-run", action="store_true", help="1 train + 1 val batch, no checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="print the command and exit")
    args, passthrough = parser.parse_known_args()
    passthrough = [p for p in passthrough if p != "--"]

    csv_path = args.dataset / "metadata.csv"
    audio_dir = args.dataset / "wavs"
    cache_dir = args.dataset / "cache"
    config_path = args.dataset / "config.json"
    root_dir = args.dataset / "checkpoints"

    for path, label in ((csv_path, "metadata.csv"), (audio_dir, "wavs/")):
        if not path.exists():
            print(f"missing {label}: {path} (run scripts/build_dataset.py first)", file=sys.stderr)
            return 1

    # scripts/train.py = upstream entry point with a single val_mel checkpoint
    # callback; see its docstring for why the stock callbacks crash offline.
    cmd = [
        sys.executable, str(Path(__file__).resolve().parent / "train.py"), "fit",
        "--data.csv_path", str(csv_path),
        "--data.audio_dir", str(audio_dir),
        "--data.cache_dir", str(cache_dir),
        "--data.config_path", str(config_path),
        "--data.voice_name", args.voice_name,
        "--data.espeak_voice", args.espeak_voice,
        "--data.batch_size", str(args.batch_size),
        "--data.num_workers", str(args.num_workers),
        "--model.sample_rate", str(args.sample_rate),
        "--model.mos_metric", "none",
        "--trainer.accelerator", "cpu" if args.no_gpu else "gpu",
        "--trainer.default_root_dir", str(root_dir),
        "--trainer.max_epochs", str(args.epochs),
        "--trainer.accumulate_grad_batches", str(args.accumulate),
    ]
    if not args.no_gpu:
        cmd += ["--trainer.precision", args.precision, "--trainer.devices", "1"]
    if args.fast_dev_run:
        cmd += ["--trainer.fast_dev_run", "true"]
    if args.ckpt_path:
        cmd += ["--ckpt_path", args.ckpt_path]
    cmd += passthrough

    print(" ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
