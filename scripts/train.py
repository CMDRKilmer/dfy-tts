"""Train entry point that works on a fresh, offline Windows install.

Two upstream problems this module works around
------------------------------------------------
1. The PyPI ``piper-tts`` wheel ships two checkpoint callbacks
   (``ModelCheckpoint(monitor="val_mel")`` and ``ModelCheckpoint(monitor="val_mos")``).
   When the UTMOS predictor cannot be downloaded (offline machine) ``val_mos`` is
   never logged, so the first epoch end raises::

       MisconfigurationException: `ModelCheckpoint(monitor='val_mos')` could not
       find the monitored key ...
       RuntimeError: Found more than one stateful callback of type
       `ModelCheckpoint` ...

   Both fire before ``last.ckpt`` is written, i.e. an unattended overnight run
   produces nothing. This module builds the LightningCLI itself with a single
   checkpoint callback (``val_mel``, which is always logged).

2. ``typeshed_client`` resolves its search path by shelling out
   (``subprocess.check_output([python, "-c", ...])``), and a restricted Windows
   sandbox denies ``_winapi.CreatePipe`` with ``PermissionError: [WinError 5]``.
   ``jsonargparse`` then quietly resolves **zero** parameters for
   ``VitsModel``/``VitsDataModule``, so ``link_arguments("data.batch_size",
   "model.batch_size")`` blows up with ``No action for key "model.batch_size"``.
   Seeding the module-level resolver singleton with an explicit ``search_path``
   skips the subprocess entirely -- no sandbox permission needed.

Usage
-----
Identical to ``python -m piper.train fit``::

    .venv/Scripts/python.exe scripts/train.py fit --data.csv_path ...

``scripts/train_voice.py`` calls this module for you.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def _patch_typeshed_resolver() -> None:
    """Resolve stub signatures without spawning a helper subprocess.

    ``jsonargparse`` asks ``typeshed_client`` for a search context, which by
    default runs ``sys.executable -c "print(sys.path)"``. In a sandbox that
    forbids child pipes the call raises, jsonargparse swallows it, and every
    piper class silently ends up with zero parameters. Giving it ``sys.path``
    directly produces the identical context with no subprocess at all.
    """
    try:
        import typeshed_client as tc
        import jsonargparse._stubs_resolver as stubs_resolver
    except ImportError:  # older jsonargparse without the stubs dependency
        return
    if getattr(stubs_resolver, "stubs_resolver", None) is not None:
        return
    search_path = [Path(p) for p in sys.path if p]
    stubs_resolver.stubs_resolver = stubs_resolver.StubsResolver(
        search_context=tc.get_search_context(search_path=search_path)
    )


_patch_typeshed_resolver()

import torch  # noqa: E402
from lightning.pytorch.callbacks import ModelCheckpoint  # noqa: E402

from piper.train.__main__ import VitsLightningCLI  # noqa: E402
from piper.train.vits.dataset import VitsDataModule  # noqa: E402
from piper.train.vits.lightning import VitsModel  # noqa: E402

#: `val_mel` is logged on every epoch, so it is safe as the only monitor.
CHECKPOINT_CALLBACKS = [
    ModelCheckpoint(
        monitor="val_mel",
        mode="min",
        save_top_k=5,
        save_last=True,
        filename="epoch={epoch}-val_mel={val_mel:.4f}",
        auto_insert_metric_name=False,
    ),
]


def cli() -> VitsLightningCLI:
    return VitsLightningCLI(
        VitsModel,
        VitsDataModule,
        trainer_defaults={"max_epochs": -1, "callbacks": CHECKPOINT_CALLBACKS},
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    cli()


if __name__ == "__main__":
    main()
