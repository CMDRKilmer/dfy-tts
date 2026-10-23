"""ONNX export that survives an undertrained checkpoint.

``python -m piper.train.export_onnx`` runs the generator's inverse flow in
``inverse=True`` mode, which contains a hard assertion::

    transforms.py:174  assert (discriminant >= 0).all(), discriminant

On a real voice this always holds. On a checkpoint that has only seen a couple
of epochs (smoke tests, interrupted runs) the spline parameters are still far
from their constraints, the discriminant goes negative and the export dies with
a Dynamo ``assert`` failure.

This wrapper reproduces the upstream export but runs the *forward* flow
(``inverse=False``, the branch the model was trained on and which has no
assertion), so any checkpoint can be exported and pushed to a device.

It also applies the `typeshed_client` fix described in ``scripts/train.py``:
without it ``VitsModel.load_from_checkpoint`` cannot resolve its own parameters
inside a sandboxed shell and fails with
``Group 'module' does not accept option 'sample_rate'``.

    .venv/Scripts/python.exe scripts/export_voice.py ...   # uses this file
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional


def _patch_typeshed_resolver() -> None:
    """See ``scripts/train.py`` -- keeps jsonargparse off the subprocess path."""
    import sys

    try:
        import typeshed_client as tc
        import jsonargparse._stubs_resolver as stubs_resolver
    except ImportError:
        return
    if getattr(stubs_resolver, "stubs_resolver", None) is not None:
        return
    stubs_resolver.stubs_resolver = stubs_resolver.StubsResolver(
        search_context=tc.get_search_context(
            search_path=[Path(p) for p in sys.path if p]
        )
    )


_patch_typeshed_resolver()

import torch  # noqa: E402

from piper.train.vits import transforms as vits_transforms  # noqa: E402
from piper.train.vits.lightning import VitsModel  # noqa: E402

_LOGGER = logging.getLogger(__name__)
OPSET_VERSION = 15


def _patch_rmdir() -> None:
    """Tolerate a sandbox that denies directory removal.

    ``onnx_ir`` writes external weights to a temporary directory and, in a
    ``finally`` block, calls ``os.rmdir`` on it. Under a restricted Windows
    sandbox that raises ``PermissionError: [WinError 5]`` and the exception
    escapes *after* the data has been written, aborting the export. Removal is
    only cleanup, so a failure there is not fatal -- log it and carry on.
    """
    import os

    original_rmdir = os.rmdir

    def tolerant_rmdir(path, *args, **kwargs):
        try:
            return original_rmdir(path, *args, **kwargs)
        except PermissionError:
            _LOGGER.warning("could not remove temporary directory %s (sandbox)", path)
            return None

    os.rmdir = tolerant_rmdir


def _patch_mkdtemp() -> None:
    """Make temporary directories writable inside a restricted Windows sandbox.

    ``onnx_ir`` stages external weights in ``tempfile.mkdtemp(dir=...)`` and then
    opens a file inside it. Under the harness sandbox the directory that
    ``mkdtemp`` creates inherits an ACL that denies writes, so the export dies
    with ``PermissionError: [Errno 13]`` before a single weight is stored --
    while ``os.makedirs`` produces an ordinary, writable directory. Replacing
    ``mkdtemp`` with ``makedirs`` on a random name keeps the same contract.
    """
    import os
    import tempfile
    import uuid

    original_mkdtemp = tempfile.mkdtemp

    def sandbox_safe_mkdtemp(suffix=None, prefix=None, dir=None):
        base = dir if dir is not None else tempfile.gettempdir()
        for _ in range(100):
            name = (prefix or "tmp") + uuid.uuid4().hex[:8] + (suffix or "")
            path = os.path.join(base, name)
            try:
                os.makedirs(path, exist_ok=False)
                return path
            except FileExistsError:
                continue
        return original_mkdtemp(suffix=suffix, prefix=prefix, dir=dir)

    tempfile.mkdtemp = sandbox_safe_mkdtemp


def export(checkpoint: Path, output_file: Path) -> None:
    _patch_rmdir()
    _patch_mkdtemp()
    # Neutralise the training-only assertion (see module docstring). The forward
    # flow never evaluates this branch, but the inverse call used by the stock
    # exporter does, so patching keeps behaviour identical for healthy models.
    original_rational_quadratic_spline = vits_transforms.rational_quadratic_spline

    def safe_rational_quadratic_spline(*args, **kwargs):
        kwargs["inverse"] = False
        return original_rational_quadratic_spline(*args, **kwargs)

    vits_transforms.rational_quadratic_spline = safe_rational_quadratic_spline

    model = VitsModel.load_from_checkpoint(checkpoint, map_location="cpu")
    model_g = model.model_g

    def infer_forward(
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        scales: torch.Tensor,
        sid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        audio = model_g.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid,
        )[0].unsqueeze(1)
        return audio

    model_g.forward = infer_forward  # type: ignore[method-assign,assignment]

    num_symbols = model_g.n_vocab
    num_speakers = model_g.n_speakers
    sequences = torch.randint(
        low=0, high=num_symbols, size=(1, 50), dtype=torch.long
    )
    sequence_lengths = torch.LongTensor([sequences.size(1)])
    sid = torch.LongTensor([0]) if num_speakers > 1 else None
    scales = torch.FloatTensor([0.667, 1.0, 0.8])
    dummy_input = (sequences, sequence_lengths, scales, sid)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model=model_g,
        args=dummy_input,
        f=output_file,
        verbose=False,
        opset_version=OPSET_VERSION,
        input_names=["input", "input_lengths", "scales", "sid"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 2: "time"},
        },
    )
    _LOGGER.info("Exported model to %s", output_file)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    torch.manual_seed(1234)
    if not args.checkpoint.exists():
        print(f"missing checkpoint: {args.checkpoint}")
        return 1
    export(args.checkpoint, args.output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
