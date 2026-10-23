"""Patch the installed ``piper-tts`` package so training works on Windows.

Why this exists
---------------
The PyPI ``piper-tts`` wheel does not contain the compiled Cython extension
``piper.train.vits.monotonic_align.monotonic_align.core``.  Upstream builds it
with ``build_monotonic_align.sh`` (requires a C compiler); this machine has no
MSVC / Windows SDK, so this script installs a drop-in numba implementation of
``maximum_path_c`` into the package instead.

Usage
-----
    python scripts/patch_monotonic_align.py        # run with the venv python

Idempotent: safe to run after every reinstall of piper-tts.
"""

from __future__ import annotations

import importlib
import importlib.util
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORE_SRC = HERE / "monotonic_align_core.py"


def piper_monotonic_align_dir() -> Path | None:
    """Locate ``piper/train/vits/monotonic_align`` without importing it.

    Importing the package executes its ``__init__.py``, which itself fails
    while the compiled extension is missing, so we resolve the file location
    through the spec machinery instead.
    """
    spec = importlib.util.find_spec("piper")
    if spec is None or not spec.submodule_search_locations:
        return None
    pkg_dir = Path(list(spec.submodule_search_locations)[0]) / "train/vits/monotonic_align"
    return pkg_dir if pkg_dir.is_dir() else None


def main() -> int:
    if not CORE_SRC.exists():
        print(f"missing {CORE_SRC}", file=sys.stderr)
        return 1

    ma_dir = piper_monotonic_align_dir()
    if ma_dir is None:
        print("piper-tts is not installed in this interpreter", file=sys.stderr)
        return 1

    inner = ma_dir / "monotonic_align"
    core = inner / "core.py"

    if core.exists():
        print(f"already patched: {core}")
    else:
        inner.mkdir(exist_ok=True)
        (inner / "__init__.py").touch()
        shutil.copyfile(CORE_SRC, core)
        print(f"patched: {core}")

    # verify the exact import the training code performs:
    #   piper/train/vits/monotonic_align/__init__.py does
    #   `from .monotonic_align.core import maximum_path_c`
    importlib.invalidate_caches()
    mod = importlib.import_module("piper.train.vits.monotonic_align")
    print(f"import OK: {mod.maximum_path_c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
