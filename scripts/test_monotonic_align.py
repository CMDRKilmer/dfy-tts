"""Sanity checks for the numba fallback of ``monotonic_align.core``.

Contract (see ``piper/train/vits/monotonic_align/__init__.py``):
``neg_cent`` is ``[b, t_y, t_x]``; row ``y`` selects one column ``x`` with
``x <= y``, columns are non-decreasing, adjacent rows differ by 0 or 1.

Runs standalone (no GPU, no training data):

    .venv/Scripts/python.exe scripts/test_monotonic_align.py
"""

from __future__ import annotations

import itertools

import numpy as np

from piper.train.vits.monotonic_align import maximum_path_c


def column_ranges(t_y: int, t_x: int) -> list[range]:
    """Allowed columns per row, mirroring the Cython loop bounds."""
    return [range(max(0, t_x + y - t_y), min(t_x, y + 1)) for y in range(t_y)]


def brute_force_max(neg_cent: np.ndarray) -> float:
    """Best achievable path cost, enumerated exhaustively (tiny inputs only)."""
    t_y, t_x = neg_cent.shape
    best = -np.inf
    for combo in itertools.product(*column_ranges(t_y, t_x)):
        if combo[0] != 0 or combo[-1] != t_x - 1:
            continue
        if any(b - a not in (0, 1) for a, b in zip(combo, combo[1:])):
            continue
        best = max(best, sum(float(neg_cent[y, x]) for y, x in enumerate(combo)))
    return best


def check_shape(t_y: int, t_x: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    neg_cent = rng.normal(size=(t_y, t_x)).astype(np.float32)
    path = np.zeros((t_y, t_x), dtype=np.int32)

    maximum_path_c(
        path[None],
        neg_cent[None].copy(),
        np.array([t_y], dtype=np.int32),
        np.array([t_x], dtype=np.int32),
    )

    assert (path.sum(axis=1) == 1).all(), f"row mask broken for {t_y}x{t_x}"
    cols = path.argmax(axis=1)
    assert (np.diff(cols) >= 0).all(), f"non-monotonic path for {t_y}x{t_x}"
    assert cols[0] == 0 and cols[-1] == t_x - 1, "path must span both ends"
    for y, x in enumerate(cols):
        assert x in column_ranges(t_y, t_x)[y], f"row {y}: col {x} out of bounds"

    got = float((neg_cent * path).sum())
    want = brute_force_max(neg_cent)
    assert abs(got - want) < 1e-3, f"{t_y}x{t_x}: got {got}, optimal {want}"
    print(f"ok {t_y}x{t_x}: total={got:.4f}")


def check_batch() -> None:
    rng = np.random.default_rng(7)
    shapes = [(5, 4), (6, 3), (7, 7)]
    b = len(shapes)
    t_y_max = max(t for t, _ in shapes)
    t_x_max = max(s for _, s in shapes)
    neg_cent = rng.normal(size=(b, t_y_max, t_x_max)).astype(np.float32)
    path = np.zeros((b, t_y_max, t_x_max), dtype=np.int32)
    t_ys = np.array([t for t, _ in shapes], dtype=np.int32)
    t_xs = np.array([s for _, s in shapes], dtype=np.int32)
    maximum_path_c(path, neg_cent.copy(), t_ys, t_xs)
    for i, (t_y, t_x) in enumerate(shapes):
        assert path[i, :t_y, :t_x].sum() == t_y, f"batch item {i} not aligned"
    print(f"ok batch of {b}")


if __name__ == "__main__":
    for seed, (t_y, t_x) in enumerate([(4, 3), (5, 4), (6, 5), (3, 3)]):
        check_shape(t_y, t_x, seed)
    check_batch()
    print("all monotonic_align checks passed")
