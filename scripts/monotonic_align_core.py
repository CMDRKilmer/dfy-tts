"""Pure-Python / numba implementation of Piper's ``monotonic_align.core``.

The official ``piper-tts`` wheel ships the Cython source of the VITS
monotonic alignment search but on Windows the compiled extension is not
included in the wheel, so ``from .monotonic_align.core import maximum_path_c``
fails. This module re-implements the exact same algorithm with numba JIT so no
C compiler / MSVC Build Tools are required.

Algorithm is a faithful port of ``src/piper/train/vits/monotonic_align/core.pyx``
from https://github.com/OHF-Voice/piper1-gpl.
"""

import numpy as np

try:  # numba ships with librosa, i.e. it is already a training dependency
    from numba import njit

    @njit(cache=True)
    def _maximum_path_each(path, value, t_y, t_x, max_neg_val):
        index = t_x - 1
        for y in range(t_y):
            x_start = max(0, t_x + y - t_y)
            x_end = min(t_x, y + 1)
            for x in range(x_start, x_end):
                if x == y:
                    v_cur = max_neg_val
                else:
                    v_cur = value[y - 1, x]
                if x == 0:
                    if y == 0:
                        v_prev = 0.0
                    else:
                        v_prev = max_neg_val
                else:
                    v_prev = value[y - 1, x - 1]
                value[y, x] += max(v_prev, v_cur)

        for y in range(t_y - 1, -1, -1):
            path[y, index] = 1
            if index != 0:
                if index == y:
                    index = index - 1
                elif y > 0 and value[y - 1, index] < value[y - 1, index - 1]:
                    index = index - 1

except ImportError:  # pragma: no cover - numba is always present in [train]
    def _maximum_path_each(path, value, t_y, t_x, max_neg_val):
        index = t_x - 1
        for y in range(t_y):
            x_start = max(0, t_x + y - t_y)
            x_end = min(t_x, y + 1)
            for x in range(x_start, x_end):
                if x == y:
                    v_cur = max_neg_val
                else:
                    v_cur = value[y - 1, x]
                if x == 0:
                    v_prev = 0.0 if y == 0 else max_neg_val
                else:
                    v_prev = value[y - 1, x - 1]
                value[y, x] += max(v_prev, v_cur)

        for y in range(t_y - 1, -1, -1):
            path[y, index] = 1
            if index != 0:
                if index == y:
                    index = index - 1
                elif y > 0 and value[y - 1, index] < value[y - 1, index - 1]:
                    index = index - 1


def maximum_path_c(paths, values, t_ys, t_xs, max_neg_val=-1e9):
    """In-place monotonic alignment, same contract as the Cython version.

    paths:  int32 array [b, t_t, t_s] (zeroed, filled with the alignment mask)
    values: float32 array [b, t_t, t_s] (negated cent cost, modified in place)
    t_ys:   int32 array [b] (number of valid t_t rows per sample)
    t_xs:   int32 array [b] (number of valid t_s columns per sample)
    """
    paths = np.ascontiguousarray(paths, dtype=np.int32)
    values = np.ascontiguousarray(values, dtype=np.float32)
    t_ys = np.ascontiguousarray(t_ys, dtype=np.int32)
    t_xs = np.ascontiguousarray(t_xs, dtype=np.int32)

    for i in range(paths.shape[0]):
        _maximum_path_each(
            paths[i], values[i], int(t_ys[i]), int(t_xs[i]), float(max_neg_val)
        )
