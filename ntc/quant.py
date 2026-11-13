"""The integer decode contract shared by the reference decoder and the Vortex kernels.

Per request:
  x[k]  = feature_byte[k] - 128                          int8, k < 32
  layer l (l = 1, 2, 3):
    acc[n] = bias_l[n] + sum_k W_l[n, k] * in[k]         int32
    hidden : out[n] = clamp((acc[n] + 2**(s_l - 1)) >> s_l, 0, 127)
    output : out[n] = clamp((acc[n] + 2**(s_3 - 1)) >> s_3, 0, 255)  (one byte per channel)

Shifts are arithmetic (floor). Every s_l >= 1. Weights are int8 in [-127, 127].
Feature bytes are ordered word-major: feature k is byte (k % 4) (little endian)
of texel word k // 4, and word w = stage * 4 + mip.
"""

import numpy as np

W_MAX = 127
HIDDEN_MAX = 127
OUT_MAX = 255


def requant(acc, shift, hi):
    acc = np.asarray(acc, dtype=np.int64)
    return np.clip((acc + (1 << (shift - 1))) >> shift, 0, hi)


def int_matmul(x, w):
    """Exact x @ w.T for small-integer arrays using float64 BLAS.

    |x| <= 255, |w| <= 127, K <= 256 keeps every partial sum below 2**53.
    """
    assert x.shape[-1] <= 256
    return np.rint(x.astype(np.float64) @ w.astype(np.float64).T).astype(np.int64)


def mlp_forward(x, layers):
    """x: (N, 32) int feature values (already byte - 128). layers: list of dicts
    with int arrays 'w' (out, in), 'b' (out,) and int 'shift'. Returns (N, C) uint8."""
    h = np.asarray(x, dtype=np.int64)
    for i, layer in enumerate(layers):
        acc = int_matmul(h, layer["w"]) + layer["b"].astype(np.int64)
        last = i == len(layers) - 1
        h = requant(acc, int(layer["shift"]), OUT_MAX if last else HIDDEN_MAX)
    return h.astype(np.uint8)
