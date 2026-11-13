"""Bit-exact NumPy port of the Vortex v3.0 TEX unit for the subset we use.

Covers A8R8G8B8 textures with power-of-two square or rectangular levels,
POINT or BILINEAR filtering, CLAMP or REPEAT wrap, and an integer mip level
(no MIP_LINEAR blend). Source of truth: vortex/sw/common/gfx_frag_tex.h
(tex_compute_request, TexAddressLinear, TexFilterLinear) and Lerp8888 in
vortex/sw/common/vx_gfx_abi.h.

Texel words are uint32 0xAARRGGBB. Every channel is filtered independently,
so we work on a (..., 4) uint8 view (byte 0 = B ... byte 3 = A, i.e. the
little-endian byte order the TCU sees) and the result is the same.
"""

import numpy as np

TEX_DIM_BITS = 15
TEX_SUBPIXEL_BITS = 8
FRAC = TEX_DIM_BITS + TEX_SUBPIXEL_BITS   # TEX_FXD_FRAC = 23
ONE = 1 << FRAC
MASK = ONE - 1
HALF = 1 << (FRAC - 1)

WRAP_CLAMP = 0
WRAP_REPEAT = 1
FILTER_POINT = 0
FILTER_BILINEAR = 1


def wrap_coord(fx, wrap):
    fx = np.asarray(fx, dtype=np.int64)
    if wrap == WRAP_CLAMP:
        return np.clip(fx, 0, MASK)
    if wrap == WRAP_REPEAT:
        return fx & MASK
    raise ValueError(f"unsupported wrap {wrap}")


def lerp8(a, b, f):
    """Lerp8888 per channel: weight f/256, round to nearest."""
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    return (a * (256 - f) + b * f + 128) >> 8


def sample_level(level, u, v, log_w, log_h, filt=FILTER_BILINEAR, wrap=WRAP_CLAMP):
    """Sample one mip level.

    level: (H, W, 4) uint8 array, H = 2**log_h, W = 2**log_w.
    u, v:  int arrays of FRAC-bit fixed-point coordinates (any shape S).
    Returns (S..., 4) uint8.
    """
    assert level.shape[:2] == (1 << log_h, 1 << log_w)
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    shift_u = FRAC - log_w
    shift_v = FRAC - log_h

    if filt == FILTER_POINT:
        x = wrap_coord(u, wrap) >> shift_u
        y = wrap_coord(v, wrap) >> shift_v
        return level[y, x]

    dx = HALF >> log_w
    dy = HALF >> log_h
    u0 = wrap_coord(u - dx, wrap)
    u1 = wrap_coord(u + dx, wrap)
    v0 = wrap_coord(v - dy, wrap)
    v1 = wrap_coord(v + dy, wrap)

    x0s = (u0 << 8) >> shift_u
    y0s = (v0 << 8) >> shift_v
    x0, y0 = x0s >> 8, y0s >> 8
    x1, y1 = u1 >> shift_u, v1 >> shift_v
    alpha = (x0s & 0xFF)[..., None]
    beta = (y0s & 0xFF)[..., None]

    c01 = lerp8(level[y0, x0], level[y0, x1], alpha)
    c23 = lerp8(level[y1, x0], level[y1, x1], alpha)
    return lerp8(c01, c23, beta).astype(np.uint8)


def bilinear_taps(u, v, log_w, log_h, wrap=WRAP_CLAMP):
    """Integer tap indices and 8-bit weights of a bilinear sample.

    Training uses these to build a differentiable sample whose forward value
    can be made bit-exact (see model.py). Returns x0, x1, y0, y1, alpha, beta.
    """
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    shift_u = FRAC - log_w
    shift_v = FRAC - log_h
    dx = HALF >> log_w
    dy = HALF >> log_h
    u0 = wrap_coord(u - dx, wrap)
    u1 = wrap_coord(u + dx, wrap)
    v0 = wrap_coord(v - dy, wrap)
    v1 = wrap_coord(v + dy, wrap)
    x0s = (u0 << 8) >> shift_u
    y0s = (v0 << 8) >> shift_v
    return x0s >> 8, u1 >> shift_u, y0s >> 8, v1 >> shift_v, x0s & 0xFF, y0s & 0xFF


def words_to_bytes(words):
    """uint32 0xAARRGGBB -> (..., 4) uint8 in little-endian byte order."""
    w = np.ascontiguousarray(np.asarray(words, dtype="<u4"))
    return w.view(np.uint8).reshape(w.shape + (4,))


def bytes_to_words(b):
    b = np.ascontiguousarray(np.asarray(b, dtype=np.uint8))
    assert b.shape[-1] == 4
    return b.view("<u4").reshape(b.shape[:-1])


def pixel_center_uv(x, y, log_w, log_h):
    """Fixed-point coordinate of the centre of texel (x, y) at a level of this size."""
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    return (((2 * x + 1) << FRAC) >> (log_w + 1)), (((2 * y + 1) << FRAC) >> (log_h + 1))
