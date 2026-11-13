"""Load an ambientCG material folder into one (N, N, 8) uint8 array.

Channel order (the decoder's 8 outputs):
  0-2 albedo RGB, 3-4 normal XY (OpenGL convention, Z is reconstructed),
  5 roughness, 6 ambient occlusion (255 when the set has none), 7 height.
"""

import glob
import os

import numpy as np
from PIL import Image

CHANNEL_NAMES = ["albedo_r", "albedo_g", "albedo_b", "normal_x", "normal_y",
                 "roughness", "ao", "height"]
GROUPS = {"albedo": [0, 1, 2], "normal": [3, 4], "roughness": [5], "ao": [6], "height": [7]}

_SUFFIX = {"color": "_Color", "normal": "_NormalGL", "roughness": "_Roughness",
           "ao": "_AmbientOcclusion", "height": "_Displacement"}


def _find(folder, suffix):
    hits = glob.glob(os.path.join(folder, f"*{suffix}.png"))
    return hits[0] if hits else None


def _load(path, mode, size):
    img = Image.open(path)
    if img.mode in ("I;16", "I;16B", "I") and mode == "L":
        # 16-bit maps (displacement): 0..65535 -> 0..255.
        a = np.asarray(img, dtype=np.float64)
        img = Image.fromarray(np.clip(np.round(a / 257.0), 0, 255).astype(np.uint8), "L")
    img = img.convert(mode)
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def load_material(folder, size=1024):
    color = _find(folder, _SUFFIX["color"])
    normal = _find(folder, _SUFFIX["normal"])
    rough = _find(folder, _SUFFIX["roughness"])
    height = _find(folder, _SUFFIX["height"])
    missing = [n for n, p in [("color", color), ("normal", normal), ("roughness", rough),
                              ("height", height)] if p is None]
    if missing:
        raise FileNotFoundError(f"{folder}: missing {missing}")
    ao = _find(folder, _SUFFIX["ao"])

    out = np.empty((size, size, 8), dtype=np.uint8)
    out[..., 0:3] = _load(color, "RGB", size)
    out[..., 3:5] = _load(normal, "RGB", size)[..., 0:2]
    out[..., 5] = _load(rough, "L", size)
    out[..., 6] = _load(ao, "L", size) if ao else 255
    out[..., 7] = _load(height, "L", size)
    return out


def psnr(a, b, channels=None):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if channels is not None:
        a, b = a[..., channels], b[..., channels]
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
