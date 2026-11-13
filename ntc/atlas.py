"""Atlas geometry shared by training, export, the reference decoder and Vortex.

TEX state is global per stage, so every material's feature pyramid lives in one
atlas per stage. Both stages use the same tile grid, so one atlas (u, v) feeds
all eight vx_tex calls of a request.

A request carries a material id and material-local (u, v) in TEX fixed point
(FRAC fractional bits, [0, 1)). The decoder clamps the local coordinate so that
no bilinear tap at any level leaves the material's tile, then maps it into the
atlas with shifts and an add. Pixels within half a coarsest-level texel of a
material edge are therefore sampled at the clamped position; training sees the
same clamp, so this is part of the model, not a deployment error.
"""

from dataclasses import dataclass, field
import math

import numpy as np

from .tex_model import FRAC, ONE

STAGES = 2
MIPS = 4
WORDS = STAGES * MIPS           # texel words per request; word w = stage * MIPS + mip
FEATS = WORDS * 4               # int8 features per request (one TCU K tile)
CHANNELS = 8                    # decoded material channels


def _log2(n):
    l = int(round(math.log2(n)))
    assert 1 << l == n, f"{n} is not a power of two"
    return l


@dataclass
class AtlasLayout:
    num_materials: int
    log_tile: tuple = (8, 7)    # level-0 tile size per stage: 256, 128
    log_tiles_x: int = field(init=False)
    log_tiles_y: int = field(init=False)

    def __post_init__(self):
        n = self.num_materials
        assert n >= 1
        self.log_tiles_x = math.ceil(math.log2(n) / 2) if n > 1 else 0
        self.log_tiles_y = max(0, math.ceil(math.log2(n)) - self.log_tiles_x)
        assert min(self.log_tile) - (MIPS - 1) >= 1, "coarsest level must be at least 2 texels"

    @property
    def tiles_x(self):
        return 1 << self.log_tiles_x

    @property
    def tiles_y(self):
        return 1 << self.log_tiles_y

    def tile_xy(self, material):
        return material % self.tiles_x, material // self.tiles_x

    def level_log_dims(self, stage, mip):
        """(log_w, log_h) of an atlas level."""
        lt = self.log_tile[stage] - mip
        return lt + self.log_tiles_x, lt + self.log_tiles_y

    def tile_log_size(self, stage, mip):
        return self.log_tile[stage] - mip

    @property
    def edge(self):
        """Half a texel of the coarsest tile level, in local fixed point."""
        coarsest = min(self.log_tile) - (MIPS - 1)
        return ONE >> (coarsest + 1)

    @property
    def local_min(self):
        return self.edge

    @property
    def local_max(self):
        return ONE - self.edge - 1

    def atlas_uv(self, material, u_local, v_local):
        """Material-local fixed point -> atlas fixed point (what the kernel computes)."""
        material = np.asarray(material, dtype=np.int64)
        u = np.clip(np.asarray(u_local, dtype=np.int64), self.local_min, self.local_max)
        v = np.clip(np.asarray(v_local, dtype=np.int64), self.local_min, self.local_max)
        tx = material % self.tiles_x
        ty = material // self.tiles_x
        ua = (tx << (FRAC - self.log_tiles_x)) + (u >> self.log_tiles_x)
        va = (ty << (FRAC - self.log_tiles_y)) + (v >> self.log_tiles_y)
        return ua, va

    def material_table(self):
        """Per-material constants the kernel needs: atlas origin in fixed point."""
        rows = []
        for m in range(self.num_materials):
            tx, ty = self.tile_xy(m)
            rows.append((tx << (FRAC - self.log_tiles_x), ty << (FRAC - self.log_tiles_y)))
        return np.array(rows, dtype=np.uint32)

    def to_json(self):
        return {
            "num_materials": self.num_materials,
            "log_tile": list(self.log_tile),
            "log_tiles_x": self.log_tiles_x,
            "log_tiles_y": self.log_tiles_y,
            "mips": MIPS,
            "stages": STAGES,
            "local_min": self.local_min,
            "local_max": self.local_max,
            "frac": FRAC,
        }


def pixel_local_uv(x, y, log_size):
    """Local fixed-point centre of pixel (x, y) of a 2**log_size square image."""
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    return ((2 * x + 1) << FRAC) >> (log_size + 1), ((2 * y + 1) << FRAC) >> (log_size + 1)
