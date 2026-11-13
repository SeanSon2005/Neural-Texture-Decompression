"""Per-material neural texture model: two 4-level latent pyramids + a 32->H->H->8 MLP.

Training runs in two phases:
  float : latents in [0, 1], plain float bilinear (TEX weights alpha/256), float MLP.
  qat   : forward pass is the exact integer contract (quant.py + TEX rounding);
          gradients flow through every rounding via straight-through estimators.

Tap positions and 8-bit weights come from the exact TEX addressing math on the
atlas coordinate (tex_model.bilinear_taps), shifted into the material's tile,
so the qat forward equals what Vortex computes, bit for bit.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tex_model
from .atlas import AtlasLayout, STAGES, MIPS, CHANNELS, FEATS
from .quant import W_MAX, HIDDEN_MAX, OUT_MAX


def ste_round(x):
    return x + (torch.round(x) - x).detach()


def ste_floor(x):
    return x + (torch.floor(x) - x).detach()


def ste_clamp(x, lo, hi):
    # Clamp in the forward pass; let gradient through only inside the range.
    return torch.clamp(x, lo, hi)


def compute_taps(layout: AtlasLayout, material, u_local, v_local):
    """Integer taps for every (stage, mip) level, local to the material's tile.

    Returns a list of WORDS tuples (x0, x1, y0, y1, alpha, beta) of int64 tensors.
    """
    ua, va = layout.atlas_uv(material, u_local, v_local)
    tx, ty = layout.tile_xy(material)
    taps = []
    for stage in range(STAGES):
        for mip in range(MIPS):
            log_w, log_h = layout.level_log_dims(stage, mip)
            x0, x1, y0, y1, a, b = tex_model.bilinear_taps(ua, va, log_w, log_h)
            ts = 1 << layout.tile_log_size(stage, mip)
            ox, oy = tx * ts, ty * ts
            local = [x0 - ox, x1 - ox, y0 - oy, y1 - oy]
            assert all((t >= 0).all() and (t < ts).all() for t in local), "tap left the tile"
            taps.append(tuple(torch.from_numpy(np.ascontiguousarray(t)) for t in local + [a, b]))
    return taps


class NTCMaterial(nn.Module):
    def __init__(self, layout: AtlasLayout, hidden=32, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.layout = layout
        self.hidden = hidden
        self.latents = nn.ParameterList()
        for stage in range(STAGES):
            for mip in range(MIPS):
                n = 1 << layout.tile_log_size(stage, mip)
                self.latents.append(nn.Parameter(0.5 + 0.1 * torch.randn(n, n, 4, generator=g)))
        dims = [FEATS, hidden, hidden, CHANNELS]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for i in range(3):
            w = torch.randn(dims[i + 1], dims[i], generator=g) * math.sqrt(2.0 / dims[i])
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(torch.zeros(dims[i + 1])))
        self.qat = False
        # Integer-domain scales, fixed when qat starts (see start_qat).
        # Parameters stay in float units so one learning rate suits both phases;
        # the integer value is round(param * 2**exp).
        self.register_buffer("w_exp", torch.zeros(3, dtype=torch.int64))
        self.register_buffer("b_exp", torch.zeros(3, dtype=torch.int64))
        self.register_buffer("shifts", torch.ones(3, dtype=torch.int64))

    # ---- features -------------------------------------------------------
    def _sample(self, lat, tap, quantized):
        x0, x1, y0, y1, a, b = tap
        a = a.to(lat.dtype)[:, None]
        b = b.to(lat.dtype)[:, None]
        t00, t01, t10, t11 = lat[y0, x0], lat[y0, x1], lat[y1, x0], lat[y1, x1]
        if not quantized:
            c01 = (t00 * (256 - a) + t01 * a) / 256
            c23 = (t10 * (256 - a) + t11 * a) / 256
            return (c01 * (256 - b) + c23 * b) / 256
        lerp = lambda p, q, f: ste_floor((p * (256 - f) + q * f + 128) / 256)
        return lerp(lerp(t00, t01, a), lerp(t10, t11, a), b)

    def latent_bytes(self, i):
        return ste_round(torch.clamp(self.latents[i], 0, 1) * 255)

    def features(self, taps):
        """(N, 32): float phase in [-1, 1); qat phase the exact int8 values byte - 128."""
        cols = []
        for i, tap in enumerate(taps):
            if self.qat:
                cols.append(self._sample(self.latent_bytes(i), tap, True) - 128)
            else:
                lat = torch.clamp(self.latents[i], 0, 1) * 255
                cols.append((self._sample(lat, tap, False) - 128) / 128)
        return torch.cat(cols, dim=1)  # word-major, byte order within a word = channel order

    # ---- MLP --------------------------------------------------------------
    def forward(self, taps):
        """Returns predicted channel bytes / 256 (float phase) or exact bytes (qat)."""
        x = self.features(taps)
        if not self.qat:
            for i in range(3):
                x = F.linear(x, self.weights[i], self.biases[i])
                if i < 2:
                    x = F.relu(x)
            return x * 256  # float phase trains in byte units too
        for i in range(3):
            w = ste_clamp(ste_round(self.weights[i] * (2.0 ** self.w_exp[i])), -W_MAX, W_MAX)
            b = ste_round(self.biases[i] * (2.0 ** self.b_exp[i]))
            acc = F.linear(x, w) + b
            s = int(self.shifts[i])
            hi = OUT_MAX if i == 2 else HIDDEN_MAX
            x = ste_clamp(ste_floor((acc + (1 << (s - 1))) / (1 << s)), 0, hi)
        return x

    # ---- float -> integer conversion ----------------------------------------
    @torch.no_grad()
    def start_qat(self, taps):
        """Fix power-of-two scales from activation statistics on a calibration batch,
        then reparameterize weights/biases into the integer domain."""
        x = self.features(taps)
        in_exp = 7  # float features are (byte - 128) / 128
        for i in range(3):
            w = self.weights[i]
            z = F.linear(x, w, self.biases[i])
            e = int(math.floor(math.log2(W_MAX / w.abs().max().clamp_min(1e-8).item())))
            if i < 2:
                zmax = torch.quantile(F.relu(z).flatten(), 0.999).clamp_min(1e-6).item()
                out_exp = int(math.floor(math.log2(HIDDEN_MAX / zmax)))
            else:
                out_exp = 8  # output byte = z * 256
            shift = in_exp + e - out_exp
            if shift < 1:
                # Keep the contract's s >= 1 by scaling weights up; the largest may
                # then clip at +-127, which qat adapts to.
                e += 1 - shift
                shift = 1
            self.w_exp[i] = e
            self.b_exp[i] = in_exp + e
            self.shifts[i] = shift
            x = F.relu(z) if i < 2 else z
            in_exp = out_exp
        self.qat = True

    # ---- export -------------------------------------------------------------
    @torch.no_grad()
    def export_layers(self):
        assert self.qat
        layers = []
        for i in range(3):
            w = torch.clamp(torch.round(self.weights[i] * (2.0 ** self.w_exp[i])), -W_MAX, W_MAX)
            layers.append({
                "w": w.to(torch.int64).numpy().astype(np.int8),
                "b": torch.round(self.biases[i] * (2.0 ** self.b_exp[i])).to(torch.int64).numpy().astype(np.int32),
                "shift": int(self.shifts[i]),
            })
        return layers

    @torch.no_grad()
    def export_latents(self):
        """List of WORDS (n, n, 4) uint8 tiles, index = stage * MIPS + mip."""
        return [torch.round(torch.clamp(l, 0, 1) * 255).to(torch.uint8).numpy() for l in self.latents]
