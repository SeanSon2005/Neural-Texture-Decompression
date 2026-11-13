"""Bit-exact reference decoder that reads only the exported files.

This is the oracle for the Vortex kernels: same atlas, same TEX math, same
integer MLP contract (quant.py). It never touches the training code.
"""

import json
import os

import numpy as np

from . import tex_model
from .atlas import AtlasLayout, STAGES, MIPS, CHANNELS
from .quant import mlp_forward


class ReferenceDecoder:
    def __init__(self, export_dir):
        with open(os.path.join(export_dir, "manifest.json")) as f:
            self.manifest = m = json.load(f)
        lj = m["layout"]
        self.layout = AtlasLayout(lj["num_materials"], tuple(lj["log_tile"]))
        assert (self.layout.log_tiles_x, self.layout.log_tiles_y) == (lj["log_tiles_x"], lj["log_tiles_y"])

        self.levels = []  # [stage][mip] -> (H, W, 4) uint8
        for stage, a in enumerate(m["atlas"]):
            words = np.fromfile(os.path.join(export_dir, a["file"]), dtype="<u4")
            levels = []
            for mip in range(MIPS):
                log_w, log_h = a["log_w"] - mip, a["log_h"] - mip
                start = a["mip_offsets"][mip] // 4
                lvl = words[start:start + (1 << (log_w + log_h))].reshape(1 << log_h, 1 << log_w)
                levels.append(tex_model.words_to_bytes(lvl))
            self.levels.append(levels)

        blob = np.fromfile(os.path.join(export_dir, "weights.bin"), dtype=np.uint8)
        self.layers = []
        for mat in m["materials"]:
            layers = []
            for l in mat["layers"]:
                n_w = l["out"] * l["in"]
                w = blob[l["w_offset"]:l["w_offset"] + n_w].view(np.int8).reshape(l["out"], l["in"])
                b = blob[l["b_offset"]:l["b_offset"] + 4 * l["out"]].view("<i4")
                layers.append({"w": w, "b": b, "shift": l["shift"]})
            self.layers.append(layers)

    def features(self, material, u_local, v_local):
        """(N, 32) int feature values (byte - 128), word-major."""
        ua, va = self.layout.atlas_uv(material, u_local, v_local)
        cols = []
        for stage in range(STAGES):
            for mip in range(MIPS):
                log_w, log_h = self.layout.level_log_dims(stage, mip)
                texel = tex_model.sample_level(self.levels[stage][mip], ua, va, log_w, log_h)
                cols.append(texel.astype(np.int64) - 128)
        return np.concatenate(cols, axis=-1)

    def texel_words(self, material, u_local, v_local):
        """(N, 8) uint32 TEX results in word order, for checking a kernel's raw samples."""
        ua, va = self.layout.atlas_uv(material, u_local, v_local)
        words = []
        for stage in range(STAGES):
            for mip in range(MIPS):
                log_w, log_h = self.layout.level_log_dims(stage, mip)
                words.append(tex_model.bytes_to_words(
                    tex_model.sample_level(self.levels[stage][mip], ua, va, log_w, log_h)))
        return np.stack(words, axis=-1)

    def decode(self, material, u_local, v_local):
        material = np.asarray(material, dtype=np.int64)
        x = self.features(material, u_local, v_local)
        out = np.zeros(material.shape + (CHANNELS,), dtype=np.uint8)
        for mat in np.unique(material):
            sel = material == mat
            out[sel] = mlp_forward(x[sel], self.layers[int(mat)])
        return out

    def decode_image(self, material, log_size, rows_per_chunk=64):
        """Decode a full 2**log_size square image of one material, chunked by rows."""
        from .atlas import pixel_local_uv
        n = 1 << log_size
        img = np.zeros((n, n, CHANNELS), dtype=np.uint8)
        xs = np.arange(n)
        for y0 in range(0, n, rows_per_chunk):
            ys = np.arange(y0, min(n, y0 + rows_per_chunk))
            yy, xx = np.meshgrid(ys, xs, indexing="ij")
            u, v = pixel_local_uv(xx, yy, log_size)
            img[ys[0]:ys[-1] + 1] = self.decode(np.full(u.shape, material), u, v)
        return img
