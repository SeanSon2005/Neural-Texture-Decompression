"""Write a trained material set in the layout the Vortex host programs read.

<out>/manifest.json   geometry, per-stage atlas levels and mip offsets, per-material
                     layer shapes, shifts and byte offsets into weights.bin
<out>/atlas_s{0,1}.bin   uint32 texel words (0xAARRGGBB), mip levels 0..3 back to back,
                     row-major; mip offsets in bytes (TEX VX_DCR_TEX_MIPOFF units)
<out>/weights.bin    per material, per layer: int8 W (out x in, row-major) then
                     int32 bias (out), each block 4-byte aligned
<out>/materials.bin  uint32 (origin_u, origin_v) per material, atlas fixed point
<out>/requests.bin   test vectors: uint32 (material, u_local, v_local) per request
<out>/expected.bin   uint8[8] decoded channels per request (reference decoder)
"""

import json
import os

import numpy as np

from . import tex_model
from .atlas import AtlasLayout, STAGES, MIPS, CHANNELS, FEATS


def build_atlas(layout: AtlasLayout, tiles_per_material):
    """tiles_per_material[m][stage * MIPS + mip] = (n, n, 4) uint8 -> per-stage
    lists of (H, W, 4) uint8 atlas levels."""
    atlas = []
    for stage in range(STAGES):
        levels = []
        for mip in range(MIPS):
            log_w, log_h = layout.level_log_dims(stage, mip)
            ts = 1 << layout.tile_log_size(stage, mip)
            level = np.zeros((1 << log_h, 1 << log_w, 4), dtype=np.uint8)
            for m, tiles in enumerate(tiles_per_material):
                tx, ty = layout.tile_xy(m)
                level[ty * ts:(ty + 1) * ts, tx * ts:(tx + 1) * ts] = tiles[stage * MIPS + mip]
            levels.append(level)
        atlas.append(levels)
    return atlas


def _align4(f):
    pad = (-f.tell()) % 4
    f.write(b"\0" * pad)


def write_export(out_dir, layout: AtlasLayout, names, tiles_per_material, layers_per_material,
                 hidden, requests=None, expected=None, extra=None):
    os.makedirs(out_dir, exist_ok=True)
    atlas = build_atlas(layout, tiles_per_material)
    manifest = {"layout": layout.to_json(), "hidden": hidden, "channels": CHANNELS,
                "features": FEATS, "materials": [], "atlas": []}

    for stage, levels in enumerate(atlas):
        fname = f"atlas_s{stage}.bin"
        offsets, off = [], 0
        with open(os.path.join(out_dir, fname), "wb") as f:
            for level in levels:
                offsets.append(off)
                words = tex_model.bytes_to_words(level)
                f.write(words.astype("<u4").tobytes())
                off += words.size * 4
        log_w, log_h = layout.level_log_dims(stage, 0)
        manifest["atlas"].append({"file": fname, "log_w": log_w, "log_h": log_h,
                                  "mip_offsets": offsets, "bytes": off})

    with open(os.path.join(out_dir, "weights.bin"), "wb") as f:
        for name, layers in zip(names, layers_per_material):
            entry = {"name": name, "layers": []}
            for layer in layers:
                _align4(f)
                w_off = f.tell()
                f.write(np.ascontiguousarray(layer["w"], dtype=np.int8).tobytes())
                _align4(f)
                b_off = f.tell()
                f.write(np.ascontiguousarray(layer["b"], dtype="<i4").tobytes())
                entry["layers"].append({"out": int(layer["w"].shape[0]), "in": int(layer["w"].shape[1]),
                                        "shift": int(layer["shift"]), "w_offset": w_off, "b_offset": b_off})
            manifest["materials"].append(entry)

    layout.material_table().astype("<u4").tofile(os.path.join(out_dir, "materials.bin"))
    if requests is not None:
        np.asarray(requests, dtype="<u4").tofile(os.path.join(out_dir, "requests.bin"))
        np.asarray(expected, dtype=np.uint8).tofile(os.path.join(out_dir, "expected.bin"))
        manifest["num_requests"] = int(len(requests))
    if extra:
        manifest.update(extra)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
