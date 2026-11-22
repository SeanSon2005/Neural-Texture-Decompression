"""Files the Vortex decoder hosts (tests/regression/ntc_{scalar,tcu}) read, and checks on what they write.

  python -m ntc.vortex_io prepare data/export/h32 [--image-log 7]
      vortex.txt                  flat key/value form of manifest.json (no JSON parser on the host)
      vec_<name>_req.bin          uint32 (material, u_local, v_local) per request
      vec_<name>_exp.bin          uint8[8] reference-decoder output per request
    vector sets: "mixed" (the export's random test vectors, all materials interleaved)
    and "img<m>_<L>" (every pixel centre of a 2**L square image of material m, raster order).

  python -m ntc.vortex_io image data/export/h32 img3_7 dump.bin --raw data/raw --png out.png
      checks a host dump (-d) against the expected file, writes a PNG sheet and PSNR vs the source.
"""

import argparse
import os

import numpy as np

from .atlas import pixel_local_uv, CHANNELS
from .reference import ReferenceDecoder


def write_host_manifest(ref: ReferenceDecoder, export_dir):
    m = ref.manifest
    lj = m["layout"]
    lines = [
        f"hidden {m['hidden']}",
        f"materials {lj['num_materials']}",
        f"log_tiles {lj['log_tiles_x']} {lj['log_tiles_y']}",
        f"local {lj['local_min']} {lj['local_max']}",
    ]
    for s, a in enumerate(m["atlas"]):
        offs = " ".join(str(o) for o in a["mip_offsets"])
        lines.append(f"stage {s} {a['file']} {a['log_w']} {a['log_h']} {a['bytes']} {offs}")
    for i, mat in enumerate(m["materials"]):
        layers = " ".join(f"{l['out']} {l['in']} {l['shift']} {l['w_offset']} {l['b_offset']}"
                          for l in mat["layers"])
        lines.append(f"material {i} {mat['name']} {len(mat['layers'])} {layers}")
    with open(os.path.join(export_dir, "vortex.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


def write_vectors(ref: ReferenceDecoder, export_dir, name, material, u, v):
    material = np.asarray(material, dtype=np.int64)
    req = np.stack([material, np.asarray(u), np.asarray(v)], axis=1).astype("<u4")
    exp = ref.decode(material, u, v)
    req.tofile(os.path.join(export_dir, f"vec_{name}_req.bin"))
    exp.astype(np.uint8).tofile(os.path.join(export_dir, f"vec_{name}_exp.bin"))
    return len(req)


def image_vectors(material, log_size):
    n = 1 << log_size
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    u, v = pixel_local_uv(xx.ravel(), yy.ravel(), log_size)
    return np.full(u.shape, material), u, v


def prepare(export_dir, image_log):
    ref = ReferenceDecoder(export_dir)
    write_host_manifest(ref, export_dir)
    req = np.fromfile(os.path.join(export_dir, "requests.bin"), dtype="<u4").reshape(-1, 3)
    n = write_vectors(ref, export_dir, "mixed", req[:, 0], req[:, 1], req[:, 2])
    print(f"mixed: {n} requests")
    for mat in range(ref.layout.num_materials):
        m, u, v = image_vectors(mat, image_log)
        n = write_vectors(ref, export_dir, f"img{mat}_{image_log}", m, u, v)
    print(f"img<m>_{image_log}: {n} requests x {ref.layout.num_materials} materials")


def _sheet(img):
    """(n, n, 8) channels -> one RGB sheet: albedo | normal | roughness | ao | height."""
    albedo = img[..., 0:3]
    nx, ny = img[..., 3].astype(np.float64) / 127.5 - 1, img[..., 4].astype(np.float64) / 127.5 - 1
    nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
    normal = np.stack([img[..., 3], img[..., 4], np.round((nz + 1) * 127.5)], axis=-1).astype(np.uint8)
    gray = [np.repeat(img[..., c:c + 1], 3, axis=-1) for c in (5, 6, 7)]
    return np.concatenate([albedo, normal] + gray, axis=1)


def check_image(export_dir, name, dump_path, raw_dir, png_path):
    from PIL import Image
    from .data import load_material, psnr, GROUPS

    req = np.fromfile(os.path.join(export_dir, f"vec_{name}_req.bin"), dtype="<u4").reshape(-1, 3)
    exp = np.fromfile(os.path.join(export_dir, f"vec_{name}_exp.bin"), dtype=np.uint8).reshape(-1, CHANNELS)
    got = np.fromfile(dump_path, dtype=np.uint8).reshape(-1, CHANNELS)
    assert got.shape == exp.shape, f"dump has {len(got)} requests, expected {len(exp)}"
    mismatches = int(np.any(got != exp, axis=1).sum())
    n = int(round(np.sqrt(len(got))))
    assert n * n == len(got)
    img = got.reshape(n, n, CHANNELS)
    ref = ReferenceDecoder(export_dir)
    mat = int(req[0, 0])
    mname = ref.manifest["materials"][mat]["name"]
    Image.fromarray(_sheet(img)).save(png_path.format(name=mname))

    src = load_material(os.path.join(raw_dir, mname))
    # source texel under each decoded pixel centre (point sample at the same uv)
    log_src = int(round(np.log2(src.shape[0])))
    from .tex_model import FRAC
    ix = (req[:, 1].astype(np.int64) << log_src) >> FRAC
    iy = (req[:, 2].astype(np.int64) << log_src) >> FRAC
    target = src[iy, ix].reshape(n, n, CHANNELS)
    row = {"material": mname, "size": n, "mismatches": mismatches,
           "psnr_all": round(float(psnr(img, target)), 3)}
    for g, ch in GROUPS.items():
        row[f"psnr_{g}"] = round(float(psnr(img, target, ch)), 3)
    return row


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("export")
    p.add_argument("--image-log", type=int, default=7)
    c = sub.add_parser("image")
    c.add_argument("export")
    c.add_argument("name")
    c.add_argument("dump")
    c.add_argument("--raw", default="data/raw")
    c.add_argument("--png", required=True)
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args.export, args.image_log)
    else:
        row = check_image(args.export, args.name, args.dump, args.raw, args.png)
        print(row)
        if row["mismatches"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
