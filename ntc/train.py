"""Train, quantize and export a material set.

  python -m ntc.train --raw data/raw --out data/export/h32 --hidden 32

One process per material (they are independent); the export is then decoded by
the reference decoder, which is what the PSNR table reports.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import time

import numpy as np
import torch

from .atlas import AtlasLayout, pixel_local_uv, ONE
from .data import load_material, psnr, GROUPS
from .export import write_export
from .model import NTCMaterial, compute_taps
from .reference import ReferenceDecoder

DEFAULT_MATERIALS = ["Bricks059", "PavingStones070", "Rock030", "Ground054",
                     "Fabric030", "Planks021", "Gravel022", "Metal032"]


def batch(layout, material, img, log_size, n, rng):
    size = 1 << log_size
    x = rng.integers(0, size, n)
    y = rng.integers(0, size, n)
    u, v = pixel_local_uv(x, y, log_size)
    taps = compute_taps(layout, np.full(n, material), u, v)
    target = torch.from_numpy(img[y, x].astype(np.float32))
    return taps, target


def run_phase(model, layout, material, img, log_size, steps, bs, lr_lat, lr_mlp, rng, log, tag):
    opt = torch.optim.Adam([
        {"params": model.latents.parameters(), "lr": lr_lat},
        {"params": list(model.weights.parameters()) + list(model.biases.parameters()), "lr": lr_mlp},
    ])
    base = [g["lr"] for g in opt.param_groups]
    t0 = time.time()
    for step in range(steps):
        scale = 0.5 * (1 + math.cos(math.pi * step / steps))  # cosine decay
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * scale
        taps, target = batch(layout, material, img, log_size, bs, rng)
        loss = torch.mean((model(taps) - target) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            p = 10 * math.log10(255.0 ** 2 / max(loss.item(), 1e-9))
            log.append({"phase": tag, "step": step, "mse": loss.item(), "psnr": p,
                        "sec": round(time.time() - t0, 1)})
            print(f"[m{material}] {tag} {step:6d} mse={loss.item():8.3f} psnr={p:6.2f}", flush=True)


def train_one(args_tuple):
    (material, folder, layout_n, hidden, steps_float, steps_qat, bs, seed, threads) = args_tuple
    torch.set_num_threads(threads)
    torch.manual_seed(seed + material)
    rng = np.random.default_rng(seed + material)
    layout = AtlasLayout(layout_n)
    img = load_material(folder)
    log_size = int(math.log2(img.shape[0]))
    model = NTCMaterial(layout, hidden=hidden, seed=seed + material)
    log = []
    run_phase(model, layout, material, img, log_size, steps_float, bs, 1e-2, 2e-3, rng, log, "float")
    calib, _ = batch(layout, material, img, log_size, 1 << 15, rng)
    model.start_qat(calib)
    run_phase(model, layout, material, img, log_size, steps_qat, bs, 2e-3, 5e-4, rng, log, "qat")
    return {
        "material": material,
        "tiles": model.export_latents(),
        "layers": model.export_layers(),
        "log": log,
        "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
    }


def make_requests(layout, n, rng):
    """Test vectors: random materials and local coordinates, edges included."""
    m = rng.integers(0, layout.num_materials, n)
    u = rng.integers(0, ONE, n)
    v = rng.integers(0, ONE, n)
    edge = rng.random(n) < 0.05  # exercise the edge clamp
    u[edge] = rng.choice([0, ONE - 1], edge.sum())
    return np.stack([m, u, v], axis=1).astype(np.uint32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--materials", nargs="*", default=DEFAULT_MATERIALS)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--steps-float", type=int, default=4000)
    ap.add_argument("--steps-qat", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--requests", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    n = len(args.materials)
    layout = AtlasLayout(n)
    threads = max(1, (os.cpu_count() or 8) // min(args.jobs, n))
    work = [(i, os.path.join(args.raw, name), n, args.hidden, args.steps_float, args.steps_qat,
             args.batch, args.seed, threads) for i, name in enumerate(args.materials)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(args.jobs, n)) as pool:
        results = sorted(pool.map(train_one, work), key=lambda r: r["material"])
    print(f"trained {n} materials in {time.time() - t0:.0f}s")

    os.makedirs(args.out, exist_ok=True)
    for r, name in zip(results, args.materials):
        torch.save(r["state"], os.path.join(args.out, f"{name}.pt"))

    rng = np.random.default_rng(args.seed + 1000)
    requests = make_requests(layout, args.requests, rng)
    tiles = [r["tiles"] for r in results]
    layers = [r["layers"] for r in results]
    write_export(args.out, layout, args.materials, tiles, layers, args.hidden)
    ref = ReferenceDecoder(args.out)
    expected = ref.decode(requests[:, 0], requests[:, 1], requests[:, 2])

    # Quality and size, measured on the exported (integer) model.
    table = []
    for i, name in enumerate(args.materials):
        img = load_material(os.path.join(args.raw, name))
        dec = ref.decode_image(i, int(math.log2(img.shape[0])))
        row = {"material": name, "psnr_all": round(float(psnr(dec, img)), 3)}
        for g, ch in GROUPS.items():
            row[f"psnr_{g}"] = round(float(psnr(dec, img, ch)), 3)
        table.append(row)
        print(row, flush=True)

    feature_bytes = sum(a["bytes"] for a in ref.manifest["atlas"])
    weight_bytes = os.path.getsize(os.path.join(args.out, "weights.bin"))
    size = {"feature_bytes": feature_bytes, "weight_bytes": weight_bytes,
            "table_bytes": 8 * n, "raw_bytes": n * (1 << 20) * 8}
    size["ratio"] = round(size["raw_bytes"] / (feature_bytes + weight_bytes + 8 * n), 2)
    write_export(args.out, layout, args.materials, tiles, layers, args.hidden,
                 requests=requests, expected=expected,
                 extra={"quality": table, "size": size,
                        "training": {"steps_float": args.steps_float, "steps_qat": args.steps_qat,
                                     "batch": args.batch, "seed": args.seed,
                                     "log": {name: r["log"] for r, name in zip(results, args.materials)}}})
    print(json.dumps(size))


if __name__ == "__main__":
    main()
