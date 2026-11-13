"""PSNR / size table across exported material sets, and the choice of H.

  python -m ntc.summarize data/export/h16 data/export/h32 data/export/h64 \
      --csv results/ntc_quality_size.csv
"""

import argparse
import csv
import json
import os

GROUPS = ["albedo", "normal", "roughness", "ao", "height"]


def load(export_dir):
    with open(os.path.join(export_dir, "manifest.json")) as f:
        m = json.load(f)
    rows = []
    for q in m["quality"]:
        row = {"hidden": m["hidden"], "material": q["material"], "psnr_all": q["psnr_all"]}
        row.update({f"psnr_{g}": q[f"psnr_{g}"] for g in GROUPS})
        row.update(m["size"])
        rows.append(row)
    return rows


def summary_row(rows):
    n = len(rows)
    s = {"hidden": rows[0]["hidden"],
         "mean_psnr_all": round(sum(r["psnr_all"] for r in rows) / n, 3),
         "min_psnr_all": min(r["psnr_all"] for r in rows)}
    s.update({f"mean_psnr_{g}": round(sum(r[f"psnr_{g}"] for r in rows) / n, 3) for g in GROUPS})
    for k in ("feature_bytes", "weight_bytes", "table_bytes", "raw_bytes", "ratio"):
        s[k] = rows[0][k]
    return s


def choose_hidden(summaries):
    """Pick the decoder width H used for every later phase.

    summaries: one dict per H with mean_psnr_all, min_psnr_all, the per-group
    mean PSNRs, weight_bytes, feature_bytes and ratio (see summary_row).
    Returns the chosen H (an int).
    """
    # TODO(human)
    raise NotImplementedError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exports", nargs="+")
    ap.add_argument("--csv")
    args = ap.parse_args()

    per_material, summaries = [], []
    for d in args.exports:
        rows = load(d)
        per_material += rows
        summaries.append(summary_row(rows))

    keys = ["hidden", "mean_psnr_all", "min_psnr_all"] + [f"mean_psnr_{g}" for g in GROUPS] + \
           ["feature_bytes", "weight_bytes", "ratio"]
    print(" | ".join(keys))
    for s in sorted(summaries, key=lambda s: s["hidden"]):
        print(" | ".join(str(s[k]) for k in keys))

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_material[0].keys()))
            w.writeheader()
            w.writerows(per_material)
        print(f"wrote {args.csv}")

    print(f"chosen H = {choose_hidden(summaries)}")


if __name__ == "__main__":
    main()
