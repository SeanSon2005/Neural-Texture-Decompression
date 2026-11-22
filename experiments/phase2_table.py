"""Collect phase2_baseline.sh run logs into one CSV and print a Markdown table.

  python experiments/phase2_table.py <points dir> [--csv results/phase2_baseline.csv]
"""

import argparse
import csv
import os
import re

COLS = ["point", "path", "hidden", "vectors", "occ_cap", "requests", "tiles", "occupancy",
        "cycles", "instrs", "ipc", "cycles_per_req", "instrs_per_req",
        "mix_alu", "mix_lsu", "mix_sfu", "mix_fpu", "mix_tcu", "wall_s", "kcycles_per_s", "passed"]


def parse(path):
    text = open(path).read()
    row = {"passed": "PASSED!" in text}
    m = re.search(r"NTC: path=(\S+) hidden=(\d+) vectors=(\S+) requests=(\d+) tiles=(\d+) "
                  r"valid_rows=(\d+) tile_rows=(\d+) occupancy_cap=(\d+)", text)
    if not m:
        return None
    path_, hidden, vectors, reqs, tiles, valid, rows, occ = m.groups()
    row.update(path=path_, hidden=int(hidden), vectors=vectors, requests=int(reqs), tiles=int(tiles))
    row["occ_cap"] = int(occ) if path_ == "tcu" else ""
    row["occupancy"] = round(int(valid) / int(rows), 3) if int(rows) else ""
    m = re.search(r"PERF: instrs=(\d+), cycles=(\d+)", text)
    if m:
        row["instrs"], row["cycles"] = int(m.group(1)), int(m.group(2))
        row["ipc"] = round(row["instrs"] / row["cycles"], 3)
        row["cycles_per_req"] = round(row["cycles"] / row["requests"], 1)
        row["instrs_per_req"] = round(row["instrs"] / row["requests"], 1)
    m = re.search(r"inst_mix:([^\n]*)", text)
    if m:
        for k, v in re.findall(r"(\w+)=(\d+)%", m.group(1)):
            row[f"mix_{k}"] = int(v)
    m = re.search(r"WALL: ([\d.]+)", text)
    if m:
        row["wall_s"] = round(float(m.group(1)), 1)
        if "cycles" in row:
            row["kcycles_per_s"] = round(row["cycles"] / row["wall_s"] / 1e3, 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("points")
    ap.add_argument("--csv")
    args = ap.parse_args()
    rows = []
    for name in sorted(os.listdir(args.points)):
        log = os.path.join(args.points, name, "run.log")
        if os.path.isfile(log):
            r = parse(log)
            if r:
                r["point"] = name
                rows.append(r)
    rows.sort(key=lambda r: (r["hidden"], r["path"], r["vectors"], r["occ_cap"] or 0))
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    show = ["path", "hidden", "vectors", "occ_cap", "occupancy", "cycles_per_req", "instrs_per_req",
            "ipc", "mix_alu", "mix_lsu", "mix_sfu", "mix_tcu", "kcycles_per_s", "passed"]
    print("| " + " | ".join(show) + " |")
    print("|" + "---|" * len(show))
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "")) for c in show) + " |")
    if not all(r["passed"] for r in rows):
        raise SystemExit("some points did not pass the bit-exact check")


if __name__ == "__main__":
    main()
