#!/usr/bin/env python3
"""Download the CC0 ambientCG materials used by the project (1K PNG sets).

Zips go to data/raw/zips/, extracted maps to data/raw/<AssetId>/. SHA-256 of
each zip is checked against data/checksums.txt; run with --record to write it
the first time. Raw textures are never committed (see .gitignore).
"""

import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile

MATERIALS = [
    "Bricks059", "PavingStones070", "Rock030", "Ground054",
    "Fabric030", "Planks021", "Gravel022", "Metal032",
]
RES = "1K-PNG"
URL = "https://ambientcg.com/get?file={id}_{res}.zip"

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
CHECKSUMS = os.path.join(HERE, "checksums.txt")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checksums():
    sums = {}
    if os.path.exists(CHECKSUMS):
        for line in open(CHECKSUMS):
            if line.strip():
                digest, name = line.split()
                sums[name] = digest
    return sums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true", help="write checksums.txt from the downloads")
    args = ap.parse_args()

    os.makedirs(os.path.join(RAW, "zips"), exist_ok=True)
    known = load_checksums()
    recorded = {}
    for asset in MATERIALS:
        name = f"{asset}_{RES}.zip"
        zpath = os.path.join(RAW, "zips", name)
        if not os.path.exists(zpath):
            print(f"downloading {name}", flush=True)
            req = urllib.request.Request(URL.format(id=asset, res=RES),
                                         headers={"User-Agent": "ntc-course-project-fetch/1.0"})
            with urllib.request.urlopen(req) as r, open(zpath + ".part", "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            os.replace(zpath + ".part", zpath)
        digest = sha256(zpath)
        recorded[name] = digest
        if name in known and known[name] != digest:
            sys.exit(f"checksum mismatch for {name}: {digest} != {known[name]}")
        out = os.path.join(RAW, asset)
        if not os.path.isdir(out):
            with zipfile.ZipFile(zpath) as z:
                for member in z.namelist():
                    # Only flat .png maps; refuse paths that would escape the folder.
                    if member.endswith(".png") and os.path.basename(member) == member:
                        z.extract(member, out)
        print(f"ok {asset}: {sorted(os.listdir(out))}")

    if args.record:
        with open(CHECKSUMS, "w") as f:
            for name, digest in sorted(recorded.items()):
                f.write(f"{digest}  {name}\n")
        print(f"wrote {CHECKSUMS}")


if __name__ == "__main__":
    main()
