# Neural Texture Decompression on Vortex

CS 254A (Fall 2026, UCLA) — Project 18. Team: Sean Son, Justin Kwong.

We reproduce NGPC-style neural texture decoding (feature lookup + small MLP;
Mubarik et al., ISCA 2023) on the Vortex v3.0 GPGPU, and study one question:
**does material-aware request batching improve tensor-unit utilization and
decoding throughput without excessive latency, and how do material access
patterns change the answer?** The architectural contribution is a configurable
request buffer in SimX.

- Proposal: [`docs/PROPOSAL.md`](docs/PROPOSAL.md)
- Execution plan: [`PROJECT_PLAN.md`](PROJECT_PLAN.md)

## Repository layout

```
vortex/        Vortex fork (submodule), branch `ntc` cut from tag v3.0
ntc/           PyTorch training, int8 QAT, export, NumPy reference decoder
data/          fetch script + checksums; exported artifacts
workloads/     request-trace generators (access patterns)
experiments/   sweep runner, configs, plotting
results/       raw CSVs + run manifests, figures
report/        final report and slides
docs/          proposal and notes
```

Directories are created as each phase starts; not all exist yet.

## Getting started

Build on Linux or WSL2 (Vortex's toolchain does not run natively on Windows).

```bash
git clone --recurse-submodules https://github.com/SeanSon2005/Neural-Texture-Decompression.git
cd Neural-Texture-Decompression
```

### Vortex (SimX)

`vortex/` is Vortex tag `v3.0` on local branch `ntc` (its remote is still
upstream until the team fork exists). Build out of tree, from a path with no
spaces; Makefiles break on spaces, so on Windows link the checkout first:

```bash
ln -sfn "/mnt/c/<path to>/Neural-Texture-Decompression" ~/ntd
mkdir -p ~/ntd-build && cd ~/ntd-build
~/ntd/vortex/configure --xlen=32 --tooldir=$HOME/tools
./ci/toolchain_install.sh          # only if $HOME/tools lacks the v3.0.1 toolchain
make -s -j16
```

Build trees on `/mnt/c` are slow (every file read crosses into Windows), so the
build directory lives in the Linux filesystem. Re-run `configure` from the
build directory after adding a test directory or editing any `.toml`.
Use `MAKEFLAGS=-j16` for test runs: a new `CONFIGS` set rebuilds SimX.

Smoke tests and the de-risking spike (TEX → int8 TCU, bit-exact):

```bash
cd ~/ntd-build
./ci/blackbox.sh --driver=simx --app=demo
./ci/blackbox.sh --driver=simx --app=gfx_tex
CONFIGS="-DVX_CFG_NUM_THREADS=8 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_INT8_ENABLE -DITYPE=int8 -DOTYPE=int32" \
  make -C tests/regression/sgemm_tcu
CONFIGS="-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_INT8_ENABLE -DITYPE=int8 -DOTYPE=int32" \
  ./ci/blackbox.sh --driver=simx --app=sgemm_tcu --threads=8
CONFIGS="-DVX_CFG_EXT_TEX_ENABLE -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_INT8_ENABLE" \
  ./ci/blackbox.sh --driver=simx --app=ntc_spike --threads=8
```

### Data and models

```bash
python3 -m venv --without-pip ~/ntd-venv     # Ubuntu without python3-venv/pip
curl -sSL https://bootstrap.pypa.io/get-pip.py | ~/ntd-venv/bin/python
~/ntd-venv/bin/pip install numpy pillow pytest
~/ntd-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu

cd ~/ntd
~/ntd-venv/bin/python data/fetch_materials.py           # 8 CC0 ambientCG sets, checksummed
~/ntd-venv/bin/python -m pytest -q ntc/tests             # TEX port vs Vortex, qat == reference
bash ntc/sweep_hidden.sh ~/ntd-venv/bin/python           # train + export H = 16, 32, 64
~/ntd-venv/bin/python -m ntc.summarize data/export/h16 data/export/h32 data/export/h64 \
  --csv results/ntc_quality_size.csv
```

Each `data/export/h<H>/` holds the atlas, weights, material table, test
requests and their expected outputs; `manifest.json` documents the layout
(see `ntc/export.py`). The integer decode contract every decoder must match
is in `ntc/quant.py`.

## References

1. Mubarik et al., *Hardware Acceleration of Neural Graphics*, ISCA 2023. arXiv:2303.05735
2. Vaidyanathan et al., *Random-Access Neural Compression of Material Textures*, SIGGRAPH 2023. arXiv:2305.17105
3. Matai et al., *Thread-Efficient Decoding for Neural Texture Compression*, arXiv 2026. arXiv:2608.27888
4. NVIDIA RTX Neural Texture Compression SDK — https://github.com/NVIDIA-RTX/RTXNTC
5. Vortex GPGPU — https://github.com/vortexgpgpu/vortex
