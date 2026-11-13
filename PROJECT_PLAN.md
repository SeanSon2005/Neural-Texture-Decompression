# Project 18 — Neural Texture Decompression on Vortex: Execution Plan

## Context

CS254A Project 18 (Sean Son, Justin Kwong) asks us to reproduce NGPC-style neural texture decoding
(feature lookup + small MLP) on Vortex v3.0 and to answer one research question: **does
material-aware request batching improve tensor-unit utilization and throughput without excessive
latency, and how do material access patterns change the answer?** The main architectural
contribution is a configurable request buffer in SimX (no RTL).

The course's Project 18 description (`cs254a_projects.pdf`, slides 72–75) and our proposal
([`docs/PROPOSAL.md`](docs/PROPOSAL.md)) deliberately leave models, formats, kernel mappings,
simulator interfaces and policies to us. This plan fixes those choices. About 4 weeks remain (target: finals week, ~Dec 7–11;
plug in the exact date).

**Decisions confirmed with you:** train our own int8 NTC-style models · SimX request queue feeding
decoder warps that run on the real TEX/TCU · Vortex tracked as a fork submodule · ~4 weeks.

**Proposal check (Nov 12):** every proposal item (three decode paths; throughput, mean/P95 latency,
tensor utilization, batch occupancy, weight traffic, PSNR; shared-decoder stretch goal) is covered
here. One deliberate change: the proposal said *pretrained* decoder data (RTXNTC); we train our own
int8 models instead so features and MLP shapes match TEX and the int8 TCU tile and we get
bit-exact references. Report this as an adaptation.

## What the codebase told us (drives the design)

- Lab checkouts are **Vortex v2.3**; the project needs **v3.0** (tag exists, 2,561 commits later,
  toolchain `v3.0.1`). Fresh clone + fresh toolchain install required; lab toolchains won't work.
- **TEX** (`sim/simx/tex/`, `vx_tex(stage,u,v,lod)` in `sw/kernel/include/vx_graphics.h`): returns one
  A8R8G8B8 word per lane = **4 packed 8-bit features**, fixed-point bilinear, up to 16 mip levels,
  2 stages. Texture state is DCR-global, so per-request material switching is impossible →
  **all materials must live in one atlas per stage**.
- **TCU** (`sim/simx/tcu/tcu_unit.cpp`, `wmma_context` in `sw/kernel/include/vx_tensor.h`): supports
  `int8→int32` (exact integer math; needs `-DVX_CFG_TCU_INT8_ENABLE`). With `NUM_THREADS=8` a WMMA
  tile is **M=8 × N=8 × K=32 int8**. For `Y = X·Wᵀ`, tile rows M are *requests* → **a tile natively
  holds 8 requests; unfilled rows are padding**. That is exactly the occupancy the project studies.
- Fragments are plain per-lane FP register arrays, and a TEX result word is already a valid int8
  A-operand word → TEX can write the layer-1 A fragment **directly in registers** (the NGPC
  "fuse encoding and MLP" idea, no memory round trip).
- TEX is the template for a new unit: per-core SFU sub-unit (`sfu_unit.cpp`) → socket-level core
  (`socket.cpp`) with request/response channels, a fire-and-wait (blocking) instruction path,
  DCR config, and `PerfStats`. `custom1` funct3 = 7 appears free (verify in `decode.cpp`).
- Existing TCU counters cover only WGMMA tile buffers; **utilization and weight-traffic counters
  must be added**.

## Architecture

```
trace file ─► NtxSource ─► NtxCore: bounded queue, bins keyed by decoder-id ─► vx_ntx_pop (blocks)
 (testbench)   open-loop     knobs: B batch size, Q capacity, W wait, S bins        │
                                                                                    ▼
          vx_ntx_done ◄─ store outputs ◄─ MLP (scalar | TCU WMMA) ◄─ vx_tex feature lookup (atlas)
```

**Model (adapted NTC).** Per material: 2 feature pyramids (TEX stages) × 4 mip levels × 4 u8
channels → 8 texel words = 32 features (= one K tile). MLP 32→H→H→8, int8 weights, int32 bias,
ReLU + fixed-point requantize; outputs albedo RGB, normal XY, roughness, +2 (AO/metal/height).
H chosen from {16, 32, 64} by a PSNR-vs-bytes sweep. u8→i8 by XOR `0x80808080`, bias folded at export.

**Three decode paths, identical feature lookup** (so differences isolate the MLP mapping):
1. *Scalar* — per-lane integer MLP; lanes may hold different materials.
2. *Unbatched tensor* — WMMA, FIFO order, dispatch whatever same-material run is at the head, pad.
3. *Batched tensor* — WMMA, material-aware bins with B/Q/W.

**Request buffer (new SimX unit `ntx`).** Entry = {req id, decoder id, material, u, v, arrival}.
Dispatch when a bin reaches B (full), its oldest entry ages past W (partial, padded), or the source
drains. Full buffer → backpressure on the source; latency is counted from *intended* arrival.
Batch key is **decoder id, not material id**, so the shared-decoder stretch goal needs no simulator
change. Enqueue/match/dispatch latencies are explicit parameters (the "batching overhead").
All knobs are **DCR-programmed at run time** → sweeps need no rebuilds.

**New instructions** (`custom1`): `vx_ntx_pop` (blocking; per-lane header: valid, material, req id),
`vx_ntx_arg` (per-lane packed u,v), `vx_ntx_done` (completion timestamp). Batches larger than 8 are
fetched in successive pops and decoded as multiple M-tiles **reusing the loaded weight fragments** —
the mechanism by which batching cuts weight traffic.

## Repository layout

```
Neural-Texture-Decompression/        (this repo)
  PROJECT_PLAN.md  README.md  .gitignore  .gitattributes
  docs/              proposal + reading notes
  vortex/            submodule → team fork, branch `ntc` cut from tag v3.0
  ntc/               PyTorch training, int8 QAT, export, bit-exact NumPy reference decoder
  data/              fetch script + checksums (raw textures not committed); exported artifacts
  workloads/         request-trace generators (access patterns)
  experiments/       sweep runner, configs, plotting
  results/           raw CSVs + run manifests, figures
  report/            report + slides
```
Inside the fork: `sim/simx/ntx/{ntx_core,ntx_unit}.{h,cpp}`, edits to `decode.cpp`, `sfu_unit.*`,
`socket.*`, `cluster.*`, `types.h`, `VX_config.toml`, `VX_types.toml`, `tcu/tcu_unit.*` (counters),
`sw/kernel/include/vx_ntx.h`, and tests `tests/regression/ntc_{scalar,tcu,batch}/`.

## Step 0 — Git

1. Repo and remote `origin = https://github.com/SeanSon2005/Neural-Texture-Decompression.git`: done.
2. `.gitignore` (build dirs, toolchain, raw textures, checkpoints, exported binaries,
   `cs254a_projects.pdf` — course material, kept local): done.
3. `vortex/` submodule at tag `v3.0`, local branch `ntc`: done, but its URL still points at
   upstream `vortexgpgpu/vortex`. **Open:** fork `vortexgpgpu/vortex` on GitHub, then
   `git submodule set-url vortex <fork URL>` and push branch `ntc` there, so teammates can fetch
   the `ntc` commits the superproject records.

## Phase 1 — Research & setup (Nov 12–18) → Checkpoint 1

- Build v3.0: `../configure --xlen=32 --tooldir=<project>/.tools`, `toolchain_install.sh`, `make -s`.
- Smoke: `demo`; `gfx_tex` on simx (`tests/graphics/gfx_tex`, its Makefile adds `-DVX_CFG_EXT_TEX_ENABLE`); `sgemm_tcu` with
  `-DVX_CFG_NUM_THREADS=8 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_INT8_ENABLE -DITYPE=int8 -DOTYPE=int32`.
- **De-risking spikes** (each a ~50-line kernel, done before anything else is built on them):
  (a) TEX and TCU enabled together; (b) `vx_tex` results moved straight into a `fragment_a` and
  multiplied correctly; (c) int8 WMMA bit-exact vs NumPy; (d) one mip level sampled per `lod`
  from a hand-built atlas matches a Python port of the TEX fixed-point bilinear math.
- Data: 6–8 CC0 PBR materials (ambientCG / Poly Haven), 1024², 8 channels.
- `ntc/`: train float → QAT int8 with TEX-exact interpolation; export atlas + mip chain, weights,
  requant params, material table; NumPy reference decoder; PSNR + byte accounting.
- Read NGPC, Vaidyanathan et al., Matai et al.; one-page notes each for the report's
  reproduced / adapted / new split.

**Exit:** simx runs TEX and int8 TCU tests; exported material set; reference decoder with stored
expected outputs and a PSNR/size table.

## Phase 2 — Decoders on Vortex (Nov 19–25) → Checkpoint 2

- `ntc_scalar`: atlas UV transform → 8× `vx_tex` → integer MLP → store 8 channels.
- `ntc_tcu`: lane→(request, feature-word) assignment so TEX fills fragment A in place; per layer:
  load weight fragment B, `mma_sync`, store C, requantize + repack (try `vx_packlb_f`) into next A.
- Host `main.cpp` for each: uploads atlas/weights, programs TEX DCRs, compares every output
  **bit-exactly** against the reference file; `-n`, material and pattern args.
- Decode a full 128²–256² image per material on Vortex → PNG + PSNR (full-res PSNR comes from the
  reference decoder, justified by bit-exactness).
- Baselines: cycles/request and instruction mix for scalar vs tensor at occupancy 1/8 … 8/8;
  measure simulator speed to size later sweeps.

**Exit:** both paths bit-identical to the reference on all materials; baseline table.

## Phase 3 — SimX batching (starts Nov 16 in parallel; done Dec 2) → Checkpoint 3

Built against a dummy fixed-delay decoder kernel first, so it does not wait on Phase 2.
- `NtxSource` (trace player, clearly labelled testbench), `NtxCore` (queue/bins/policies/timers),
  `NtxUnit` (SFU sub-unit), decode + DCR range + config flag `VX_CFG_EXT_NTX_ENABLE`.
- Counters: enqueued/completed, batches by reason (full/timeout/drain), valid vs capacity rows,
  backpressure cycles, queue occupancy; per-request CSV log (arrive/enqueue/dispatch/done) for P95.
- TCU: uops executed + busy block-cycles. LSU/dcache: bytes loaded and miss-fill bytes inside a
  DCR-set address window (the weight region) → weight traffic at **two named levels**.
- `ntc_batch` kernel: decoder-warp loop `pop → decode (Phase 2 code) → done`, 1–4 decoder warps.
- Directed tests: single material; mixed materials; partial batch via timeout; W=0; Q smaller than
  B; backpressure with conservation check (every request completes exactly once); drain at end of
  trace; more materials than bins. All still bit-exact.

**Exit:** all directed tests pass; knobs change timing measurably; outputs unchanged by policy.

## Phase 4 — Evaluation & submission (Dec 3–9, buffer to due date) → Checkpoint 4

Fixed platform (1 core, 4 warps, 8 threads) unless stated. `experiments/sweep.py` runs points in
parallel across the 32 host cores and records git SHAs + configs per run.

| Study | Sweep |
|---|---|
| E1 Baselines | scalar / unbatched tensor / batched tensor; 1 material and mixed; saturated |
| E2 Batch size | B ∈ {8,16,32,64} × material mix |
| E3 Wait vs load | W ∈ {0,64,256,1k,4k,∞} × arrival rate {0.3,0.6,0.9 of capacity, saturated} |
| E4 Capacity | Q ∈ {16…256}, bins S ∈ {1,2,4,8} |
| E5 Access patterns | #materials {1,2,4,8,16}; uniform vs Zipf; run length {1,4,16,64}; screen-space scan of a material-ID map |
| E6 Weight traffic | E5 × policy, both memory levels |
| E7 Quality & size | PSNR per material vs originals; bytes: features, weights, tables |

Every required metric maps to a counter above: throughput, mean/P95 latency, utilization (raw and
padding-adjusted), occupancy, weight bytes, PSNR, total bytes. Deliverables: raw CSVs, plots,
`README` reproduce instructions (one command per figure), report, slides.

**Stretch, in priority order:** shared decoder per material group (training only) · fp16 vs int8
precision at fixed PSNR · `NUM_THREADS=16` (M=16) sensitivity · closed-loop shader warps issuing
`vx_ntx_req`.

## Suggested split

- **Track A (ML + evaluation):** data, training, export, reference decoder, trace generators,
  sweep runner, plots.
- **Track B (Vortex):** build, spikes, decoder kernels, SimX `ntx` unit, counters.
- Joint: Phase 3 integration tests, report, slides.

## Main risks

| Risk | Mitigation |
|---|---|
| TEX + TCU don't co-exist cleanly, or atlas mip sampling misbehaves | Spikes (a),(d) in the first days; fallback = software bilinear via LSU loads |
| Requantize/repack between layers dominates tensor time | Measure in Phase 2; it is a legitimate reported bottleneck; WGMMA shared-memory operands as follow-up |
| int8 quality too low | QAT, widen H, more feature channels; fp16 path as fallback |
| Scalar baseline too slow to simulate | Smaller N for scalar, H=32 default, parallel sweeps |
| v3.0 toolchain/build trouble on WSL2 | Do Phase 1 build first; follow `docs/install_vortex.md` exactly |
| Blocking `pop` deadlocks at end of run | Drain flag in the pop header; conservation test in Phase 3 |

## Verification

- `ntc/`: pytest — reference decoder vs float model within tolerance; export round-trip.
- Vortex: `./ci/blackbox.sh --driver=simx --app=ntc_scalar|ntc_tcu|ntc_batch` exits 0 only on
  bit-exact output for every request; stock `sgemm_tcu`, `gfx_tex`, `demo` still pass (no regressions).
- Batching: conservation (enqueued = completed), no request older than W + service bound when
  unloaded, policy changes timing but never outputs.
- Results: re-running a sweep point reproduces identical cycle counts (SimX is deterministic).
