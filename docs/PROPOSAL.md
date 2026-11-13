# Project 18: Neural Texture Decompression — Proposal

**Team:** Sean Son, Justin Kwong

*Transcribed from the submitted proposal PDF for reference. Where the
execution plan departs from it, see "Changes since the proposal" at the end.*

## Why this project

We chose Neural Texture Decompression because we are interested in machine
learning, GPU architecture, and hardware acceleration. Sean is interested in
efficiently executing neural networks on GPUs. Justin is interested in GPU
architecture and performance optimization. We want to explore how hardware and
ML techniques can complement each other to improve neural inference
performance.

## Problem and goal

Neural texture compression reduces memory usage by representing textures with
compressed features and small neural networks. However, each texture request
requires neural network inference, which may not efficiently utilize GPU
tensor units.

Our goal is to investigate **whether batching neural texture requests can
improve tensor utilization and decoding throughput without introducing
excessive latency.** We also want to understand how requests from different
materials affect batching efficiency.

## Approach

We plan to implement neural texture decompression on Vortex v3.0, building on
the NGPC architecture (ISCA 2023).

1. **Baseline:** Implement feature lookup and scalar MLP decoding using
   pretrained neural texture data, then map the decoder onto Vortex's tensor
   unit.
2. **SimX optimization:** Implement a request-buffering mechanism in SimX that
   groups texture requests by material and dispatches them in batches.
3. **Batching experiments:** Evaluate different batch sizes, queue capacities,
   and waiting thresholds. Study how material access patterns affect batching
   efficiency.
4. **Performance analysis:** Compare scalar decoding, unbatched tensor
   decoding, and batched tensor decoding to identify performance bottlenecks
   and trade-offs.

Our primary architectural contribution will be the SimX request-buffering
mechanism.

## Expected outcome

- A working neural texture decoder on Vortex.
- A configurable batching mechanism in SimX.
- Performance comparisons across batch sizes and material access patterns.
- A final report analyzing the benefits and limitations of batching.

**Evaluation metrics**

- Decoding throughput (requests/cycle)
- Average and P95 request latency (cycles)
- Tensor-unit utilization (%)
- Average batch occupancy
- Decoder-weight memory traffic
- Reconstruction accuracy (PSNR)

Success will mean demonstrating correct decoding and identifying when batching
improves performance, including cases where its overhead outweighs its
benefits.

## Stretch goal: shared neural decoders

Normally, different materials use different neural decoders, making it
difficult to batch their requests together. We propose training shared neural
decoders that can reconstruct multiple materials using the same MLP weights.
We would identify materials that frequently produce underfilled batches, train
shared decoders for selected groups, and evaluate whether this improves
batching efficiency, comparing throughput, latency, and reconstruction quality
against the original per-material decoders.

## Milestones

- **Checkpoint 1 — Research and setup:** read the NGPC and neural texture
  compression papers; set up Vortex and obtain pretrained decoder data;
  establish a reference decoder.
- **Checkpoint 2 — Decoder implementation:** implement scalar and tensor-unit
  decoding; verify correctness and establish performance baselines.
- **Checkpoint 3 — SimX optimization:** implement request buffering and
  material-aware batching; evaluate batch sizes and queue configurations.
- **Final checkpoint — Evaluation:** complete performance experiments; analyze
  batching inefficiencies; explore shared decoders if time permits; complete
  the final report and presentation.

## Risks and questions

Potential challenges include mapping small MLPs efficiently onto tensor units,
integrating pretrained decoder data with Vortex, and accurately modeling
batching overhead in SimX. The shared-decoder extension may also introduce
reconstruction-quality loss or require additional model capacity.

Open question for the instructor: is a SimX-level batching implementation
sufficient for the architectural contribution, or is an RTL implementation
expected?

## References

1. Mubarik et al., *Hardware Acceleration of Neural Graphics*, ISCA 2023. https://arxiv.org/abs/2303.05735
2. Vaidyanathan et al., *Random-Access Neural Compression of Material Textures*, SIGGRAPH 2023. https://arxiv.org/abs/2305.17105
3. Matai et al., *Thread-Efficient Decoding for Neural Texture Compression*, arXiv 2026. https://arxiv.org/abs/2608.27888
4. NVIDIA RTX Neural Texture Compression SDK. https://github.com/NVIDIA-RTX/RTXNTC

---

## Changes since the proposal

- **Decoder data:** the proposal planned to use pretrained neural texture data
  (e.g. from RTXNTC). The execution plan instead trains our own int8
  NTC-style models, so the feature format and MLP shape match Vortex's TEX
  unit (A8R8G8B8 texels) and int8 tensor-unit tiles exactly, and so we have
  bit-exact reference outputs. This should be stated in the report as an
  adaptation.
- **Filtering and mip levels:** the course's default scope lists filtering and
  mip levels as stretch goals. Our baseline uses TEX bilinear filtering and
  four mip levels per feature pyramid, because the TEX unit provides both
  natively and our trained models depend on them. Report this as an adaptation.
- Everything else in the proposal (three decode paths, all six metrics, the
  shared-decoder stretch goal, SimX-only scope) is covered by `PROJECT_PLAN.md`.

### Clarifications to the proposal text

The submitted text above is unchanged. These points pin down what it left open
(added Nov 12 after a review against the course's Project 18 slides).

- **Shared decoders are prior work.** Matai et al. (reference 3) already train
  a shared decoder MLP with texture clustering, to reduce GPU thread
  divergence. Our stretch goal reuses that idea and must cite it. The new part
  is measuring its effect on tensor-unit batch occupancy, throughput and
  latency in SimX.
- **Reproduced, adapted, new.** *Reproduced:* NGPC's two accelerated stages,
  input encoding and MLP, as Vortex kernels (scalar first, then on the tensor
  unit). *Adapted:* the encoding is a multi-resolution dense feature grid
  stored as TEX mip levels (NGPC's hash grids are not implemented), and the
  decoder is our own int8 model. *New:* the SimX request buffer and the
  batching study.
- **Total bytes stored.** The course lists bytes stored as a required
  measurement. The metrics list above has only decoder-weight memory traffic,
  which is a different quantity. We also report total stored bytes per
  material (features, weights, tables); see E7 in the plan.
- **Request source.** Requests come from synthetic traces replayed open-loop
  by a SimX testbench source, not from a running shader. The traces vary
  material count, uniform versus Zipf popularity, run length, and a
  screen-space scan of a material-ID map. Our batching conclusions hold only
  for these arrival patterns; state this as a limitation. Closed-loop shader
  warps are a stretch goal.
- **"Unbatched tensor decoding"** serves requests in arrival order: a tile
  takes whatever run of same-material requests is at the head of the queue,
  padded to 8 rows. It does not mean one request per tile.
- **Two batching mechanisms.** With 8 threads, one tensor tile holds 8
  requests. Up to B = 8, batching only fills the tile; beyond 8, any gain
  comes from reusing loaded decoder weights across tiles. Report the two
  effects separately.
- **"Excessive latency"** is defined relative to the unbatched tensor path: a
  batching configuration is acceptable if its P95 request latency is at most
  2x the unbatched tensor path's P95 at the same arrival rate and material
  mix. Latency is counted from intended arrival, so queueing delay under
  backpressure is included. We also report the full throughput-versus-P95
  curve, so the conclusion does not rest on the 2x figure alone. The 2x is a
  provisional default; revisit it once the Phase 2 baselines exist.

### Questions still open with the instructor

- SimX-only versus RTL. Asked in the proposal; the Project 18 slide lists no
  RTL deliverable, but no answer is recorded yet.
- Whether training our own int8 models is acceptable in place of the
  pretrained decoder that the default scope assumes.
