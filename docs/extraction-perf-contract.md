# Extraction performance executable contract

Status: pilot baseline and harness checks complete; full floor survey and GPU
pilot running; no optimized lane certified. Authority: Anamnesis
extraction-perf brief (2026-09-19), optimization-agent ferry of the same date,
and Luxia's authorization. This document consolidates those rulings; it does
not relax a gate. P remains Palimpsest-owned. Registry edits remain out of scope.

## Deliverable and scope

Reduce measured seconds per row for 70B probe-free capture with identical
feature definitions and a characterized arithmetic backend. Ship a separate
lane, the fidelity harness, provenance and measurements. Keep the NumPy
reference GPU-independent. Excess/adapter scoring is NOT certified until the
separate adapter-bearing capture exists and passes the same harness.

Order: deployment pin -> consumer inventory and CPU baseline/path floors ->
harness with adversarial negative controls -> GPU lane -> batching, gated again.
P is independent and cannot be assumed in our baseline. Shard-size changes
require durable intermediate banking and belong to the caller.

## Deployment and reproducibility

Use an isolated checkout, not the shared node copy. Pin every imported
Anamnesis and frozen Palimpsest Python source, feature configuration, ordered
names, calibration/standardizer/bank hashes, model configuration/checkpoint
identity, Python/NumPy/SciPy/Torch/Transformers/CUDA stack, deterministic and
TF32 settings, thread environment, node/GPU identifiers and device layout.
Verify local/deployed source hashes before running. Recheck at run completion.

The frozen battery disables path-signature features. The original node
deployment lacks that optional dispatch; the isolated baseline includes the
current upstream block but leaves it disabled. This is recorded explicitly,
and bank-vs-recomputed CPU drift is measured rather than assumed zero.

## Source-grounded consumer inventory

The active battery is `sigbridge.battery_configs(70b-base)` and the feature
schema is the bank's ordered 4,086 names. Preserve conversion order as well as
formulas. Captured bf16 surfaces become float32 before the reference reductions.
Standard deviations use population normalization. All feature outputs are f32.

| Consumer | Surfaces and coverage | Temporal/cross-layer retention | Numerical contract |
|---|---|---|---|
| Activation norms | position-corrected residual, every decoder layer | scalar series and trajectory samples | f32 subtraction/norm/aggregation; hidden index l+1 |
| Logit/token statistics | logits and next chosen token, every step | scalar series, surprise moving windows | f64 exp/log reductions -> f32 series; surprise uses f32 clipped probability; strict greater-than rank |
| Attention entropy | all attention layers | entropy scalars at existing stride (T//60) | f64 clipped/normalized entropy; preserve sampling |
| Head agreement | all attention layers | scalars at existing stride (T//30) | f64 row normalization and generalized JSD; f32 agreement series |
| Residual deltas | adjacent decoder hidden outputs | adjacent layers' corrected residuals | f32 delta/norm; f64 cosine -> f32 series |
| Attention spectral | sampled layers | sampled head-mean attention rows, full causal width | head mean in f32 THEN f64; padded cosine Gram, symmetric unnormalized Laplacian +1e-10 diagonal, eigvalsh and existing summaries |
| Cache profiles/decay | sampled head-mean attention | scalar series; decay sample sufficient statistics | f32 head mean THEN f64; ratio of sums, threshold counts, BOS excluded from decay; polyfit semantics |
| Key geometry/epochs | sampled pre-RoPE keys | head means [T,128], windows/cumulative means | f32 head means; f64 SVD/cosines, explicit f32 round trips for drift/novelty; epoch windows 50/stride25 |
| PCA | corrected residual at five sampled layers/times | sampled hidden rows and fixed basis | correction in f32; centering and projection f64 -> f32; no refit |
| Residual trajectory | five trajectory layers | adjacent steps/velocities plus scalar series/endpoints | corrected f32 -> f64; velocity cosine round trip through f32; window and STFT rules |
| Attention flow | seven sampled layers, all rows/heads | mass/diversity scalar series | f32 head mean -> f64; per-head masses f64; decay polyfit; temporal input cast f32 |
| Gate features | seven sampled gate_proj outputs | consecutive activated vectors/top-k sets, layer means, scalar series | SiLU f64; drift and layer-mean cosine f32 round trip; NumPy argsort tie behavior is a parity risk |
| Per-head | sampled attention and pre-RoPE keys | head entropy/sink series; key head trajectories | f64 normalization/entropy/correlation/cosines, explicit short-series fallbacks |
| Value geometry | sampled values + keys | head means [T,128], scalar series | f32 head mean -> f64; SVD participation rank, cumulative novelty, drift correlation, explicit casts |
| Query/key geometry | sampled pre-RoPE Q,K | head means [T,128], cumulative K, scalar series | same mean convention; f64 SVD/cosine; NOT post-RoPE attention logits |
| KV CKA | sampled keys and values | head-mean matrices across seven layers | f32 head mean -> f64; centered cross products, Frobenius norms, existing pair order |
| Shared temporal operators | derived scalar series | whole short [T] series | incoming f32 -> f64; four floor-sized windows with remainder last, polyfit slope; SciPy STFT Hann/64/32, zero boundary/padding, five power summaries |
| Ancillary outputs | mean teacher-forced logprob; optional kNN final residual | scalar / one final residual | must explicitly preserve caller outputs; 4,086 floats alone do not describe the API |

Inactive for this contract: temporal_dynamics, learned contrastive projection,
path_signature, binding_probe (untracked local module, not part of this
deployment), AttnRes and expert routing. Their existence forbids a claim of
general feature-pipeline compatibility. Reject unsupported configurations rather
than silently dropping a family. `attn_outputs` is captured by the historical
path but not read by this active suite; removing its capture still requires
the declared parity and resource gates.

## GPU design and priced retention exceptions

Eager attention remains mandatory. Install per-layer attention hooks which
reduce the returned rows while alive. The verified Transformers 4.57.6 Llama
eager implementation returns weights to the attention-module hook even with
`output_attentions=False`; the decoder then discards them. Use that behavior
without changing returned tensors or attention arithmetic. Do not accumulate
`out.attentions`.

Named exception SPECTRAL: retain only subsampled head-mean causal rows until
the within-layer cross-step Gram is computed. For T=127, stride10, C<=299,
13 rows × (C+T) × 8 bytes is <=44,304 bytes per sampled layer, excluding
temporaries/Gram. This is a dimension-derived estimate, NOT measured peak VRAM.
Finish each layer's spectral reduction and release that store when possible.
Named exception CKA: seven layers' head-mean key/value matrices must coexist
until cross-layer CKA finishes (2×7×T×128×8 bytes, about 1.74 MiB at T=127).
Name and measure gate, residual adjacency and temporal retention separately.
The one-layer attention tensor still exists during eager computation; its
peak and reduction workspaces count. Do not upload the full positional-means
bank blindly: stage needed positions/layers, and report H2D bytes/time as well
as D2H. Do not hide shifted transfer costs in feature-only timing.

Named v1 exception GATE-TOPK: preserve NumPy's historical unstable argsort tie
ordering for gate Jaccard. Transfer sampled pre-SiLU gates as float32 and run
the reference SiLU/argsort/top-k-set intersection on CPU; all other gate
statistics stay on device. Price: 7×127×28672×4 = 101,957,632 bytes/span
(97.2 MiB) at the banked shape, plus CPU sort time. This is an explicit hybrid
exception, not an all-GPU claim. CUDA topk without verified tie equivalence
would change the features. The withdrawn 32-KB target is not met by this v1.
Positional correction stages only the needed positions, one layer at a time:
80×127×8192×4 = 332,922,880 H2D bytes/span before any caching optimization.
These dimension-derived costs must be confirmed in telemetry and included in
end-to-end timing. PCA components/mean are uploaded once per lane instance.

No handwritten CUDA. Start with deterministic Torch reductions with the
reference dtype transitions. Compilation/graphs are later measured variants.
The old <=32 KB D2H target and G speed forecast are withdrawn. Re-price only
after inventory and baseline profiles; no numerical forecast is an exit gate.

## Frozen fidelity gates

- G0: independent full replays, fresh caches and hook state, identical tokens,
  identical stack/lane -> bit-identical finite f32 signatures on every
  reference row. Re-reducing one tensor is insufficient. Record determinism
  scope; no claim of universality across software/GPU/batch layouts.
- G0.1: bank incremental-vs-cached teacher-forced replay floor(b) on the
  actual model, 4,086 names and pinned standardizer, per row.
- G1: `z(e)=w*(e/sigma_cal)`, no mean subtraction; L2. Match by
  `(doc_id,cell,replicate,span)` with a bank namespace. Require
  `||z(cpu-gpu)|| <= min(floor_b_row, 0.1*||z(d_row)||)` for EVERY row.
  Report prefix-length terciles separately, including their worst row.
  Also require each `abs(cpu_j-gpu_j) <= 0.1*sigma_cal_j`. This multiplier is
  provisionally ratified; an implementer cannot relax it after seeing results.
  Anchor-A condition-delta definition has been asked of the governing agent:
  A-A is zero and would otherwise impose an exact cross-backend bound. Until
  resolved, no certification based on an invented nonzero comparator.
- G2: exact same included documents/clusters and frozen analysis code;
  delete-one-cluster jackknife coherence ratio of F-B document deltas.
  Require both ratio and SE absolute changes <=0.25*reference_SE; report each
  change in SE units. Pin reference artifact, do not encode headline constants.
  Current artifact is rubric-140 final (383 documents); the n=750 delivery
  supersedes it when available, on matched-n data only.
- G3: probe-free certification only, length extremes, E, null forks, B-E and
  F-E near-floor contrasts, and the named off-target pair. Missing coverage is
  visible, never an implicit pass. Corpus currently inspected has 12 shards,
  2,719 emitted rows (not the brief's 3,447); resolve that discrepancy before
  declaring full-reference coverage. All inspected holder spans are length128;
  short-span coverage cannot be claimed from this bank alone.
- G4: signatures and analysis inputs carry lane and source/config/calibration
  identity. Refuse mixed lanes inside a scientific contrast. CPU-vs-GPU
  validation is an explicitly labeled equivalence comparison.
- G5: per-device/phase peak allocated AND reserved VRAM, with units and process
  sampling recorded. The historical receipt is resolved: process-group NVML
  peak **135.29 GiB**, printed as 135.3 GB by Heimdall. Gate the process metric
  against that receipt, and compare matched current CPU measurements. A
  145-GiB scheduler budget is separate. Allocated/reserved counters are not
  substituted for the process metric; the first inspected historical shard
  records 133.09 allocated and 136.41 reserved GiB, a different run/metric.

Harness negative controls: nonfinite values, feature-name permutation,
duplicate/missing row, mixed lanes, changed calibration, changed n/clusters,
single-feature excess, whole-vector excess, nonrepeatability and VRAM excess.
CPU-self must pass. Failures stop promotion; retain their receipts.

## Measurement and diary

Historical 5.46 s/row is capture-shard wall divided by emitted rows. A replay
microbenchmark does not rederive that number. Record both boundaries separately;
the optimization directly changes replay+features. Model/calibration/pool startup
and generation remain separate costs. Record warmup policy, synchronized timing,
per-row vectors, peak VRAM, parent/worker RSS, throughput and transfer bytes.
Use matched inputs/stack/device placement for comparisons. Plot all attempts
chronologically, marking rejected candidates and baseline provenance; use
matched-scope baseline ratios only. Preserve diary and machine-readable ledger,
including failed setup/submission attempts. Never overwrite prior measurements.

Delivery includes source changes/tests, complete gate receipts, measured-vs-priced
ratios for any forecast made, scope limitations, and the registry wording issue
as a hand-back. A contract-mandated stop is reported as such, not as success.

## Measured receipts as of the first pilot

- Node1 isolated deployment matched local Python source hashes; frozen caller
  and calibration/standardization inputs were pinned before GPU work.
- CPU pilot: 25/25 vectors byte-identical to their banked counterpart; 25/25
  fresh-cache repeat vectors byte-identical. Floor(b): 0.503943–3.652998.
- Replay+pooled-feature timing on this pilot: mean 7.5454 s/span, median 5.7600,
  range 5.3464–17.0933. Outliers are retained. Pool startup 435.65 s; maximum
  observed worker RSS 95.08 GiB; peak allocated 132.8142, reserved 132.9902 GiB.
- Full bank audit: 2,719 rows in 12 shards. Recorded weighted shard-wall time
  12.4611 s/emitted row; replay time 6.5317. The source's 3,447 was a priced
  future extension in notes/199, not this bank. The reported 5.46 denominator
  remains unreproduced and is not relabeled as a fresh measurement.
- CPU-self G2 reproduces the owner's frozen estimator: ratio 1.1952486631696086,
  SE 0.08761776823641536, matched n=383. This is not a GPU G2 pass.
- Full per-row floors are being surveyed in two disjoint partitions. GPU trial
  is a pilot on known-floor rows. No batch>1 work or certification yet.
- Execution-stack control: `CUBLAS_WORKSPACE_CONFIG=:4096:8` moves the CPU
  reference by 0.850491 L2 on the first row, even with deterministic mode off.
  `:16:8` preserves its exact banked bytes with deterministic algorithms ON.
  Pin `:16:8` and validate broader coverage; do not change a threshold to
  accommodate the incompatible setting. Failed attempts remain in the ledger.
