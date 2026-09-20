# Extraction lane: deployment and consumer handoff

Status: single-row owner-roster qualification complete; expanded batch
qualification and final delivery audit still in progress. Read the executable
contract and results record alongside this document.

## Choose the implementation, not just the branch name

| Use | Source / evidence | Qualification |
|---|---|---|
| 70B cached single-row replay | Frozen `gpu-v3`; numerical files match `6feb4bf`; `roster-certification-001.json` | All 3,467 owner-roster rows, G0–G5, current n383 primary/null statistics |
| 70B independent-prefill batch2 | Frozen `batch-v7`/`batch-v8`; numerical files match `cba271c`; `verification-batch-bank-002.json` | Original 2,719 rows pass; expanded roster pending |
| Native 3B/8B full replay | Current experimental CLI | Three-input/model characterization, not a tolerance certificate |

The `gpu_features.py` source SHA for the batch implementation is
`0f759dfe9f13ad292c9348b211a9f6c6275a04cc885ea4535599b26057abf9c2`;
`batch_layout.py` is
`018cdfe51ae55e766af787cdc14e95e286ec20183278f73920b7cf1ae2dd7a07`.
Both were checked against the deployed manifest and git `cba271c`. The single
lane's four numerical hashes are listed in `extraction-perf-results.md`.
These identify numerical modules, not a claim that every deployment file equals
that commit. Preserve the complete deployment manifests, including caller code.

**Do not import current HEAD and label its single-row output `gpu-v3`.** Later
full-replay and batch work changed the module and its computed lane identity.
Use a separate checkout of the qualified implementation and verify its source
manifest, or requalify the implementation you actually intend to run. Do not
overwrite the frozen remote snapshots to make their names point at new code.

## 70B caller API

The integration point is `anamnesis.extraction.gpu_features.GpuFeatureLane`.
Construct it once per loaded model using the frozen complete extraction/family
configs, ordered feature names, positional means, PCA basis/mean, device and
calibration SHA. The qualification runner `scripts/extraction_perf_gpu_trial.py`
shows model construction and the required all-layer hook coverage.

```python
lane = GpuFeatureLane(
    extraction, families, names,
    calibration.positional_means,
    calibration.pca_components, calibration.pca_mean,
    device="cuda:0", calibration_sha256=calibration_hash,
)
result = lane.replay_span(loaded, token_ids, start, end)
```

`start`/`end` are absolute span boundaries; the caller supplies the prefix tokens
as well as the continuation. The returned object includes f32 features, ordered
names, teacher-forced mean logprob, optional kNN baseline and extraction metadata.
Persist the metadata with every signature, not only once in a run-level README.
This is a feature-only path: it does not produce a raw-tensor bank.

Use eager attention, TF32 off and deterministic algorithms on with
`CUBLAS_WORKSPACE_CONFIG=:16:8` set before starting Python. The tested node stack
is B200 / Torch2.9.1+cu128 / Transformers4.57.6. Carry the full pinned environment,
checkpoint and calibration identity from the deployment receipt. Requalify after
stack/hardware/config changes; the flag alone is not evidence of fidelity.

The current batch API is `lane.replay_batch(loaded, spans,
prefill_policy="independent")`, where spans are `ReplaySpan` objects from
`anamnesis.extraction.batch_layout`, not raw tuples.
The qualified schedule uses batch2 and exact shapes, with independent unpadded
prefills and batched continuation. Other padding/composition policies do not
inherit this evidence. A filler is a computational duplicate, never an extra
scientific row, and its full cost must be charged to the emitted row.

## Analysis boundary

The current delivery branch provides `require_single_lane` in
`anamnesis.analysis.lane_guard`, wired into the shared geometric loader. Custom
consumers must apply the same check before forming contrasts. It rejects mixed
lane IDs and tagged/untagged mixtures. All-untagged legacy input remains unknown,
not automatically certified CPU data. Do not combine single and batch outputs
inside a scientific contrast. Explicit CPU/GPU validation is a different purpose.

The frozen old extraction checkout does not acquire later analysis guards by
magic: callers must use the guarded analysis path or enforce equivalent checks.

## Owner hand-back, not sibling edits

- Palimpsest owns driver integration/P, shard sizing and any future n750/m-mix
  reference delivery. Recompute G2 on its matched cohort when delivered.
- Preserve the roster and current null references; separately record s101's
  coincident null draws and excluded A endpoint for future null-lane review.
- Registry claim: deterministic replay must name the arithmetic lane and pinned
  stack; CPU/GPU byte-identity is not claimed. The registry owner makes that edit.
- Adapter/excess scoring needs the owner-supplied adapter-bearing capture.
- The old32KB D2H target was withdrawn; the named gate-tie CPU exception remains.
- Batching is not currently recommended as a universal speed improvement:
  balanced pilot and operational sparse timings disagree. Both are retained.

No consumer deployment, default switch, registry edit or sibling integration has
been performed by this work. Those actions are distinct from delivering the lane.
