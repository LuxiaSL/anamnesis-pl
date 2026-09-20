# Extraction performance: measured status

This is a scoped results record, not a claim that the entire work is finished.
Artifact paths below are relative to the experiment repository's
`outputs/extraction_perf/` directory. P and registry edits remain owner tasks.

## Qualified frozen single-row lane

`roster-certification-001.json` passes G0–G5 for the owner-selected3467 rows,
with explicit workspace evidence and the ratified base B/F/E contrast minima.
The current primary F-B and three separate null-fork statistic checks use the
sha-pinned n383 anchor. Replica/m-mix statistics are not inferred from that
anchor. The certificate is limited to probe-free Llama3.1-70B,4086 features,
the pinned p05 standardizer, cache263–299 and continuation128.

The qualified arithmetic implementation is frozen `gpu-v3`, lane
`torch-eager-reduce-v1-512a99842820d6fced4e`. Its four numerical implementation
files match git commit `6feb4bf` (also unchanged at `39ff3f4`):

| File | SHA256 |
|---|---|
| gpu_features.py | 439a1858f84d651301068f3cd0104ca6d70b058c3f8ee009f411f0a0cbd04c6c |
| gpu_ops.py | e6920e84ef2713817071da67d3ef2f8eb4d2427c7cc961ad1c700951738ef536 |
| gpu_attention.py | 674ad9cbe463977adb5c4b1b7714c4cef132f2a8c4772f38e673fe464a546208 |
| gpu_families.py | b85f55b1655db8a7583a7784c90d3522e89ae886b86fff4aded75a8c0b3ed993 |

Do not transfer this certificate to arbitrary current-head code, native full
replay, batch layouts, checkpoints, adapters, software stacks or hardware.
The full deployment/source/calibration manifests travel with the receipts.

Measured replay+feature timing:4-row matched pilot5.8546→1.2376s/span (~4.7x).
Original2719-row GPU bank mean1.29953s/span. Its historical CPU replay mean
6.53165s is a historical comparison, not a fresh matched full-bank benchmark.
The expanded roster reuses2693 byte/input-matched rows and measures774 additional
rows; dual-replay supplemental capture timings are not relabeled as single-call
throughput. Process peaks133.31445/133.53125GiB remain below135.29GiB.

## Batching: fidelity qualified, optional for performance

- Fully batched ragged replay failed at cache297 padded to299.
- Exact-shape batching passed a12-row pilot, then failed at cache290 in the full
  bank. Neither failed attempt is promoted or omitted from the record.
- Independent single-row prefill plus batched continuation clears the failing
  pair and passes its12-row pilot. Balanced warm timing retained36 paired samples:
  single1.16358 versus batch1.11542s/row (1.04318x); all six pairs and both
  execution orders improve. This is not bank-workload-weighted timing.
- `batch-bank-002` completed and `verification-batch-bank-002.json` independently
  passes G0–G5 on the original 2,719 rows under the tightened contrast map.
  Its 1,367 batches emit every row once, with 15 computational fillers charged
  to timing. The supplemental 774-row capture also completed; the final
  `roster-batch-certification-001.json` passes G0–G5 on all 3,467 owner-roster
  rows. Both captures peak at 134.52930GiB. Supplemental schedule:394 batches,
  14 computational fillers. No expanded population or threshold was changed.
- `batch-timing-bank-002-v2.json` measures 1.27355s/emitted row on first calls
  and 1.22602 on independent repeats, including filler costs. These are separate
  from setup and from the full dual-replay validation job's wall time.
  The 91 sparse paired rows show single1.28984 versus batch1.41206s/row
  (0.91344x). Excluding filler calls still gives only0.95555x on86 rows;
  the five sampled filler calls give0.49722x. This sample is neither balanced
  nor workload-weighted, but it contradicts a blanket batching-speedup claim.
  Retain the positive balanced pilot and this negative operational sample.
  Do not promote batching to a default performance recommendation on fidelity
  success alone; no batch-wide measured speedup over matched singles is claimed.
- Supplemental timing:1.25886s/row first calls,1.21357 independent repeats.
  Its58 sparse paired rows give0.90796x overall and0.99931x without fillers.
  This independently supports keeping batching optional, not a default win.

## Broader operating evidence and limitations

The length sweep repeats exactly at all30 points. Cross-lane error stays near
5e-5 over tested cache128–1024 points; cache64 shows broad continuous arithmetic
drift before GPU reduction, and2048 has increased but much smaller error.
Continuation length was128 throughout: this is not a minimum-span-length result.
No wider calibrated envelope is silently certified.

Native full-pass characterization on three inputs/model measured1.68x for3B
and3.15x for8B, with exact repeats. The3B first-call outlier is retained. These
3358/3414-feature measurements have no transferred70B sigma/floor certificate.
The experimental feature-only CLI requires explicit opt-in and a new directory;
it does not replace raw-tensor banking or accept adapters/interventions.

Remaining costs include eager attention, ~97.2MiB gate-tie CPU transfer per70B
span, and~317.5MiB positional-calibration upload. The original32KB D2H target
and speed forecast were withdrawn by the governing ferry, not met silently.
Lane tags and the shared analysis guard prevent ordinary loader joins from
mixing tagged arithmetic lanes; legacy untagged data remains explicitly unknown.

## Light residual-cost profile

`single-profile-report-001.json` preserves a one-input diagnostic against frozen
gpu-v3. The unprofiled warm replay takes1.16542s; the traced repeat is byte-exact.
Profiler instrumentation expands the host timeline to1.60069s, so its shares
must not be treated as exact percentages of unprofiled runtime or saved time.
Exclusive traced host shares: NumPy gate sorting21.92%, other gate work19.17%,
forward excluding feature hooks15.30%, attention reductions10.15%, value7.01%,
query6.56%, tensor construction/upload scopes5.49%. Device H2D27.17ms and
D2H16.68ms are separate activity durations, not additive host components.
Gate-feature work is a better next profiling target than bulk transfer alone.
No optimization or new numerical certificate follows from this trace.

The final audit and consumer handoff are separate from optional future speed work.
