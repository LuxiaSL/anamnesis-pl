"""Sign teacher-forced kotodama rows on the tsumugi GPU lane, batch of one.

Reads a replay-rows JSONL (a header row, then one row per span:
``{cand_id, input_ids, prompt_length, ...}``) and signs each row with
``kotodama.model.capture.replay_features_gpu`` under a pleroma ``KotoGpuLane``,
the lane pleroma's tsumugi banks are harvested on. Every row's full lane vector is
kept with its names, together with a readout in a banked space's ``FULL_names``
order (NaN where the row's schema lacks a name). Rows whose span is too short for
the lane (fewer than two generated tokens) are recorded as refused, never padded.

Output (``--out-dir``):
  ``signatures-<shard>.npz``: per schema (the names a row's step count yields),
    ``names_<k>``, ``features_<k>`` [rows, names] and ``cand_ids_<k>``;
    ``full_readout`` [rows, FULL] in manifest order; ``cand_ids``; ``lane_ids``;
    ``mean_logprob``.
  ``receipts-<shard>.jsonl``: one line per row (cand_id, lane_id, steps, status).
  ``run_meta-<shard>.json``: model, preset, calibration provenance, code roots,
    counts and wall time.

Resumable: rows already present in the shard's receipts are skipped, and a
partial file is checkpointed every ``--checkpoint`` rows.

Environment (as pleroma's tsumugi GPU-lane job): ``KOTODAMA_ATTNRES_EPS_FP64_COMPAT=1``,
``CUBLAS_WORKSPACE_CONFIG=:16:8``, ``ANAMNESIS_MODELS`` naming pleroma's profile,
PYTHONPATH = pleroma tree, its vendored anamnesis and kotodama ``src``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def _load_rows(path: Path, shard: int, n_shards: int) -> tuple[dict, list[dict]]:
    header: dict = {}
    rows: list[dict] = []
    with path.open() as fh:
        for i, line in enumerate(fh):
            rec = json.loads(line)
            if i == 0 and ("_header" in rec or "cand_id" not in rec):
                header = rec
                continue
            rows.append(rec)
    return header, [r for k, r in enumerate(rows) if k % n_shards == shard]


def _done(receipts: Path) -> set[str]:
    if not receipts.exists():
        return set()
    return {json.loads(line)["cand_id"] for line in receipts.open() if line.strip()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--checkpoint-path", required=True, help="tsumugi model.pt")
    ap.add_argument("--calib-dir", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path, help="banked space npz with FULL_names")
    ap.add_argument("--preset", default="tsumugi-1b-release-s52450")
    ap.add_argument("--koto-size", default="1b-tsumugi")
    ap.add_argument("--koto-boundaries", default="tsumugi_loop")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--checkpoint", type=int, default=500)
    args = ap.parse_args(argv)

    import torch
    from kotodama.model.capture import replay_features_gpu
    from pleroma.config.anamnesis_registry import ensure_registry
    from pleroma.harvest.koto_gpu_lane import KotoGpuLane
    from pleroma.model import koto

    if not torch.cuda.is_available():
        print("REFUSED: needs a GPU", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.shard:02d}of{args.n_shards:02d}"
    receipts_path = args.out_dir / f"receipts-{tag}.jsonl"
    header, rows = _load_rows(args.rows, args.shard, args.n_shards)
    done = _done(receipts_path)
    todo = [r for r in rows if r["cand_id"] not in done]
    print(f"[sign] shard {tag}: {len(rows)} rows, {len(done)} done, {len(todo)} to sign", flush=True)

    ensure_registry()
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = koto.load_koto(args.checkpoint_path, size=args.koto_size,
                           boundaries=koto.boundaries_for(args.koto_boundaries), device="cuda:0",
                           rope_fp32=True, eps_fp64_compat=True)
    manifest = np.load(args.manifest, allow_pickle=False)
    full_names = [str(n) for n in manifest["FULL_names"]]
    lanes = KotoGpuLane(args.preset, args.calib_dir, full_names, "cuda:0")
    reach = lanes.provenance["positions_calibrated"]

    partial_path = args.out_dir / f"partial-{tag}.npz"
    store: dict[str, dict] = {}
    if partial_path.exists():
        prev = np.load(partial_path, allow_pickle=True)
        store = prev["store"].item()
    t0 = time.time()
    counts = {"signed": 0, "refused_short": 0, "refused_reach": 0, "failed": 0}

    def checkpoint() -> None:
        np.savez(partial_path, store=np.array(store, dtype=object))

    with receipts_path.open("a") as receipts:
        for k, row in enumerate(todo, 1):
            ids, P = row["input_ids"], int(row["prompt_length"])
            steps = len(ids) - P - 1
            status, lane_id = "signed", None
            if len(ids) - P < 2:
                status = "refused_short"
            elif len(ids) - 2 >= reach:
                status = "refused_reach"
            else:
                try:
                    result = replay_features_gpu(model, ids, P, lanes.lane(steps))
                    names = [str(n) for n in result.feature_names]
                    key = hashlib.sha256(json.dumps(names).encode()).hexdigest()[:12]
                    group = store.setdefault(key, {"names": names, "features": [], "cand_ids": [],
                                                   "lane_ids": [], "mean_logprob": []})
                    group["features"].append(np.asarray(result.features, dtype=np.float32))
                    group["cand_ids"].append(row["cand_id"])
                    lane_id = (result.metadata or {}).get("lane_id")
                    group["lane_ids"].append(lane_id)
                    group["mean_logprob"].append(float(result.mean_logprob))
                except Exception as exc:  # recorded per row; the run continues
                    status = f"failed: {type(exc).__name__}: {exc}"
            counts[status if status in counts else "failed"] += 1
            receipts.write(json.dumps({"cand_id": row["cand_id"], "lane_id": lane_id,
                                       "steps": steps, "status": status}) + "\n")
            if k % args.checkpoint == 0:
                receipts.flush()
                checkpoint()
                rate = k / (time.time() - t0)
                print(f"[sign] {k}/{len(todo)} rows, {rate:.2f} rows/s, {counts}", flush=True)
    checkpoint()

    out: dict[str, np.ndarray] = {}
    all_ids, all_lanes, all_lp, readout = [], [], [], []
    for key, group in store.items():
        out[f"names_{key}"] = np.array(group["names"])
        out[f"features_{key}"] = np.stack(group["features"])
        out[f"cand_ids_{key}"] = np.array(group["cand_ids"])
        index = {n: i for i, n in enumerate(group["names"])}
        cols = [index.get(n, -1) for n in full_names]
        for vec in group["features"]:
            readout.append(np.array([vec[c] if c >= 0 else np.nan for c in cols], dtype=np.float32))
        all_ids += group["cand_ids"]
        all_lanes += group["lane_ids"]
        all_lp += group["mean_logprob"]
    out["full_readout"] = np.stack(readout) if readout else np.zeros((0, len(full_names)), np.float32)
    out["full_names"] = np.array(full_names)
    out["cand_ids"] = np.array(all_ids)
    out["lane_ids"] = np.array([str(x) for x in all_lanes])
    out["mean_logprob"] = np.array(all_lp, dtype=np.float64)
    np.savez(args.out_dir / f"signatures-{tag}.npz", **out)
    meta = dict(rows=str(args.rows), rows_header=header, preset=args.preset,
                checkpoint=args.checkpoint_path, calibration=lanes.provenance,
                manifest=str(args.manifest), shard=args.shard, n_shards=args.n_shards,
                counts=counts, lane_ids=sorted({str(x) for x in all_lanes}),
                schemas={k: len(g["names"]) for k, g in store.items()},
                wall_seconds=round(time.time() - t0, 1),
                code={m: sys.modules[m].__file__ for m in ("anamnesis", "kotodama", "pleroma")})
    (args.out_dir / f"run_meta-{tag}.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[sign] done shard {tag}: {counts}; lane ids {meta['lane_ids']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
