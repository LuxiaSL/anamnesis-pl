"""Independently recompute the batch pilot, preserving its limited scope."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from extraction_perf_phase0 import sha, write_json


def verify(folder: Path, sigma_path: Path, rulers_path: Path) -> dict:
    pin = json.loads((folder / "deployment.json").read_text())
    complete = json.loads((folder / "complete.json").read_text())
    if complete["rows"] != 12 or pin["batch_size"] != 2:
        raise ValueError("incomplete or wrong batch pilot")
    if (
        sha(sigma_path) != pin["standardizer_sha256"]
        or sha(rulers_path) != pin["rulers_sha256"]
    ):
        raise ValueError("ruler identity changed")
    ruler = json.loads(rulers_path.read_text())
    bound = {tuple(r["key"][1:]): r["bound"] for r in ruler["vectors"]["rows"]}
    with np.load(sigma_path) as z:
        sigma, weights = z["sigma_cal"].astype(float), z["weights"].astype(float)
        schema = hashlib.sha256(
            json.dumps(z["feature_names"].astype(str).tolist()).encode()
        ).hexdigest()
    records = [json.loads(p.read_text()) for p in sorted(folder.glob("row-*.json"))]
    if len(records) != 12 or [r["key"] for r in records] != [
        r["key"] for r in pin["selected"]
    ]:
        raise ValueError("pilot population mismatch")
    memory = json.loads((folder / "process-vram.json").read_text())
    if memory["errors"] or any(
        not memory["phases"].get(p, {}).get("samples")
        for p in ("warmup", "batch", "batch_repeat")
    ):
        raise ValueError("incomplete process memory")
    peak = (
        max(
            p["peak_bytes"]
            for p in memory["phases"].values()
            if p["peak_bytes"] is not None
        )
        / 2**30
    )
    replay_ids = set()
    batch_invocations = set()
    lanes = set()
    summaries = []
    for i, record in enumerate(records):
        path = folder / f"row-{i:05d}.npz"
        with np.load(path) as z:
            for name in ("gpu", "repeat", "bank", "single"):
                if (
                    z[name].dtype != np.float32
                    or z[name].shape != sigma.shape
                    or not np.isfinite(z[name]).all()
                ):
                    raise ValueError("invalid feature array")
            exact = z["gpu"].tobytes() == z["repeat"].tobytes()
            error = (z["gpu"].astype(float) - z["bank"]) / sigma
            distance = float(np.linalg.norm(error * weights))
            maximum = float(np.abs(error).max())
            single_error = float(
                np.linalg.norm((z["gpu"].astype(float) - z["single"]) * weights / sigma)
            )
        a, b = record["telemetry"], record["repeat_telemetry"]
        for meta in (a, b):
            if (
                not meta["fresh_cache"]
                or meta["batch_size"] != 2
                or meta["feature_schema_sha256"] != schema
                or meta["calibration_sha256"] != pin["calibration_sha256"]
                or meta.get("prefill_policy", "batched")
                != pin.get("prefill_policy", "batched")
            ):
                raise ValueError("batch replay metadata mismatch")
            if meta["replay_id"] in replay_ids:
                raise ValueError("reused replay")
            replay_ids.add(meta["replay_id"])
            lanes.add(meta["lane_id"])
        if (
            a["batch_invocation_id"] == b["batch_invocation_id"]
            or a["batch_composition_sha256"] != b["batch_composition_sha256"]
        ):
            raise ValueError("batch repetition/composition mismatch")
        if a["batch_row"] != i % 2 or b["batch_row"] != i % 2:
            raise ValueError("batch row alignment mismatch")
        if i % 2 == 0:
            for invocation in (a["batch_invocation_id"], b["batch_invocation_id"]):
                if invocation in batch_invocations:
                    raise ValueError("reused batch invocation")
                batch_invocations.add(invocation)
        if i % 2:
            previous = records[i - 1]
            if any(
                record[k] != previous[k]
                for k in ("batch_seconds", "batch_repeat_seconds")
            ):
                raise ValueError("inconsistent shared timing")
            if a["batch_invocation_id"] != previous["telemetry"]["batch_invocation_id"]:
                raise ValueError("rows were not jointly replayed")
        if record["seconds_per_row"] != record["batch_seconds"] / 2:
            raise ValueError("incorrect per-row timing attribution")
        summaries.append(
            dict(
                key=record["key"],
                G0=exact,
                G1=bool(distance <= bound[tuple(record["key"])] and maximum <= 0.1)
                if exact
                else None,
                floor_c=distance,
                max_feature_sigma_error=maximum,
                batch_vs_single_l2=single_error,
                arrays_sha256=sha(path),
            )
        )
    if len(lanes) != 1:
        raise ValueError("mixed batch lanes")
    single_seconds = float(np.mean([r["single_seconds"] for r in records]))
    batch_seconds = float(np.mean([r["seconds_per_row"] for r in records]))
    return dict(
        pilot_vector_memory_checks_pass=all(r["G0"] and r["G1"] for r in summaries)
        and peak <= 135.29,
        rows=summaries,
        process_peak_gib=peak,
        mean_single_seconds=single_seconds,
        mean_batch_seconds_per_row=batch_seconds,
        paired_speedup=single_seconds / batch_seconds,
        deployment_sha256=sha(folder / "deployment.json"),
        lane_id=next(iter(lanes)),
        prefill_policy=pin.get("prefill_policy", "batched"),
        certification=False,
        scope="12-row batch2 pilot; full bank and downstream gates still required",
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("pilot", "sigma", "rulers", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = verify(args.pilot, args.sigma, args.rulers)
    write_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
