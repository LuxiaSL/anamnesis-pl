"""Recompute synthetic length-sweep evidence, without inventing a flatness gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from extraction_perf_phase0 import sha, write_json
from extraction_perf_sweep import LENGTHS


def verify(cpu: Path, gpu: Path, sigma_path: Path) -> dict:
    pins = [json.loads((p / "deployment.json").read_text()) for p in (cpu, gpu)]
    for folder in (cpu, gpu):
        complete = json.loads((folder / "complete.json").read_text())
        if complete["rows"] != 30:
            raise ValueError("incomplete sweep")
    for field in (
        "inputs_sha256",
        "bank_pin_sha256",
        "calibration_sha256",
        "standardizer_sha256",
        "caller",
        "torch",
        "transformers",
        "numpy",
        "runner_sha256",
    ):
        if pins[0][field] != pins[1][field]:
            raise ValueError(f"cross-lane provenance mismatch: {field}")
    if pins[0]["mode"] != "cpu" or pins[1]["mode"] != "gpu":
        raise ValueError("wrong sweep modes")
    if pins[0]["workspace"] is not None or pins[1]["workspace"] != ":16:8":
        raise ValueError("unqualified arithmetic workspace")
    if (
        pins[0]["deterministic"] is not False
        or pins[1]["deterministic"] is not True
        or any(p["tf32"] is not False for p in pins)
    ):
        raise ValueError("unqualified deterministic/TF32 settings")
    for p, pin in zip((cpu, gpu), pins, strict=True):
        if sha(p / "inputs.json") != pin["inputs_sha256"]:
            raise ValueError("input manifest changed")
    if sha(sigma_path) != pins[0]["standardizer_sha256"]:
        raise ValueError("standardizer changed")
    with np.load(sigma_path) as z:
        sigma, weights = z["sigma_cal"].astype(float), z["weights"].astype(float)
        names = z["feature_names"].astype(str).tolist()
    schema = hashlib.sha256(json.dumps(names).encode()).hexdigest()
    inputs = json.loads((cpu / "inputs.json").read_text())
    expected = {("synthetic", length, seed) for length in LENGTHS for seed in range(5)}
    if len(inputs) != 30 or {tuple(r["key"]) for r in inputs} != expected:
        raise ValueError("sweep population differs from declared six-by-five grid")
    results, file_pins, replay_ids, lane_ids = [], {}, set(), set()
    for folder in (cpu, gpu):
        if (
            len(list(folder.glob("row-*.json"))) != 30
            or len(list(folder.glob("row-*.npz"))) != 30
        ):
            raise ValueError("missing/extra sweep rows")
    for i, row in enumerate(inputs):
        if (
            row["start"] != row["key"][1]
            or row["end"] != row["start"] + 128
            or len(row["tokens"]) != row["end"]
        ):
            raise ValueError("synthetic span geometry mismatch")
        records, arrays = [], []
        for folder in (cpu, gpu):
            path = folder / f"row-{i:05d}.json"
            rec = json.loads(path.read_text())
            if rec["key"] != row["key"] or rec["cache_length"] != row["start"]:
                raise ValueError("row alignment mismatch")
            with np.load(path.with_suffix(".npz")) as z:
                values, repeat = z["features"], z["repeat"]
                if (
                    values.dtype != np.float32
                    or values.shape != sigma.shape
                    or repeat.dtype != np.float32
                    or repeat.shape != values.shape
                    or not np.isfinite(values).all()
                ):
                    raise ValueError("invalid feature array")
                if (
                    values.tobytes() != repeat.tobytes()
                    or not rec["independent_repeat_exact"]
                ):
                    raise ValueError("independent replay mismatch")
                arrays.append(values.copy())
            records.append(rec)
            if rec["last_feature_position"] != row["end"] - 2:
                raise ValueError("feature position mismatch")
            for artifact in (path, path.with_suffix(".npz")):
                file_pins[str(artifact)] = sha(artifact)
        input_sha = hashlib.sha256(
            np.asarray(row["tokens"], dtype="<i8").tobytes()
        ).hexdigest()
        for meta in records[1]["metadata"]:
            if (
                not meta["fresh_cache"]
                or meta["input_tokens_sha256"] != input_sha
                or meta["feature_schema_sha256"] != schema
                or meta["calibration_sha256"] != pins[1]["calibration_sha256"]
            ):
                raise ValueError("GPU replay provenance mismatch")
            if meta["replay_id"] in replay_ids:
                raise ValueError("reused replay invocation")
            replay_ids.add(meta["replay_id"])
            lane_ids.add(meta["lane_id"])
        if len(records[1]["metadata"]) != 2:
            raise ValueError("two replay receipts required")
        for field in (
            "inside_calibration",
            "all_feature_positions_count_gt5",
            "last_feature_position",
        ):
            if records[0][field] != records[1][field]:
                raise ValueError("calibration scope differs between lanes")
        delta = arrays[1].astype(float) - arrays[0].astype(float)
        results.append(
            dict(
                key=row["key"],
                cache_length=row["start"],
                floor_c=float(np.linalg.norm(delta * weights / sigma)),
                delta_self=0.0,
                max_feature_sigma_error=float(np.max(np.abs(delta) / sigma)),
                calibrated_positions=records[0]["inside_calibration"]
                and records[0]["all_feature_positions_count_gt5"],
                cpu_seconds=records[0]["seconds"],
                gpu_seconds=records[1]["seconds"],
            )
        )
    if len(lane_ids) != 1 or len(replay_ids) != 60:
        raise ValueError("mixed lane or incomplete independent replays")
    lengths = []
    for length in LENGTHS:
        group = [r for r in results if r["cache_length"] == length]
        lengths.append(
            dict(
                cache_length=length,
                n=len(group),
                delta_self=0.0,
                floor_c_min=min(r["floor_c"] for r in group),
                floor_c_max=max(r["floor_c"] for r in group),
                floor_c_mean=float(np.mean([r["floor_c"] for r in group])),
                calibrated_positions=all(r["calibrated_positions"] for r in group),
            )
        )
    return dict(
        coverage_complete=True,
        independent_repeats_exact=True,
        per_feature_pass=all(r["max_feature_sigma_error"] <= 0.1 for r in results),
        lengths=lengths,
        rows=results,
        files_sha256=file_pins,
        deployment_sha256=[sha(p / "deployment.json") for p in (cpu, gpu)],
        lane_id=next(iter(lane_ids)),
        certification=False,
        scope="Instrument length-sweep evidence only; consumer scope remains banked regime.",
        flatness="No numerical flatness threshold was ratified; report trend for scope review.",
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--cpu", type=Path, required=True)
    p.add_argument("--gpu", type=Path, required=True)
    p.add_argument("--sigma", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = verify(args.cpu, args.gpu, args.sigma)
    write_json(args.out, result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("files_sha256", "rows")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
