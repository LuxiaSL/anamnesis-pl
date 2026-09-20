"""Native model measurements, with no invented cross-model tolerance ruler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from extraction_perf_phase0 import sha, write_json


def compare(cpu: Path, gpu: Path) -> dict:
    folders = (cpu, gpu)
    pins = [json.loads((p / "deployment.json").read_text()) for p in folders]
    for field in (
        "source",
        "model",
        "model_files",
        "calibration_files",
        "manifest_sha256",
        "selected",
        "extraction",
        "families",
        "torch",
    ):
        if pins[0][field] != pins[1][field]:
            raise ValueError(f"native reference mismatch: {field}")
    if pins[0]["mode"] != "cpu" or pins[1]["mode"] != "gpu":
        raise ValueError("wrong native modes")
    if sha(cpu / "inputs.json") != sha(gpu / "inputs.json"):
        raise ValueError("native inputs differ")
    for p in folders:
        if json.loads((p / "complete.json").read_text())["rows"] != 3:
            raise ValueError("incomplete native capture")
    results = []
    for i, key in enumerate(pins[0]["selected"]):
        arrays = []
        records = []
        names = None
        hashes = []
        for p in folders:
            path = p / f"row-{i:05d}.npz"
            record = json.loads(path.with_suffix(".json").read_text())
            if record["generation_id"] != key:
                raise ValueError("native row mismatch")
            with np.load(path) as z:
                current = z["feature_names"].astype(str).tolist()
                if names is not None and current != names:
                    raise ValueError("native schema differs across lanes")
                names = current
                if (
                    z["features"].dtype != np.float32
                    or not np.isfinite(z["features"]).all()
                ):
                    raise ValueError("invalid native features")
                if z["features"].tobytes() != z["repeat"].tobytes() or not record["G0"]:
                    raise ValueError("native repeat failure")
                arrays.append(z["features"].copy())
            records.append(record)
            hashes.append(sha(path))
        a, b = records[1]["metadata"]
        if (
            a["replay_id"] == b["replay_id"]
            or not a["fresh_cache"]
            or not b["fresh_cache"]
            or a["replay_path"] != "full"
            or b["replay_path"] != "full"
        ):
            raise ValueError("invalid independent GPU replay provenance")
        delta = arrays[1].astype(float) - arrays[0].astype(float)
        worst = np.argsort(np.abs(delta))[-5:][::-1]
        results.append(
            dict(
                generation_id=key,
                feature_count=len(names),
                prefix=records[0]["prefix"],
                continuation=records[0]["continuation"],
                G0=True,
                cpu_seconds=records[0]["seconds"],
                gpu_seconds=records[1]["seconds"],
                raw_l2=float(np.linalg.norm(delta)),
                raw_max_abs=float(np.abs(delta).max()),
                differing_coordinates=int(np.count_nonzero(delta)),
                arrays_sha256=hashes,
                largest_raw_errors=[
                    dict(
                        name=names[j],
                        cpu=float(arrays[0][j]),
                        gpu=float(arrays[1][j]),
                        absolute_error=float(abs(delta[j])),
                    )
                    for j in worst
                ],
            )
        )
    cpu_mean = float(np.mean([r["cpu_seconds"][0] for r in results]))
    gpu_mean = float(np.mean([r["gpu_seconds"][0] for r in results]))
    peaks = {}
    for folder, label in zip(folders, ("cpu", "gpu"), strict=True):
        memory = json.loads((folder / "process-vram.json").read_text())
        if memory["errors"]:
            raise ValueError("native process memory sampling failed")
        peaks[label] = (
            max(
                p["peak_bytes"]
                for p in memory["phases"].values()
                if p["peak_bytes"] is not None
            )
            / 2**30
        )
    return dict(
        model=pins[0]["model"],
        rows=results,
        mean_cpu_seconds=cpu_mean,
        mean_gpu_seconds=gpu_mean,
        paired_speedup=cpu_mean / gpu_mean,
        process_peak_gib=peaks,
        deployment_sha256=[sha(p / "deployment.json") for p in folders],
        certification=False,
        numerical_scope="raw feature units only; no model-specific sigma, path floor or condition gate evaluated",
        timing_scope="full-pass replay plus feature computation; first measured call per input after one warmup",
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ("cpu", "gpu", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = compare(args.cpu, args.gpu)
    write_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
