"""Summarize every predeclared paired timing sample, without order cherry-picking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from extraction_perf_phase0 import sha, write_json
from extraction_perf_verify_batch_trial import verify


def summarize(records, rounds):
    if rounds < 2 or rounds % 2 or len(records) != 6:
        raise ValueError("six pairs and even balanced rounds required")
    all_samples = []
    pairs = []
    for index, record in enumerate(records):
        if (
            record["batch"] != index
            or not record["all_samples_retained"]
            or not record["feature_bytes_unchanged"]
        ):
            raise ValueError("incomplete/unstable timing evidence")
        samples = record["samples"]
        if len(samples) != rounds:
            raise ValueError("missing timing round")
        for i, sample in enumerate(samples):
            expected = ["single", "batch"] if i % 2 == 0 else ["batch", "single"]
            if sample["round"] != i or sample["order"] != expected:
                raise ValueError("timing order differs from predeclared balance")
            if not all(
                np.isfinite(sample[k]) and sample[k] > 0
                for k in ("single_seconds_per_row", "batch_seconds_per_row")
            ):
                raise ValueError("invalid timing")
        single = float(np.mean([s["single_seconds_per_row"] for s in samples]))
        batch = float(np.mean([s["batch_seconds_per_row"] for s in samples]))
        pairs.append(
            dict(
                shape=record["shape"],
                keys=record["keys"],
                single_mean=single,
                batch_mean=batch,
                speedup=single / batch,
            )
        )
        all_samples.extend(samples)

    def stats(samples):
        single = float(np.mean([s["single_seconds_per_row"] for s in samples]))
        batch = float(np.mean([s["batch_seconds_per_row"] for s in samples]))
        return dict(
            n=len(samples), single_mean=single, batch_mean=batch, speedup=single / batch
        )

    return dict(
        all_samples=stats(all_samples),
        by_pair=pairs,
        by_first_mode={
            mode: stats([s for s in all_samples if s["order"][0] == mode])
            for mode in ("single", "batch")
        },
        weighting="six fixed input pairs equally weighted; not full-bank workload weighting",
        statistical_scope="descriptive measured timing; no population confidence claim",
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ("pilot", "sigma", "rulers", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    pilot = verify(args.pilot, args.sigma, args.rulers)
    if not pilot["pilot_vector_memory_checks_pass"]:
        raise ValueError("pilot failed fidelity/resource gate")
    pin = json.loads((args.pilot / "deployment.json").read_text())
    paths = sorted(args.pilot.glob("timing-batch-*.json"))
    result = summarize([json.loads(p.read_text()) for p in paths], pin["timing_rounds"])
    result.update(
        pilot=pilot, files_sha256={str(p): sha(p) for p in paths}, certification=False
    )
    write_json(args.out, result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("pilot", "files_sha256")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
