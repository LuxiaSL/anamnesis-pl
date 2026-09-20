"""Aggregate complete batch captures without double-counting calls or fillers."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from extraction_perf_phase0 import sha, write_json


def summarize(rows: list[dict]) -> dict:
    if not rows or len({tuple(r["key"]) for r in rows}) != len(rows):
        raise ValueError("empty capture or duplicate emitted row")
    if len({r["lane"] for r in rows}) != 1:
        raise ValueError("mixed arithmetic lanes")
    batches = defaultdict(list)
    for row in rows:
        batches[row["batch_index"]].append(row)
    first_total = repeat_total = sampled_single = sampled_batch = 0.0
    fillers = sample_rows = 0
    paired_by_fill = {}
    for group in batches.values():
        first = group[0]
        for field in (
            "batch_seconds",
            "batch_repeat_seconds",
            "emitted_rows_in_batch",
            "filler_rows",
            "filler_identity_exact",
        ):
            if any(r[field] != first[field] for r in group):
                raise ValueError(f"inconsistent batch receipt: {field}")
        if first["emitted_rows_in_batch"] != len(group):
            raise ValueError("incomplete batch")
        if first["filler_rows"] < 0 or not first["filler_identity_exact"]:
            raise ValueError("invalid filler evidence")
        for field in ("batch_seconds", "batch_repeat_seconds"):
            if not math.isfinite(first[field]) or first[field] <= 0:
                raise ValueError("invalid elapsed time")
        for row in group:
            if not math.isclose(
                row["seconds_per_row"],
                first["batch_seconds"] / len(group),
                rel_tol=1e-12,
                abs_tol=0,
            ):
                raise ValueError("filler cost not charged to emitted rows")
        measured = [r["single_seconds"] is not None for r in group]
        if any(measured) and not all(measured):
            raise ValueError("partial paired timing batch")
        if all(measured):
            times = [r["single_seconds"] for r in group]
            if any(not math.isfinite(t) or t <= 0 for t in times):
                raise ValueError("invalid single elapsed time")
            sampled_single += sum(times)
            sampled_batch += first["batch_seconds"]
            sample_rows += len(group)
            label = "with_filler" if first["filler_rows"] else "no_filler"
            subtotal = paired_by_fill.setdefault(
                label, dict(rows=0, single_seconds=0.0, batch_seconds=0.0)
            )
            subtotal["rows"] += len(group)
            subtotal["single_seconds"] += sum(times)
            subtotal["batch_seconds"] += first["batch_seconds"]
        first_total += first["batch_seconds"]
        repeat_total += first["batch_repeat_seconds"]
        fillers += first["filler_rows"]
    for subtotal in paired_by_fill.values():
        subtotal["speedup"] = subtotal["single_seconds"] / subtotal["batch_seconds"]
    return dict(
        rows=len(rows),
        batches=len(batches),
        fillers=fillers,
        lane_id=rows[0]["lane"],
        first_call_seconds=first_total,
        repeat_call_seconds=repeat_total,
        first_call_seconds_per_row=first_total / len(rows),
        repeat_call_seconds_per_row=repeat_total / len(rows),
        paired_sample_rows=sample_rows,
        paired_single_seconds_per_row=sampled_single / sample_rows
        if sample_rows
        else None,
        paired_batch_seconds_per_row=sampled_batch / sample_rows
        if sample_rows
        else None,
        paired_sample_speedup=sampled_single / sampled_batch if sample_rows else None,
        paired_sample_by_filler=paired_by_fill,
        scope="model replay + reductions + transfers; first/repeat calls separate; fillers charged",
        caveat="sparse paired samples are not a full-population matched benchmark or balanced-order experiment",
    )


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    complete = json.loads((args.capture / "complete.json").read_text())
    paths = sorted(args.capture.glob("row-*.json"))
    rows = [json.loads(p.read_text()) for p in paths]
    if complete["rows"] != len(rows):
        raise ValueError("completion count mismatch")
    result = summarize(rows)
    result["verification_sha256"] = sha(args.verification)
    result["verification"] = str(args.verification)
    result["source_receipts"] = {p.name: sha(p) for p in paths}
    result["deployment_sha256"] = sha(args.capture / "deployment.json")
    result["script_sha256"] = sha(Path(__file__))
    write_json(args.out, result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "source_receipts"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
