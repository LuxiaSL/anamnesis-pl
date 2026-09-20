"""Plan roster evidence reuse only after matching inputs AND banked feature bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from extraction_perf_phase0 import sha, write_json


def matches(row, record, bank_sha):
    return (
        row["key"] == record["key"]
        and row["bank_vector_sha256"] == bank_sha
        and record["telemetry"]["input_tokens_sha256"] == row["input_sha256"]
        and record["telemetry"]["span_start"] == row["start"]
        and record["telemetry"]["span_end"] == row["end"]
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ("audit", "candidate", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    audit = json.loads(args.audit.read_text())
    pin = json.loads((args.candidate / "deployment.json").read_text())
    if audit["standardizer_sha256"] != pin["standardizer_sha256"]:
        raise ValueError("roster/candidate standardizer mismatch")
    if not (args.candidate / "complete.json").exists():
        raise ValueError("only completed candidate banks may be planned for reuse")
    existing = {}
    for path in sorted(args.candidate.glob("row-*.json")):
        record = json.loads(path.read_text())
        key = tuple(record["key"])
        if key in existing:
            raise ValueError("duplicate candidate row")
        existing[key] = (record, path)
    reused = []
    missing = []
    changed = []
    for row in audit["row_manifest"]:
        key = tuple(row["key"])
        if key not in existing:
            missing.append(row)
            continue
        record, path = existing[key]
        with np.load(path.with_suffix(".npz")) as data:
            bank_sha = hashlib.sha256(data["bank"].tobytes()).hexdigest()
            if (
                data["gpu"].tobytes() != data["repeat"].tobytes()
                or record["G0"] is not True
            ):
                raise ValueError("cannot reuse failed repeatability evidence")
        if not matches(row, record, bank_sha):
            changed.append(row)
            continue
        if record["telemetry"]["calibration_sha256"] != pin["calibration_sha256"]:
            raise ValueError("candidate calibration receipt inconsistent")
        if (
            record["telemetry"]["feature_schema_sha256"]
            != audit["feature_schema_sha256"]
        ):
            raise ValueError("roster/candidate schema mismatch")
        reused.append(
            dict(
                reference=row,
                receipt=str(path),
                receipt_sha256=sha(path),
                arrays_sha256=sha(path.with_suffix(".npz")),
            )
        )
    required = missing + changed
    result = dict(
        roster_audit_sha256=sha(args.audit),
        candidate_deployment_sha256=sha(args.candidate / "deployment.json"),
        total_rows=len(audit["row_manifest"]),
        reusable_rows=len(reused),
        new_keys=len(missing),
        changed_reference_rows=len(changed),
        measurement_rows=len(required),
        reuse=reused,
        required_measurements=required,
        certification=False,
        note="Reuse plan only; new rows, floors, downstream cohort and owner workspace prerequisite remain separate",
    )
    write_json(args.out, result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("reuse", "required_measurements")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
