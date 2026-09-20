"""Join exact owner-roster vector/path evidence; leave unresolved gates explicit."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from extraction_perf_phase0 import sha, write_json
from extraction_perf_verify_bank import row_files
from anamnesis.extraction.path_floor import coordinate_lower_bound
import anamnesis.extraction.path_floor as path_floor_module


def choose_evidence(key, reused, original, supplemental):
    source = original if key in reused else supplemental
    if key not in source:
        raise ValueError(f"missing required roster evidence: {key}")
    return source[key]


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in (
        "audit",
        "reuse",
        "original",
        "original-floors",
        "supplement",
        "supplement-floors",
        "sigma",
        "memory-reference",
        "conditions",
        "out",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workspace-probe", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--anchor", type=Path)
    parser.add_argument("--bank-scope-ruling", type=Path)
    parser.add_argument(
        "--candidate-kind", choices=("single", "batch"), default="single"
    )
    parser.add_argument("--reuse-origin", type=Path)
    parser.add_argument("--roster", type=Path)
    parser.add_argument("--extension", type=Path)
    for name, target in (("reference-prefix", "reference"), ("extension-prefix", "extension")):
        parser.add_argument(
            f"--{name}",
            help=(
                f"Absolute path prefix, exactly as recorded in the roster's file entries, "
                f"that --{target} mirrors locally. Required for --candidate-kind batch."
            ),
        )
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    for folder in (
        args.original,
        args.original_floors,
        args.supplement,
        args.supplement_floors,
    ):
        if not (folder / "complete.json").exists():
            raise ValueError(f"capture not complete: {folder}")
    audit = json.loads(args.audit.read_text())
    reuse = json.loads(args.reuse.read_text())
    if (
        sha(args.audit) != reuse["roster_audit_sha256"]
        or sha(args.sigma) != audit["standardizer_sha256"]
    ):
        raise ValueError("roster audit/ruler identity changed")
    original_pin = json.loads((args.original / "deployment.json").read_text())
    supplemental_pin = json.loads((args.supplement / "deployment.json").read_text())
    old_floor_pin = json.loads((args.original_floors / "deployment.json").read_text())
    new_floor_pin = json.loads((args.supplement_floors / "deployment.json").read_text())
    validation = json.loads((args.original_floors / "validation-pass.json").read_text())
    if validation["bit_exact"] is not True or validation["n"] != 945:
        raise ValueError("validated first-step proof origin missing")
    if (
        new_floor_pin["proof_origin_sha256"]
        != sha(args.original_floors / "deployment.json")
        or new_floor_pin["cpu_pin_sha256"]
        != old_floor_pin["reference_deployment_sha256"]
        or new_floor_pin["helper_sha256"] != old_floor_pin["helper_sha256"]
        or old_floor_pin["helper_sha256"] != sha(Path(path_floor_module.__file__))
        or new_floor_pin["source"] != old_floor_pin["source"]
        or new_floor_pin["caller"] != old_floor_pin["caller"]
        or new_floor_pin["standardizer_sha256"] != audit["standardizer_sha256"]
        or new_floor_pin["reuse_sha256"] != sha(args.reuse)
        or new_floor_pin["selected"]
        != [
            r["key"] if isinstance(r, dict) else r for r in supplemental_pin["selected"]
        ]
        or new_floor_pin["mode"] != "cpu-floor"
        or new_floor_pin["workspace"] is not None
        or new_floor_pin["deterministic"] is not False
    ):
        raise ValueError("CPU floor proof lineage/arithmetic mismatch")
    reuse_origin = args.reuse_origin or args.original
    if (
        sha(reuse_origin / "deployment.json") != reuse["candidate_deployment_sha256"]
        or (
            args.candidate_kind == "single"
            and supplemental_pin["origin_sha256"]
            != reuse["candidate_deployment_sha256"]
        )
        or supplemental_pin["reuse_sha256"] != sha(args.reuse)
        or supplemental_pin["audit_sha256"] != sha(args.audit)
    ):
        raise ValueError("supplement lineage mismatch")
    for field in ("source", "caller", "calibration_sha256", "standardizer_sha256"):
        if original_pin[field] != supplemental_pin[field]:
            raise ValueError(f"candidate identity changed: {field}")
    old = row_files([args.original])
    new = row_files([args.supplement])
    batch_schedules = None
    if args.candidate_kind == "batch":
        if not all(
            (
                args.reference,
                args.extension,
                args.roster,
                args.reuse_origin,
                args.reference_prefix,
                args.extension_prefix,
            )
        ):
            raise ValueError(
                "batch roster join requires explicit reference mirrors "
                "(--reference/--extension with their --*-prefix roster keys) "
                "and reuse origin"
            )
        from extraction_perf_phase0 import rows_from_bank
        from extraction_perf_roster import roster_rows
        from extraction_perf_roster_supplement import select_supplement
        from extraction_perf_verify_bank import validate_batch_schedule

        if sha(args.roster) != audit["roster_sha256"]:
            raise ValueError("batch roster identity mismatch")
        original_rows, _ = rows_from_bank(args.reference)
        owner_rows, _, _ = roster_rows(
            args.roster,
            {
                args.reference_prefix: args.reference,
                args.extension_prefix: args.extension,
            },
        )
        supplement_rows = select_supplement(owner_rows, reuse)
        batch_schedules = [
            validate_batch_schedule(args.original, original_pin, original_rows, old),
            validate_batch_schedule(
                args.supplement, supplemental_pin, supplement_rows, new
            ),
        ]
    old_floors = row_files([args.original_floors])
    new_floors = row_files([args.supplement_floors])
    reused = {tuple(item["reference"]["key"]): item for item in reuse["reuse"]}
    with np.load(args.sigma) as z:
        sigma, weights = z["sigma_cal"].astype(float), z["weights"].astype(float)
        names = z["feature_names"].astype(str).tolist()
    if (
        hashlib.sha256(json.dumps(names).encode()).hexdigest()
        != audit["feature_schema_sha256"]
    ):
        raise ValueError("ruler schema differs")
    indices = np.array([names.index(f"activation_norm_traj0_L{i}") for i in range(80)])
    rows = []
    reference_vectors = []
    candidate_vectors = []
    repeat_vectors = []
    candidate_metadata = []
    lane_ids = set()
    replay_ids = set()
    for reference in audit["row_manifest"]:
        key = tuple(reference["key"])
        record, path = choose_evidence(key, reused, old, new)
        floor_record, floor_path = choose_evidence(key, reused, old_floors, new_floors)
        if key in reused and args.candidate_kind == "single":
            saved = reused[key]
            if (
                sha(path) != saved["arrays_sha256"]
                or sha(path.with_suffix(".json")) != saved["receipt_sha256"]
            ):
                raise ValueError("reused evidence changed")
        with np.load(path) as values, np.load(floor_path) as floor:
            bank, candidate, repeat = values["bank"], values["gpu"], values["repeat"]
            if any(
                v.dtype != np.float32
                or v.shape != sigma.shape
                or not np.isfinite(v).all()
                for v in (bank, candidate, repeat)
            ):
                raise ValueError("invalid roster vector")
            if (
                hashlib.sha256(bank.tobytes()).hexdigest()
                != reference["bank_vector_sha256"]
            ):
                raise ValueError("reference vector does not match owner roster")
            if (
                floor["coordinate_indices"].tolist() != indices.tolist()
                or floor["reference_coordinates"].tobytes() != bank[indices].tobytes()
            ):
                raise ValueError("floor coordinates do not match reference")
            bound = coordinate_lower_bound(
                floor["incremental_coordinates"],
                floor["reference_coordinates"],
                sigma[indices],
                weights[indices],
            )["floor_b_lower_bound"]
            if bound != floor_record["floor_b_lower_bound"]:
                raise ValueError("path bound does not reproduce")
            exact = candidate.tobytes() == repeat.tobytes()
            error = (candidate.astype(float) - bank) / sigma
            distance = float(np.linalg.norm(error * weights))
            max_error = float(np.abs(error).max())
            reference_vectors.append(bank.copy())
            candidate_vectors.append(candidate.copy())
            repeat_vectors.append(repeat.copy())
        for field in ("telemetry", "repeat_telemetry"):
            meta = record[field]
            if (
                meta["input_tokens_sha256"] != reference["input_sha256"]
                or meta["span_start"] != reference["start"]
                or meta["span_end"] != reference["end"]
                or meta["calibration_sha256"] != original_pin["calibration_sha256"]
                or meta["feature_schema_sha256"] != audit["feature_schema_sha256"]
                or not meta["fresh_cache"]
            ):
                raise ValueError("replay input/calibration/schema mismatch")
            if meta["replay_id"] in replay_ids:
                raise ValueError("reused replay invocation")
            replay_ids.add(meta["replay_id"])
            lane_ids.add(meta["lane_id"])
        candidate_metadata.append(record["telemetry"])
        rows.append(
            dict(
                key=list(key),
                G0=exact,
                floor_c=distance,
                floor_b_lower_bound=bound,
                path_bound_proven=bool(exact and distance <= bound),
                requires_full_floor=bool(exact and distance > bound),
                max_feature_sigma_error=max_error,
                feature_bound_pass=bool(exact and max_error <= 0.1),
                source_arrays_sha256=sha(path),
                floor_arrays_sha256=sha(floor_path),
            )
        )
    if len(lane_ids) != 1 or len(rows) != audit["rows"]:
        raise ValueError("mixed lane or incomplete roster")
    peaks = []
    for folder in (args.original, args.supplement):
        memory = json.loads((folder / "process-vram.json").read_text())
        if memory["errors"] or any(
            not p["samples"] for name, p in memory["phases"].items() if name != "setup"
        ):
            raise ValueError("missing process memory evidence")
        peaks.append(
            max(
                p["peak_bytes"]
                for p in memory["phases"].values()
                if p["peak_bytes"] is not None
            )
            / 2**30
        )
    cap = json.loads(args.memory_reference.read_text())["cap_gib"]
    condition_manifest = json.loads(args.conditions.read_text())
    if condition_manifest.get(
        "format"
    ) != "declared-contrasts-v2" or condition_manifest["roster_audit_sha256"] != sha(
        args.audit
    ):
        raise ValueError("conditions are not pinned to this roster audit")
    contrast_receipts = None
    g1 = None
    if all(r["G0"] for r in rows):
        from extraction_perf_contrast_rules import resolve_entries

        keys = tuple(("rubric-140", *r["key"]) for r in audit["row_manifest"])
        _, distances, _, contrast_receipts = resolve_entries(
            condition_manifest["rows"],
            keys,
            np.stack(reference_vectors),
            np.stack(candidate_vectors),
            np.stack(repeat_vectors),
            sigma,
            weights,
            {
                k: r["input_sha256"]
                for k, r in zip(keys, audit["row_manifest"], strict=True)
            },
            {
                k: (r["start"], r["end"])
                for k, r in zip(keys, audit["row_manifest"], strict=True)
            },
        )
        for key, row in zip(keys, rows, strict=True):
            condition = distances[key]
            row["condition_distance"] = condition
            row["condition_bound_pass"] = (
                condition is None or row["floor_c"] <= 0.1 * condition
            )
            row["G1"] = (
                False
                if not row["condition_bound_pass"] or not row["feature_bound_pass"]
                else True
                if row["path_bound_proven"]
                else None
            )
        g1 = (
            False
            if any(r["G1"] is False for r in rows)
            else None
            if any(r["G1"] is None for r in rows)
            else True
        )
    workspace = None
    if args.workspace_probe:
        from extraction_perf_workspace_evidence import assess_workspace_probe

        workspace = assess_workspace_probe(
            json.loads(args.workspace_probe.read_text()), audit
        )
        workspace["receipt_sha256"] = sha(args.workspace_probe)
    downstream = None
    coverage = None
    if bool(args.reference) != bool(args.anchor):
        raise ValueError("reference and matched anchor must be provided together")
    if args.reference:
        from extraction_perf_downstream import evaluate

        downstream = evaluate(
            SimpleNamespace(
                reference=args.reference, gpu=args.original, anchor=args.anchor
            )
        )
        for name, statistic in downstream["results"].items():
            for doc in statistic["documents"]:
                involved = [(doc, "own_cont", 0, "h1")]
                if name == "F-B":
                    involved.extend(
                        (doc, cell, 0, "h1")
                        for cell in ("xfam_paraphrase", "paraphrase_own")
                    )
                else:
                    involved.append(
                        (doc, "null::own_cont", int(name.rsplit("#", 1)[1]), "h1")
                    )
                for key in involved:
                    if key not in reused:
                        raise ValueError(
                            "downstream cohort is not contained in byte-matched reused roster evidence"
                        )
                    origin = reused[key]["reference"]["origin_shard"]
                    roster_hash = next(
                        f["shard_sha256"]
                        for f in audit["files"]
                        if f["origin"] == origin
                    )
                    if original_pin["reference_files"].get(origin) != roster_hash:
                        raise ValueError(
                            "downstream cluster/exclusion provenance changed"
                        )
        downstream["scope"] = (
            "pinned matched primary F-B and three null forks; replica/m-mix statistics not inferred"
        )
    if args.bank_scope_ruling:
        from anamnesis.extraction.fidelity import (
            ReplayBatch,
            verify_bank_scope_coverage,
        )

        keys = tuple(("rubric-140", *r["key"]) for r in audit["row_manifest"])
        batch = ReplayBatch(
            np.stack(candidate_vectors),
            keys,
            tuple(names),
            tuple(m["lane_id"] for m in candidate_metadata),
            tuple(r["input_sha256"] for r in audit["row_manifest"]),
            tuple(m["replay_id"] for m in candidate_metadata),
            hashlib.sha256(
                json.dumps(original_pin["source"], sort_keys=True).encode()
            ).hexdigest(),
            original_pin["calibration_sha256"],
            audit["feature_schema_sha256"],
        )
        regimes = {
            "E": [k for k in keys if k[2] == "rung_control"],
            **{
                f"null-fork-{i}": [
                    k for k in keys if k[2] == "null::own_cont" and k[3] == i
                ]
                for i in range(3)
            },
        }
        for label, name in (("near-floor-B-E", "B-E"), ("near-floor-F-E", "F-E")):
            regimes[label] = sorted(
                {
                    tuple(term[side])
                    for entry in condition_manifest["rows"]
                    for term in entry["contrasts"]
                    if term["name"] == name
                    for side in ("left", "right")
                }
            )
        coverage = verify_bank_scope_coverage(
            batch,
            keys,
            regimes,
            {
                k: (r["start"], r["end"])
                for k, r in zip(keys, audit["row_manifest"], strict=True)
            },
            sha(args.bank_scope_ruling),
        )
    pending = []
    if downstream is None:
        pending.append("matched downstream statistics")
    if coverage is None:
        pending.append("bank-regime coverage composition")
    if workspace is None or not workspace["owner_probe_pass"]:
        pending.append("owner workspace prerequisite")
    result = dict(
        rows=rows,
        n_rows=len(rows),
        G0=all(r["G0"] for r in rows),
        G1=g1,
        G2=downstream["all_numeric_bounds_pass"] if downstream else None,
        G3=coverage["G3"] if coverage else None,
        G4=True,
        G5=max(peaks) <= cap,
        all_path_bounds_proven=all(r["path_bound_proven"] for r in rows),
        all_feature_bounds_pass=all(r["feature_bound_pass"] for r in rows),
        process_peaks_gib=peaks,
        lane_id=next(iter(lane_ids)),
        roster_audit_sha256=sha(args.audit),
        reuse_plan_sha256=sha(args.reuse),
        condition_manifest_sha256=sha(args.conditions),
        declared_contrast_evidence=contrast_receipts,
        workspace_evidence=workspace,
        batch_schedules=batch_schedules,
        downstream=downstream,
        coverage=coverage,
        certification=False,
        pending=pending,
    )
    result["certification"] = not pending and all(
        result[g] is True for g in ("G0", "G1", "G2", "G3", "G4", "G5")
    )
    result["certification_scope"] = (
        (
            "frozen single-row gpu-v3"
            if args.candidate_kind == "single"
            else "frozen exact-shape batch2 with independent prefill"
        )
        + "; owner roster3467; cache263–299/continuation128; current n383 primary/null statistics; no other batch-layout/adapter/native-model scope transfer"
    )
    write_json(args.out, result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("rows", "declared_contrast_evidence", "downstream")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
