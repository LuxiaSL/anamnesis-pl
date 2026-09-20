"""Join complete immutable banks to the frozen fidelity harness.

Missing rows/rulings are reported as incomplete, never intersected away or
assigned a permissive default. An approved condition map names both endpoints
for EVERY reference row, including anchor A.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from extraction_perf_phase0 import rows_from_bank, sha, write_json
from anamnesis.extraction.path_floor import coordinate_lower_bound
import anamnesis.extraction.path_floor as path_floor_module
from anamnesis.extraction.fidelity import (
    ReplayBatch,
    Ruler,
    DownstreamAnchor,
    verify_vectors,
    verify_downstream,
    verify_coverage,
    verify_sweep_coverage,
    verify_bank_scope_coverage,
    verify_memory,
    verify_within_tol,
)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def row_files(folders):
    found = {}
    for folder in folders:
        for path in sorted(folder.glob("row-*.json")):
            record = json.loads(path.read_text())
            key = tuple(record["key"])
            if key in found:
                raise ValueError(f"duplicate receipt row {key}")
            found[key] = (record, path.with_suffix(".npz"))
    return found


def validate_floor_origin(adaptive, cpu, candidate, origin, origin_sha, helper_sha):
    """A CPU-path lower bound is candidate-independent, but its lineage is not."""
    if (
        adaptive["source"] != cpu["anamnesis"]
        or adaptive["caller"] != cpu["caller"]
        or adaptive["gpu_deployment_sha256"] != origin_sha
        or adaptive["helper_sha256"] != helper_sha
    ):
        raise ValueError("adaptive reference/helper/origin provenance mismatch")
    for field in (
        "calibration_sha256",
        "standardizer_sha256",
        "reference_files",
        "caller",
    ):
        if candidate[field] != origin[field]:
            raise ValueError(f"floor reuse reference mismatch: {field}")
    if adaptive["standardizer_sha256"] != candidate["standardizer_sha256"]:
        raise ValueError("adaptive standardizer mismatch")


def validate_batch_schedule(folder, pin, rows, receipts):
    from extraction_perf_batch_plan import exact_batches
    from anamnesis.extraction.batch_layout import ReplaySpan, pack_spans

    path = folder / "batch-plan.json"
    if sha(path) != pin["batch_plan_sha256"]:
        raise ValueError("batch schedule changed")
    plans = json.loads(path.read_text())
    if pin["pairing_policy"] != "exact" or plans != exact_batches(rows):
        raise ValueError(
            "batch schedule is not the declared complete exact-shape policy"
        )
    invocations = set()
    for batch_index, plan in enumerate(plans):
        spans = tuple(
            ReplaySpan(tuple(rows[i]["tokens"]), rows[i]["start"], rows[i]["end"])
            for i in plan["indices"]
        )
        vocab = max(max(s.tokens[: s.end]) for s in spans) + 1
        composition = pack_spans(spans, vocab).composition_sha256
        candidate_ids = set()
        repeat_ids = set()
        timings = set()
        for slot, index in enumerate(plan["emitted_indices"]):
            record, _ = receipts[tuple(rows[index]["key"])]
            if (
                record["batch_index"] != batch_index
                or record["emitted_rows_in_batch"] != len(plan["emitted_indices"])
                or record["filler_rows"] != plan["filler_count"]
                or not record["filler_identity_exact"]
            ):
                raise ValueError("batch membership/filler accounting mismatch")
            for key, ids in (
                ("telemetry", candidate_ids),
                ("repeat_telemetry", repeat_ids),
            ):
                meta = record[key]
                if (
                    meta["batch_size"] != 2
                    or meta["batch_row"] != slot
                    or meta["batch_composition_sha256"] != composition
                    or meta.get("prefill_policy", "batched")
                    != pin.get("prefill_policy", "batched")
                ):
                    raise ValueError("batch composition mismatch")
                ids.add(meta["batch_invocation_id"])
            if record["seconds_per_row"] != record["batch_seconds"] / len(
                plan["emitted_indices"]
            ):
                raise ValueError("singleton overhead hidden from timing")
            timings.add((record["batch_seconds"], record["batch_repeat_seconds"]))
        if (
            len(candidate_ids) != 1
            or len(repeat_ids) != 1
            or len(timings) != 1
            or candidate_ids & repeat_ids
        ):
            raise ValueError("inconsistent/reused batch invocation")
        current = candidate_ids | repeat_ids
        if current & invocations:
            raise ValueError("batch invocation reused across groups")
        invocations.update(current)
    return dict(
        batches=len(plans),
        fillers=sum(p["filler_count"] for p in plans),
        plan_sha256=sha(path),
        all_reference_rows_emitted_once=True,
    )


def main():
    import palimpsest.genpair as genpair

    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--gpu", type=Path, required=True)
    ap.add_argument("--floors", type=Path, nargs="+", required=True)
    ap.add_argument("--adaptive", type=Path)
    ap.add_argument(
        "--floor-candidate-origin",
        type=Path,
        help="Original candidate deployment pinned by adaptive proofs; never substitute its errors for the current candidate",
    )
    ap.add_argument("--anchor", type=Path, required=True)
    ap.add_argument("--memory-reference", type=Path, required=True)
    ap.add_argument("--condition-map", type=Path)
    ap.add_argument("--identical-input-zero-rule", type=Path)
    ap.add_argument("--coverage", type=Path)
    ap.add_argument("--sweep-cpu", type=Path)
    ap.add_argument("--sweep-gpu", type=Path)
    ap.add_argument("--bank-scope-ruling", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows, names = rows_from_bank(args.reference)
    keys = [tuple(r["key"]) for r in rows]
    gpu = row_files([args.gpu])
    floors = row_files(args.floors)
    adaptive = row_files([args.adaptive]) if args.adaptive else {}
    path_rows = dict(floors)
    path_rows.update(adaptive)
    missing = {
        label: [list(k) for k in keys if k not in found]
        for label, found in (("gpu", gpu), ("floor", path_rows))
    }
    extra = {
        label: [list(k) for k in found if k not in set(keys)]
        for label, found in (("gpu", gpu), ("floor", path_rows))
    }
    if any(extra.values()):
        raise ValueError(f"unexpected receipt rows: {extra}")
    known_failures = [
        dict(key=list(k), kind="candidate", record=r)
        for k, (r, _) in gpu.items()
        if r.get("G0") is False or r.get("fidelity_status") == "failed"
    ]
    known_failures.extend(
        dict(key=list(k), kind="reference", record=r)
        for k, (r, _) in floors.items()
        if r.get("bank_max_abs", 0) != 0 or r.get("cpu_repeat_bit_exact") is False
    )
    known_failures.extend(
        dict(key=list(k), kind="path-proof", record=r)
        for k, (r, _) in adaptive.items()
        if r.get("path_gate_proven") is False
    )
    if known_failures:
        write_json(
            args.out,
            dict(
                passed=False,
                status="failed",
                known_failures=known_failures,
                missing=missing,
            ),
        )
        print("FAILED", len(known_failures), "known row failures; no certification")
        return
    if any(missing.values()):
        write_json(
            args.out,
            dict(
                passed=False,
                status="incomplete",
                expected=len(keys),
                gpu_rows=len(gpu),
                floor_rows=len(floors),
                missing=missing,
            ),
        )
        print("INCOMPLETE", len(gpu), len(floors), "of", len(keys))
        return
    if not (args.gpu / "complete.json").exists() or not all(
        (p / "complete.json").exists() or (p / "stopped.json").exists()
        for p in args.floors
    ):
        raise ValueError(
            "complete-row files exist but end-of-run source verification is missing"
        )
    for folder in args.floors:
        if (folder / "stopped.json").exists():
            sealed = json.loads((folder / "stopped.json").read_text())
            for name, value in sealed["files"].items():
                if sha(folder / name) != value:
                    raise ValueError("sealed full-floor receipt changed")
    adaptive_pin = None
    if args.adaptive:
        if (
            not (args.adaptive / "complete.json").exists()
            or not (args.adaptive / "validation-pass.json").exists()
        ):
            raise ValueError("adaptive validation/completion evidence missing")
        adaptive_pin = json.loads((args.adaptive / "deployment.json").read_text())
        validation = json.loads((args.adaptive / "validation-pass.json").read_text())
        if not validation["bit_exact"] or validation["n"] != len(floors):
            raise ValueError(
                "adaptive proof was not validated against all full-floor rows"
            )
    gpu_pin = json.loads((args.gpu / "deployment.json").read_text())
    batch_schedule = (
        validate_batch_schedule(args.gpu, gpu_pin, rows, gpu)
        if gpu_pin.get("full_bank")
        else None
    )
    origin_path = (args.floor_candidate_origin or args.gpu) / "deployment.json"
    origin_pin = json.loads(origin_path.read_text())
    cpu_pins = [json.loads((p / "deployment.json").read_text()) for p in args.floors]
    if adaptive_pin is not None:
        validate_floor_origin(
            adaptive_pin,
            cpu_pins[0],
            gpu_pin,
            origin_pin,
            sha(origin_path),
            sha(Path(path_floor_module.__file__)),
        )
        if set(tuple(k) for k in adaptive_pin["row_keys"]) != set(keys):
            raise ValueError("adaptive reference population mismatch")
    for pin in cpu_pins[1:]:
        for field in (
            "anamnesis",
            "caller",
            "files",
            "torch",
            "numpy",
            "transformers",
            "extraction",
            "families",
        ):
            if pin[field] != cpu_pins[0][field]:
                raise ValueError(f"CPU floor partitions differ in {field}")
    if sha(Path(genpair.__file__)) != gpu_pin["caller"]["genpair.py"]:
        raise ValueError("downstream estimator is not the frozen caller version")
    sigma_path = args.reference / "standardization_floor_p05.npz"
    if sha(sigma_path) != gpu_pin["standardizer_sha256"]:
        raise ValueError("standardizer differs from capture")
    with np.load(sigma_path) as z:
        sigma = z["sigma_cal"].astype(float)
        weights = z["weights"].astype(float)
        if z["feature_names"].astype(str).tolist() != names:
            raise ValueError("ruler schema mismatch")
    cpu = []
    candidate = []
    repeat = []
    metadata = []
    repeats = []
    floor_values = {}
    floor_kinds = {}
    floor_proofs = {}
    for row in rows:
        key = tuple(row["key"])
        grecord, gpath = gpu[key]
        if key in floors:
            frecord, fpath = floors[key]
            with np.load(fpath, allow_pickle=False) as f:
                if f["cpu"].tobytes() != row["bank"].tobytes():
                    raise ValueError(f"reference bytes differ for {key}")
                measured = float(
                    np.linalg.norm(
                        (f["incremental"].astype(float) - f["cpu"]) * weights / sigma
                    )
                )
                if not np.isclose(measured, frecord["floor_b"], rtol=1e-12, atol=1e-12):
                    raise ValueError(f"floor receipt does not reproduce for {key}")
                floor_values[key] = min(measured, frecord["floor_b"])
                floor_kinds[key] = "measured"
                if key in adaptive:
                    _, apath = adaptive[key]
                    with np.load(apath, allow_pickle=False) as a:
                        if (
                            a["incremental_coordinates"].tobytes()
                            != f["incremental"][a["coordinate_indices"]].tobytes()
                        ):
                            raise ValueError(
                                "adaptive/full coordinate equality does not reproduce"
                            )
        else:
            arecord, apath = adaptive[key]
            with np.load(apath, allow_pickle=False) as a:
                idx = a["coordinate_indices"]
                expected = [
                    names.index(f"activation_norm_traj0_L{layer}")
                    for layer in range(80)
                ]
                if (
                    idx.tolist() != expected
                    or a["reference_coordinates"].tobytes()
                    != row["bank"][idx].tobytes()
                ):
                    raise ValueError("coordinate proof schema/reference mismatch")
                if arecord["proof_kind"] == "full_floor_fallback":
                    floor_values[key] = float(
                        np.linalg.norm(
                            (a["incremental_full"].astype(float) - row["bank"])
                            * weights
                            / sigma
                        )
                    )
                    floor_kinds[key] = "measured"
                else:
                    bound = coordinate_lower_bound(
                        a["incremental_coordinates"],
                        a["reference_coordinates"],
                        sigma[idx],
                        weights[idx],
                    )
                    if bound["floor_b_lower_bound"] != arecord["floor_b_lower_bound"]:
                        raise ValueError("lower-bound certificate does not reproduce")
                    floor_values[key] = bound["floor_b_lower_bound"]
                    floor_kinds[key] = "lower_bound"
                    floor_proofs[key] = digest(
                        {
                            "record": sha(apath.with_suffix(".json")),
                            "arrays": sha(apath),
                            "deployment": sha(args.adaptive / "deployment.json"),
                        }
                    )
        with np.load(gpath, allow_pickle=False) as g:
            if g["bank"].tobytes() != row["bank"].tobytes():
                raise ValueError(f"reference bytes differ for {key}")
            cpu.append(row["bank"].copy())
            candidate.append(g["gpu"].copy())
            repeat.append(g["repeat"].copy())
        metadata.append(grecord["telemetry"])
        repeats.append(grecord["repeat_telemetry"])
    cpu, candidate, repeat = [np.stack(x) for x in (cpu, candidate, repeat)]
    namespace = "rubric-140"
    full_keys = tuple((namespace, *k) for k in keys)
    schema = hashlib.sha256(json.dumps(names).encode()).hexdigest()
    inputs = tuple(
        hashlib.sha256(
            np.asarray(r["tokens"][: r["end"]], dtype="<i8").tobytes()
        ).hexdigest()
        for r in rows
    )
    if any(
        m["input_tokens_sha256"] != h or r["input_tokens_sha256"] != h
        for m, r, h in zip(metadata, repeats, inputs, strict=True)
    ):
        raise ValueError("captured token inputs do not match frozen bank")
    calib = gpu_pin["calibration_sha256"]
    if any(
        m["calibration_sha256"] != calib or m["feature_schema_sha256"] != schema
        for m in [*metadata, *repeats]
    ):
        raise ValueError("per-row calibration/schema receipt mismatch")
    cpub = ReplayBatch(
        cpu,
        full_keys,
        tuple(names),
        ("cpu-reference",) * len(keys),
        inputs,
        tuple(f"cpu/{k}" for k in keys),
        digest(cpu_pins[0]["anamnesis"]),
        calib,
        schema,
    )
    gpub = ReplayBatch(
        candidate,
        full_keys,
        tuple(names),
        tuple(m["lane_id"] for m in metadata),
        inputs,
        tuple(m["replay_id"] for m in metadata),
        digest(gpu_pin["source"]),
        calib,
        schema,
        fresh_cache=all(m["fresh_cache"] for m in metadata),
    )
    repeatb = ReplayBatch(
        repeat,
        full_keys,
        tuple(names),
        tuple(m["lane_id"] for m in repeats),
        inputs,
        tuple(m["replay_id"] for m in repeats),
        digest(gpu_pin["source"]),
        calib,
        schema,
        fresh_cache=all(m["fresh_cache"] for m in repeats),
    )
    for batch in (cpub, gpub, repeatb):
        batch.validate()
    if set(gpub.replay_ids) & set(repeatb.replay_ids):
        raise ValueError("same replay invocation reused")
    if gpub.lane_ids[0] != repeatb.lane_ids[0]:
        raise ValueError("repeat lane differs")
    exact = candidate.tobytes() == repeat.tobytes()
    vectors = dict(G0=exact, G1=None, G4=True)
    if not exact:
        write_json(
            args.out,
            dict(passed=False, status="G0 failed; G1 not evaluated", vectors=vectors),
        )
        return
    if args.condition_map is not None:
        manifest = json.loads(args.condition_map.read_text())
        entries = manifest["rows"]
        contrast_receipts = None
        if manifest.get("format") == "declared-contrasts-v2":
            from extraction_perf_contrast_rules import resolve_entries

            pairs, distances, null_reasons, contrast_receipts = resolve_entries(
                entries,
                full_keys,
                cpu,
                candidate,
                repeat,
                sigma,
                weights,
                dict(zip(full_keys, inputs, strict=True)),
                {
                    k: (r["start"], r["end"])
                    for k, r in zip(full_keys, rows, strict=True)
                },
            )
            zero_pairs = set()
        else:
            pairs = {
                tuple(e["key"]): None
                if e.get("contrast_status") == "null-by-rule"
                else (tuple(e["left"]), tuple(e["right"]))
                for e in entries
            }
            null_reasons = {
                tuple(e["key"]): e["reason"]
                for e in entries
                if e.get("contrast_status") == "null-by-rule"
            }
            if any(k[2] != "own_cont" for k in null_reasons):
                raise ValueError(
                    "this consumer ruling permits null-by-rule only for anchor A"
                )
            if len(pairs) != len(entries):
                raise ValueError("duplicate condition mapping")
            lookup = {k: i for i, k in enumerate(full_keys)}
            distances = {
                k: None
                if pair is None
                else float(
                    np.linalg.norm(
                        (cpu[lookup[pair[0]]].astype(float) - cpu[lookup[pair[1]]])
                        * weights
                        / sigma
                    )
                )
                for k, pair in pairs.items()
            }
            zero_pairs = set()
            if args.identical_input_zero_rule:
                for k, pair in pairs.items():
                    if pair is None or distances[k] != 0.0:
                        continue
                    left, right = [rows[lookup[p]] for p in pair]
                    if (
                        left["start"] == right["start"]
                        and left["end"] == right["end"]
                        and left["tokens"][: left["end"]]
                        == right["tokens"][: right["end"]]
                    ):
                        zero_pairs.add(k)
        ruler = Ruler(
            sigma,
            weights,
            sha(sigma_path),
            tuple(names),
            {(namespace, *k): v for k, v in floor_values.items()},
            distances,
            pairs,
            {k: r["start"] for k, r in zip(full_keys, rows, strict=True)},
            full_keys,
            calib,
            schema,
            digest(gpu_pin["reference_files"]),
            null_contrast_reasons=null_reasons,
            floor_kinds={(namespace, *k): v for k, v in floor_kinds.items()},
            floor_proof_sha256={(namespace, *k): v for k, v in floor_proofs.items()},
            identical_input_zero_pairs=frozenset(zero_pairs),
            condition_manifest_sha256=sha(args.condition_map),
        )
        vectors = verify_vectors(cpub, gpub, repeatb, ruler)
        vectors["declared_contrast_evidence"] = contrast_receipts
    else:
        vectors["reason"] = "approved per-row condition manifest missing"
    by_key = {k: i for i, k in enumerate(keys)}
    docs = []
    clusters = []
    dc = []
    dg = []
    scale = weights / sigma
    for doc in sorted({k[0] for k in keys}):
        triplet = [
            (doc, c, 0, "h1") for c in ("own_cont", "xfam_paraphrase", "paraphrase_own")
        ]
        if not all(k in by_key and not rows[by_key[k]]["excluded"] for k in triplet):
            continue
        ia, iff, ib = [by_key[k] for k in triplet]
        for values, deltas in ((cpu, dc), (candidate, dg)):
            a, f, b = [values[i].astype(float) * scale for i in (ia, iff, ib)]
            deltas.append((f - a) - (b - a))
        docs.append(doc)
        clusters.append(rows[ia]["cluster"])
    anchor_doc = json.loads(args.anchor.read_text())
    ref = anchor_doc["fb_contrast"]
    if ref["n_documents"] != len(docs):
        raise ValueError("anchor population differs; matched-n required")
    stat_sha = sha(Path(genpair.__file__))
    anchor = DownstreamAnchor(
        ref["ratio"],
        ref["se_corrected"],
        tuple(docs),
        tuple(clusters),
        sha(args.anchor),
        stat_sha,
    )
    downstream = verify_downstream(
        np.stack(dc),
        np.stack(dg),
        docs,
        clusters,
        anchor,
        genpair.jackknife_coherence_ratio,
        stat_sha,
    )
    from extraction_perf_downstream import evaluate

    all_downstream = evaluate(args)
    if all_downstream["results"]["F-B"]["G2"] != downstream["G2"]:
        raise ValueError("downstream implementations disagree")
    downstream = dict(
        G2=all_downstream["all_numeric_bounds_pass"],
        separate_forks=True,
        results=all_downstream["results"],
        anchor_sha256=all_downstream["anchor_sha256"],
    )
    if args.sweep_cpu or args.sweep_gpu or args.bank_scope_ruling:
        if bool(args.sweep_cpu) != bool(args.sweep_gpu):
            raise ValueError("both sweep lanes required")
        from extraction_perf_verify_sweep import verify as verify_sweep

        sweep = (
            verify_sweep(args.sweep_cpu, args.sweep_gpu, sigma_path)
            if args.sweep_cpu
            else None
        )
        regimes = {
            "E": [k for k in full_keys if k[2] == "rung_control"],
            **{
                f"null-fork-{i}": [
                    k for k in full_keys if k[2] == "null::own_cont" and k[3] == i
                ]
                for i in range(3)
            },
        }
        # Include every B-E/F-E pair, hence also the weakest/near-floor cells;
        # never select a favorable subset after observing GPU error.
        for name, cell in (
            ("near-floor-B-E", "paraphrase_own"),
            ("near-floor-F-E", "xfam_paraphrase"),
        ):
            endpoints = set()
            for key in full_keys:
                if key[2] == cell:
                    control = (namespace, key[1], "rung_control", 0, key[4])
                    if control not in full_keys:
                        raise ValueError("bank coverage lacks matched E endpoint")
                    endpoints.update((key, control))
            regimes[name] = sorted(endpoints)
        if args.bank_scope_ruling:
            coverage = verify_bank_scope_coverage(
                gpub,
                full_keys,
                regimes,
                {
                    key: (row["start"], row["end"])
                    for key, row in zip(full_keys, rows, strict=True)
                },
                sha(args.bank_scope_ruling),
            )
            if sweep is not None:
                coverage["wider_sweep_evidence"] = sweep
        else:
            coverage = verify_sweep_coverage(gpub, full_keys, regimes, sweep)
            coverage["sweep_evidence"] = sweep
    elif args.coverage:
        regimes = {
            name: [tuple(k) for k in value]
            for name, value in json.loads(args.coverage.read_text()).items()
        }
        coverage = verify_coverage(gpub, full_keys, regimes)
    else:
        coverage = dict(G3=None, reason="approved coverage manifest missing")
    memory = json.loads((args.gpu / "process-vram.json").read_text())
    if memory["errors"] or any(
        not v["samples"] for k, v in memory["phases"].items() if k != "setup"
    ):
        raise ValueError("missing/failed NVML evidence")
    measured = max(
        v["peak_bytes"]
        for v in memory["phases"].values()
        if v["peak_bytes"] is not None
    )
    historical = json.loads(args.memory_reference.read_text())
    cap = round(historical["cap_gib"] * 2**30)
    resource = verify_memory(
        {"all_phases": [measured]},
        {"all_phases": [cap]},
        cap_bytes=cap,
        metric="process",
        baseline_sha256=sha(args.memory_reference),
    )
    result = verify_within_tol(vectors, downstream, coverage, resource)
    result["n_rows"] = len(keys)
    result["batch_schedule"] = batch_schedule
    result["floor_certificate_origin_sha256"] = sha(origin_path)
    result["condition_manifest_sha256"] = (
        sha(args.condition_map) if args.condition_map else None
    )
    result["zero_rule_sha256"] = (
        sha(args.identical_input_zero_rule) if args.identical_input_zero_rule else None
    )
    result["n_documents"] = len(docs)
    result["certification_scope"] = "probe-free only; no adapter/excess coverage"
    write_json(args.out, result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("vectors", "coverage", "downstream", "memory")
            }
        )
    )


if __name__ == "__main__":
    main()
