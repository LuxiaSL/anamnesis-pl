"""Capture the exact missing owner-roster inputs on frozen arithmetic lanes.

CPU path certificates and GPU repeats run in separate processes. This records
evidence, not final cohort/statistic certification. Insufficient lower bounds
remain unresolved, never mislabeled as a failed measured full floor.
"""

from __future__ import annotations
import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from extraction_perf_phase0 import sha, source_pin, stamp, write_json
from extraction_perf_roster import roster_rows
from extraction_perf_memory import ProcessMemorySampler


def select_supplement(rows, plan):
    required = {tuple(r["key"]): r for r in plan["required_measurements"]}
    selected = [r for r in rows if tuple(r["key"]) in required]
    if len(selected) != plan["measurement_rows"] or len(required) != len(selected):
        raise ValueError("incomplete supplement population")
    for row in selected:
        expected = required[tuple(row["key"])]
        if (
            hashlib.sha256(row["bank"].tobytes()).hexdigest()
            != expected["bank_vector_sha256"]
            or hashlib.sha256(
                np.asarray(row["tokens"][: row["end"]], dtype="<i8").tobytes()
            ).hexdigest()
            != expected["input_sha256"]
            or row["start"] != expected["start"]
            or row["end"] != expected["end"]
        ):
            raise ValueError("supplement input/reference changed")
    return selected


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--mode", choices=("cpu-floor", "gpu"), required=True)
    for name in (
        "roster",
        "reuse",
        "audit",
        "origin",
        "cpu-pin",
        "proof-origin",
        "out",
    ):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--floors", type=Path)
    args = p.parse_args()
    if args.mode == "cpu-floor":
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    import torch
    import anamnesis
    import palimpsest.sigbridge as bridge
    from anamnesis.config import ModelConfig
    from anamnesis.extraction.model_loader import load_model
    from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data
    from first_step_floor import first_incremental_coordinates, coordinate_lower_bound

    args.out.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(args.mode == "gpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cpu_pin = json.loads(args.cpu_pin.read_text())
    origin = json.loads((args.origin / "deployment.json").read_text())
    plan = json.loads(args.reuse.read_text())
    audit = json.loads(args.audit.read_text())
    if (
        sha(args.audit) != plan["roster_audit_sha256"]
        or sha(args.roster) != audit["roster_sha256"]
    ):
        raise ValueError("roster/reuse plan identity mismatch")
    if sha(args.origin / "deployment.json") != plan["candidate_deployment_sha256"]:
        raise ValueError("wrong candidate origin")
    rows, names, provenance = roster_rows(
        args.roster,
        {
            str(Path(i["path"]).parents[1]): Path(i["path"]).parents[1]
            for i in json.loads(args.roster.read_text())["files"]
        },
    )
    selected = select_supplement(rows, plan)
    source = source_pin(Path(anamnesis.__file__).parent)
    expected_source = (
        cpu_pin["anamnesis"] if args.mode == "cpu-floor" else origin["source"]
    )
    if (
        source != expected_source
        or source_pin(Path(bridge.__file__).parent) != cpu_pin["caller"]
    ):
        raise ValueError("frozen implementation/caller mismatch")
    for filename, digest in cpu_pin["files"].items():
        if sha(Path(filename)) != digest:
            raise ValueError("calibration/ruler changed")
    proof = json.loads((args.proof_origin / "deployment.json").read_text())
    validation = json.loads((args.proof_origin / "validation-pass.json").read_text())
    helper_sha = sha(Path(__file__).with_name("first_step_floor.py"))
    if (
        not validation["bit_exact"]
        or validation["n"] != 945
        or proof["helper_sha256"] != helper_sha
    ):
        raise ValueError("validated first-step helper identity missing")
    sigma_path = Path(cpu_pin["arguments"]["sigma"])
    with np.load(sigma_path) as z:
        sigma, weights = z["sigma_cal"].astype(float), z["weights"].astype(float)
        if z["feature_names"].astype(str).tolist() != names:
            raise ValueError("ruler schema changed")
    indices = np.array([names.index(f"activation_norm_traj0_L{i}") for i in range(80)])
    if args.mode == "gpu":
        if args.floors is None or not (args.floors / "complete.json").exists():
            raise ValueError("completed CPU certificate capture required")
        floor_pin = json.loads((args.floors / "deployment.json").read_text())
        if floor_pin["selected"] != [r["key"] for r in selected] or floor_pin[
            "reuse_sha256"
        ] != sha(args.reuse):
            raise ValueError("CPU/GPU supplement populations differ")
    pin = dict(
        utc=stamp(),
        mode=args.mode,
        source=source,
        caller=cpu_pin["caller"],
        roster_sha256=sha(args.roster),
        reuse_sha256=sha(args.reuse),
        audit_sha256=sha(args.audit),
        origin_sha256=sha(args.origin / "deployment.json"),
        cpu_pin_sha256=sha(args.cpu_pin),
        helper_sha256=helper_sha,
        proof_origin_sha256=sha(args.proof_origin / "deployment.json"),
        runner_sha256=sha(Path(__file__)),
        roster_files=provenance,
        standardizer_sha256=sha(sigma_path),
        calibration_sha256=origin["calibration_sha256"],
        selected=[r["key"] for r in selected],
        workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        deterministic=args.mode == "gpu",
        certification=False,
    )
    write_json(args.out / "deployment.json", pin)
    preset = bridge.resolve_preset("70b-base")
    ec, fc = bridge.battery_configs(preset)
    calibration = bridge.CalibrationBundle.load(
        cpu_pin["arguments"]["calibration"], expect_model_path="/models/Llama-3.1-70B"
    )
    cfg = ModelConfig(
        model_id="/models/Llama-3.1-70B",
        torch_dtype=preset.torch_dtype,
        num_layers=preset.num_layers,
        hidden_dim=preset.hidden_dim,
        num_attention_heads=preset.num_attention_heads,
        num_kv_heads=preset.num_kv_heads,
        head_dim=preset.head_dim,
        device_map="cuda:0",
        attn_implementation="eager",
    )
    memory = ProcessMemorySampler()
    atexit.register(lambda: write_json(args.out / "process-vram.json", memory.close()))
    layers = list(range(80))
    with memory.phase("model_load"):
        loaded = load_model(
            cfg,
            sampled_layers=preset.sampled_layers,
            register_gate_hooks=True,
            key_layers=layers,
            value_layers=layers,
            query_layers=layers,
            attn_output_layers=layers,
        )
    lane = None
    if args.mode == "gpu":
        from anamnesis.extraction.gpu_features import GpuFeatureLane

        lane = GpuFeatureLane(
            ec,
            fc,
            names,
            calibration.positional_means,
            calibration.pca_components,
            calibration.pca_mean,
            device="cuda:0",
            calibration_sha256=origin["calibration_sha256"],
        )
        if lane.identity != json.loads((args.origin / "lane.json").read_text()):
            raise ValueError("supplement GPU lane differs from original bank")
        write_json(args.out / "lane.json", lane.identity)
    for i, row in enumerate(selected):
        torch.cuda.synchronize()
        start = time.perf_counter()
        record = dict(utc=stamp(), key=row["key"], origin_shard=row["origin_shard"])
        if lane is None:
            # Positive controls on the new bank, not a refit of its reference.
            if i < 4:
                raw = bridge.replay_span(
                    loaded,
                    row["tokens"],
                    bridge.SpanRef(
                        span_id="h1",
                        start=row["start"],
                        length=row["end"] - row["start"],
                    ),
                    positional_means=calibration.positional_means,
                )
                reference = compute_features_v2_from_data(
                    raw, ec, fc, calibration.pca_components, calibration.pca_mean
                )
                del raw
                np.savez(
                    args.out / f"cpu-control-{i:05d}.npz",
                    cpu=reference.features,
                    bank=row["bank"],
                )
                if reference.features.tobytes() != row["bank"].tobytes():
                    write_json(args.out / "cpu-reference-failure.json", record)
                    raise RuntimeError(
                        "literal CPU arithmetic does not reproduce supplemental bank"
                    )
            values, coordinate_names = first_incremental_coordinates(
                loaded,
                row["tokens"],
                row["start"],
                row["end"],
                calibration.positional_means,
                ec,
            )
            if coordinate_names != [names[j] for j in indices]:
                raise ValueError("coordinate schema changed")
            bound = coordinate_lower_bound(
                values, row["bank"][indices], sigma[indices], weights[indices]
            )
            np.savez(
                args.out / f"row-{i:05d}.npz",
                incremental_coordinates=values,
                reference_coordinates=row["bank"][indices],
                coordinate_indices=indices,
            )
            record.update(bound, proof_kind="lower_bound", helper_sha256=helper_sha)
        else:
            with memory.phase("candidate"):
                first = lane.replay_span(
                    loaded, row["tokens"], row["start"], row["end"]
                )
            with memory.phase("repeat"):
                second = lane.replay_span(
                    loaded, row["tokens"], row["start"], row["end"]
                )
            np.savez(
                args.out / f"row-{i:05d}.npz",
                gpu=first.features,
                repeat=second.features,
                bank=row["bank"],
            )
            exact = first.features.tobytes() == second.features.tobytes()
            if not exact:
                write_json(
                    args.out / f"row-{i:05d}.json", dict(record, G0=False, G1=None)
                )
                raise RuntimeError("supplement G0 failure")
            error = (first.features.astype(float) - row["bank"]) / sigma
            distance = float(np.linalg.norm(error * weights))
            floor_record = json.loads((args.floors / f"row-{i:05d}.json").read_text())
            if floor_record["key"] != row["key"]:
                raise ValueError("floor row mismatch")
            mem = memory.snapshot()
            if mem["errors"] or any(
                not mem["phases"].get(p, {}).get("samples")
                for p in ("candidate", "repeat")
            ):
                raise RuntimeError("missing process memory")
            peak = (
                max(
                    p["peak_bytes"]
                    for p in mem["phases"].values()
                    if p["peak_bytes"] is not None
                )
                / 2**30
            )
            record.update(
                G0=True,
                G1=None,
                floor_c=distance,
                max_feature_sigma_error=float(np.abs(error).max()),
                path_bound_proven=distance <= floor_record["floor_b_lower_bound"],
                requires_full_floor=distance > floor_record["floor_b_lower_bound"],
                floor_b_lower_bound=floor_record["floor_b_lower_bound"],
                telemetry=first.metadata,
                repeat_telemetry=second.metadata,
                process_peak_gib=peak,
                G5=peak <= 135.29,
            )
            if np.abs(error).max() > 0.1 or peak > 135.29:
                write_json(
                    args.out / f"row-{i:05d}.json",
                    dict(record, fidelity_status="failed"),
                )
                raise RuntimeError("supplement feature/resource failure")
        torch.cuda.synchronize()
        record["capture_seconds"] = time.perf_counter() - start
        write_json(args.out / f"row-{i:05d}.json", record)
        if i % 25 == 0:
            print(
                "ROW",
                i,
                "OF",
                len(selected),
                json.dumps(
                    {
                        k: v
                        for k, v in record.items()
                        if k not in ("telemetry", "repeat_telemetry")
                    }
                ),
                flush=True,
            )
    if source_pin(Path(anamnesis.__file__).parent) != source:
        raise RuntimeError("supplement source changed")
    write_json(
        args.out / "complete.json",
        dict(utc=stamp(), rows=len(selected), certification=False),
    )


if __name__ == "__main__":
    main()
