"""Validate first-step certificates against full receipts, then cover remaining rows.

Uses the immutable original CPU deployment, not a changed GPU arithmetic stack.
The lower-bound helper is copied verbatim into this experiment's scripts folder
as first_step_floor.py; its hash is part of the receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from extraction_perf_phase0 import (
    rows_from_bank,
    sha,
    stamp,
    write_json,
    source_pin,
    incremental_raw,
)
from first_step_floor import first_incremental_coordinates, coordinate_lower_bound


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--full-floors", type=Path, nargs="+", required=True)
    ap.add_argument("--gpu", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import torch
    import anamnesis
    import palimpsest.sigbridge as bridge
    from anamnesis.config import ModelConfig
    from anamnesis.extraction.model_loader import load_model
    from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data

    if (
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") is not None
        or torch.are_deterministic_algorithms_enabled()
    ):
        raise ValueError(
            "first-step reference must use original floor-survey arithmetic settings"
        )
    if not (args.gpu / "complete.json").exists():
        raise ValueError("wait for the independently repeated GPU bank to complete")
    rows, names = rows_from_bank(args.reference)
    by_key = {tuple(r["key"]): r for r in rows}
    full = {}
    seal_hashes = {}
    for folder in args.full_floors:
        seal = json.loads((folder / "stopped.json").read_text())
        seal_hashes[str(folder)] = sha(folder / "stopped.json")
        for name, digest in seal["files"].items():
            if sha(folder / name) != digest:
                raise ValueError(f"sealed receipt changed: {folder / name}")
            if name.endswith(".json"):
                record = json.loads((folder / name).read_text())
                key = tuple(record["key"])
                if key in full:
                    raise ValueError("duplicate full-floor row")
                full[key] = (folder / name, record)
    if not full:
        raise ValueError("real-model validation requires completed full-floor receipts")
    reference_pin = json.loads((args.full_floors[0] / "deployment.json").read_text())
    if source_pin(Path(anamnesis.__file__).parent) != reference_pin["anamnesis"]:
        raise ValueError("CPU implementation changed")
    if source_pin(Path(bridge.__file__).parent) != reference_pin["caller"]:
        raise ValueError("reference caller changed")
    for path, digest in reference_pin["files"].items():
        if sha(Path(path)) != digest:
            raise ValueError(f"calibration/ruler changed: {path}")
    for path, digest in reference_pin["bank_files"].items():
        if sha(Path(path)) != digest:
            raise ValueError(f"bank changed: {path}")
    with np.load(reference_pin["arguments"]["sigma"]) as z:
        sigma = z["sigma_cal"].astype(float)
        weights = z["weights"].astype(float)
    selected_names = [f"activation_norm_traj0_L{layer}" for layer in range(80)]
    indices = np.asarray([names.index(name) for name in selected_names])
    gpu = {}
    for path in sorted(args.gpu.glob("row-*.json")):
        record = json.loads(path.read_text())
        if not record["G0"] or record["fidelity_status"] == "failed":
            raise ValueError("GPU bank has failed evaluated gates")
        key = tuple(record["key"])
        if key in gpu:
            raise ValueError("duplicate GPU row")
        gpu[key] = (path, record)
    if set(gpu) != set(by_key):
        raise ValueError("GPU/reference population mismatch")
    validation = [r for r in rows if tuple(r["key"]) in full]
    remaining = [r for r in rows if tuple(r["key"]) not in full]
    order = validation + remaining
    pin = dict(
        utc=stamp(),
        helper_sha256=sha(Path(__file__).with_name("first_step_floor.py")),
        runner_sha256=sha(Path(__file__)),
        reference_deployment_sha256=sha(args.full_floors[0] / "deployment.json"),
        source=reference_pin["anamnesis"],
        caller=reference_pin["caller"],
        sealed_floors=seal_hashes,
        gpu_deployment_sha256=sha(args.gpu / "deployment.json"),
        standardizer_sha256=sha(Path(reference_pin["arguments"]["sigma"])),
        coordinates=selected_names,
        coordinate_indices=indices.tolist(),
        n_validation=len(validation),
        n_remaining=len(remaining),
        row_keys=[r["key"] for r in order],
        method="first-position activation norms; max-coordinate lower bound on full standardized L2",
        artifact_semantics="lower-bound certificates, not full floor estimates",
    )
    write_json(args.out / "deployment.json", pin)
    print(
        "PINNED", len(validation), "validation", len(remaining), "remaining", flush=True
    )
    preset = bridge.resolve_preset("70b-base")
    extraction, families = bridge.battery_configs(preset)
    calibration = bridge.CalibrationBundle.load(
        reference_pin["arguments"]["calibration"],
        expect_model_path="/models/Llama-3.1-70B",
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
    layers = list(range(preset.num_layers))
    loaded = load_model(
        cfg,
        sampled_layers=preset.sampled_layers,
        register_gate_hooks=True,
        key_layers=layers,
        value_layers=layers,
        query_layers=layers,
        attn_output_layers=layers,
    )
    first_incremental_coordinates(
        loaded,
        order[0]["tokens"],
        order[0]["start"],
        order[0]["end"],
        calibration.positional_means,
        extraction,
    )
    fallback_count = 0
    for i, row in enumerate(order):
        key = tuple(row["key"])
        calibration.assert_covers(row["end"] - 1)
        torch.cuda.synchronize()
        started = time.perf_counter()
        first, got_names = first_incremental_coordinates(
            loaded,
            row["tokens"],
            row["start"],
            row["end"],
            calibration.positional_means,
            extraction,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if got_names != selected_names:
            raise ValueError("first-step coordinate schema changed")
        if key in full:
            path, record = full[key]
            with np.load(path.with_suffix(".npz")) as data:
                if first.tobytes() != data["incremental"][indices].tobytes():
                    np.savez(
                        args.out / "validation-failure.npz",
                        first=first,
                        expected=data["incremental"][indices],
                    )
                    write_json(
                        args.out / "validation-failure.json",
                        dict(key=row["key"], full_reference=str(path)),
                    )
                    raise RuntimeError(
                        "first-step coordinates differ from full incremental path; stop"
                    )
        gpath, grecord = gpu[key]
        with np.load(gpath.with_suffix(".npz")) as g:
            if g["bank"].tobytes() != row["bank"].tobytes():
                raise ValueError("GPU bank/reference bytes mismatch")
            floor_c = float(
                np.linalg.norm((g["gpu"].astype(float) - row["bank"]) * weights / sigma)
            )
        bound = coordinate_lower_bound(
            first, row["bank"][indices], sigma[indices], weights[indices]
        )
        if key in full and bound["floor_b_lower_bound"] > full[key][1]["floor_b"]:
            raise RuntimeError("purported lower bound exceeds full measured floor")
        result = dict(
            utc=stamp(),
            key=row["key"],
            first_step_s=elapsed,
            validated_against_full=key in full,
            floor_c=floor_c,
            **bound,
            proof_kind="lower_bound",
            path_gate_proven=floor_c <= bound["floor_b_lower_bound"],
            gpu_row_sha256=sha(gpath),
            full_reference=str(full[key][0]) if key in full else None,
        )
        arrays = dict(
            incremental_coordinates=first,
            reference_coordinates=row["bank"][indices],
            coordinate_indices=indices,
        )
        if not result["path_gate_proven"]:
            if key in full:
                exact = full[key][1]["floor_b"]
            else:
                fallback_count += 1
                t = time.perf_counter()
                raw = incremental_raw(loaded, row, calibration.positional_means)
                features = compute_features_v2_from_data(
                    raw,
                    extraction,
                    families,
                    calibration.pca_components,
                    calibration.pca_mean,
                )
                del raw
                if features.feature_names != names:
                    raise ValueError("fallback feature schema mismatch")
                exact = float(
                    np.linalg.norm(
                        (features.features.astype(float) - row["bank"])
                        * weights
                        / sigma
                    )
                )
                arrays["incremental_full"] = features.features
                result["fallback_s"] = time.perf_counter() - t
            result.update(
                proof_kind="full_floor_fallback",
                floor_b=exact,
                path_gate_proven=floor_c <= exact,
            )
        np.savez(args.out / f"row-{i:05d}.npz", **arrays)
        write_json(args.out / f"row-{i:05d}.json", result)
        print("ROW", i, json.dumps(result), flush=True)
        if not result["path_gate_proven"]:
            raise RuntimeError(
                "G1 path bound failed, including full-floor fallback; stop"
            )
        if i + 1 == len(validation):
            write_json(
                args.out / "validation-pass.json",
                dict(utc=stamp(), n=len(validation), bit_exact=True),
            )
    if source_pin(Path(anamnesis.__file__).parent) != pin["source"]:
        raise RuntimeError("reference source changed during run")
    write_json(
        args.out / "complete.json",
        dict(
            utc=stamp(),
            rows=len(order),
            n_full_validations=len(validation),
            n_full_fallbacks=fallback_count,
            all_path_inequalities_proven=True,
            certification=False,
            note="condition ceilings, feature tolerances, G2 and G3 remain separate",
        ),
    )


if __name__ == "__main__":
    main()
