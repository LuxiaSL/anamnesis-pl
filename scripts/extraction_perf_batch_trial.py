"""Bank-regime batch-size-two pilot; no inherited or automatic certification."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import time
from pathlib import Path

import numpy as np

from extraction_perf_phase0 import rows_from_bank, sha, source_pin, stamp, write_json
from extraction_perf_memory import ProcessMemorySampler
from extraction_perf_batch_plan import exact_batches


def feature_gate(distance: float, bound: float, maximum_sigma_error: float) -> bool:
    """Return a JSON-safe bool even when a NumPy scalar reaches the boundary."""
    return bool(distance <= bound and maximum_sigma_error <= 0.1)


def select_rows(rows, pairing="ragged"):
    if pairing == "exact":
        buckets = {}
        for row in rows:
            buckets.setdefault((row["start"], row["end"] - row["start"]), []).append(
                row
            )
        # Start with the failing cache297 geometry; cover other geometries
        # deterministically, without inspecting their candidate errors.
        shapes = sorted(
            (s for s, group in buckets.items() if len(group) >= 2),
            key=lambda s: (s[0] != 297, s[0] != 299, s),
        )
        selected = []
        for shape in shapes[:6]:
            group = sorted(
                buckets[shape],
                key=lambda r: (
                    r["key"] != ["s000", "paraphrase_own", 0, "h1"],
                    r["key"],
                ),
            )
            selected.extend(group[:2])
        if len(selected) != 12:
            raise ValueError("six populated shape buckets required for this control")
        return selected
    if pairing != "ragged":
        raise ValueError("unknown pairing policy")
    selected = {}
    for row in (
        min(rows, key=lambda r: r["start"]),
        max(rows, key=lambda r: r["start"]),
    ):
        selected[tuple(row["key"])] = row
    for cell in (
        "own_cont",
        "paraphrase_own",
        "xfam_paraphrase",
        "rung_control",
        "null::own_cont",
    ):
        for row in [r for r in rows if r["key"][1] == cell][:2]:
            selected[tuple(row["key"])] = row
    for row in rows:
        if len(selected) >= 12:
            break
        selected[tuple(row["key"])] = row
    if len(selected) != 12:
        raise ValueError("twelve unique pilot rows required")
    return list(selected.values())


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("bank", "calibration", "sigma", "baseline", "rulers", "ruling", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--pairing", choices=("ragged", "exact"), default="ragged")
    parser.add_argument("--full-bank", action="store_true")
    parser.add_argument("--batch-pilot", type=Path)
    args = parser.parse_args()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":16:8":
        raise ValueError("pinned workspace required")
    args.out.mkdir(parents=True, exist_ok=False)
    import torch
    import anamnesis
    import palimpsest.sigbridge as bridge
    from anamnesis.config import ModelConfig
    from anamnesis.extraction.model_loader import load_model
    from anamnesis.extraction.gpu_features import GpuFeatureLane
    from anamnesis.extraction.batch_layout import ReplaySpan

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows, names = rows_from_bank(args.bank)
    if args.full_bank and args.pairing != "exact":
        raise ValueError("full bank requires the passing exact-shape policy")
    selected = rows if args.full_bank else select_rows(rows, args.pairing)
    plans = (
        exact_batches(selected)
        if args.full_bank
        else [
            dict(indices=[i, i + 1], emitted_indices=[i, i + 1], filler_count=0)
            for i in range(0, len(selected), 2)
        ]
    )
    write_json(args.out / "batch-plan.json", plans)
    baseline = json.loads(args.baseline.read_text())
    for filename, digest in {**baseline["files"], **baseline["bank_files"]}.items():
        if sha(Path(filename)) != digest:
            raise ValueError(f"reference changed: {filename}")
    rulers = json.loads(args.rulers.read_text())
    if not all(rulers["gates"][g] for g in ("G0", "G1", "G2", "G4", "G5")):
        raise ValueError("bank prerequisite gates did not pass")
    bounds = {tuple(r["key"][1:]): r["bound"] for r in rulers["vectors"]["rows"]}
    with np.load(args.sigma) as z:
        sigma, weights = z["sigma_cal"].astype(float), z["weights"].astype(float)
        if z["feature_names"].astype(str).tolist() != names:
            raise ValueError("schema mismatch")
    import hashlib

    calhash = hashlib.sha256(
        json.dumps(baseline["files"], sort_keys=True).encode()
    ).hexdigest()
    source = source_pin(Path(anamnesis.__file__).parent)
    if args.full_bank:
        if (
            args.batch_pilot is None
            or not (args.batch_pilot / "complete.json").exists()
        ):
            raise ValueError("completed batch pilot required before full bank")
        pilot_pin = json.loads((args.batch_pilot / "deployment.json").read_text())
        if pilot_pin["source"] != source:
            raise ValueError("feature source differs from passing batch pilot")
        pilot_rows = [
            json.loads(p.read_text()) for p in args.batch_pilot.glob("row-*.json")
        ]
        if len(pilot_rows) != 12 or not all(
            r["G0"] and r["G1"] and r["G5"] for r in pilot_rows
        ):
            raise ValueError("batch pilot has missing/failed gates")
    pin = dict(
        utc=stamp(),
        source=source,
        caller=source_pin(Path(bridge.__file__).parent),
        runner_sha256=sha(Path(__file__)),
        rulers_sha256=sha(args.rulers),
        ruling_sha256=sha(args.ruling),
        calibration_sha256=calhash,
        standardizer_sha256=sha(args.sigma),
        batch_size=2,
        pairing_policy=args.pairing,
        full_bank=args.full_bank,
        batch_plan_sha256=sha(args.out / "batch-plan.json"),
        planner_sha256=sha(Path(__file__).with_name("extraction_perf_batch_plan.py")),
        batch_pilot_sha256=sha(args.batch_pilot / "deployment.json")
        if args.batch_pilot
        else None,
        reference_files=baseline["bank_files"],
        selected=[dict(key=r["key"], start=r["start"], end=r["end"]) for r in selected],
        scope="bank cache263–299; full reference validation"
        if args.full_bank
        else "bank cache263–299; batch pilot only",
        certification=False,
    )
    write_json(args.out / "deployment.json", pin)
    preset = bridge.resolve_preset("70b-base")
    extraction, families = bridge.battery_configs(preset)
    calibration = bridge.CalibrationBundle.load(
        args.calibration, expect_model_path="/models/Llama-3.1-70B"
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
    layers = list(range(preset.num_layers))
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
    lane = GpuFeatureLane(
        extraction,
        families,
        names,
        calibration.positional_means,
        calibration.pca_components,
        calibration.pca_mean,
        device="cuda:0",
        calibration_sha256=calhash,
    )
    write_json(args.out / "lane.json", lane.identity)
    if args.full_bank and lane.identity != json.loads(
        (args.batch_pilot / "lane.json").read_text()
    ):
        raise ValueError("lane identity differs from passing batch pilot")

    def timed(fn, phase):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with memory.phase(phase):
            result = fn()
            torch.cuda.synchronize()
        return result, time.perf_counter() - start

    def spans(group):
        return tuple(
            ReplaySpan(tuple(r["tokens"]), r["start"], r["end"]) for r in group
        )

    _, warm = timed(
        lambda: lane.replay_batch(
            loaded, spans([selected[i] for i in plans[0]["indices"]])
        ),
        "warmup",
    )
    write_json(args.out / "startup.json", dict(gpu_warmup_s=warm, pool_s=0))
    measured_shapes = set()
    for batch, plan in enumerate(plans):
        group = [selected[i] for i in plan["indices"]]
        n_real = len(plan["emitted_indices"])
        singles = []
        shape = (group[0]["start"], group[0]["end"] - group[0]["start"])
        measure_single = (
            not args.full_bank or shape not in measured_shapes or batch % 64 == 0
        )
        measured_shapes.add(shape)
        for row in group if measure_single else []:
            value, seconds = timed(
                lambda: lane.replay_span(
                    loaded, row["tokens"], row["start"], row["end"]
                ),
                "single",
            )
            singles.append((value, seconds))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        first, elapsed = timed(lambda: lane.replay_batch(loaded, spans(group)), "batch")
        second, repeat_s = timed(
            lambda: lane.replay_batch(loaded, spans(group)), "batch_repeat"
        )
        mem = memory.snapshot()
        write_json(args.out / "process-vram.json", mem)
        if mem["errors"] or any(
            not mem["phases"].get(p, {}).get("samples")
            for p in ("warmup", "batch", "batch_repeat")
        ):
            raise RuntimeError("missing process memory evidence")
        peak = (
            max(
                v["peak_bytes"]
                for v in mem["phases"].values()
                if v["peak_bytes"] is not None
            )
            / 2**30
        )
        failed = False
        physical_repeat_exact = all(
            a.features.tobytes() == b.features.tobytes()
            for a, b in zip(first, second, strict=True)
        )
        filler_identity_exact = n_real == 2 or (
            first[0].features.tobytes() == first[1].features.tobytes()
            and second[0].features.tobytes() == second[1].features.tobytes()
        )
        for i, row in enumerate(group[:n_real]):
            row_index = plan["emitted_indices"][i]
            error = (first[i].features.astype(float) - row["bank"]) / sigma
            exact = physical_repeat_exact and filler_identity_exact
            distance = float(np.linalg.norm(error * weights))
            bound = bounds[tuple(row["key"])]
            passed = (
                exact
                and distance <= bound
                and np.abs(error).max() <= 0.1
                and peak <= 135.29
            )
            np.savez(
                args.out / f"row-{row_index:05d}.npz",
                gpu=first[i].features,
                repeat=second[i].features,
                bank=row["bank"],
                **({"single": singles[i][0].features} if singles else {}),
            )
            record = dict(
                utc=stamp(),
                key=row["key"],
                lane=first[i].metadata["lane_id"],
                scope="batched_prefill+gpu_features+feature_D2H",
                seconds_per_row=elapsed / n_real,
                batch_index=batch,
                emitted_rows_in_batch=n_real,
                filler_rows=plan["filler_count"],
                filler_identity_exact=filler_identity_exact,
                batch_seconds=elapsed,
                batch_repeat_seconds=repeat_s,
                single_seconds=singles[i][1] if singles else None,
                floor_c=distance,
                bound=bound,
                max_feature_sigma_error=float(np.abs(error).max()),
                G0=exact,
                G1=feature_gate(distance, bound, np.abs(error).max())
                if exact
                else None,
                G5=peak <= 135.29,
                G5_historical_cap_check=peak <= 135.29,
                process_peak_gib=peak,
                allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                telemetry=first[i].metadata,
                repeat_telemetry=second[i].metadata,
                fidelity_status=(
                    "awaiting_full_gates" if args.full_bank else "pilot_only"
                )
                if passed
                else "failed",
            )
            write_json(args.out / f"row-{row_index:05d}.json", record)
            print(
                "ROW",
                row_index,
                json.dumps(
                    {
                        k: v
                        for k, v in record.items()
                        if k not in ("telemetry", "repeat_telemetry")
                    }
                ),
                flush=True,
            )
            failed |= not passed
        if failed:
            raise RuntimeError("batch pilot gate failure; stop and preserve both rows")
    if source_pin(Path(anamnesis.__file__).parent) != source:
        raise RuntimeError("source changed during pilot")
    write_json(
        args.out / "complete.json",
        dict(
            utc=stamp(),
            rows=len(selected),
            certification=False,
            note="Full bank/statistic gates remain required",
        ),
    )


if __name__ == "__main__":
    main()
