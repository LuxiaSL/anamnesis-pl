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


def feature_gate(distance: float, bound: float, maximum_sigma_error: float) -> bool:
    """Return a JSON-safe bool even when a NumPy scalar reaches the boundary."""
    return bool(distance <= bound and maximum_sigma_error <= 0.1)


def select_rows(rows):
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
    selected = select_rows(rows)
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
        selected=[dict(key=r["key"], start=r["start"], end=r["end"]) for r in selected],
        scope="bank cache263–299; batch pilot only",
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

    _, warm = timed(lambda: lane.replay_batch(loaded, spans(selected[:2])), "warmup")
    write_json(args.out / "startup.json", dict(gpu_warmup_s=warm, pool_s=0))
    for batch in range(6):
        group = selected[2 * batch : 2 * batch + 2]
        singles = []
        for row in group:
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
        for i, row in enumerate(group):
            error = (first[i].features.astype(float) - row["bank"]) / sigma
            exact = first[i].features.tobytes() == second[i].features.tobytes()
            distance = float(np.linalg.norm(error * weights))
            bound = bounds[tuple(row["key"])]
            passed = (
                exact
                and distance <= bound
                and np.abs(error).max() <= 0.1
                and peak <= 135.29
            )
            np.savez(
                args.out / f"row-{2 * batch + i:05d}.npz",
                gpu=first[i].features,
                repeat=second[i].features,
                bank=row["bank"],
                single=singles[i][0].features,
            )
            record = dict(
                utc=stamp(),
                key=row["key"],
                lane=first[i].metadata["lane_id"],
                scope="batched_prefill+gpu_features+feature_D2H",
                seconds_per_row=elapsed / 2,
                batch_seconds=elapsed,
                batch_repeat_seconds=repeat_s,
                single_seconds=singles[i][1],
                floor_c=distance,
                bound=bound,
                max_feature_sigma_error=float(np.abs(error).max()),
                G0=exact,
                G1=feature_gate(distance, bound, np.abs(error).max())
                if exact
                else None,
                G5=peak <= 135.29,
                process_peak_gib=peak,
                allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                telemetry=first[i].metadata,
                repeat_telemetry=second[i].metadata,
                fidelity_status="pilot_only" if passed else "failed",
            )
            write_json(args.out / f"row-{2 * batch + i:05d}.json", record)
            print(
                "ROW",
                2 * batch + i,
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
            rows=12,
            certification=False,
            note="Full bank/statistic gates remain required",
        ),
    )


if __name__ == "__main__":
    main()
