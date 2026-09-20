"""Frozen six-length instrument sweep; CPU and GPU run in separate processes.

No generation, no new consumer capture, no per-length floor(b), no certification.
Use floor-v1 for CPU and the qualified immutable gpu-v3 package for GPU.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from extraction_perf_phase0 import rows_from_bank, sha, source_pin, stamp, write_json

LENGTHS = (64, 128, 299, 512, 1024, 2048)


def sweep_rows(bank_rows):
    """Five fixed distinct token streams; cyclic extension is explicitly synthetic."""
    seeds = []
    for row in bank_rows:
        if row["key"][1] == "own_cont" and not row["excluded"]:
            seeds.append(row)
        if len(seeds) == 5:
            break
    if len(seeds) != 5:
        raise ValueError("five anchor input streams required")
    output = []
    for length in LENGTHS:
        for index, row in enumerate(seeds):
            stream = row["tokens"][: row["end"]]
            tokens = (stream * ((length + 128) // len(stream) + 1))[: length + 128]
            output.append(
                dict(
                    key=["synthetic", length, index],
                    start=length,
                    end=length + 128,
                    tokens=tokens,
                    seed_key=row["key"],
                )
            )
    return output


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--mode", choices=("cpu", "gpu", "cpu-control"), required=True)
    p.add_argument(
        "--indices", help="Diagnostic subset only; cannot establish full sweep coverage"
    )
    p.add_argument(
        "--blas-library", choices=("default", "cublas", "cublaslt"), default="default"
    )
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--bank-pin", type=Path, required=True)
    p.add_argument("--floor-pin", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--sigma", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="/models/Llama-3.1-70B")
    args = p.parse_args()
    if args.blas_library != "default" and args.mode != "cpu-control":
        raise ValueError("BLAS backend variants are diagnostic CPU controls only")
    if args.mode == "cpu":
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    import torch
    import transformers
    import anamnesis
    import palimpsest.sigbridge as bridge
    from anamnesis.config import ModelConfig
    from anamnesis.extraction.model_loader import load_model
    from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data
    from extraction_perf_memory import ProcessMemorySampler

    args.out.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(args.mode != "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.blas_library != "default":
        torch.backends.cuda.preferred_blas_library(args.blas_library)
    rows, names = rows_from_bank(args.bank)
    selected = sweep_rows(rows)
    if args.indices:
        indices = [int(i) for i in args.indices.split(",")]
        if len(set(indices)) != len(indices) or any(
            i < 0 or i >= len(selected) for i in indices
        ):
            raise ValueError("invalid diagnostic selection")
        selected = [selected[i] for i in indices]
    bank_pin = json.loads(args.bank_pin.read_text())
    floor_pin = json.loads(args.floor_pin.read_text())
    for filename, digest in floor_pin["files"].items():
        if sha(Path(filename)) != digest:
            raise ValueError(f"calibration/reference changed: {filename}")
    calibration_hash = hashlib.sha256(
        json.dumps(floor_pin["files"], sort_keys=True).encode()
    ).hexdigest()
    if calibration_hash != bank_pin["calibration_sha256"]:
        raise ValueError("calibration identity differs from bank")
    if sha(args.sigma) != bank_pin["standardizer_sha256"]:
        raise ValueError("standardizer changed")
    with np.load(args.sigma) as z:
        if z["feature_names"].astype(str).tolist() != names:
            raise ValueError("schema changed")
    source = source_pin(Path(anamnesis.__file__).parent)
    expected = bank_pin["source"] if args.mode == "gpu" else floor_pin["anamnesis"]
    if source != expected:
        raise ValueError("source differs from qualified reference")
    if source_pin(Path(bridge.__file__).parent) != bank_pin["caller"]:
        raise ValueError("caller changed")
    write_json(args.out / "inputs.json", selected)
    pin = dict(
        utc=stamp(),
        source=source,
        caller=bank_pin["caller"],
        bank_pin_sha256=sha(args.bank_pin),
        runner_sha256=sha(Path(__file__)),
        inputs_sha256=sha(args.out / "inputs.json"),
        mode=args.mode,
        preferred_blas_library=str(torch.backends.cuda.preferred_blas_library()),
        torch=torch.__version__,
        transformers=transformers.__version__,
        numpy=np.__version__,
        workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        deterministic=args.mode != "cpu",
        tf32=False,
        standardizer_sha256=sha(args.sigma),
        calibration_sha256=bank_pin["calibration_sha256"],
    )
    write_json(args.out / "deployment.json", pin)
    preset = bridge.resolve_preset("70b-base")
    extraction, families = bridge.battery_configs(preset)
    calibration = bridge.CalibrationBundle.load(
        args.calibration, expect_model_path=args.model, expect_preset="70b-base"
    )
    max_pos = calibration.positional_means.shape[1]
    with np.load(args.calibration / "positional_means.npz") as archive:
        counts = archive["pos_counts"]
    if counts.shape != calibration.positional_means.shape[:2]:
        raise ValueError("position-count schema mismatch")
    write_json(
        args.out / "calibration-scope.json",
        dict(shape=list(calibration.positional_means.shape), max_pos=max_pos),
    )
    cfg = ModelConfig(
        model_id=args.model,
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
    memory = ProcessMemorySampler()
    atexit.register(lambda: write_json(args.out / "process-vram.json", memory.close()))
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
            extraction,
            families,
            names,
            calibration.positional_means,
            calibration.pca_components,
            calibration.pca_mean,
            device="cuda:0",
            calibration_sha256=bank_pin["calibration_sha256"],
        )
        write_json(args.out / "lane.json", lane.identity)

    def replay(row):
        if lane is not None:
            result = lane.replay_span(loaded, row["tokens"], row["start"], row["end"])
            return result.features, result.metadata
        raw = bridge.replay_span(
            loaded,
            row["tokens"],
            bridge.SpanRef(span_id="h1", start=row["start"], length=128),
            positional_means=calibration.positional_means,
        )
        result = compute_features_v2_from_data(
            raw, extraction, families, calibration.pca_components, calibration.pca_mean
        )
        if result.feature_names != names:
            raise ValueError("feature schema changed")
        return result.features, {}

    with memory.phase("warmup"):
        replay(selected[0])
    for i, row in enumerate(selected):
        outputs, metadata, seconds = [], [], []
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for repeat in range(2):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with memory.phase(args.mode):
                values, receipt = replay(row)
                torch.cuda.synchronize()
            seconds.append(time.perf_counter() - start)
            outputs.append(values)
            metadata.append(receipt)
        exact = outputs[0].tobytes() == outputs[1].tobytes()
        np.savez(args.out / f"row-{i:05d}.npz", features=outputs[0], repeat=outputs[1])
        record = dict(
            utc=stamp(),
            key=row["key"],
            cache_length=row["start"],
            last_feature_position=row["end"] - 2,
            inside_calibration=row["end"] - 2 < max_pos,
            all_feature_positions_count_gt5=bool(
                row["end"] - 2 < max_pos
                and (counts[1:, row["start"] : row["end"] - 1] > 5).all()
            ),
            independent_repeat_exact=exact,
            seconds=seconds,
            metadata=metadata,
            allocated_bytes=torch.cuda.max_memory_allocated(),
            reserved_bytes=torch.cuda.max_memory_reserved(),
        )
        write_json(args.out / f"row-{i:05d}.json", record)
        write_json(args.out / "process-vram.json", memory.snapshot())
        print("ROW", i, json.dumps(record), flush=True)
        if not exact or not all(np.isfinite(x).all() for x in outputs):
            raise RuntimeError("repeatability/nonfinite failure; stop")
    if source != source_pin(Path(anamnesis.__file__).parent):
        raise RuntimeError("source changed during sweep")
    write_json(
        args.out / "complete.json",
        dict(utc=stamp(), rows=len(selected), certification=False),
    )


if __name__ == "__main__":
    main()
