"""Native full-pass CPU/GPU measurement, explicitly not a transferred 70B certificate."""

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
from extraction_perf_memory import ProcessMemorySampler


def select_entries(entries):
    ordered = sorted(entries, key=lambda k: (len(entries[k]["input_ids"]), int(k)))
    if len(ordered) < 3:
        raise ValueError("three native reference inputs required")
    return [ordered[0], ordered[len(ordered) // 2], ordered[-1]]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--model", choices=("3b", "8b"), required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--mode", choices=("cpu", "gpu"), required=True)
    for name in ("calibration", "manifest", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    if args.mode == "cpu":
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    import torch
    import anamnesis
    from anamnesis.config import MODEL_PRESETS, ModelConfig
    from anamnesis.extraction.model_loader import load_model
    from anamnesis.extraction.replay_config import native_replay_configs
    from anamnesis.extraction.replay_extract import replay_extract
    from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data
    from anamnesis.extraction.gpu_schema import resolve_gpu_schema
    from anamnesis.extraction.gpu_features import GpuFeatureLane
    from anamnesis.scripts.run_replay_extraction import _load_calibration

    args.out.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(args.mode == "gpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    preset = MODEL_PRESETS[args.model]
    ec, fc = native_replay_configs(preset)
    entries = json.loads(args.manifest.read_text())["entries"]
    selected = select_entries(entries)
    calibration_files = {
        p.name: sha(p)
        for p in (
            args.calibration / "positional_means.npz",
            args.calibration / "pca_model.pkl",
        )
    }
    calhash = hashlib.sha256(
        json.dumps(calibration_files, sort_keys=True).encode()
    ).hexdigest()
    weights = sorted(args.model_path.glob("*.safetensors"))
    if not weights:
        raise ValueError("local safetensors checkpoint required")
    model_files = {p.name: sha(p) for p in [args.model_path / "config.json", *weights]}
    pin = dict(
        utc=stamp(),
        source=source_pin(Path(anamnesis.__file__).parent),
        runner_sha256=sha(Path(__file__)),
        model=args.model,
        model_files=model_files,
        mode=args.mode,
        calibration_files=calibration_files,
        manifest_sha256=sha(args.manifest),
        selected=selected,
        extraction=ec.model_dump(mode="json"),
        families=fc.model_dump(mode="json"),
        workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        torch=torch.__version__,
        certification=False,
        ruler="no model-specific sigma or path-floor gate evaluated",
    )
    write_json(args.out / "deployment.json", pin)
    write_json(args.out / "inputs.json", {key: entries[key] for key in selected})
    pm, components, mean = _load_calibration(args.calibration, True)
    if any(v is None for v in (pm, components, mean)):
        raise ValueError("complete native calibration required")
    if any(len(entries[k]["input_ids"]) - 2 >= pm.shape[1] for k in selected):
        raise ValueError("native trial exceeds calibration extent")
    cfg = ModelConfig(
        model_id=str(args.model_path),
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
    lanes = {}
    if args.mode == "gpu":
        by_schema = {}
        for key in selected:
            steps = len(entries[key]["input_ids"]) - entries[key]["prompt_length"] - 1
            schema = resolve_gpu_schema(preset.num_layers, steps, ec, fc, components)
            if schema.feature_names not in by_schema:
                by_schema[schema.feature_names] = GpuFeatureLane(
                    ec,
                    fc,
                    list(schema.feature_names),
                    pm,
                    components,
                    mean,
                    device="cuda:0",
                    calibration_sha256=calhash,
                    replay_path="full",
                )
            lanes[steps] = by_schema[schema.feature_names]

    def replay(key):
        row = entries[key]
        tokens, start = row["input_ids"], row["prompt_length"]
        if args.mode == "cpu":
            raw = replay_extract(loaded, tokens, start, pm)
            result = compute_features_v2_from_data(raw, ec, fc, components, mean)
            return (
                result.features,
                result.feature_names,
                {"replay_path": "full", "raw_tensors_saved": False},
            )
        steps = len(tokens) - start - 1
        result = lanes[steps].replay_span(loaded, tokens, start, len(tokens))
        return result.features, list(result.feature_names), result.metadata

    with memory.phase("warmup"):
        replay(selected[0])
    for i, key in enumerate(selected):
        features = []
        receipts = []
        times = []
        names = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for repeat in range(2):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with memory.phase(args.mode):
                values, current, receipt = replay(key)
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            if names is not None and names != current:
                raise ValueError("repeat schema mismatch")
            names = current
            features.append(values)
            receipts.append(receipt)
        exact = features[0].tobytes() == features[1].tobytes()
        np.savez(
            args.out / f"row-{i:05d}.npz",
            features=features[0],
            repeat=features[1],
            feature_names=np.array(names),
        )
        record = dict(
            utc=stamp(),
            generation_id=key,
            G0=exact,
            seconds=times,
            feature_count=len(names),
            prefix=entries[key]["prompt_length"],
            continuation=len(entries[key]["input_ids"]) - entries[key]["prompt_length"],
            metadata=receipts,
            allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            certification=False,
        )
        write_json(args.out / f"row-{i:05d}.json", record)
        print("ROW", i, json.dumps(record), flush=True)
        if not exact or not np.isfinite(features[0]).all():
            raise RuntimeError("native repeatability/nonfinite failure")
    if source_pin(Path(anamnesis.__file__).parent) != pin["source"]:
        raise RuntimeError("native source changed during trial")
    write_json(
        args.out / "complete.json", dict(utc=stamp(), rows=3, certification=False)
    )


if __name__ == "__main__":
    main()
