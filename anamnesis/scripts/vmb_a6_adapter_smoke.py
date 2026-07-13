"""A6 adapter-loading smoke (block ops item, BLOCKING for cell-i):
1. PEFT merge happens before hooks (load_model adapter_path) — verified by replaying
   2 banked base-model continuations through a MERGED checkpoint and checking the
   signature (a) differs from the banked BASE signature (adapter took effect through
   the hook surface), (b) reproduces bitwise on a second replay (merged-model
   determinism — the 12b assumption extends to adapted weights).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from anamnesis.config import MODEL_PRESETS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import torch

    from anamnesis.config import ExtractionConfig, FeaturePipelineConfig, ModelConfig
    from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data
    from anamnesis.extraction.model_loader import load_model
    from anamnesis.extraction.replay_extract import replay_extract

    preset = MODEL_PRESETS[args.model]
    all_layers = list(range(preset.num_layers))
    extraction_config = ExtractionConfig(
        sampled_layers=preset.sampled_layers, pca_layers=preset.pca_layers,
        early_layer_cutoff=preset.early_layer_cutoff, late_layer_cutoff=preset.late_layer_cutoff,
        enable_tier3=True)
    family_config = FeaturePipelineConfig(
        include_baseline_tiers=True, enable_residual_trajectory=True,
        enable_attention_flow=True, enable_gate_features=True,
        enable_temporal_dynamics=False, enable_per_head=True, enable_stft=True,
        enable_contrastive_projection=False, enable_value_geometry=True,
        enable_qk_geometry=True, enable_kv_cka=True,
        trajectory_layers=preset.trajectory_layers, contrastive_layers=preset.contrastive_layers)
    model_config = ModelConfig(
        model_id=args.model_path, torch_dtype=preset.torch_dtype,
        num_layers=preset.num_layers, hidden_dim=preset.hidden_dim,
        num_attention_heads=preset.num_attention_heads, num_kv_heads=preset.num_kv_heads,
        head_dim=preset.head_dim)
    loaded = load_model(
        model_config, sampled_layers=preset.sampled_layers, register_gate_hooks=True,
        key_layers=all_layers, value_layers=all_layers, query_layers=all_layers,
        attn_output_layers=all_layers, adapter_path=args.adapter_path)

    pm = np.load(args.calib_dir / "positional_means.npz")["positional_means"].astype(np.float32)
    entries = json.loads((args.stage0_run / "replay_manifest.json").read_text())["entries"]

    results = {"adapter": args.adapter_path, "checks": {}}
    ok = True
    for gid in (0, 10):
        e = entries[str(gid)]
        f1 = compute_features_v2_from_data(
            replay_extract(loaded, e["input_ids"], int(e["prompt_length"]), positional_means=pm),
            extraction_config, family_config, None, None).features
        f2 = compute_features_v2_from_data(
            replay_extract(loaded, e["input_ids"], int(e["prompt_length"]), positional_means=pm),
            extraction_config, family_config, None, None).features
        bitwise = bool(np.array_equal(np.asarray(f1), np.asarray(f2)))
        # banked BASE signature for the same gid
        base = np.load(args.stage0_run / "signatures_v3" / f"gen_{gid:03d}.npz")["features"]
        differs = not np.array_equal(np.asarray(f1)[: len(base)], base)
        results["checks"][str(gid)] = {"bitwise_repeat": bitwise, "differs_from_base": differs}
        ok &= bitwise and differs
        logger.info(f"gen {gid}: bitwise={bitwise} differs_from_base={differs}")
    results["PASS"] = ok
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    logger.info(f"A6 adapter smoke: {'PASS' if ok else 'FAIL'} -> {args.out}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
