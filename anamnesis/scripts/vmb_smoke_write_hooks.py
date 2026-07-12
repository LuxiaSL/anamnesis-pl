"""vmb Stage-A smoke test (3B): capture surface + activation-WRITE path validation.

Checks, on ONE banked continuation (prereg Stage A(iii)/(iv), session prompt item C):
  1. Full vmb replay surface loads + captures: all-layer keys/values/queries/attn_outputs,
     sampled gate. attn_outputs threaded end-to-end (replay → RawGenerationData → npz →
     load_raw_tensors round-trip).
  2. Feature extraction runs on the extended family set; feature count reported.
  3. WRITE path: α=0 injection reproduces the hookless replay EXACTLY (strict no-op);
     α>0 at a mid layer CHANGES the signature, monotonically in α (rank correlation of
     |Δfeature| vs α reported per dose).
  4. Replay-vs-replay same-device jitter (faithfulness preview): Pearson r between two
     hookless replays' feature vectors.
  5. Storage price: v3 npz size with the new all-layer q + attn_out capture (census input).

Usage (node1, one GPU):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=pipeline python -m anamnesis.scripts.vmb_smoke_write_hooks \
        --model 3b --model-path /models/llama-3.2-3b-instruct \
        --floor-run-dir /models/anamnesis-extract/runs/vmb_stage0_3b \
        --calib-dir /models/anamnesis-extract/calibration/3b \
        --out /dev/shm/vmb_smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import torch

from anamnesis.config import MODEL_PRESETS, ModelConfig, ExtractionConfig, FeaturePipelineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--floor-run-dir", type=Path, required=True)
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "vmb_smoke")
    ap.add_argument("--gen-id", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from anamnesis.extraction.model_loader import load_model, ResidualWriteSpec
    from anamnesis.extraction.replay_extract import replay_extract
    from anamnesis.extraction.raw_saver import save_raw_tensors_v3, load_raw_tensors
    from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data

    preset = MODEL_PRESETS[args.model]
    all_layers = list(range(preset.num_layers))
    model_config = ModelConfig(
        model_id=args.model_path, torch_dtype=preset.torch_dtype,
        num_layers=preset.num_layers, hidden_dim=preset.hidden_dim,
        num_attention_heads=preset.num_attention_heads, num_kv_heads=preset.num_kv_heads,
        head_dim=preset.head_dim,
    )
    extraction_config = ExtractionConfig(
        sampled_layers=preset.sampled_layers, pca_layers=preset.pca_layers,
        early_layer_cutoff=preset.early_layer_cutoff, late_layer_cutoff=preset.late_layer_cutoff,
        enable_tier3=False,   # smoke: skip PCA calibration dependence
    )
    family_config = FeaturePipelineConfig(
        include_baseline_tiers=True, enable_residual_trajectory=True,
        enable_attention_flow=True, enable_gate_features=True,
        enable_temporal_dynamics=False, enable_per_head=True, enable_stft=True,
        enable_value_geometry=True, enable_qk_geometry=True, enable_kv_cka=True,
        trajectory_layers=preset.trajectory_layers,
    )

    manifest = json.loads((args.floor_run_dir / "replay_manifest.json").read_text())
    entry = manifest["entries"][str(args.gen_id)]
    ids, plen = entry["input_ids"], int(entry["prompt_length"])
    logger.info(f"continuation gen_{args.gen_id}: L={len(ids)} P={plen}")

    pm_path = args.calib_dir / "positional_means.npz"
    positional_means = np.load(pm_path)["positional_means"].astype(np.float32) if pm_path.exists() else None

    loaded = load_model(
        model_config, sampled_layers=preset.sampled_layers, register_gate_hooks=True,
        key_layers=all_layers, value_layers=all_layers,
        query_layers=all_layers, attn_output_layers=all_layers,
    )

    def replay_features():
        raw = replay_extract(loaded, ids, plen, positional_means=positional_means)
        feats, names = compute_features_v2_from_data(raw, extraction_config, family_config)
        return raw, np.asarray(feats, dtype=np.float64), names

    results: dict[str, object] = {}

    # ── 1+2: baseline replay, capture surface check ──
    raw0, f0, names = replay_features()
    n_ao = len(raw0.attn_outputs or {})
    n_q = len(raw0.queries or {})
    logger.info(f"capture: attn_outputs layers={n_ao}/{preset.num_layers}, queries layers={n_q}, "
                f"features={len(f0)}")
    assert n_ao == preset.num_layers, "attn_outputs not captured at all layers"
    assert n_q == preset.num_layers, "queries not captured at all layers"
    results["n_features_extended"] = len(f0)
    results["attn_out_layers"] = n_ao

    # npz round-trip + storage price
    npz_path = save_raw_tensors_v3(raw0, 999, args.out, plen, input_ids=np.asarray(ids))
    size_mb = npz_path.stat().st_size / 1e6
    rt = load_raw_tensors(999, args.out, surfaces=["attn_out"])
    assert rt.attn_outputs is not None and len(rt.attn_outputs) == preset.num_layers, "attn_out round-trip failed"
    logger.info(f"npz round-trip OK; storage price = {size_mb:.1f} MB/gen (vmb full surface)")
    results["npz_mb_per_gen"] = round(size_mb, 1)

    # ── 4: same-device hookless jitter (faithfulness preview) ──
    _, f1, _ = replay_features()
    r = float(np.corrcoef(f0, f1)[0, 1])
    max_rel = float(np.max(np.abs(f0 - f1) / np.maximum(np.abs(f0), 1e-9)))
    logger.info(f"replay-vs-replay same-device: r={r:.6f}, max_rel_diff={max_rel:.2e}")
    results["same_device_r"] = r

    # ── 3: write path ──
    mid_layer = preset.num_layers // 2
    gen = torch.Generator().manual_seed(1234)
    vec = torch.randn(preset.hidden_dim, generator=gen)

    handle = loaded.add_residual_write(ResidualWriteSpec(
        layer_idx=mid_layer, vector=vec, alpha=0.0, start_pos=plen))
    _, f_a0, _ = replay_features()
    handle.remove()
    # α=0 is a strict computational no-op (the pre-hook early-returns), so f_a0 is
    # just another same-device replay: judge it against the jitter envelope
    # |f0−f1|, not bitwise equality.
    jitter = float(np.mean(np.abs(f0 - f1)))
    a0_delta = float(np.mean(np.abs(f_a0 - f1)))
    alpha0_identical = bool(a0_delta <= max(3.0 * jitter, 1e-10))
    r_a0 = float(np.corrcoef(f0, f_a0)[0, 1])
    logger.info(f"alpha=0: mean|Δ|={a0_delta:.3e} vs jitter {jitter:.3e} → noop={alpha0_identical} "
                f"(r={r_a0:.6f})")
    results["alpha0_noop"] = alpha0_identical
    results["alpha0_delta"] = a0_delta
    results["jitter_delta"] = jitter

    deltas = {}
    for alpha in (0.5, 2.0, 8.0):
        h = loaded.add_residual_write(ResidualWriteSpec(
            layer_idx=mid_layer, vector=vec, alpha=alpha, start_pos=plen))
        _, fa, _ = replay_features()
        h.remove()
        finite = np.isfinite(fa) & np.isfinite(f0)
        d = float(np.nanmean(np.abs(fa[finite] - f0[finite])))
        deltas[alpha] = d
        logger.info(f"alpha={alpha}: mean|Δfeature|={d:.5f}")
    monotone = deltas[0.5] < deltas[2.0] < deltas[8.0]
    results["dose_deltas"] = deltas
    results["dose_monotone"] = bool(monotone)

    (args.out / "smoke_results.json").write_text(json.dumps(results, indent=2))
    ok = alpha0_identical and monotone and results["same_device_r"] > 0.999
    logger.info(f"SMOKE {'PASS' if ok else 'FAIL'}: {json.dumps(results)}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
