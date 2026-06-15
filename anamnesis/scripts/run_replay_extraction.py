"""Per-worker replay extraction: re-process a banked run to v3 signatures + full raw.

Loads the model ONCE with the v3 capture surface (all-layer k_proj + v_proj hooks,
sampled-layer q_proj + gate_proj hooks), the run's calibration (positional_means + PCA),
and the replay manifest; then for each assigned gen_id:
    replay_extract -> save_raw_tensors_v3 -> compute_features_v2_from_data -> save signature.
Resume-aware: skips gens whose signature json already exists. Designed to be spawned
N-per-GPU by parallel_replay.py with CUDA_VISIBLE_DEVICES set per worker.

Usage (one worker, all gens, resume):
    PYTHONPATH=. python -m anamnesis.scripts.run_replay_extraction \
        --model 8b --model-path /models/llama-3.1-8b-instruct \
        --run-dir /models/anamnesis-extract/runs/8b_fat_01 \
        --calib-dir /models/anamnesis-extract/calibration/llama31_8b \
        --manifest /models/anamnesis-extract/runs/8b_fat_01/replay_manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from anamnesis.config import MODEL_PRESETS

F32 = NDArray[np.float32]
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_calibration(calib_dir: Path, enable_tier3: bool) -> tuple[F32 | None, F32 | None, F32 | None]:
    """Load positional_means + PCA (components, mean) from a calibration dir."""
    positional_means = pca_components = pca_mean = None
    pm_path = calib_dir / "positional_means.npz"
    if pm_path.exists():
        positional_means = np.load(pm_path)["positional_means"].astype(np.float32)
        logger.info(f"positional_means {positional_means.shape}")
    else:
        logger.warning(f"NO positional_means at {pm_path} — corrected features will be wrong")
    pca_path = calib_dir / "pca_model.pkl"
    if enable_tier3 and pca_path.exists():
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
        if isinstance(pca, dict):
            pca_components = np.asarray(pca["components"], dtype=np.float32)
            pca_mean = np.asarray(pca["mean"], dtype=np.float32)
        else:
            pca_components = np.asarray(pca.components_, dtype=np.float32)
            pca_mean = np.asarray(pca.mean_, dtype=np.float32)
        logger.info(f"pca components {pca_components.shape}")
    return positional_means, pca_components, pca_mean


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay-extract a banked run to v3 sigs + raw")
    parser.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    parser.add_argument("--model-path", type=str, required=True, help="Local model dir (overrides preset HF id)")
    parser.add_argument("--run-dir", type=Path, required=True, help="Output run dir (holds metadata.json + outputs)")
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gen-ids", type=int, nargs="+", default=None, help="Subset of gen ids (default: all in manifest)")
    parser.add_argument("--raw-subdir", default="raw_tensors_v3")
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="Absolute raw output dir (overrides run-dir/raw-subdir; e.g. /dev/shm scratch)")
    parser.add_argument("--sig-subdir", default="signatures_v3")
    parser.add_argument("--no-raw", action="store_true", help="Skip raw banking (signatures only)")
    parser.add_argument("--no-tier3", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="Recompute even if a signature exists")
    parser.add_argument("--label", default="w", help="Worker label for logs")
    args = parser.parse_args()

    preset = MODEL_PRESETS[args.model]
    all_layers = list(range(preset.num_layers))

    # ── Configs (v3 capture surface) ──
    from anamnesis.config import ExtractionConfig, FeaturePipelineConfig, ModelConfig

    extraction_config = ExtractionConfig(
        sampled_layers=preset.sampled_layers,
        pca_layers=preset.pca_layers,
        early_layer_cutoff=preset.early_layer_cutoff,
        late_layer_cutoff=preset.late_layer_cutoff,
        enable_tier3=not args.no_tier3,
    )
    family_config = FeaturePipelineConfig(
        include_baseline_tiers=True,
        enable_residual_trajectory=True,
        enable_attention_flow=True,
        enable_gate_features=True,
        enable_temporal_dynamics=False,   # v3: temporal_dynamics ignored
        enable_per_head=True,             # v3: new surface
        enable_stft=True,
        enable_contrastive_projection=False,  # contrastive is a separate addon
        trajectory_layers=preset.trajectory_layers,
        contrastive_layers=preset.contrastive_layers,
    )
    model_config = ModelConfig(
        model_id=args.model_path,
        torch_dtype=preset.torch_dtype,
        num_layers=preset.num_layers,
        hidden_dim=preset.hidden_dim,
        num_attention_heads=preset.num_attention_heads,
        num_kv_heads=preset.num_kv_heads,
        head_dim=preset.head_dim,
    )

    # ── Load model with the full v3 capture surface ──
    from anamnesis.extraction.model_loader import load_model
    from anamnesis.extraction.replay_extract import replay_extract
    from anamnesis.extraction.raw_saver import save_raw_tensors_v3
    from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data, save_features

    loaded = load_model(
        model_config,
        sampled_layers=preset.sampled_layers,
        register_gate_hooks=True,
        key_layers=all_layers,        # all-layer keys
        value_layers=all_layers,      # all-layer values (v_proj)
        query_layers=preset.sampled_layers,  # sampled pre-RoPE queries (QK geometry)
    )

    positional_means, pca_components, pca_mean = _load_calibration(
        args.calib_dir, enable_tier3=not args.no_tier3,
    )

    # ── Manifest + source metadata ──
    with open(args.manifest) as f:
        manifest = json.load(f)
    entries = manifest["entries"]

    src_meta: dict[int, dict] = {}
    meta_path = args.run_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            md = json.load(f)
        gens = md["generations"] if isinstance(md, dict) and "generations" in md else md
        src_meta = {int(g["generation_id"]): g for g in gens}

    raw_dir = args.raw_dir if args.raw_dir is not None else (args.run_dir / args.raw_subdir)
    sig_dir = args.run_dir / args.sig_subdir
    sig_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    # ── Determine work set (subset + resume) ──
    avail = sorted(int(k) for k in entries)
    todo = [g for g in avail if (args.gen_ids is None or g in set(args.gen_ids))]
    if not args.no_resume:
        todo = [g for g in todo if not (sig_dir / f"gen_{g:03d}.json").exists()]

    logger.info(f"[{args.label}] {len(todo)} gens to process ({len(avail)} in manifest)")

    n_done = n_fail = 0
    t0 = time.time()
    for i, gid in enumerate(todo):
        try:
            e = entries[str(gid)]
            input_ids = e["input_ids"]
            plen = int(e["prompt_length"])

            raw_data = replay_extract(loaded, input_ids, plen, positional_means=positional_means)

            if not args.no_raw:
                save_raw_tensors_v3(raw_data, gid, raw_dir, prompt_length=plen, input_ids=input_ids)

            result = compute_features_v2_from_data(
                raw_data, extraction_config, family_config, pca_components, pca_mean,
            )

            metadata = dict(src_meta.get(gid, {"generation_id": gid}))
            metadata["num_features"] = int(len(result.features))
            metadata["tier_slices"] = {k: list(v) for k, v in result.tier_slices.items()}
            metadata["extraction_version"] = 3
            save_features(gid, result, metadata, sig_dir)

            n_done += 1
            if (i + 1) % 10 == 0 or i == 0:
                el = time.time() - t0
                rate = (i + 1) / el if el > 0 else 0
                eta = (len(todo) - i - 1) / rate if rate > 0 else 0
                logger.info(
                    f"[{args.label}] {i + 1}/{len(todo)} gen_{gid:03d}: "
                    f"{len(result.features)} feats, {el:.0f}s, ETA {eta:.0f}s"
                )
        except Exception as exc:  # noqa: BLE001 — keep going, report failures
            n_fail += 1
            logger.error(f"[{args.label}] gen_{gid:03d} FAILED: {exc}", exc_info=True)

    logger.info(f"[{args.label}] done: {n_done} ok, {n_fail} failed in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
