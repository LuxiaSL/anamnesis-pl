"""Recompute v3 signatures from banked v3 raw with a given PCA (e.g. the C5 corrected
per-layer PCA). CPU only — the offline feature loop on banked raw, with positional_means
injected (v3 raw is pos_means-deduped). Resume-aware via recompute (overwrites output dir).

Usage:
    OMP_NUM_THREADS=1 python -m anamnesis.scripts.run_recompute_v3 --model 3b \
        --run-dir /models/anamnesis-extract/runs/3b_fat_01 \
        --calib-dir /models/anamnesis-extract/calibration/3b \
        --pca-model /models/anamnesis-extract/calibration/3b/pca_model_corrected.pkl \
        --out-subdir signatures_v3_c5 --workers 48
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from anamnesis.config import ExtractionConfig, FeaturePipelineConfig, MODEL_PRESETS
from anamnesis.extraction.feature_pipeline import _load_pca_model, recompute_all_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="Recompute v3 sigs from banked raw with a given PCA (CPU)")
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--pca-model", type=Path, default=None, help="default: calib-dir/pca_model.pkl")
    ap.add_argument("--raw-subdir", default="raw_tensors_v3")
    ap.add_argument("--out-subdir", default="signatures_v3_c5")
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    p = MODEL_PRESETS[args.model]
    ec = ExtractionConfig(sampled_layers=p.sampled_layers, pca_layers=p.pca_layers,
                          early_layer_cutoff=p.early_layer_cutoff, late_layer_cutoff=p.late_layer_cutoff,
                          enable_tier3=True)
    fc = FeaturePipelineConfig(include_baseline_tiers=True, enable_residual_trajectory=True,
                               enable_attention_flow=True, enable_gate_features=True,
                               enable_temporal_dynamics=False, enable_per_head=True, enable_stft=True,
                               enable_contrastive_projection=False, trajectory_layers=p.trajectory_layers,
                               contrastive_layers=p.contrastive_layers)

    pos = np.load(args.calib_dir / "positional_means.npz")["positional_means"].astype(np.float32)
    pca_path = args.pca_model or (args.calib_dir / "pca_model.pkl")
    pca_c, pca_m = _load_pca_model(pca_path)
    logger.info(f"PCA: {pca_path.name} ({'per-layer' if isinstance(pca_c, dict) else 'pooled'})")

    recompute_all_features(
        raw_dir=args.run_dir / args.raw_subdir,
        output_dir=args.run_dir / args.out_subdir,
        config=ec, pca_components=pca_c, pca_mean=pca_m,
        metadata_dir=args.run_dir / "signatures_v3",  # source per-gen metadata json
        family_config=fc, n_workers=args.workers, positional_means=pos,
    )
    logger.info(f"Recompute done → {args.run_dir / args.out_subdir}")


if __name__ == "__main__":
    main()
