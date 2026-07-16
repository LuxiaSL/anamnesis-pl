"""ANNEX (session 7, route-5 tenancy): bank dir0 as a consumable axis file + its
shuffled-label null axes — charter v1.1's gauge and null target of record.

The gauge and its null come from ONE construction differing in one argument:
  gauge = LDA(lsqr, shrinkage=auto) on z-scored pure-mode corpora, REAL labels
          (identical to vmb_a5_frozen_directional.build_axes's lda_axis(*DIR0_PAIR))
  null  = the same fit with mode labels SHUFFLED (k independent draws, fixed seeds)

Consumable format = arms/C2/c2_orphaned_axis_3b.npz's contract, full feature set:
  {axis (d,), feature_indices (d,), feature_names (d,), med (D,), scale (D,)}
(annex_samefamily_lever.py --axis-npz eats exactly these keys.)

Sanity gates (script fails loudly if any is violated):
  1. banked gauge reproduces the in-script construction bit-for-bit (same code path)
  2. every shuffled draw is near-orthogonal to the gauge (|cos| < 0.15)
  3. draws are near-orthogonal to each other
  4. unit norms

CPU + banked data only. Writes to outputs/battery/annex/ ONLY.

Usage (from pipeline/):
    python -m anamnesis.scripts.annex_bank_dir0 \
        --battery-root ../outputs/battery --out-dir ../outputs/battery/annex --model 3b
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from anamnesis.analysis.battery.deltas import ConditionCorpus, load_floor_scale
from anamnesis.analysis.battery.floors import load_signature_matrix
from anamnesis.analysis.battery.manifest import MODEL_META

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]

DIR0_PAIR = ("analogical", "contrastive")  # matches vmb_a5_frozen_directional.DIR0_PAIR
N_NULL_DRAWS = 8
NULL_SEED_BASE = 20260716
ORTHO_BAR = 0.15


def fit_axis(X: F32, y: F32) -> F32:
    """The lda_axis construction, verbatim (vmb_a5_frozen_directional.py:171)."""
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)
    a = clf.coef_[0].astype(np.float64)
    n = np.linalg.norm(a)
    if n <= 1e-12:
        raise SystemExit("degenerate LDA axis (zero norm)")
    return (a / n).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default="3b", choices=list(MODEL_META.keys()))
    args = ap.parse_args()

    mm = MODEL_META[args.model]
    stage0 = args.battery_root / mm.stage0_dir
    med, scale = load_floor_scale(stage0 / "signatures_v3")

    corpora: dict[str, ConditionCorpus] = {}
    census: dict[str, dict] = {}
    for m_ in DIR0_PAIR:
        d = args.battery_root / f"vmb_a2_{args.model}_pure_{m_}"
        corpora[m_] = ConditionCorpus(d / "signatures_v3", d / "metadata.json",
                                      med, scale, f"pure-{m_}")
        meta = json.loads((d / "metadata.json").read_text())["generations"]
        modes = sorted({g["mode"] for g in meta})
        census[m_] = {"n_rows": int(corpora[m_].Z.shape[0]), "n_meta": len(meta),
                      "modes_in_metadata": modes}
        logger.info(f"{m_}: {census[m_]}")

    # feature names of record = the modal set the loader actually returned
    _, names, _ = load_signature_matrix(
        args.battery_root / f"vmb_a2_{args.model}_pure_{DIR0_PAIR[0]}" / "signatures_v3")
    n_feat = corpora[DIR0_PAIR[0]].Z.shape[1]
    if len(names) != n_feat or len(med) != n_feat:
        raise SystemExit(f"feature-dim mismatch: names={len(names)} Z={n_feat} med={len(med)}")

    X = np.vstack([corpora[DIR0_PAIR[0]].Z, corpora[DIR0_PAIR[1]].Z])
    y = np.r_[np.ones(corpora[DIR0_PAIR[0]].Z.shape[0]),
              np.zeros(corpora[DIR0_PAIR[1]].Z.shape[0])]

    gauge = fit_axis(X, y)

    # sanity 1: same code path reproduces bit-for-bit
    if not np.array_equal(gauge, fit_axis(X, y)):
        raise SystemExit("gauge not deterministic under refit — investigate before banking")

    nulls, seeds, cos_to_gauge = [], [], []
    for k in range(N_NULL_DRAWS):
        seed = NULL_SEED_BASE + k
        rng = np.random.default_rng(seed)
        ax = fit_axis(X, rng.permutation(y))
        c = float(ax @ gauge)
        if abs(c) >= ORTHO_BAR:
            raise SystemExit(f"null draw {k} (seed {seed}) |cos|={abs(c):.3f} >= {ORTHO_BAR}")
        nulls.append(ax)
        seeds.append(seed)
        cos_to_gauge.append(c)
    N = np.stack(nulls)
    pair_cos = (N @ N.T)[np.triu_indices(N_NULL_DRAWS, k=1)]
    if np.abs(pair_cos).max() >= ORTHO_BAR:
        raise SystemExit(f"null draws inter-cos max {np.abs(pair_cos).max():.3f} >= {ORTHO_BAR}")

    idx = np.arange(n_feat, dtype=np.int64)
    nm = np.array(names)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gauge_path = args.out_dir / f"annex_dir0_axis_{args.model}.npz"
    np.savez(gauge_path, axis=gauge, feature_indices=idx, feature_names=nm,
             med=med, scale=scale)
    null_path = args.out_dir / f"annex_dir0_shufnull_axis_{args.model}.npz"
    np.savez(null_path, axes=N, seeds=np.array(seeds, dtype=np.int64),
             cos_to_gauge=np.array(cos_to_gauge, dtype=np.float32),
             feature_indices=idx, feature_names=nm, med=med, scale=scale)
    # draw 0 additionally in the single-axis consumable shape, for --axis-npz drop-in
    null0_path = args.out_dir / f"annex_dir0_shufnull0_axis_{args.model}.npz"
    np.savez(null0_path, axis=N[0], feature_indices=idx, feature_names=nm,
             med=med, scale=scale)

    report = {
        "construction": "LinearDiscriminantAnalysis(solver=lsqr, shrinkage=auto) on "
                        f"z-scored pures {DIR0_PAIR}, coef normalized "
                        "(verbatim vmb_a5_frozen_directional.lda_axis)",
        "corpora_census": census,
        "n_features": int(n_feat),
        "stage0_floor": str(stage0),
        "null_draws": N_NULL_DRAWS, "null_seed_base": NULL_SEED_BASE,
        "null_cos_to_gauge": [round(c, 5) for c in cos_to_gauge],
        "null_pairwise_cos_max_abs": float(np.abs(pair_cos).max()),
        "files": {"gauge": gauge_path.name, "nulls": null_path.name,
                  "null0_consumable": null0_path.name},
    }
    rp = args.out_dir / f"annex_dir0_bank_report_{args.model}.json"
    rp.write_text(json.dumps(report, indent=1))
    logger.info(f"banked: {gauge_path.name}, {null_path.name}, {null0_path.name}")
    logger.info(f"report -> {rp}")


if __name__ == "__main__":
    main()
