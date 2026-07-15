"""C2 — the orphaned readout coordinate (PREFLIGHT §2; session-4 Part C).

The steering-target COORDINATE for the synthetic-temperature cell: the t03-vs-t09 LDA
discriminant computed on NON-TRIVIAL SURFACES ONLY — attention + gate + keys + qk families,
EXCLUDING source:output (logit conditioning) and source:residual (self-read), per C§1. This
axis is "orphaned": no text-only party can even define temperature (A1: content detectors at
chance), so the coordinate is defined purely by the sampler-knob metadata.

Masking is PRIMARY; length-residualization default-on; GroupKFold-by-topic (leak-proof).
The likelihood-residualization ROBUSTNESS column needs per-token surprisal — reported if the
signature carries it, else flagged owed (never load-bearing under the primary claim).

The frozen axis is banked BEFORE any steered generation exists (C3 not yet run). CPU-only.
First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from anamnesis.analysis.battery.deltas import load_floor_scale
from anamnesis.analysis.battery.floors import load_signature_matrix
from anamnesis.analysis.battery.manifest import MODEL_META
from anamnesis.analysis.feature_map import FeatureMap, Source

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]
NONTRIVIAL = [Source.attention, Source.gate, Source.keys, Source.qk]
TRIVIAL = [Source.output, Source.residual]


def _meta_field(run: Path, field: str) -> dict[int, object]:
    md = json.loads((run / "metadata.json").read_text())
    gens = md["generations"] if "generations" in md else md
    out = {}
    for g in gens:
        gid = int(g.get("generation_id", g.get("gen_id", -1)))
        out[gid] = g.get(field)
    return out


def _length_residualize(X: F32, length: NDArray, train: NDArray) -> F32:
    """Per-feature OLS residualization against length; β fit on `train` rows, applied to all."""
    L = np.c_[np.ones(len(length)), length.astype(np.float64)]
    Lt = L[train]
    beta, *_ = np.linalg.lstsq(Lt, X[train].astype(np.float64), rcond=None)
    return (X.astype(np.float64) - L @ beta).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--cold-run", type=Path, required=True, help="vmb_a1_3b_t03")
    ap.add_argument("--hot-run", type=Path, required=True, help="vmb_a1_3b_t09")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default="3b")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mm = MODEL_META[args.model]

    stage0 = args.battery_root / mm.stage0_dir
    med, scale = load_floor_scale(stage0 / "signatures_v3")

    def load(run: Path, y: int):
        X, names, gids = load_signature_matrix(run / "signatures_v3")
        Z = (X - med) / scale
        topic = _meta_field(run, "topic")
        length = _meta_field(run, "num_generated_tokens")
        return Z, names, gids, [str(topic.get(g)) for g in gids], \
            np.array([length.get(g) or 0 for g in gids], dtype=float), np.full(len(gids), y)

    Zc, names, _, tc, lc, yc = load(args.cold_run, 0)
    Zh, names_h, _, th, lh, yh = load(args.hot_run, 1)
    assert list(names) == list(names_h), "feature-name fork between t03/t09"
    Z = np.vstack([Zc, Zh]); y = np.r_[yc, yh]
    topics = np.array(tc + th); length = np.r_[lc, lh]
    logger.info(f"t03 n={len(yc)} t09 n={len(yh)}; {Z.shape[1]} features")

    fm = FeatureMap(list(names), mm.n_layers)
    masks = {
        "nontrivial": np.any([fm.mask(source=s) for s in NONTRIVIAL], axis=0),
        "whole_vector": np.ones(len(names), dtype=bool),
        "trivial_output_residual": np.any([fm.mask(source=s) for s in TRIVIAL], axis=0),
    }
    per_source = {s.value: int(fm.mask(source=s).sum()) for s in NONTRIVIAL}
    logger.info(f"nontrivial mask = {int(masks['nontrivial'].sum())} feats {per_source}")

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    def oof_auc(Xm: F32, resid: bool) -> float:
        scores = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(Xm, y, topics):
            Xf = _length_residualize(Xm, length, tr) if resid else Xm
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xf[tr], y[tr])
            scores[te] = clf.decision_function(Xf[te])
        return float(roc_auc_score(y, scores))

    results = {}
    for name, m in masks.items():
        Xm = Z[:, m]
        results[name] = {"n_feats": int(m.sum()),
                         "auc_lengthresid": round(oof_auc(Xm, True), 4),
                         "auc_raw": round(oof_auc(Xm, False), 4)}
        logger.info(f"[{name}] AUC len-resid {results[name]['auc_lengthresid']} "
                    f"(raw {results[name]['auc_raw']})")

    # ── FROZEN AXIS: fit on full nontrivial (length-resid globally) — banked before any steered gen ──
    m = masks["nontrivial"]
    Xm = _length_residualize(Z[:, m], length, np.arange(len(y)))
    axis_clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xm, y)
    axis = axis_clf.coef_[0].astype(np.float64)
    axis /= np.linalg.norm(axis)
    feat_idx = np.where(m)[0]
    np.savez(args.out_dir / f"c2_orphaned_axis_{args.model}.npz",
             axis=axis.astype(np.float32), feature_indices=feat_idx,
             feature_names=np.array([names[i] for i in feat_idx]),
             med=med, scale=scale)

    out = {"model": args.model, "arm": "C2-orphaned-coordinate",
           "STATUS": "FIRST_READ_PENDING (C§8) — frozen BEFORE any steered generation (C3 not run)",
           "law": "t03-vs-t09 LDA on non-trivial surfaces {attention,gate,keys,qk}; C§1 excludes "
                  "{output,residual}; GroupKFold-by-topic; length-residualization default-on (fold-safe); "
                  "masking PRIMARY, likelihood-residualization robustness column OWED (needs per-token surprisal)",
           "n_cold": int(len(yc)), "n_hot": int(len(yh)),
           "nontrivial_source_feats": per_source,
           "discriminant_auc": results,
           "frozen_axis_npz": f"c2_orphaned_axis_{args.model}.npz",
           "robustness_column_note": "likelihood-residualization NOT run — t03/t09 signatures do not "
                                     "carry per-token surprisal; owed as robustness col (never load-bearing)"}
    (args.out_dir / f"c2_orphaned_coordinate_{args.model}.json").write_text(json.dumps(out, indent=1))
    logger.info(f"banked frozen axis + readout -> {args.out_dir}")


if __name__ == "__main__":
    main()
