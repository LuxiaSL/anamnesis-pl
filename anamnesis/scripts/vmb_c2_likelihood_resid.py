"""C2 — likelihood-residualization robustness column (the owed col; session-11 Part A.1).

Closes the debt self-scoped in `vmb_c2_orphaned_coordinate.py` ("likelihood-residualization
NOT run — t03/t09 signatures do not carry per-token surprisal; owed as robustness col (never
load-bearing)"). That scope note was written before checking the aggregate grain: the v3
signature DOES bank `mean_surprise` = mean −log p(chosen token) under the model's own logits
over the generated span. For the t03/t09 corpora (no injection; temperature acts at the
sampler, logits untouched) this is exactly the base-model NLL of the generated text — A1's
likelihood ruler — so the covariate is banked and the column is CPU-computable.

Question the column answers: is the C2 orphaned-coordinate readability (t03-vs-t09 LDA on
non-trivial surfaces, banked AUC .678 len-resid) merely a linear read of text surprisal?
Method: identical to the banked C2 readout (same masks, GroupKFold-by-topic, fold-safe
per-feature OLS residualization with β fit on train rows), with the covariate design extended
from [1, length] to [1, length, mean_surprise] (and a surprisal-only variant reported).

Scope notes (stated in the artifact):
- `mean_surprise` is itself an output-source feature (idx in names: 'mean_surprise'), so it is
  ABSENT from the nontrivial mask by construction (C§1 excludes source:output) — the primary
  row is a clean covariate-vs-features design. In the whole_vector/trivial rows the covariate
  is also a column of X; residualizing a feature against itself zeroes it (self-consistent,
  noted, those rows are context only).
- This column does NOT extend to the C3 steered cells: under injection, chosen-prob-derived
  surprisal is surprisal under the STEERED model, not base-model NLL; the per-gen unsteered
  NLL for those cells was never banked (only cell means in c3_certifying_c_likelihood_auc).
  The steered-side likelihood question is already certified separately on cell means (14f
  likelihood-AUC item, INSIDE its filed P).

Self-scoped NON-LOAD-BEARING (the primary C2 claim rests on masking; this is descriptive
robustness per 14q item 7). CPU-only. First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

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
    out: dict[int, object] = {}
    for g in gens:
        gid = int(g.get("generation_id", g.get("gen_id", -1)))
        out[gid] = g.get(field)
    return out


def _residualize(X: F32, C: NDArray, train: NDArray) -> F32:
    """Per-feature OLS residualization against covariate design C ([n, k] incl. intercept);
    β fit on `train` rows, applied to all (the C2 fold-safe convention)."""
    Ct = C[train]
    beta, *_ = np.linalg.lstsq(Ct, X[train].astype(np.float64), rcond=None)
    return (X.astype(np.float64) - C @ beta).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--cold-run", type=Path, required=True, help="vmb_a1_3b_t03")
    ap.add_argument("--hot-run", type=Path, required=True, help="vmb_a1_3b_t09")
    ap.add_argument("--banked-json", type=Path, required=True,
                    help="c2_orphaned_coordinate_3b.json (echo comparison rows)")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--model", default="3b")
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    mm = MODEL_META[args.model]

    stage0 = args.battery_root / mm.stage0_dir
    med, scale = load_floor_scale(stage0 / "signatures_v3")

    def load(run: Path, y: int):
        X, names, gids = load_signature_matrix(run / "signatures_v3")
        Z = (X - med) / scale
        topic = _meta_field(run, "topic")
        length = _meta_field(run, "num_generated_tokens")
        return (X, Z, list(names), gids,
                [str(topic.get(g)) for g in gids],
                np.array([length.get(g) or 0 for g in gids], dtype=float),
                np.full(len(gids), y))

    Xc, Zc, names, _, tc, lc, yc = load(args.cold_run, 0)
    Xh, Zh, names_h, _, th, lh, yh = load(args.hot_run, 1)
    if names != names_h:
        raise SystemExit("feature-name fork between t03/t09")
    try:
        i_sur = names.index("mean_surprise")
    except ValueError as e:
        raise SystemExit("mean_surprise not in signature names — covariate not banked") from e

    Z = np.vstack([Zc, Zh])
    y = np.r_[yc, yh]
    topics = np.array(tc + th)
    length = np.r_[lc, lh]
    # covariate from RAW (un-z-scored) values — affine-equivalent for OLS, plainer to read
    surprisal = np.r_[Xc[:, i_sur], Xh[:, i_sur]].astype(np.float64)
    logger.info(f"t03 n={len(yc)} t09 n={len(yh)}; {Z.shape[1]} feats; "
                f"mean_surprise t03 {surprisal[y == 0].mean():.4f} vs t09 {surprisal[y == 1].mean():.4f}")

    fm = FeatureMap(names, mm.n_layers)
    masks = {
        "nontrivial": np.any([fm.mask(source=s) for s in NONTRIVIAL], axis=0),
        "whole_vector": np.ones(len(names), dtype=bool),
        "trivial_output_residual": np.any([fm.mask(source=s) for s in TRIVIAL], axis=0),
    }
    if masks["nontrivial"][i_sur]:
        raise SystemExit("mean_surprise leaked into the nontrivial mask — C§1 violated, refuse to run")

    ones = np.ones(len(y))
    designs = {
        "lengthresid": np.c_[ones, length],                       # banked row (echo check)
        "surprisalresid": np.c_[ones, surprisal],                 # the owed column, alone
        "length_plus_surprisal": np.c_[ones, length, surprisal],  # the owed column, on the standing base
    }

    def oof_auc(Xm: F32, C: NDArray | None) -> float:
        scores = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(Xm, y, topics):
            Xf = _residualize(Xm, C, tr) if C is not None else Xm
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xf[tr], y[tr])
            scores[te] = clf.decision_function(Xf[te])
        return float(roc_auc_score(y, scores))

    # covariate-alone strength (the confound's own ceiling on this split)
    sur_auc = oof_auc(surprisal.reshape(-1, 1).astype(np.float32), None)

    results: dict[str, dict] = {}
    for mname, m in masks.items():
        row = {"n_feats": int(m.sum()), "auc_raw": round(oof_auc(Z[:, m], None), 4)}
        for dname, C in designs.items():
            row[f"auc_{dname}"] = round(oof_auc(Z[:, m], C), 4)
        results[mname] = row
        logger.info(f"[{mname}] {row}")

    banked = json.loads(args.banked_json.read_text())["discriminant_auc"]
    echo = {k: {"banked_lengthresid": banked[k]["auc_lengthresid"],
                "recomputed_lengthresid": results[k]["auc_lengthresid"],
                "match": abs(banked[k]["auc_lengthresid"] - results[k]["auc_lengthresid"]) < 5e-4}
            for k in results}

    out = {
        "model": args.model,
        "arm": "C2 — likelihood-residualization robustness column (owed col, session-11 Part A.1)",
        "STATUS": "FIRST_READ_PENDING (C§8)",
        "law": "t03-vs-t09 LDA, same masks/folds as banked C2; per-feature OLS residualization "
               "fold-safe (β on train); covariate = mean_surprise (banked v3 output feature = "
               "base-model NLL of generated text on unsteered corpora); NON-LOAD-BEARING by the "
               "C2 script's own scope — descriptive robustness (14q item 7)",
        "covariate": {"name": "mean_surprise", "feature_index": int(i_sur),
                      "mean_t03": round(float(surprisal[y == 0].mean()), 4),
                      "mean_t09": round(float(surprisal[y == 1].mean()), 4),
                      "covariate_alone_auc": round(sur_auc, 4)},
        "n_cold": int(len(yc)), "n_hot": int(len(yh)),
        "echo_check_vs_banked": echo,
        "discriminant_auc": results,
        "scope_notes": [
            "mean_surprise is source:output → absent from nontrivial mask by construction; "
            "nontrivial row is the clean covariate-vs-features design",
            "whole_vector/trivial rows contain the covariate as a column of X; residualizing a "
            "feature against itself zeroes it (context rows only)",
            "does NOT extend to C3 steered cells: chosen-prob surprisal under injection is "
            "steered-model surprisal, not base NLL; per-gen unsteered NLL never banked there",
        ],
    }
    args.out_json.write_text(json.dumps(out, indent=1))
    logger.info(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
