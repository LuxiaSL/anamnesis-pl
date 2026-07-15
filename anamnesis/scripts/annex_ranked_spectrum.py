"""ANNEX — THE RANK BASIS: does a top axis survive when 8 rows CANNOT own a principal component?

⚠ ANNEX LANE: nothing here is quotable until it graduates through a frozen prereg cell.
Ledger: outputs/battery/annex/ANNEX-LEDGER.md

WHY THIS IS NO LONGER A "REPRESENTATION NICETY" (promoted, session 2 → session 3). The spec asked
for a rank transform and session 1 skipped it as cosmetic. It is not cosmetic: it is **the
PROPHYLACTIC for the artifact class that has now bitten this annex TWICE**, and it would have
prevented BOTH outright.

  session 1: cell-PC1 (25% var, gate enrichment 10.83, split-half .988) = the 512-token CAP,
             kurtosis 14.1, minority 28/800.
  session 2: A4-power PC1/PC2 (58% of variance, attn_res enrichment 15.64) = **8-17 rows of
             23,758**, kurtosis **11,861**, middle-98% spanning 0.02 sd — with BOTH nuisance
             channels certifying clean.

A variance-maximizing method hands you the outliers as PC1 every time. A rank/normal-score
transform maps every feature to a Gaussian marginal, which **destroys an outlier row's ability to
own a component by sheer magnitude**. What survives that is an axis of ORDERING.

★ THE CONFOUND THIS RUNG MUST NOT WALK INTO — and the reason `winsor` exists.
A rank transform does **TWO** things at once: (a) it kills outliers, and (b) it forces every
feature to unit variance (⇒ PCA-on-correlation rather than PCA-on-covariance). If the top axes
change, (a) and (b) are confounded and the rung answers nothing. So three bases run, not two:

    raw    — floor-z as banked. THE ARTIFACT BASIS. What sessions 1-2 actually ran.
    rank   — per-feature rank → normal score (van der Waerden). Kills outliers AND equalizes scale.
    winsor — per-feature clip at ±WINSOR_SD. Kills outliers, PRESERVES relative scale.

  · rank ≈ winsor, both ≠ raw  ⇒ the artifact was OUTLIERS. The rank basis is doing what it
    claims and the answer is trustworthy.
  · rank ≠ winsor              ⇒ scale-equalization is doing the work, not outlier removal, and
    "the rank basis fixes it" would have been a false story. Report both; name the mechanism.

⚠ THE HONEST COST, stated up front and NOT argued away: rank-transforming DISCARDS genuine
magnitude information. It answers "what is the axis of ORDERING", not "what is the axis of
VARIANCE". These are different questions. Both bases get reported; neither is "the truth".

    PYTHONPATH=pipeline python -m anamnesis.scripts.annex_ranked_spectrum --corpus power --variant cell
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm, rankdata

from anamnesis.scripts.annex_corpus import REPO, load_power, load_venue, prepare
from anamnesis.scripts.annex_shape_audit import audit_axis
from anamnesis.scripts.annex_spectrum import pca

logger = logging.getLogger(__name__)
F32 = NDArray[np.float32]
OUT = REPO / "outputs/battery/annex"
WINSOR_SD = 5.0


def rank_normal(X: F32) -> F32:
    """Per-feature rank → normal score (van der Waerden): r/(n+1) through the Gaussian quantile.

    Ties get average ranks, so a constant feature maps to all-zeros (harmless, and it drops out of
    the covariance rather than exploding it). After this, EVERY feature is marginally Gaussian —
    the maximum |z| any row can reach is ~Φ⁻¹(n/(n+1)) ≈ 4 at n=23,758, versus the 129.6 the raw
    basis allows. That bound is the whole point: it is what an outlier row would need to own a PC.
    """
    n = X.shape[0]
    r = rankdata(X.astype(np.float64), axis=0, method="average")
    return norm.ppf(r / (n + 1.0)).astype(np.float32)


def winsorize(X: F32, sd: float = WINSOR_SD) -> F32:
    """Per-feature clip at ±sd ROBUST sd's of the feature's own centre.

    The scale-preserving twin of `rank_normal`: it removes outliers WITHOUT equalizing feature
    variances, so comparing the two separates 'the artifact was outliers' from 'the artifact was
    scale'. Uses median/MAD rather than mean/std because the outliers being clipped are exactly
    what would inflate a non-robust scale estimate and hide themselves.
    """
    Xd = X.astype(np.float64)
    med = np.median(Xd, axis=0)
    mad = np.median(np.abs(Xd - med), axis=0) * 1.4826
    scale = np.maximum(mad, 1e-9)
    return np.clip(Xd, med - sd * scale, med + sd * scale).astype(np.float32)


def subspace_cos(A: F32, B: F32) -> float:
    """Mean principal cosine between two top-k subspaces. Reads the SUBSPACE, never PC identity —
    near-degenerate eigenvalues rotate freely (session 1's soup-3e4 trap)."""
    s = np.linalg.svd(A @ B.T, compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["venue", "venuecap", "power"], default="power")
    ap.add_argument("--variant", default="cell")
    ap.add_argument("--partner", default=None)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    c = (load_power(partner=args.partner) if args.corpus == "power"
         else load_venue(capped_only=(args.corpus == "venuecap")))
    fam_of = c.family_of_index()
    covs: dict[str, NDArray] = dict(c.covariates)
    for k, v in c.factors.items():
        if k != "conv":
            covs[k] = v

    # ⚠ MEMORY: this corpus is 23,758 x 2,252 and `prepare` makes float64 copies (~430 MB each).
    # Holding all three bases at once OOM-killed the first run on a 38 GB box already at 13 GB.
    # Build each basis INSIDE the loop and drop it before the next — only the k x d components
    # (6 x 2,252, negligible) are carried across iterations.
    BUILD = {"raw": lambda X: X, "rank": rank_normal, "winsor": winsorize}
    print(f"corpus={args.corpus}/{args.variant}  n={c.n} d={c.d}  bases={list(BUILD)}")

    # ⚠ The outlier-headroom diagnostic must NOT divide by MAD: this corpus contains features that
    # are CONSTANT over >50% of rows, so their MAD is exactly 0 and a MAD-scaled z is meaningless
    # (the first run printed max|z| = 2.9e12 for `raw` and 4.2e9 for `rank` — an artifact of the
    # diagnostic, not the data; `rank` cannot exceed ~4 by construction). Report the honest thing:
    # the largest standardized value each basis ALLOWS, plus how many features are degenerate.
    mad0 = int((np.median(np.abs(c.X - np.median(c.X, axis=0)), axis=0) == 0).sum())
    print(f"⚠ features with MAD == 0 (constant over >50% of rows): {mad0}/{c.d} "
          f"— a MAD-scaled z is undefined for these; do not report one.")
    print("max |value| per basis (the outlier headroom each basis ALLOWS):")
    for name, fn in BUILD.items():
        Xb = fn(c.X)
        print(f"   {name:8s} max|value| = {float(np.abs(Xb).max()):12.2f}")
        del Xb

    results: dict[str, dict] = {}
    comps: dict[str, F32] = {}
    for name, fn in BUILD.items():
        cb = c.model_copy(update={"X": np.ascontiguousarray(fn(c.X))})
        Xp, df = prepare(cb, args.variant)
        del cb
        sp = pca(Xp, df, k=args.k)
        print(f"\n{'='*78}\n=== BASIS: {name}   (df={df})\n{'='*78}")
        comps[name] = sp.components[:args.k]
        axes = {}
        for j in range(args.k):
            score = (Xp @ sp.components[j]).astype(np.float32)
            a = audit_axis(score, covs)
            a["var_ratio"] = round(float(sp.var_ratio[j]), 4)
            # family enrichment: share of the axis's squared mass vs the family's share of d
            w2 = sp.components[j] ** 2
            enr = {}
            for f in c.families:
                share = float(w2[f.start:f.end].sum())
                enr[f.name] = round(share / max((f.end - f.start) / c.d, 1e-12), 2)
            top = sorted(enr.items(), key=lambda x: -x[1])[:3]
            a["family_enrichment_top3"] = dict(top)
            # structure coefficients: corr(feature, score) — never raw loadings
            sc = [(c.feature_names[i], float(np.corrcoef(Xp[:, i], score)[0, 1]))
                  for i in np.argsort(-np.abs(sp.components[j]))[:4]]
            a["structure_coefficients_top4"] = {k2: round(v, 3) for k2, v in sc}
            axes[f"PC{j+1}"] = a
            print(f"  PC{j+1}  var {a['var_ratio']:.4f}  kurt {a['kurtosis']:9.2f}  "
                  f"minority {a['minority_frac']*100:5.2f}%  span {a['p1_p99_span_sd']:6.2f} sd  "
                  f"| {top[0][0]} x{top[0][1]}")
            if a["flags"]:
                print(f"        ⚠ {'; '.join(a['flags'])}")
        results[name] = {"df": int(df), "axes": axes}
        del Xp, sp

    # ── ★ THE DECISIVE CONTRAST: does the top subspace MOVE, and which mechanism moved it? ──
    print(f"\n{'='*78}\n=== ★ TOP-{args.k} SUBSPACE AGREEMENT BETWEEN BASES (mean principal cosine)\n{'='*78}")
    pairs = {}
    for a in bases:
        for b in bases:
            if a < b:
                pairs[f"{a}_vs_{b}"] = round(subspace_cos(comps[a], comps[b]), 4)
                print(f"  {a:8s} vs {b:8s}   {pairs[f'{a}_vs_{b}']:.4f}")
    def _pair(a: str, b: str) -> float:
        """Order-free lookup. ⚠ The keys are built with `if a < b`, which is a STRING compare:
        'rank' < 'raw' ('n' < 'w'), so the key is `rank_vs_raw`, NOT `raw_vs_rank`. Asking for the
        wrong one returned None, and the `is not None` guard then rendered a DECISIVE result as
        'INCONCLUSIVE' — a silent miss that looks exactly like an answer. Never let a missing key
        reach a verdict: raise."""
        v = pairs.get(f"{a}_vs_{b}", pairs.get(f"{b}_vs_{a}"))
        if v is None:
            raise KeyError(f"no subspace cosine for ({a}, {b}); have {sorted(pairs)}")
        return v

    rw, rr, ww = _pair("rank", "winsor"), _pair("raw", "rank"), _pair("raw", "winsor")
    verdict = "INCONCLUSIVE"
    if True:
        if rw > 0.8 and rr < 0.6:
            verdict = ("OUTLIERS were the artifact — rank and winsor agree with each other and "
                       "BOTH depart from raw. The rank basis is removing outliers, not merely "
                       "rescaling, and its answer is trustworthy.")
        elif rw < 0.6:
            verdict = ("SCALE-EQUALIZATION is doing the work, NOT outlier removal — rank and "
                       "winsor DISAGREE. 'The rank basis fixes it' would have been a false story. "
                       "Report the mechanism, do not adopt rank as the default basis.")
        elif rr > 0.8:
            verdict = ("The top subspace is BASIS-INVARIANT — raw's axes are not outlier-driven "
                       "after all, or the transform did not bite. Re-read the shape audit.")
    print(f"\n  ⇒ {verdict}")

    p = args.out or OUT / f"annex_ranked_spectrum_{args.corpus}_{args.variant}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "ANNEX_RULE": "NOT QUOTABLE until graduated via a frozen prereg cell",
        "rung": "the rank basis — PROMOTED from 'representation nicety' to prophylactic",
        "why": "a variance-maximizing method hands you outliers as PC1 every time. This annex has "
               "been bitten twice (session 1 cap: kurtosis 14; session 2 A4-power: kurtosis "
               "11,861 = 8 rows of 23,758). A rank transform bounds any row at ~4 sd, so 8 rows "
               "CANNOT own a component.",
        "three_bases": {
            "raw": "floor-z as banked — THE ARTIFACT BASIS (what sessions 1-2 ran)",
            "rank": "per-feature rank -> normal score. Kills outliers AND equalizes scale.",
            "winsor": f"per-feature clip at +-{WINSOR_SD} robust sd. Kills outliers, PRESERVES scale.",
        },
        "why_winsor_exists": "a rank transform confounds outlier-removal with scale-equalization. "
                             "winsor isolates the first. rank≈winsor≠raw ⇒ outliers were the "
                             "artifact; rank≠winsor ⇒ scale did the work and the rank story is "
                             "false. Without this control the rung answers nothing.",
        "honest_cost": "rank-transforming DISCARDS genuine magnitude information: it answers 'what "
                       "is the axis of ORDERING', not 'of VARIANCE'. Different questions. Both "
                       "bases reported; neither is 'the truth'.",
        "corpus": args.corpus, "variant": args.variant, "partner": args.partner,
        "n": int(c.n), "d": int(c.d), "k": args.k,
        "subspace_agreement": pairs,
        "mechanism_verdict": verdict,
        "bases": results,
    }, indent=1))
    print(f"\n  → {p}")


if __name__ == "__main__":
    main()
