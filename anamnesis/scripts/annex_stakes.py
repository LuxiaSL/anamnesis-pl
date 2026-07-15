"""ANNEX — A2 (LOAD-BEARING): where do our named stakes live in nature's basis?

⚠ ANNEX LANE: nothing here is quotable until it graduates through a frozen prereg cell.
Ledger: outputs/battery/annex/ANNEX-LEDGER.md — the A2 PREDICTION IS FROZEN THERE, written
before this script was run. Read it before reading any output.

A2 is load-bearing because of the 2026-07-15 reframe: the probe tests whether dir0's
read-write asymmetry is dir0-LOCAL or MAP-GENERAL. dir0's rank in the unsupervised spectrum
is the evidence.

  branch (i)  dir0 sits LOW  -> V4's failure is explained by low-variance write-surface;
                               steelman ALIVE (hand the GPU cell a top PC)
  branch (ii) dir0 in top-3  -> we already know V4 fails on a dominant on-manifold natural
                               axis; steelman DEAD ON ARRIVAL; the closed ladder hardens
  branch (iii) rank ~4-20    -> no clean read; report as such, do NOT retrofit a story

⚠⚠ TWO dir0 CONSTRUCTIONS EXIST IN THIS PROGRAM — a collision the decoder ring would warn
about if it listed it. Do not conflate:
  dir0_a5   = analogical<->contrastive PAIR LDA in floor-z (vmb_a5_frozen_directional.
              DIR0_PAIR). ***THIS is the coordinate V3/V4/V3sel-bare actually steered, and
              the one WAIS §5's asymmetry claim is about.*** PRIMARY for the frozen prediction.
  lda5_dir{0,1,2} = top-3 discriminants of the 5-WAY mode LDA (the factor-naming memo's
              "~3 how-axis directions"). Related, NOT identical. Secondary/diagnostic.
We report cos(dir0_a5, lda5_dir0) so the relationship is a measured number, not an assumption.

A2" (--remainder): project out EVERY known stake, then PCA what is left. WAIS §3's "the
remainder is not noise — it is unexplored territory" as one line of linear algebra. Kills the
third horn of a negative's ambiguity (structure nested in a subspace we never looked at).

    PYTHONPATH=pipeline python -m anamnesis.scripts.annex_stakes --corpus venue
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from anamnesis.analysis.battery.deltas import ConditionCorpus
from anamnesis.scripts.annex_corpus import (
    REPO, VENUE_DIR, AnnexCorpus, load_venue, prepare, residualize,
)
from anamnesis.scripts.annex_spectrum import Spectrum, pca

F32 = NDArray[np.float32]
OUT = REPO / "outputs/battery/annex"
DIR0_PAIR = ("analogical", "contrastive")
ALL_MODES = ["analogical", "contrastive", "dialectical", "linear", "socratic"]


def _unit(v: NDArray) -> F32:
    v = np.asarray(v, dtype=np.float64)
    return (v / max(np.linalg.norm(v), 1e-12)).astype(np.float32)


def _pure_corpora(med: F32, scale: F32) -> dict[str, ConditionCorpus]:
    """The 5 pure-mode corpora on the venue's frozen floor-z scale (the exact V3sel recipe:
    dir0 is calibrated ONCE from labels, in the Stage-0 z space)."""
    out = {}
    for m in ALL_MODES:
        d = REPO / "outputs/battery" / f"vmb_a2_3b_pure_{m}"
        md = json.loads((d / "metadata.json").read_text())
        for g in md["generations"]:                       # 14a §2 hereditary gate
            if str(g.get("condition", "")) != "standard":
                raise AssertionError(f"{d.name} gen {g['generation_id']}: not unsteered")
            if not str(g.get("mode", "")).startswith("pure_"):
                raise AssertionError(f"{d.name} gen {g['generation_id']}: mode not pure_*")
        out[m] = ConditionCorpus(d / "signatures_v3", d / "metadata.json", med, scale, m)
    return out


def build_stakes(c: AnnexCorpus, med: F32, scale: F32) -> dict[str, F32]:
    """Every known stake as a unit direction in the venue's floor-z feature space."""
    stakes: dict[str, F32] = {}
    pure = _pure_corpora(med, scale)

    def lda_pair(a: str, b: str) -> F32:
        X = np.vstack([pure[a].Z, pure[b].Z])
        y = np.r_[np.ones(len(pure[a].Z)), np.zeros(len(pure[b].Z))]
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)
        return _unit(clf.coef_[0])

    # THE steered coordinate (frozen A5 construction) — primary for the prediction.
    stakes["dir0_a5"] = lda_pair(*DIR0_PAIR)
    # Same contrast, NO whitening — the construction V3 actually is. The gap between these
    # two rows IS the whitening's contribution to dir0's spectral location.
    stakes["dir0_meandiff"] = _meandiff_direction(pure, *DIR0_PAIR)

    # The 5-way LDA's top-3 (factor-naming). eigen solver exposes scalings_ (the discriminant
    # directions). Naming these would need STRUCTURE coefficients, not raw weights (the
    # 2026-06-14 lesson) — but for LOCATION (which is all A2 asks) the direction is the object.
    Xa = np.vstack([pure[m].Z for m in ALL_MODES])
    ya = np.concatenate([[m] * len(pure[m].Z) for m in ALL_MODES])
    lda5 = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto",
                                      n_components=3).fit(Xa, ya)
    for i in range(3):
        stakes[f"lda5_dir{i}"] = _unit(lda5.scalings_[:, i])

    # Artifact class.
    stakes["length"] = _unit(_regress_direction(c.X, c.covariates["glen"]))
    stakes["cap"] = _unit(_regress_direction(c.X, c.covariates["cap"]))

    # Content factors: MANNER (template — a candidate, not a nuisance) and SUBJECT (topic).
    stakes["template"] = _unit(_multiclass_direction(c.X, c.factors["template"]))
    stakes["topic"] = _unit(_multiclass_direction(c.X, c.factors["topic"]))
    return stakes


def _regress_direction(X: F32, y: F32) -> F32:
    """The direction along which a continuous covariate moves the cloud (OLS slope vector).

    ⚠ KNOWN-SPURIOUS FOR OVERLAP ANALYSIS (caught 2026-07-15, annex): beta ∝ Cov(X, y) is
    VARIANCE-WEIGHTED, so it aligns with high-variance directions no matter what y is. A
    RANDOM covariate's beta scores cos ≈ 0.35 (p95 0.73) against PC1 of this corpus; the real
    glen beta scores 0.41, i.e. p = 0.38 — indistinguishable from noise. Its `overlap()` row
    is therefore MEANINGLESS unless read against `null_random_covariate_direction`. Kept only
    so the null can be shown next to it; never read the raw row.
    """
    yc = (y - y.mean()).astype(np.float64)
    Xc = X.astype(np.float64) - X.mean(axis=0)
    return (Xc.T @ yc) / max(float(yc @ yc), 1e-12)


def _meandiff_direction(pure: dict[str, ConditionCorpus], a: str, b: str) -> F32:
    """UNWHITENED diff-of-means — the construction V3 actually is.

    dir0_a5 is an LDA axis: Σ⁻¹(μa − μb). The Σ⁻¹ whitening suppresses high-variance
    directions and boosts low-variance ones BY CONSTRUCTION, so an LDA axis is tail-biased
    before it ever meets the data — which is exactly the confound that could make the frozen
    A2 prediction fire on an artifact. This is the same contrast WITHOUT the whitening, so the
    two can be compared and the whitening's contribution read off directly.

    (It is also the read/write mismatch in miniature: the READ coordinate dir0 is whitened;
    the WRITE vector V3 is a diff-of-means over residuals. Whether §B.5's "writable content
    in the low-variance tail" is partly the whitening putting it there is a HYPOTHESIS — see
    the ledger. Not claimed here.)
    """
    return _unit(pure[a].Z.mean(axis=0) - pure[b].Z.mean(axis=0))


def null_shuffled_lda(pure: dict[str, ConditionCorpus], a: str, b: str, sp: Spectrum,
                      *, n_draw: int = 40, seed: int = 0) -> list[dict]:
    """dir0's OWN construction on PERMUTED labels — the matched null for its rank.

    If a random-label LDA also lands at rank ~100 with ~0.5% top-10 mass, then dir0's rank is
    telling us about LDA, not about dir0, and the A2 table cannot be read.
    """
    rng = np.random.default_rng(seed)
    X = np.vstack([pure[a].Z, pure[b].Z])
    n_a = len(pure[a].Z)
    out = []
    for i in range(n_draw):
        y = np.zeros(len(X))
        y[rng.permutation(len(X))[:n_a]] = 1.0        # same class sizes, random membership
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)
        out.append(overlap(_unit(clf.coef_[0]), sp))
    return out


def null_random_direction(sp: Spectrum, d: int, *, n_draw: int = 200,
                          seed: int = 0) -> list[dict]:
    """A random unit direction's mass profile — the floor every stake must be read against.

    For random u in R^d and an r-dim PC basis, E[mass in top-k] ≈ k/d and E[captured_total]
    ≈ r/d. Without this row, "0.5% of mass in the top-10" is an uninterpretable number.
    """
    rng = np.random.default_rng(seed)
    return [overlap(_unit(rng.normal(size=d)), sp) for _ in range(n_draw)]


def _summarize_null(rows: list[dict]) -> dict:
    """Median + 5th/95th percentile of the null's overlap statistics."""
    def q(key, f=None):
        v = np.array([(f(r) if f else r[key]) for r in rows], dtype=np.float64)
        return {"median": round(float(np.median(v)), 4),
                "p05": round(float(np.percentile(v, 5)), 4),
                "p95": round(float(np.percentile(v, 95)), 4)}
    return {
        "n_draw": len(rows),
        "best_pc_rank": q("best_pc_rank"),
        "best_pc_abscos": q("best_pc_abscos"),
        "mass_top10": q(None, lambda r: r["mass_in_topk"]["top10"]),
        "mass_top50": q(None, lambda r: r["mass_in_topk"]["top50"]),
        "rank_for_50pct_mass": q("rank_for_50pct_mass"),
        "captured_total": q("captured_total"),
    }


def _multiclass_direction(X: F32, labels: NDArray) -> F32:
    """Top discriminant of a categorical factor (its single best axis in feature space)."""
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto",
                                     n_components=1).fit(X, labels)
    return lda.scalings_[:, 0]


def overlap(stake: F32, sp: Spectrum, ks=(1, 3, 5, 10, 20, 50)) -> dict:
    """Where a stake lives in the PC basis: mass in the top-k subspaces + its best-aligned
    component and that component's RANK (the number the frozen prediction turns on).

    `mass_topk` = ||P_topk u||^2, the fraction of the stake's (unit) length captured by the
    top-k PCs. `rank_50pct_mass` = how deep you must go to capture half of it.
    """
    u = _unit(stake)
    cs = sp.components @ u                       # [r] cos with each PC (components are unit)
    cum = np.cumsum(cs ** 2)
    best = int(np.argmax(np.abs(cs)))
    total = float(cum[-1])
    r50 = int(np.searchsorted(cum, 0.5 * max(total, 1e-12)) + 1)
    return {
        "best_pc_rank": best + 1,                       # 1-indexed: "dir0 is PC #n"
        "best_pc_abscos": round(float(abs(cs[best])), 4),
        "best_pc_var_share": round(float(sp.var_ratio[best]), 4),
        "mass_in_topk": {f"top{k}": round(float(cum[min(k, len(cum)) - 1]), 4) for k in ks},
        "rank_for_50pct_mass": r50,
        "captured_total": round(total, 4),   # <1 when the PC basis is rank-deficient (df < d)
    }


def project_out(X: F32, dirs: list[F32]) -> F32:
    """Remove the span of the given directions from every row (A2\")."""
    if not dirs:
        return X
    B = np.column_stack([_unit(d) for d in dirs]).astype(np.float64)
    Q, _ = np.linalg.qr(B)                       # orthonormal basis of the stake span
    Xd = X.astype(np.float64)
    return (Xd - (Xd @ Q) @ Q.T).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["venue", "venue2108"], default="venue",
                    help="power corpus has no dir0 (different model) — A2 is venue-only")
    ap.add_argument("--variants", default="raw,topic,cell")
    ap.add_argument("--remainder", action="store_true",
                    help='A2": project out every stake, then PCA the remainder')
    ap.add_argument("--n-null-random", type=int, default=200)
    ap.add_argument("--n-null-lda", type=int, default=40)
    args = ap.parse_args()

    shared = args.corpus == "venue2108"
    c = load_venue(shared_2108=shared)
    # Stakes are built in the FULL 3358-d floor-z space (that is where dir0 is defined), then
    # restricted to the analysis space — never rebuilt inside a truncated space.
    cfull = load_venue()
    from anamnesis.analysis.battery.deltas import load_floor_scale
    med, scale = load_floor_scale(VENUE_DIR / "signatures_v3")
    stakes_full = build_stakes(cfull, med, scale)
    if shared:
        idx = np.array([cfull.feature_names.index(n) for n in c.feature_names])
        stakes = {k: _unit(v[idx]) for k, v in stakes_full.items()}
    else:
        stakes = stakes_full

    print(f"[{c.name} d={c.d}] stakes: {list(stakes)}")
    print(f"  cos(dir0_a5, lda5_dir0) = "
          f"{abs(float(stakes['dir0_a5'] @ stakes['lda5_dir0'])):.4f}  "
          "(the two dir0 constructions — measured, not assumed)")
    for a in ("dir0_a5", "lda5_dir0", "template"):
        for b in ("length", "cap", "topic"):
            print(f"  cos({a}, {b}) = {abs(float(stakes[a] @ stakes[b])):.4f}")

    pure_for_null = _pure_corpora(med, scale)
    if shared:
        idx_s = np.array([cfull.feature_names.index(n) for n in c.feature_names])
        for k_ in pure_for_null:
            pure_for_null[k_].Z = np.ascontiguousarray(pure_for_null[k_].Z[:, idx_s])

    results = []
    for variant in args.variants.split(","):
        Xp, df = prepare(c, variant)
        sp = pca(Xp, df)
        row = {"variant": variant, "df": df, "n_components": int(len(sp.eigenvalues)),
               "stakes": {k: overlap(v, sp) for k, v in stakes.items()}}
        # ── MATCHED NULLS — without these the table above cannot be read at all ──
        row["nulls"] = {
            "random_direction": _summarize_null(
                null_random_direction(sp, c.d, n_draw=args.n_null_random)),
            "shuffled_label_lda": _summarize_null(
                null_shuffled_lda(pure_for_null, *DIR0_PAIR, sp, n_draw=args.n_null_lda)),
        }
        if args.remainder:
            # A2": drop the span of every stake, re-PCA. Length/cap already residualized by
            # prepare(); this additionally removes the mode/template/topic stakes.
            Xr = project_out(Xp, list(stakes.values()))
            spr = pca(Xr, df - len(stakes))
            row["remainder"] = {
                "n_dirs_removed": len(stakes),
                "variance_retained": round(float(spr.eigenvalues.sum()
                                                 / max(sp.eigenvalues.sum(), 1e-12)), 4),
                "effective_rank_pr": round(spr.effective_rank_pr, 2),
                "n_90pct_var": spr.n_90,
                "var_ratio_top10": [round(float(v), 4) for v in spr.var_ratio[:10]],
            }
        results.append(row)

        print(f"\n  === variant={variant} (df={df}, {len(sp.eigenvalues)} PCs) ===")
        print(f"  {'stake':14s} {'bestPC':>7s} {'|cos|':>7s} {'top10':>7s} {'top50':>7s} "
              f"{'r50%':>6s} {'captot':>7s}")
        for k, o in row["stakes"].items():
            print(f"  {k:14s} {o['best_pc_rank']:7d} {o['best_pc_abscos']:7.3f} "
                  f"{o['mass_in_topk']['top10']:7.4f} {o['mass_in_topk']['top50']:7.4f} "
                  f"{o['rank_for_50pct_mass']:6d} {o['captured_total']:7.3f}")
        print("  ── MATCHED NULLS (median [p05, p95]) — read every row above against these ──")
        for nm, nl in row["nulls"].items():
            print(f"  {nm:14s} {nl['best_pc_rank']['median']:7.0f} "
                  f"{nl['best_pc_abscos']['median']:7.3f} {nl['mass_top10']['median']:7.4f} "
                  f"{nl['mass_top50']['median']:7.4f} "
                  f"{nl['rank_for_50pct_mass']['median']:6.0f} "
                  f"{nl['captured_total']['median']:7.3f}")
            print(f"  {'':14s} {'':>7s} [{nl['best_pc_abscos']['p05']:.3f},"
                  f"{nl['best_pc_abscos']['p95']:.3f}] "
                  f"[{nl['mass_top10']['p05']:.4f},{nl['mass_top10']['p95']:.4f}] "
                  f"[{nl['mass_top50']['p05']:.4f},{nl['mass_top50']['p95']:.4f}] "
                  f"[{nl['rank_for_50pct_mass']['p05']:.0f},"
                  f"{nl['rank_for_50pct_mass']['p95']:.0f}]")
        if args.remainder:
            print(f"  remainder: var retained {row['remainder']['variance_retained']:.3f} "
                  f"eff_rank {row['remainder']['effective_rank_pr']:.1f} "
                  f"n90 {row['remainder']['n_90pct_var']}")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"annex_a2_stakes_{args.corpus}.json"
    p.write_text(json.dumps({
        "ANNEX_RULE": "NOT QUOTABLE until graduated via a frozen prereg cell",
        "rung": "A2", "corpus": args.corpus,
        "frozen_prediction": "see outputs/battery/annex/ANNEX-LEDGER.md — written before this run",
        "dir0_note": "dir0_a5 = the STEERED coordinate (analogical<->contrastive pair LDA, "
                     "vmb_a5_frozen_directional) = primary. lda5_dir* = 5-way LDA top-3 "
                     "(factor-naming) = secondary. NOT the same object.",
        "cos_dir0a5_lda5dir0": round(abs(float(stakes["dir0_a5"] @ stakes["lda5_dir0"])), 4),
        "results": results}, indent=1))
    print(f"\n  → {p}")


if __name__ == "__main__":
    main()
