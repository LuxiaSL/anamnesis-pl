"""ANNEX — A3: does any top axis show real separation, against nulls that actually bite?

⚠ ANNEX LANE: nothing here is quotable until it graduates through a frozen prereg cell.
Ledger: outputs/battery/annex/ANNEX-LEDGER.md

The spec's original null (random orthonormal axes in the full 3,358-d space) was replaced
during design review: random projections in high dimension are near-Gaussian by concentration
(Diaconis-Freedman), so EVERY structured axis "beats" that null and A3 becomes a rubber stamp.
The two replacements ask questions that can actually fail:

  NULL-A  subspace rotation — random directions inside the span of the top-k PCs, drawn
          WHITENED so no direction is privileged by variance. Asks: is PC_j special, or is
          any direction in the top subspace equally lumpy? (Unwhitened would be near-useless
          when PC1 dominates: a random direction in the raw subspace is ~PC1, so the null
          would be testing PC1 against itself.)
  NULL-B  parametric Gaussian — resample from a Gaussian with the SAME covariance spectrum
          at the same n/df, re-PCA, score its PCs. Asks: is the lumpiness more than finite-n
          PCA manufactures from a perfectly elliptical cloud?

Split-half stability (annex_spectrum) is the third leg: an axis that does not reproduce is
not an axis, whatever it scores here.

Statistics (primary first). All computed on the STANDARDIZED 1-d projection, so scale-free:
  bic_gain    BIC(1-component GMM) - BIC(2-component), per sample. >0 favours two modes.
  silhouette  2-means silhouette on the 1-d projection.
  bimod_coeff (skew^2 + 1) / (kurt + 3(n-1)^2/((n-2)(n-3))); >0.555 is the classic heuristic.
Absolute calibration does not matter — every statistic is read as a percentile against the
nulls, which is the whole point of having nulls.

    PYTHONPATH=pipeline python -m anamnesis.scripts.annex_separation --corpus venue
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from anamnesis.scripts.annex_corpus import REPO, load_power, load_venue, prepare
from anamnesis.scripts.annex_spectrum import Spectrum, apply_weighting, pca

F32 = NDArray[np.float32]
OUT = REPO / "outputs/battery/annex"


def _std1d(x: NDArray) -> NDArray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return (x - x.mean()) / max(x.std(), 1e-12)


def bic_gain(x: NDArray, seed: int = 0) -> float:
    """Per-sample BIC improvement of a 2-component GMM over 1. The primary statistic:
    directly interpretable as 'how much better do two modes fit than one'."""
    z = _std1d(x).reshape(-1, 1)
    g1 = GaussianMixture(1, random_state=seed).fit(z)
    g2 = GaussianMixture(2, random_state=seed, n_init=2).fit(z)
    return float((g1.bic(z) - g2.bic(z)) / len(z))


# silhouette_score materialises pairwise distances => O(n^2). At the venue's n=707 that is free
# (500k pairs); on the POWER corpus (n=23,758) it is 564M pairs PER CALL, and A3 makes ~450 calls
# across the nulls => hours, not minutes. This is why A3 had only ever run on the venue.
# sample_size estimates the SAME quantity with sampling noise, applied IDENTICALLY to the real
# axes and to both nulls — and every statistic here is read as a percentile against those nulls,
# so a consistent estimator is all the design requires. Exact below the cutoff, so the venue's
# banked numbers are untouched.
SILHOUETTE_EXACT_MAX_N = 5000
SILHOUETTE_SAMPLE = 4000


def silhouette_2means(x: NDArray, seed: int = 0) -> float:
    z = _std1d(x).reshape(-1, 1)
    lab = KMeans(2, n_init=4, random_state=seed).fit_predict(z)
    if len(set(lab.tolist())) < 2:
        return 0.0
    if len(z) <= SILHOUETTE_EXACT_MAX_N:
        return float(silhouette_score(z, lab))
    return float(silhouette_score(z, lab, sample_size=SILHOUETTE_SAMPLE, random_state=seed))


def bimod_coeff(x: NDArray) -> float:
    z = _std1d(x)
    n = len(z)
    if n < 4:
        return float("nan")
    g, k = float(skew(z)), float(kurtosis(z, fisher=True))
    return float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


STATS = {"bic_gain": bic_gain, "silhouette": silhouette_2means, "bimod_coeff": bimod_coeff}


def null_a_subspace(sp: Spectrum, k: int, n_draw: int, seed: int) -> dict[str, NDArray]:
    """Random WHITENED directions inside the top-k PC subspace."""
    rng = np.random.default_rng(seed)
    W = sp.scores[:, :k] / np.maximum(sp.scores[:, :k].std(axis=0), 1e-12)   # whiten
    out = {s: np.zeros(n_draw) for s in STATS}
    for i in range(n_draw):
        u = rng.normal(size=k)
        u /= np.linalg.norm(u)
        proj = W @ u
        for s, fn in STATS.items():
            out[s][i] = fn(proj, seed=i) if s != "bimod_coeff" else fn(proj)
    return out


def null_b_gaussian(sp: Spectrum, k: int, n_rows: int, n_draw: int,
                    seed: int) -> dict[str, NDArray]:
    """Elliptical Gaussian with the SAME covariance spectrum; re-PCA'd, its PCs scored.

    Built in the empirical PC basis (the adopted upgrade says exactly this) because the
    ambient covariance is rank-deficient whenever df < d — you cannot sample from it directly,
    but you can sample its nonzero spectrum, which is the part the null is about.
    """
    rng = np.random.default_rng(seed)
    sd = np.sqrt(np.maximum(sp.eigenvalues.astype(np.float64), 0.0))
    out = {s: np.zeros((n_draw, k)) for s in STATS}
    for i in range(n_draw):
        Y = rng.normal(size=(n_rows, len(sd))) * sd                # same spectrum, Gaussian
        Y -= Y.mean(axis=0)
        spy = pca(Y.astype(np.float32), df=max(n_rows - 1, 1), k=k)
        for j in range(min(k, spy.scores.shape[1])):
            for s, fn in STATS.items():
                out[s][i, j] = (fn(spy.scores[:, j], seed=i) if s != "bimod_coeff"
                                else fn(spy.scores[:, j]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["venue", "venue2108", "power", "venuecap"], required=True)
    ap.add_argument("--partner", default=None)
    ap.add_argument("--variant", default="cell")
    ap.add_argument("--weighting", default="raw")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-draw", type=int, default=200, help="NULL-A draws")
    ap.add_argument("--n-sim", type=int, default=40, help="NULL-B simulations")
    ap.add_argument("--transform", choices=["none", "rank", "winsor"], default="none",
                    help="feature basis. 'none' = floor-z as banked (what sessions 1-3 ran). "
                         "'rank' = per-feature normal scores. ★ On the RANK basis every marginal "
                         "is Gaussian BY CONSTRUCTION, so NULL-B (elliptical Gaussian with the "
                         "same covariance) becomes the sharpest possible question: is there "
                         "non-Gaussian JOINT structure once every marginal has been Gaussianised "
                         "— i.e. structure beyond the covariance? Marginal Gaussianity does NOT "
                         "imply joint Gaussianity, so this can genuinely fail.")
    args = ap.parse_args()

    c = (load_power(partner=args.partner) if args.corpus == "power"
         else load_venue(shared_2108=args.corpus == "venue2108",
                         capped_only=args.corpus == "venuecap"))
    if args.transform != "none":
        from anamnesis.scripts.annex_ranked_spectrum import rank_normal, winsorize
        fn = rank_normal if args.transform == "rank" else winsorize
        c = c.model_copy(update={"X": np.ascontiguousarray(fn(c.X))})
        print(f"  basis: {args.transform} (max|value| {float(np.abs(c.X).max()):.2f})")
    Xp, df = prepare(c, args.variant)
    Xw, _ = apply_weighting(Xp, c, args.weighting)
    sp = pca(Xw, df)
    k = min(args.k, len(sp.eigenvalues))

    real = {s: np.array([fn(sp.scores[:, j], seed=0) if s != "bimod_coeff"
                         else fn(sp.scores[:, j]) for j in range(k)])
            for s, fn in STATS.items()}
    na = null_a_subspace(sp, k, args.n_draw, seed=1)
    nb = null_b_gaussian(sp, k, c.n, args.n_sim, seed=2)

    rows = []
    for j in range(k):
        r = {"pc": j + 1, "var_ratio": round(float(sp.var_ratio[j]), 4)}
        for s in STATS:
            v = float(real[s][j])
            # NULL-A: one pooled distribution over the subspace (not rank-specific).
            pa = float((na[s] >= v).mean())
            # NULL-B: rank-matched — PC_j of real data vs PC_j of the Gaussian sims.
            pb = float((nb[s][:, j] >= v).mean())
            r[s] = {"value": round(v, 4),
                    "p_nullA_subspace": round(pa, 4),
                    "p_nullB_gaussian": round(pb, 4),
                    "nullB_median": round(float(np.median(nb[s][:, j])), 4)}
        rows.append(r)

    print(f"[{c.name}/{args.variant}/{args.weighting}] n={c.n} df={df} k={k} "
          f"(NULL-A {args.n_draw} draws, NULL-B {args.n_sim} sims)")
    print(f"  {'PC':>3s} {'var':>7s} | {'bic_gain':>9s} {'pA':>6s} {'pB':>6s} | "
          f"{'silh':>6s} {'pA':>6s} {'pB':>6s} | {'bimod':>6s} {'pA':>6s} {'pB':>6s}")
    for r in rows:
        b, s_, m = r["bic_gain"], r["silhouette"], r["bimod_coeff"]
        print(f"  {r['pc']:3d} {r['var_ratio']:7.4f} | {b['value']:9.3f} "
              f"{b['p_nullA_subspace']:6.3f} {b['p_nullB_gaussian']:6.3f} | "
              f"{s_['value']:6.3f} {s_['p_nullA_subspace']:6.3f} {s_['p_nullB_gaussian']:6.3f} | "
              f"{m['value']:6.3f} {m['p_nullA_subspace']:6.3f} {m['p_nullB_gaussian']:6.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    stem = args.corpus + (f"_{args.partner}" if args.partner else "")
    tsuf = "" if args.transform == "none" else f"_{args.transform}"
    p = OUT / f"annex_a3_separation_{stem}_{args.variant}_{args.weighting}{tsuf}.json"
    p.write_text(json.dumps({
        "ANNEX_RULE": "NOT QUOTABLE until graduated via a frozen prereg cell",
        "rung": "A3", "corpus": args.corpus, "partner": args.partner,
        "variant": args.variant, "weighting": args.weighting, "transform": args.transform,
        "n": c.n, "df": df, "n_draw_nullA": args.n_draw, "n_sim_nullB": args.n_sim,
        "null_note": "NULL-A = whitened random directions in the top-k PC subspace (pooled, "
                     "not rank-specific). NULL-B = rank-matched Gaussian with identical "
                     "covariance spectrum, re-PCA'd. p = fraction of null >= real.",
        "transform_note": (
            "basis = floor-z as banked (sessions 1-3)." if args.transform == "none" else
            f"basis = {args.transform}. ★ On the RANK basis every marginal is Gaussian BY "
            "CONSTRUCTION, so NULL-B (elliptical Gaussian, same covariance spectrum) asks the "
            "sharpest available question: is there non-Gaussian JOINT structure once every "
            "marginal has been Gaussianised — i.e. structure BEYOND the covariance? Marginal "
            "Gaussianity does not imply joint Gaussianity, so this can genuinely fail. A rank "
            "axis that beats NULL-B is not a magnitude artifact and not an elliptical-cloud "
            "artifact."),
        "silhouette_note": (
            f"n={c.n} > {SILHOUETTE_EXACT_MAX_N} => silhouette estimated on a "
            f"sample_size={SILHOUETTE_SAMPLE} subsample (it is O(n^2); ~450 calls at n=23,758 "
            "would take hours). Applied IDENTICALLY to the real axes and both nulls, and every "
            "statistic is read as a percentile against those nulls, so a consistent estimator is "
            "all the design needs. SHORTCUT, NAMED."
            if c.n > SILHOUETTE_EXACT_MAX_N else "silhouette exact (n below the sampling cutoff)"),
        "rows": rows}, indent=1))
    print(f"  → {p}")


if __name__ == "__main__":
    main()
