"""ANNEX A5 — THE SIX-SURFACE RAW SPECTRUM: which substrate carries nature's dominant axes?

⚠ ANNEX LANE: nothing here is quotable until it graduates through a frozen prereg cell.
Ledger: outputs/battery/annex/ANNEX-LEDGER.md

STANDALONE BY DESIGN — numpy + sklearn only, no `anamnesis` import. It runs ON node1 against
/models/kotodama-data/rider_day2 (11.8 GB of raw), so nothing is pulled local and nothing is
written outside the caller's own output path.

    ssh node1
    source ~/luxi-files/.venv-shared/bin/activate     # system python3 has NO numpy
    OMP_NUM_THREADS=32 python annex_rawspectrum.py --out ~/luxi-files/annex_raw_spectrum.json

═══ WHY THIS IS *NOT* THE "RAW vs HAND" COMPARISON THE BATON ASKED FOR ═══

The baton (§5 item 2) framed A5 as raw-vs-hand: "is the RAW substrate's natural spectrum also
trivial-channel-dominated, or was the hand vector the limiting factor?" **That comparison is
CONFOUNDED and cannot be run as specified.** `taste_replay._pool_raw` is `.mean(0)` / `.mean(1)`
over producing positions — the raw P is TOKEN-MEAN-POOLED, while the hand vector is 2nd-order
(entropies, drifts, std, band energies). So raw-vs-hand varies SUBSTRATE and AGGREGATION at once,
and this program's own NO-MEANS law says aggregation is precisely the thing that matters. A
difference between the two spectra could not be attributed to either cause.

**What IS clean: the six surfaces are pooled IDENTICALLY.** So the CROSS-SURFACE comparison
inside raw has the aggregation confound cancelled by construction. The question becomes the
unsupervised twin of the source ranking (attention ≫ residual > gate > keys > output):

    WHICH SUBSTRATE CARRIES NATURE'S DOMINANT AXES?

And the sharp part: **the raw surface has NO logit/output block at all.** The venue's winner
("nature's top three axes are output-distribution confidence") CANNOT reproduce here — the
channel structurally does not exist. That is the cleanest available form of "was the hand vector
the limiting factor": remove the channel that won, and see what wins instead.

═══ ⚠ THE MEAN-POOLING HETEROSCEDASTICITY TRAP — session 1's artifact, structural this time ═══

`_pool_raw` divides by the number of producing positions, so var(pooled) ∝ 1/glen: SHORT
generations have systematically HIGHER pooled variance. That is EXACTLY the artifact class that
produced session 1's retracted PC1 (the 512-token cap) — a scale effect that location-based
audits certify as clean — except here it is not incidental, it is baked into the pooling
operator. Session 1's lesson, verbatim:

    Residualizing a covariate removes its effect on the MEAN and leaves its effect on the
    VARIANCE. Audit location AND scale, or the audit is decorative.

So this rung audits BOTH `corr(score, glen)` and `corr(|score|, glen)` on every axis, and offers
`--glen-band` as the artifact-controlled variant of record (the analog of `capped_only`).
Restricting to one length regime is what removes a scale effect; residualizing cannot.

═══ NAMING — the bridge, and the one durable job of the hand vector ═══

Raw dimensions have no semantic names (`resid_L14_dim_2001` means nothing). Two nameable reads:
  1. DEPTH PROFILE — which of the 7 sampled layers carries the axis's mass (the `depth` axis of
     source×method×depth, read directly off the block structure).
  2. THE HAND BRIDGE — correlate the raw axis's score against the 2,252 NAMED hand features on
     the same generations (matched by key). This is the hand vector doing the job the June
     encoder work found is its durable one: **interpretive VOCABULARY, not a competing
     extractor.** Structure coefficients (corr(feature, score)), never raw loadings.

Slice map verified against kotodama `v3_raw_decomp.py:57-64` (itself checked against
configs/model.yaml: hidden 3072, 24 q heads, 8 kv heads, head_dim 128, ffn 8192,
SAMPLED_LAYERS [0,7,14,18,21,24,27], DD3B_BOUNDARIES [0,1,3,7,15,19,24]).
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.utils.extmath import randomized_svd

logger = logging.getLogger(__name__)
F32 = NDArray[np.float32]

RAW_GLOB = "/models/kotodama-data/rider_day2/v3_raw_*.npz"
SIG_GLOB = "/models/kotodama-data/rider_day2/sig_cache16_*.npz"
RAW_DIMS = 136192
SIG_DIMS = 2252
SL = [0, 7, 14, 18, 21, 24, 27]          # sampled layers
BD = [0, 1, 3, 7, 15, 19, 24]            # AttnRes committed boundaries

# (start, end, per-layer width, layer labels) — six substrates, all pooled identically
SURFACES: dict[str, tuple[int, int, int, list[int]]] = {
    "residual":  (0, 21504, 3072, SL),      # post-ffn, sampled layers
    "committed": (21504, 43008, 3072, BD),  # AttnRes committed streams
    "keys":      (43008, 50176, 1024, SL),  # 8 kv heads x 128
    "values":    (50176, 57344, 1024, SL),  # 8 kv heads x 128
    "queries":   (57344, 78848, 3072, SL),  # 24 q heads x 128
    "gate":      (78848, 136192, 8192, SL),  # SwiGLU
}


def robust_scale(X: F32) -> tuple[F32, F32]:
    """Floor-z, matching `anamnesis.analysis.battery.floors.robust_scale`:
    scale = max(MAD*1.4826, 0.05*std). The 0.05*std guard stops near-constant features from
    exploding; session 1 verified it binds for only a handful of features and is not the
    pathology behind heavy tails."""
    med = np.median(X, axis=0).astype(np.float32)
    mad = np.median(np.abs(X - med), axis=0).astype(np.float32) * 1.4826
    std = X.std(axis=0).astype(np.float32)
    return med, np.maximum(np.maximum(mad, 0.05 * std), 1e-6).astype(np.float32)


def load_raw(pattern: str) -> tuple[F32, list[str], NDArray[np.int32], NDArray[np.bool_]]:
    """Load + DEDUP-BY-KEY + shape-filter the v3_raw shards.

    ⚠ THE DEDUP IS MANDATORY HERE (unlike base_elicit, which was audited clean). kotodama's own
    RESULTS-curator-v3-2026-07-05.md: "sig_cache16 contains 265 duplicate keys (multi-shard
    overlap; global-resume artifact) + 1 bad-shape row (2262-d) — dedup-by-key + shape-filter
    required on load". Duplicate rows are PERFECTLY correlated — precisely what a
    variance-maximizing method seizes on. Their "36,247 clean" count is post-dedup.

    ⚠ ALSO RETURNS THE FINITE-COLUMN MASK — fp16 OVERFLOW, measured 2026-07-15, not anticipated
    by the baton. The residual and committed surfaces contain ±inf: `max|finite|` = 65,280 sits
    exactly at fp16's 65,504 ceiling, so this is MASSIVE-ACTIVATION OVERFLOW, not corruption.
    It is tiny and structured — 9/21,504 residual dims (0.042%), 8/21,504 committed:
      · residual offset 4 overflows at L0 (100% of rows), L14 (0.05%), L27 (98.6%) — the same
        offset at every depth: the classic persistent outlier / attention-sink register.
      · the other 6 are L0-only embedding outliers (offsets 292, 716, 2071, 2591, 2610, 2671);
        `committed` mirrors that exact offset set at its first boundary.
    keys / values / queries / gate are perfectly finite (max|finite| 3.6 / 364 / 5.0 / 128).
    Excluding never-representable columns is the honest minimum: those dims are not measurable
    from this bank at all, so no spectrum may be read through them. It is NOT a claim that
    massive activations are uninteresting — only that this fp16 bank cannot represent them.
    ⚠ CROSS-PROJECT POINTER (kotodama owns this, not us): `deploy_encoder_residual.npz` scores
    `((pooled[:21504]-mean)/std) @ w` over this SAME overflowing block. The encoder demonstrably
    works, so their path must tolerate it — but it is worth their eyes. Pointer, not a fix.
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no shards matched {pattern}")
    rows: list[F32] = []
    keys: list[str] = []
    glens: list[int] = []
    seen: set[str] = set()
    n_dup = n_shape = 0
    finite = np.ones(RAW_DIMS, dtype=bool)             # AND-accumulated over the FULL corpus
    for i, f in enumerate(files):
        z = np.load(f, allow_pickle=True)
        P, ks, gl = z["P"], z["keys"], z["glen"]
        if P.shape[1] != RAW_DIMS:                     # shape filter
            n_shape += len(P)
            logger.warning(f"{Path(f).name}: dim {P.shape[1]} != {RAW_DIMS} — shard dropped")
            continue
        finite &= np.isfinite(P).all(axis=0)
        for j, k in enumerate(ks):
            ks_ = str(k)
            if ks_ in seen:                            # dedup-by-key, keep-first
                n_dup += 1
                continue
            seen.add(ks_)
            rows.append(P[j])
            keys.append(ks_)
            glens.append(int(gl[j]))
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(files)} shards, {len(rows)} rows kept", flush=True)
    X = np.stack(rows)
    print(f"  raw: {X.shape} | dropped {n_dup} duplicate keys, {n_shape} bad-shape rows")
    print(f"  fp16 overflow: {int((~finite).sum())}/{RAW_DIMS} columns non-finite somewhere "
          f"⇒ excluded (see load_raw docstring — massive activations, not corruption)")
    return X, keys, np.array(glens, dtype=np.int32), finite


def load_sigs(pattern: str) -> tuple[F32, list[str]]:
    """Hand signatures for the same keys — the NAMING VOCABULARY, not a competing extractor."""
    files = sorted(glob.glob(pattern))
    rows: list[F32] = []
    keys: list[str] = []
    seen: set[str] = set()
    n_dup = n_shape = 0
    for f in files:
        z = np.load(f, allow_pickle=True)
        X, ks = z["X"], z["keys"]
        if X.shape[1] != SIG_DIMS:
            n_shape += len(X)
            continue
        for j, k in enumerate(ks):
            ks_ = str(k)
            if ks_ in seen:
                n_dup += 1
                continue
            seen.add(ks_)
            rows.append(X[j])
            keys.append(ks_)
    print(f"  hand sigs: {len(rows)} rows | dropped {n_dup} dup keys, {n_shape} bad-shape")
    return (np.stack(rows).astype(np.float32) if rows else np.zeros((0, SIG_DIMS), np.float32)), keys


def cell_center(Z: F32, cell: NDArray[np.int64]) -> tuple[F32, int]:
    """Remove each (arm, prompt, turn) cell's mean ⇒ pure seed-side variation.
    Cells lie entirely within one arm, so this removes arm/prompt/turn/content by construction.
    df = n - n_cells.

    Sorted `add.reduceat` groupby, NOT a per-cell Python loop: at ~8.7k cells x 43k rows x 57k
    columns the naive loop is O(n_cells * n) boolean masks over a 10 GB array and does not finish.
    """
    order = np.argsort(cell, kind="stable")
    Zs = Z[order]
    cs = cell[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    counts = np.diff(np.r_[starts, len(cs)])
    means = np.add.reduceat(Zs, starts, axis=0) / counts[:, None]
    Zs -= np.repeat(means, counts, axis=0)
    out = np.empty_like(Z)
    out[order] = Zs
    return out, len(Z) - len(starts)


def eff_rank_gram(Xc: F32, rng: np.random.Generator, n_sub: int = 15000) -> float:
    """Participation ratio (Σλ)²/Σλ², exactly, via the Gram trick — no eigendecomposition.

    Σλ  = trace(Cov)  = ‖Xc‖_F² / df
    Σλ² = ‖Cov‖_F²    = ‖Xc Xcᵀ‖_F² / df²      (since ‖AᵀA‖_F = ‖AAᵀ‖_F)

    Row-subsampled to keep the n×n Gram tractable on a SHARED box. The SAME subsample is used
    for every surface, so the cross-surface comparison — the only thing this rung claims — is
    fair. Absolute eff_rank at n_sub is not the same statistic as at full n; do not compare it
    across corpora.
    """
    n = len(Xc)
    idx = rng.choice(n, size=min(n_sub, n), replace=False)
    A = Xc[idx]
    df = max(len(A) - 1, 1)
    G = A @ A.T
    return float((np.trace(G) / df) ** 2 / max((G ** 2).sum() / df ** 2, 1e-30))


def split_half_stability(Xc: F32, cell: NDArray[np.int64], k: int,
                         rng: np.random.Generator, n_splits: int = 6) -> list[float]:
    """|cos| of PC_j across random halves, split by whole CELL (never within a cell — that would
    leak the cell mean across the split). Greedy top-k matching, because PC IDENTITY is not an
    object under near-degenerate eigenvalues (session 1's soup-3e4 trap); the subspace is."""
    uc = np.unique(cell)
    out: list[list[float]] = [[] for _ in range(k)]
    for _ in range(n_splits):
        perm = rng.permutation(uc)
        a = set(perm[: len(perm) // 2].tolist())
        ma = np.array([c in a for c in cell])
        if ma.sum() < k + 2 or (~ma).sum() < k + 2:
            continue
        _, _, VA = randomized_svd(Xc[ma], n_components=k, random_state=0)
        _, _, VB = randomized_svd(Xc[~ma], n_components=k, random_state=0)
        M = np.abs(VA @ VB.T)
        for j in range(k):
            out[j].append(float(M[j].max()))          # greedy: best match anywhere in the other half
    return [round(float(np.median(v)), 4) if v else float("nan") for v in out]


def audit(score: F32, glen: NDArray[np.int32]) -> dict[str, float]:
    """LOCATION *and* SCALE. Session 1 certified a truncation artifact as clean by testing only
    location: corr(score, cap) = .012 ✅ while corr(|score|, cap) = .829 ❌.
    Here the scale channel is STRUCTURAL: var(mean-pooled) ∝ 1/glen."""
    s = (score - score.mean()) / max(score.std(), 1e-12)
    g = glen.astype(np.float64)
    return {
        "corr_score_glen": round(float(np.corrcoef(s, g)[0, 1]), 4),        # location
        "corr_absscore_glen": round(float(np.corrcoef(np.abs(s), g)[0, 1]), 4),  # scale ← the tell
        "kurtosis": round(float(((s - s.mean()) ** 4).mean() / max(s.var() ** 2, 1e-30)), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=10, help="components per surface")
    ap.add_argument("--glen-band", type=str, default=None,
                    help="MIN,MAX — the artifact-controlled variant (restricts the pooling "
                         "heteroscedasticity to one length regime). e.g. 150,400")
    ap.add_argument("--n-name", type=int, default=12, help="top named hand correlates per axis")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rng = np.random.default_rng(args.seed)

    print("=== LOAD (dedup-by-key + shape-filter + fp16-overflow mask) ===")
    P, keys, glen, finite = load_raw(RAW_GLOB)
    S, skeys = load_sigs(SIG_GLOB)

    band_note = "all generations (POOLING HETEROSCEDASTICITY UNCONTROLLED)"
    if args.glen_band:
        lo, hi = (int(x) for x in args.glen_band.split(","))
        m = (glen >= lo) & (glen <= hi)
        print(f"\n  glen band [{lo},{hi}]: {m.sum()}/{len(m)} rows kept "
              f"({100*m.mean():.1f}%) — the artifact-controlled variant")
        P, keys, glen = P[m], [k for k, x in zip(keys, m) if x], glen[m]
        band_note = f"glen restricted to [{lo},{hi}] — one length regime (the capped_only analog)"

    arm = np.array([k.split("|")[0] for k in keys])
    cell_key = np.array(["|".join(k.split("|")[:3]) for k in keys])   # arm|prompt|turn
    uc = {c: i for i, c in enumerate(sorted(set(cell_key.tolist())))}
    cell = np.array([uc[c] for c in cell_key], dtype=np.int64)
    print(f"\n  n={len(P)} | arms {sorted(set(arm.tolist()))} | cells={len(uc)} "
          f"| glen {glen.min()}-{glen.max()}")

    # hand-signature alignment for the naming bridge
    smap = {k: i for i, k in enumerate(skeys)}
    have = np.array([k in smap for k in keys])
    sidx = np.array([smap[k] for k in keys if k in smap], dtype=np.int64)
    print(f"  hand-sig match: {have.sum()}/{len(keys)} raw rows have a hand twin")

    results: dict[str, dict] = {}
    for name, (a, b, width, layers) in SURFACES.items():
        keep = finite[a:b]
        layer_of = (np.arange(b - a) // width)[keep]   # survives the overflow exclusion
        n_drop = int((~keep).sum())
        print(f"\n=== SURFACE: {name}  [{a}:{b}]  d={b-a}→{int(keep.sum())} "
              f"({len(layers)} x {width}; {n_drop} fp16-overflow dims excluded) ===", flush=True)
        X = P[:, a:b][:, keep].astype(np.float32)
        if not np.isfinite(X).all():                   # belt and braces: never analyse non-finite
            raise AssertionError(f"{name}: non-finite survived the overflow mask")
        med, sc = robust_scale(X)
        Z = (X - med) / sc
        Xc, df = cell_center(Z, cell)
        del X, Z

        total_var = float((Xc ** 2).sum() / max(df, 1))
        U, Sv, Vt = randomized_svd(Xc, n_components=args.k, random_state=args.seed)
        var = (Sv ** 2) / max(df, 1)
        var_ratio = var / max(total_var, 1e-30)      # ← relative to the FULL spectrum (the
        #                                              session-1 var_ratio bug, not repeated)
        er = eff_rank_gram(Xc, rng)
        stab = split_half_stability(Xc, cell, min(args.k, 5), rng)

        # depth profile: where each axis's loading mass lives across the sampled layers
        depth = []
        for j in range(min(3, args.k)):
            v = Vt[j] ** 2
            per = [round(float(v[layer_of == i].sum()), 4) for i in range(len(layers))]
            depth.append({"pc": j + 1, "layers": layers, "mass_per_layer": per,
                          "peak_layer": layers[int(np.argmax(per))]})

        axes = []
        for j in range(min(3, args.k)):
            score = (Xc @ Vt[j]).astype(np.float32)
            row = {"pc": j + 1, "var_ratio": round(float(var_ratio[j]), 4),
                   "split_half_abscos": stab[j] if j < len(stab) else None,
                   "audit": audit(score, glen)}
            # THE HAND BRIDGE — structure coefficients in the NAMED vocabulary
            if have.sum() > 100:
                sc_sub = score[have]
                H = S[sidx]
                Hc, _ = cell_center(H, cell[have])
                hs = (Hc - Hc.mean(0)) / np.maximum(Hc.std(0), 1e-9)
                z = (sc_sub - sc_sub.mean()) / max(sc_sub.std(), 1e-12)
                r = (hs * z[:, None]).mean(0)
                top = np.argsort(-np.abs(r))[: args.n_name]
                row["hand_correlates"] = [{"idx": int(t), "r": round(float(r[t]), 3)} for t in top]
                row["max_abs_hand_r"] = round(float(np.abs(r).max()), 3)
            axes.append(row)

        results[name] = {
            "dims_nominal": b - a, "dims_analysed": int(keep.sum()),
            "fp16_overflow_dims_excluded": n_drop,
            "n": int(len(Xc)), "df": int(df),
            "pc1_var_ratio": round(float(var_ratio[0]), 4),
            "topk_var_ratio": round(float(var_ratio.sum()), 4),
            "eff_rank_subsampled": round(er, 2),
            "split_half_abscos": stab,
            "depth_profile": depth,
            "axes": axes,
        }
        print(f"  PC1 var {var_ratio[0]:.4f} | top-{args.k} var {var_ratio.sum():.4f} | "
              f"eff_rank(sub) {er:.1f} | split-half {stab}")
        for d_ in depth:
            print(f"    PC{d_['pc']} depth: peak L{d_['peak_layer']}  mass {d_['mass_per_layer']}")
        for r_ in axes:
            au = r_["audit"]
            flag = "  ⚠ SCALE" if abs(au["corr_absscore_glen"]) > 0.3 else ""
            print(f"    PC{r_['pc']}: corr(score,glen) {au['corr_score_glen']:+.3f} | "
                  f"corr(|score|,glen) {au['corr_absscore_glen']:+.3f} | "
                  f"kurt {au['kurtosis']:.1f} | max|r| hand {r_.get('max_abs_hand_r')}{flag}")
        del Xc

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "ANNEX_RULE": "NOT QUOTABLE until graduated via a frozen prereg cell",
        "rung": "A5 — six-surface raw spectrum (kotodama rider_day2, DPO arms A1/A5)",
        "question": "which SUBSTRATE carries nature's dominant axes? (the unsupervised twin of "
                    "the source ranking attention >> residual > gate > keys > output)",
        "why_not_raw_vs_hand": "CONFOUNDED and not run: _pool_raw is .mean() over positions, so "
                               "raw is TOKEN-MEAN-POOLED while the hand vector is 2nd-order. "
                               "raw-vs-hand varies substrate AND aggregation at once, and the "
                               "NO-MEANS law says aggregation is what matters. The six surfaces "
                               "are pooled IDENTICALLY, so the CROSS-SURFACE contrast is clean.",
        "structural_absence": "the raw surface has NO logit/output block. The venue's winner "
                              "(output-distribution confidence at t4/t3/t2) CANNOT reproduce "
                              "here — the channel does not exist. That is the point.",
        "pooling_trap": "var(mean-pooled) ∝ 1/glen ⇒ short gens have higher pooled variance — "
                        "session 1's artifact class, STRUCTURAL here. Location audits cannot see "
                        "it; corr(|score|,glen) can, and --glen-band controls it.",
        "variant": band_note,
        "hygiene": "dedup-by-key (keep-first) + shape-filter — MANDATORY for rider_day2 per "
                   "kotodama RESULTS-curator-v3-2026-07-05.md (265 dup keys + 1 bad-shape row). "
                   "Distinct from base_elicit, which this session audited CLEAN (23,758/23,758 "
                   "unique keys, 0 dup, 4 byte-identical rows = 0.017%). Measured here: 7,250 "
                   "duplicate raw keys dropped (16.7% of the bank) landing on EXACTLY kotodama's "
                   "documented 36,247-clean count — an independent confirmation of the dedup. "
                   "PLUS: fp16-overflow columns excluded (see load_raw docstring).",
        "fp16_overflow": "residual + committed carry ±inf at the MASSIVE-ACTIVATION dims "
                         "(max|finite| 65,280 vs fp16's 65,504 ceiling). 9/21,504 residual dims, "
                         "8/21,504 committed; offset 4 recurs at L0/L14/L27 (the persistent "
                         "outlier register), the rest are L0-only embedding outliers. Excluded, "
                         "because a dim the bank cannot represent cannot carry a spectrum. This "
                         "is NOT a claim that massive activations are uninteresting — it is a "
                         "REPRESENTATION LIMIT OF THE BANK and it scopes every residual/committed "
                         "number here. keys/values/queries/gate are perfectly finite.",
        "naming": "raw dims are unnamed; two nameable reads — (1) DEPTH profile off the block "
                  "structure, (2) THE HAND BRIDGE: corr(raw axis score, named hand feature) on "
                  "matched keys. Structure coefficients, never raw loadings. This is the hand "
                  "vector's durable role: interpretive VOCABULARY, not a competing extractor.",
        "shortcuts": "eff_rank row-subsampled to 15k (SAME subsample every surface ⇒ the "
                     "cross-surface contrast is fair; the absolute value is not comparable across "
                     "corpora). Split-half at 6 splits. Cell-centering is fold-free. "
                     "hand_correlates carry feature INDICES — join to feature_names from "
                     "base_elicit/taste_signatures.npz (identical 2252 layout).",
        "surface_map_provenance": "kotodama v3_raw_decomp.py:57-64, verified vs configs/model.yaml",
        "surfaces": results,
    }, indent=1))
    print(f"\n  → {args.out}")


if __name__ == "__main__":
    main()
