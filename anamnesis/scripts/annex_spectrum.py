"""ANNEX — A1: the spectrum. PCA + Horn's + split-half stability + family-weighting controls.

⚠ ANNEX LANE: nothing here is quotable until it graduates through a frozen prereg cell.
Ledger: outputs/battery/annex/ANNEX-LEDGER.md

Answers spec question (1) "is the bare cloud low-rank?" — which CANNOT be read off a variance
curve when d > df (venue: 3358 vs 717), because a decaying curve is exactly what noise looks
like at that aspect ratio (Marchenko-Pastur). Hence Horn's parallel analysis, and hence the
insistence on df rather than n.

Three weightings, per the adopted upgrades — an axis surviving all three is not a census
artifact; an axis that exists ONLY under `raw` tells us which family's redundancy was
impersonating structure (itself a finding):
  raw       — every feature votes (the census-biased default)
  sqrt      — each feature down-weighted 1/sqrt(family size)
  onevote   — per-family first-PC representatives, then PCA over the ~11 representatives

    PYTHONPATH=pipeline python -m anamnesis.scripts.annex_spectrum --corpus venue
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from anamnesis.scripts.annex_corpus import (
    REPO, AnnexCorpus, load_power, load_venue, prepare,
)

F32 = NDArray[np.float32]
Weighting = Literal["raw", "sqrt", "onevote"]
Variant = Literal["raw", "topic", "cell"]
OUT = REPO / "outputs/battery/annex"


class Spectrum(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    eigenvalues: F32          # [r] variance along each component (descending)
    var_ratio: F32            # [r] fraction of total variance
    components: F32           # [r, d] unit row-vectors
    scores: F32               # [n, r]
    df: int
    d: int

    @property
    def effective_rank_pr(self) -> float:
        """Participation ratio (sum l)^2 / sum(l^2) — a smooth 'how many axes' with no cutoff."""
        e = self.eigenvalues.astype(np.float64)
        return float(e.sum() ** 2 / np.maximum((e ** 2).sum(), 1e-30))

    @property
    def n_90(self) -> int:
        return int(np.searchsorted(np.cumsum(self.var_ratio), 0.90) + 1)


def _svd(Xd: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """SVD with a robust fallback driver.

    numpy's default LAPACK driver (gesdd, divide-and-conquer) intermittently fails to
    converge on perfectly finite, well-scaled inputs — observed 2026-07-15 on the venue's
    tier3 block (400x1250, all finite, max|z|=7.4). gesvd is slower but does not have the
    bug. Verified it is NOT a data problem before adding this: no NaN/Inf anywhere in either
    corpus (checked pre- and post-hygiene).
    """
    try:
        return np.linalg.svd(Xd, full_matrices=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as _scipy_svd
        return _scipy_svd(Xd, full_matrices=False, lapack_driver="gesvd")


def _gram_eig(Xd: NDArray, *, want_vectors: bool) -> tuple[NDArray, NDArray | None]:
    """Singular values (and right singular vectors) via the SMALLER Gram matrix.

    Both corpora are lopsided (venue 800x3358, power 23758x2252), so a full SVD wastes most
    of its work. eigh on min(n,d)^2 is 20-40x faster and is what makes Horn's affordable at
    all. Trade-off named: the Gram squares the condition number, so the TAIL eigenvalues are
    less accurate than an SVD's — acceptable here because every question this rung asks is
    about the top spectrum and the effective rank, never the tail's precise values.
    """
    n, d = Xd.shape
    if n <= d:
        G = Xd @ Xd.T                                    # [n, n]
        w, U = np.linalg.eigh(G)
        w, U = w[::-1], U[:, ::-1]
        S = np.sqrt(np.maximum(w, 0.0))
        if not want_vectors:
            return S, None
        nz = S > (S.max() * 1e-10 if S.max() > 0 else 1)
        Vt = np.zeros((len(S), d))
        Vt[nz] = (U[:, nz].T @ Xd) / S[nz, None]
        return S, Vt
    C = Xd.T @ Xd                                        # [d, d]
    w, V = np.linalg.eigh(C)
    w, V = w[::-1], V[:, ::-1]
    S = np.sqrt(np.maximum(w, 0.0))
    return (S, V.T) if want_vectors else (S, None)


def singular_values(X: F32) -> NDArray:
    """Eigenvalues-only fast path (Horn's inner loop doesn't need components)."""
    try:
        S, _ = _gram_eig(np.asarray(X, dtype=np.float64), want_vectors=False)
        return S
    except np.linalg.LinAlgError:
        return _svd(np.asarray(X, dtype=np.float64))[1]


def pca(X: F32, df: int, k: int | None = None) -> Spectrum:
    """PCA of the (already centered) design matrix. Variance uses df, not n-1 —
    group-centering burns one dimension per group and the spectrum must be read against
    the residual df or every eigenvalue is biased low."""
    Xd = np.asarray(X, dtype=np.float64)
    try:
        S, Vt = _gram_eig(Xd, want_vectors=True)
    except np.linalg.LinAlgError:                        # Gram failed → dense SVD fallback
        _U, S, Vt = _svd(Xd)
    assert Vt is not None
    eig = (S ** 2) / max(df, 1)
    nz = S > (S.max() * 1e-10) if S.max() > 0 else np.zeros(len(S), bool)
    # var_ratio is ALWAYS relative to the FULL spectrum, computed BEFORE k-truncation.
    # (Bug caught 2026-07-15: normalising after truncation made `pca(..., k=6)` report
    # top-6-RELATIVE shares — annex_rhyme printed PC1 "var .349" when the true share was far
    # smaller. Components were always correct; only the reported share was wrong. A caller
    # asking for k components must still be told what fraction of the WHOLE cloud they are.)
    total = float(eig[nz].sum())
    keep = nz & (np.arange(len(S)) < k) if k is not None else nz
    eig, Vt = eig[keep], Vt[keep]
    return Spectrum(eigenvalues=eig.astype(np.float32),
                    var_ratio=(eig / max(total, 1e-30)).astype(np.float32),
                    components=Vt.astype(np.float32),
                    scores=(Xd @ Vt.T).astype(np.float32), df=df, d=int(X.shape[1]))


# ── weightings (the authoring-bias controls) ─────────────────────────────────────

def apply_weighting(X: F32, c: AnnexCorpus, how: Weighting) -> tuple[F32, list[str]]:
    """Return the design matrix under one family-weighting, plus its column labels."""
    if how == "raw":
        return X, list(c.feature_names)
    if how == "sqrt":
        w = np.ones(X.shape[1], dtype=np.float64)
        for f in c.families:
            w[f.start:f.end] = 1.0 / np.sqrt(max(f.size, 1))
        return (X * w).astype(np.float32), list(c.feature_names)
    if how == "onevote":
        # Each family contributes ONE column: its own first PC (unit-variance scored).
        # ~11 columns total, so no family can win by member count.
        cols, labels = [], []
        for f in c.families:
            blk = X[:, f.start:f.end]
            if blk.shape[1] == 0:
                continue
            sp = pca(blk, df=max(blk.shape[0] - 1, 1), k=1)
            s = sp.scores[:, 0]
            cols.append(s / max(float(s.std()), 1e-12))
            labels.append(f.name)
        return np.column_stack(cols).astype(np.float32), labels
    raise ValueError(f"unknown weighting {how!r}")


# ── Horn's parallel analysis ─────────────────────────────────────────────────────

def horn(X: F32, df: int, *, n_perm: int = 20, pct: float = 95.0,
         seed: int = 0) -> tuple[int, F32]:
    """Column-permutation null: how many components exceed the noise spectrum?

    Permuting each feature independently preserves marginals and destroys ALL inter-feature
    correlation. ⚠ NECESSARY, NOT SUFFICIENT (adopted upgrade #3): a redundant family's
    internal co-mode is real correlation, so a census artifact passes Horn's trivially. Read
    this ONLY alongside the weighting controls.
    """
    rng = np.random.default_rng(seed)
    real = (singular_values(X) ** 2) / max(df, 1)
    null = np.zeros((n_perm, len(real)), dtype=np.float64)
    Xd = np.asarray(X, dtype=np.float64)
    for i in range(n_perm):
        # permuted(axis=0) shuffles every column INDEPENDENTLY in one vectorized call —
        # the Python per-column shuffle it replaces was the whole cost of this rung.
        Xp = rng.permuted(Xd, axis=0)
        Xp -= Xp.mean(axis=0)          # re-center: permutation breaks group centering
        e = (singular_values(Xp.astype(np.float32)) ** 2) / max(df, 1)
        null[i, :min(len(e), len(real))] = e[:len(real)]
    thresh = np.percentile(null, pct, axis=0)
    n_real = int(np.argmax(real <= thresh)) if (real <= thresh).any() else len(real)
    return n_real, thresh.astype(np.float32)


# ── split-half axis stability (adopted upgrade #2, outer-loop add) ───────────────

def split_half_stability(c: AnnexCorpus, variant: Variant, how: Weighting, *, k: int = 10,
                         n_splits: int = 10, seed: int = 0) -> F32:
    """|cos| of each of the top-k PCs across random half-corpora. An axis that does not
    reproduce is not an axis, whatever its bimodality score says.

    Splits by whole CELL — never within one. For the cell-centered variant, rows inside a cell
    sum to zero by construction; tearing a cell across halves would leave each half with a
    fake nonzero cell mean and manufacture agreement. Each half re-derives its own centering
    and residualization from scratch.

    Matching is greedy by |cos| over the other half's top-k, so a PC that reappears at a
    DIFFERENT rank (near-degenerate eigenvalues rotate freely) still counts as stable — the
    subspace is what reproduces, not the ordering.
    """
    rng = np.random.default_rng(seed)
    cells = np.unique(c.cell)
    out = np.zeros((n_splits, k), dtype=np.float64)
    for s in range(n_splits):
        perm = rng.permutation(cells)
        a = np.isin(c.cell, perm[:len(perm) // 2])
        halves = []
        for m in (a, ~a):
            sub = c.subset(m)
            Xp, df = prepare(sub, variant)
            Xw, _ = apply_weighting(Xp, sub, how)
            halves.append(pca(Xw, df, k=k).components)
        kk = min(k, len(halves[0]), len(halves[1]))
        M = np.abs(halves[0][:kk] @ halves[1][:kk].T)     # [k, k] cross-half |cos|
        out[s, :kk] = M.max(axis=1)
    return out.mean(axis=0).astype(np.float32)


def run(c: AnnexCorpus, variant: Variant, how: Weighting, *, k: int = 10,
        n_perm: int = 20, n_splits: int = 10) -> dict:
    Xp, df = prepare(c, variant)
    Xw, labels = apply_weighting(Xp, c, how)
    sp = pca(Xw, df)
    n_horn, _ = horn(Xw, df, n_perm=n_perm)
    stab = split_half_stability(c, variant, how, k=min(k, len(sp.eigenvalues)),
                                n_splits=n_splits)
    return {
        "corpus": c.name, "variant": variant, "weighting": how,
        "n": c.n, "d_effective": int(Xw.shape[1]), "df": df,
        "df_over_d": round(df / max(Xw.shape[1], 1), 3),
        "effective_rank_pr": round(sp.effective_rank_pr, 2),
        "n_90pct_var": sp.n_90,
        "n_above_horn": n_horn,
        "var_ratio_top10": [round(float(v), 4) for v in sp.var_ratio[:10]],
        "cumvar_top10": [round(float(v), 4) for v in np.cumsum(sp.var_ratio)[:10]],
        "split_half_abscos_top10": [round(float(v), 3) for v in stab[:10]],
        "onevote_labels": labels if how == "onevote" else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["venue", "venue2108", "power", "venuecap"], required=True)
    ap.add_argument("--partner", default=None, help="power only: stratify to one weight-state")
    ap.add_argument("--variants", default="raw,topic,cell")
    ap.add_argument("--weightings", default="raw,sqrt,onevote")
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.corpus == "power":
        c = load_power(partner=args.partner)
    else:
        c = load_venue(shared_2108=args.corpus == "venue2108",
                       capped_only=args.corpus == "venuecap")
    print(c.describe(), flush=True)

    rows = []
    for variant in args.variants.split(","):
        for how in args.weightings.split(","):
            r = run(c, variant, how, n_perm=args.n_perm, n_splits=args.n_splits)
            rows.append(r)
            print(f"  {variant:5s}/{how:7s} df/d={r['df_over_d']:5.2f} "
                  f"eff_rank={r['effective_rank_pr']:6.2f} n90={r['n_90pct_var']:4d} "
                  f"horn={r['n_above_horn']:4d} pc1={r['var_ratio_top10'][0]:.3f} "
                  f"stab={r['split_half_abscos_top10'][:3]}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = args.corpus + (f"_{args.partner}" if args.partner else "") + (args.tag or "")
    p = OUT / f"annex_a1_spectrum_{stem}.json"
    p.write_text(json.dumps({
        "ANNEX_RULE": "NOT QUOTABLE until graduated via a frozen prereg cell",
        "rung": "A1", "corpus": args.corpus, "partner": args.partner,
        "notes": c.notes, "results": rows}, indent=1))
    print(f"  → {p}")


if __name__ == "__main__":
    main()
