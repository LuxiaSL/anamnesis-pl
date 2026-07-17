"""fast_lda — the shared LDA constructor for NEW confirmatory cells (backlog #4).

Why this exists: ``LinearDiscriminantAnalysis(shrinkage="auto")`` runs Ledoit-Wolf
covariance estimation PER FIT — a measured multi-minute crawl at 3,358 dims on big
grids (22 analyzer files still carry it). The vmb_arm_a4_analyze.py:98 fix
(``solver="lsqr", shrinkage=0.1``) is the constructor of record for new work; this
helper is that fix as one importable, declared choice.

⚠ THE LAW (Luxia, canonical-ops 2026-07-16; same principle as the analyzer-freeze
rule): auto→0.1 CHANGES NUMBERS. This helper is for cells DECLARED AT AUTHORING
time — NEVER retro-swap it into an already-scored cell's analyzer; the 22 existing
``shrinkage="auto"`` sites are scored records and stay untouched.

Usage:
    from anamnesis.analysis.fast_lda import fast_lda
    clf = fast_lda().fit(X[tr], y[tr])          # default shrinkage 0.1
    clf = fast_lda(shrinkage=0.2)               # declare a different value at authoring
"""
from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def fast_lda(shrinkage: float = 0.1) -> LinearDiscriminantAnalysis:
    """LDA with lsqr + FIXED shrinkage (no per-fit Ledoit-Wolf).

    Parameters
    ----------
    shrinkage : float
        Fixed shrinkage intensity in [0, 1]. 0.1 is the a4-analyzer value of
        record. Passing a float (never "auto") is the point of this helper.
    """
    if not (isinstance(shrinkage, float) and 0.0 <= shrinkage <= 1.0):
        raise ValueError(
            f"fast_lda wants a FIXED float shrinkage in [0,1], got {shrinkage!r} "
            f"— 'auto' (Ledoit-Wolf per fit) is exactly what this helper replaces"
        )
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)
