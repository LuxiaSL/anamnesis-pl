"""ANNEX — THE GATE: you may not state a direction's spectral position without ITS OWN matched null.

★★ BEFORE BUILDING ON ANY MAIN-LANE ARTIFACT, READ THE ⛔ VERDICT SECTIONS OF THAT ARM'S FIRST-READ
RECORDS AND THE CAMPAIGN LOG FOR THE THREAD. That is where the VOIDS and SCOPINGS live. The annex
lost two hours twice over — re-deriving `arms/A5/a5_covariance_screen_3b.json` (already banked,
better, unread) and quoting **voided** V3selbare rows — on things those records had already settled.
This is reflex #1 with its missing half: **it is not the results doc, it is the VERDICT record.**

⚠ ANNEX LANE: nothing here is quotable until it graduates through a frozen prereg cell.
Ledger: outputs/battery/annex/ANNEX-LEDGER.md

────────────────────────────────────────────────────────────────────────────────────────────────
WHY THIS FILE EXISTS, AND WHY IT IS CODE INSTEAD OF A PARAGRAPH

Session 1 wrote the law:

    "Every stake direction carries a construction-induced spectral bias, and it points in DIFFERENT
     directions for different constructions — LDA/Σ⁻¹ biases toward the tail, diff-of-means biases
     toward the top. NO STAKE'S SPECTRAL RANK IS READABLE WITHOUT ITS OWN SHUFFLED-LABEL NULL."

It is the best-written sentence in the ledger. It is four sessions old. **It has never once failed
to catch us, because we keep not applying it.** It caught session 4 THREE TIMES IN ONE DAY, in three
different directions:
  · isotropic null vs a diff-of-means  → "V3 is top-heavy, so §B.5's tail claim is wrong"  (FALSE:
    against the correct Σ-null, V3 is TAIL-SHIFTED at the 0th percentile)
  · top-only statistic vs a full spectrum → "V4 ≈ random"  (FALSE: V4 is ANTI-manifold, 61% of its
    mass in the bottom quartile, Mahalanobis 636 vs random's 172)
  · no null at all vs six gradients     → "gradients of normalized gauges are tail-seeking"  (FALSE:
    4 of 6 are TOP-biased)

★ THE AUTOPSY'S FINDING (AUTOPSY-prediction-failures-2026-07-15.md), which is why this is a module:

    Reflexes that became TOOLING have a PERFECT record:
        annex_shape_audit.py  → caught the A4-power 8-row artifact
        annex_ladder_control.py's zero control → validated the MT bank; RATIFIED as addendum 14h
        the `winsor` arm in annex_ranked_spectrum.py → caught rank's variance-equalization inflation
    Reflexes that stayed PROSE have a PERFECT record of FAILING:
        "every claim needs its own matched null"     → failed 6x
        "do not report before the control lands"     → failed 3x
        "a refutation is a claim and needs a control" → written one morning, broken 3x by lunch

    A lesson must be REMEMBERED AT THE MOMENT OF TEMPTATION. A gate fires whether or not you
    remember. ⇒ Build the gate; do not write another lesson.

────────────────────────────────────────────────────────────────────────────────────────────────
★ THE DERIVATION THAT MAKES THE DIFF-OF-MEANS NULL FREE (session 4)

Under a random split, dmu = mean(A) - mean(B) with both samples drawn from the SAME distribution:

    dmu ~ N(0, Sigma * (1/n_a + 1/n_b))

⇒ **A LABEL-FREE DIFF-OF-MEANS IS EXACTLY A Σ-WEIGHTED RANDOM VECTOR.** So for a unit-normalized
draw, E[energy in the top-k eigendirections] = (sum of the top-k eigenvalues) / trace — an ANALYTIC
null that needs NO DATA, NO LABELS, and NO SHUFFLING. It is one line, and not having it is what
made "V3 is top-heavy" look like a finding instead of a construction.

    PYTHONPATH=pipeline python -m anamnesis.scripts.annex_null --self-test
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)
F64 = NDArray[np.float64]

# A construction MUST declare itself. There is no default, on purpose: the whole failure mode is
# asking "where does this vector sit?" without saying how it was built.
Construction = Literal["diff_of_means", "lda_whitened", "gradient", "isotropic_random"]

_NULL_AVAILABLE: dict[str, str] = {
    "diff_of_means": "ANALYTIC: dmu ~ N(0, cSigma) under a random split ⇒ E[top-k energy] = "
                     "top-k share of trace. Free, no data needed.",
    "isotropic_random": "ANALYTIC: E[top-k energy] = k/d.",
    "lda_whitened": "REQUIRES a shuffled-LABEL null (Sigma^-1 biases toward the tail BY "
                    "CONSTRUCTION). Not derivable from Sigma alone — you must refit the "
                    "construction on shuffled labels. NOT provided here.",
    "gradient": "NO KNOWN NULL. The spectral position of a gauge-gradient depends on the gauge, "
                "and session 4 measured 4-of-6 banked gradients TOP-biased and 1 (S_mass = V4) "
                "TAIL-biased at Mahalanobis 636 — i.e. the class has no characteristic position. "
                "Do not state a spectral position for a gradient.",
}


@dataclass(frozen=True)
class Spectrum:
    """Σ's eigendecomposition, descending. Build once, reuse."""
    evals: F64      # [d] descending
    evecs: F64      # [d, d] columns aligned to evals
    ridge: float = 0.0

    @staticmethod
    def from_npz(path, evals_key: str = "evals", evecs_key: str = "evecs",
                 ridge_key: str = "ridge") -> "Spectrum":
        """⚠ NEVER trust the stored order. `a5_sigma_L14_3b.npz` stores evals ASCENDING; sorting
        explicitly is the difference between reading the top of the spectrum and the bottom."""
        z = np.load(path)
        ev = np.asarray(z[evals_key], dtype=np.float64)
        U = np.asarray(z[evecs_key], dtype=np.float64)
        o = np.argsort(-ev)
        return Spectrum(evals=ev[o], evecs=U[:, o],
                        ridge=float(z[ridge_key]) if ridge_key in z else 0.0)

    @property
    def d(self) -> int:
        return int(len(self.evals))

    def energy_profile(self, v: F64) -> F64:
        """[d] squared projection of a unit v onto each eigendirection. Sums to 1."""
        v = np.asarray(v, dtype=np.float64)
        v = v / np.linalg.norm(v)
        return (self.evecs.T @ v) ** 2

    def mahalanobis(self, v: F64) -> float:
        v = np.asarray(v, dtype=np.float64)
        v = v / np.linalg.norm(v)
        return float(v @ (self.evecs @ ((self.evecs.T @ v) / (self.evals + self.ridge))))


def analytic_null_topk(sp: Spectrum, construction: Construction, k: int) -> float:
    """E[top-k energy] for the null of `construction`. Raises where no analytic null exists."""
    if construction == "diff_of_means":
        return float(sp.evals[:k].sum() / sp.evals.sum())      # ← the free null
    if construction == "isotropic_random":
        return float(k / sp.d)
    raise NotImplementedError(
        f"no analytic null for construction={construction!r}. {_NULL_AVAILABLE[construction]}")


def empirical_null_topk(sp: Spectrum, construction: Construction, k: int,
                        n_draws: int = 200, seed: int = 0) -> F64:
    """[n_draws] top-k energies under the construction's null. Gives a percentile, not just a mean."""
    rng = np.random.default_rng(seed)
    if construction == "diff_of_means":
        sd = np.sqrt(np.maximum(sp.evals, 0.0))
        draw = lambda: sd * rng.normal(size=sp.d)              # v ~ N(0, Σ) in the eigenbasis
    elif construction == "isotropic_random":
        draw = lambda: rng.normal(size=sp.d)
    else:
        raise NotImplementedError(
            f"no samplable null for construction={construction!r}. {_NULL_AVAILABLE[construction]}")
    out = np.empty(n_draws)
    for i in range(n_draws):
        c = draw()
        c = c / np.linalg.norm(c)
        out[i] = float((c[:k] ** 2).sum())                     # already in the eigenbasis
    return out


@dataclass(frozen=True)
class NullVerdict:
    construction: str
    k: int
    observed_topk: float
    null_mean: float
    null_p05: float
    null_p95: float
    percentile: float
    mahalanobis: float
    verdict: str

    def __str__(self) -> str:
        return (f"{self.construction:16s} top{self.k}={self.observed_topk:.4f}  "
                f"null={self.null_mean:.4f} [{self.null_p05:.4f},{self.null_p95:.4f}]  "
                f"pct={self.percentile:5.1f}%  mahal={self.mahalanobis:8.1f}  → {self.verdict}")


def assert_against_own_null(v: F64, construction: Construction, sp: Spectrum, *,
                            k: int = 256, n_draws: int = 200, seed: int = 0) -> NullVerdict:
    """★ THE GATE. Returns a direction's spectral position ONLY alongside its OWN matched null.

    `construction` is REQUIRED and has no default — that is the entire point. The failure this
    prevents is asking "where does this vector sit in the spectrum?" without declaring how it was
    built, and then reading a construction artifact as a finding.

    RAISES (never returns a number) when the construction has no matched null. A silent None that
    renders as a verdict is worse than a crash: it looks exactly like an answer. Session 4's rank
    basis printed "INCONCLUSIVE" for a DECISIVE result because a key-lookup miss fell through an
    `is not None` guard — the cosines were 0.826/0.204/0.210 the whole time.
    """
    if construction not in _NULL_AVAILABLE:
        raise ValueError(f"unknown construction {construction!r}; "
                         f"declare one of {sorted(_NULL_AVAILABLE)}")
    prof = sp.energy_profile(v)
    obs = float(prof[:k].sum())
    null = empirical_null_topk(sp, construction, k, n_draws=n_draws, seed=seed)  # may raise
    analytic = analytic_null_topk(sp, construction, k)
    pct = 100.0 * float((null < obs).mean())
    if pct < 5:
        verdict = f"BELOW its own null (p<.05) — {construction} does NOT explain this position"
    elif pct > 95:
        verdict = f"ABOVE its own null (p<.05) — {construction} does NOT explain this position"
    else:
        verdict = f"INSIDE its own null — UNREADABLE; the position IS the construction"
    return NullVerdict(construction=construction, k=k, observed_topk=obs,
                       null_mean=float(analytic), null_p05=float(np.percentile(null, 5)),
                       null_p95=float(np.percentile(null, 95)), percentile=pct,
                       mahalanobis=sp.mahalanobis(v), verdict=verdict)


def _self_test() -> None:
    """Reproduce session 4's table from banked artifacts. If this drifts, the gate is broken."""
    from anamnesis.scripts.annex_corpus import REPO
    sp = Spectrum.from_npz(REPO / "outputs/battery/arms/A5/a5_sigma_L14_3b.npz")
    V = np.load(REPO / "outputs/battery/a5_vectors_3b/a5_vectors.npz")
    print(f"Σ_L14: d={sp.d}  ridge={sp.ridge:.3g}  top-256 trace share={analytic_null_topk(sp,'diff_of_means',256):.4f}")
    print(f"\n★ THE GATE, on the banked bank (k=256):\n")
    for name, key, con in (("V3 (data, WORKS)", "V3_L14", "diff_of_means"),
                           ("V1 (data, WORKS)", "V1_L14", "diff_of_means"),
                           ("R1 (random)", "R1", "isotropic_random")):
        print(f"  {name:20s} {assert_against_own_null(V[key].astype(np.float64), con, sp)}")
    print(f"\n  ⛔ and the one the gate REFUSES to answer:")
    try:
        assert_against_own_null(V["V4_L14"].astype(np.float64), "gradient", sp)
        raise AssertionError("gate FAILED to refuse a gradient — this is a bug")
    except NotImplementedError as e:
        print(f"  V4 (formula)         RAISED, correctly:\n    {e}")
    print(f"\n  V4's Mahalanobis (always readable, no null needed): {sp.mahalanobis(V['V4_L14'].astype(np.float64)):.1f}"
          f"   vs random ~172, V3 59.4")
    print("\n  ⇒ session-4 expectation: V3 sits BELOW its own diff-of-means null (0th pct) — i.e.")
    print("    tail-shifted relative to its construction, which is §B.5's claim, CONFIRMED once")
    print("    the null is the RIGHT one. Against an ISOTROPIC null V3 looks top-heavy and §B.5")
    print("    looks wrong. Same vector, opposite readings, and only one of them is licensed.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true",
                    help="reproduce session 4's table from banked artifacts")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.self_test:
        _self_test()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
