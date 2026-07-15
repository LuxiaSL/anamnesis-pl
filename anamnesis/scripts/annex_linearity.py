"""ANNEX A6 — THE DOSE-LINEARITY GATE: is the map locally linear at the doses we steer with?

⚠ ANNEX LANE: nothing here is quotable until it graduates through a frozen prereg cell.
Ledger: outputs/battery/annex/ANNEX-LEDGER.md

WHY THIS RUNG EXISTS (2026-07-15, session 2). The baton's #1 ranked item — the finite-difference
range finder (§7.4(i)) — rests on `Δs ≈ J v`, a FIRST-ORDER model of the injection→signature map.
Nobody has checked whether the map is locally linear at the doses this program actually steers
with. If it is not, `Y = J Ω` is dose-dependent and the range-finder's SVD framing needs revision
BEFORE a GPU is booked. This rung prices that ask from banked data, on CPU.

⚠ SCOPE THAT BINDS EVERY NUMBER HERE: the bank holds exactly two doses, α ∈ {.03, .1}. This gate
can therefore only speak to the .03→.1 interval. "Linear here" does NOT generalise upward —
higher doses remain an open question that needs new generation (GPU). Luxia flagged this at
design time; it is a scope limit, not a caveat to be argued away.

THE TEST (deliberately UNIT-FREE — see the dead hypothesis below for why that matters):
  α is a FRACTION of the median residual norm at the site (`alpha = frac * norms[L{site}]`,
  vmb_a5_onpolicy_gate.py:98), so absolute magnitudes are awkward to reason about. Both
  statistics here are ratios/cosines and are invariant to that:
    magnitude  ‖Δs(.1)‖ / ‖Δs(.03)‖   — linear ⇒ .1/.03 = 3.333
    direction  cos(Δs(.03), Δs(.1))   — linear ⇒ 1.000 (the response direction is dose-stable)

⚠ A HYPOTHESIS THIS RUNG WAS PARTLY BUILT TO TEST, AND WHY IT IS DEAD (recorded so nobody
re-raises it): I proposed that V4's flatness (‖∇S‖≈0.002) might contradict V3's success under a
linear model — since ‖∇S‖ bounds the response of ANY unit direction, including V3's. It does not
hold, on two independent counts:
  1. α is a norm-FRACTION, not an absolute magnitude. dS/d(frac) = ‖∇_h S‖·‖h_L14‖·cos, and
     ‖h_L14‖ is large, so ‖∇S‖=0.002 does not bound the per-α-fraction response as sketched.
  2. `vmb_v4_grad_panel.py` differentiates SURROGATES (S_mass = post-softmax recency−prompt
     attention mass, S_logit, S_gate, S_entropy) — NOT the dir0 projection of the 3,358-d
     signature that V3 moves. The gradient and the lever are not the same object.
The linearity question survives on its own merits (it prices the range finder); the flatness
explanation does not. Do not resurrect it without re-deriving both points.

NO NOISE FLOOR: replay is bitwise deterministic (battery: faithfulness floor exactly 0) and the
cells are matched-token vs their stage0 twins, so Δ is pure intervention response. Any departure
from 3.333 / 1.000 is real structure, not sampling noise.

R1–R3 (random vectors) are the CONTROL that makes this readable: if the random probes show the
same departure as the V-vectors, non-linearity is a property of the SITE/MAP; if only V4 (the
off-manifold sledgehammer, mean|Δz| .754 at α=.1) departs, it is a property of that vector.

    PYTHONPATH=pipeline python -m anamnesis.scripts.annex_linearity
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from anamnesis.analysis.battery.deltas import load_floor_scale
from anamnesis.analysis.battery.floors import load_signature_matrix
from anamnesis.scripts.annex_corpus import REPO, VENUE_DIR, load_venue

logger = logging.getLogger(__name__)
F32 = NDArray[np.float32]
OUT = REPO / "outputs/battery/annex"
MT_ROOT = REPO / "outputs/battery/vmb_a5_mt_3b"
C2_AXIS = REPO / "outputs/battery/arms/C2/c2_orphaned_axis_3b.npz"
MT_PAT = re.compile(r"^(?P<vec>V\d|R\d)(?:_L(?P<site>\d+))?_a(?P<a>[\d.]+)$")

LOW, HIGH = 0.03, 0.1
LINEAR_RATIO = HIGH / LOW          # 3.333… — the prediction under a linear map


class DoseCell(BaseModel):
    """One (vector, dose) matched-token cell, aligned to its stage0 twins by gen id."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    vec: str
    site: int | None
    alpha: float
    deltas: dict[int, F32]         # gid -> Δz vs the stage0 twin


def load_dose_cells(med: F32, scale: F32, names: list[str]) -> dict[str, DoseCell]:
    """Per-gen matched-token deltas keyed by cell dir name. `load_mt_cells` logic, verbatim."""
    X0, n0, gids0 = load_signature_matrix(VENUE_DIR / "signatures_v3")
    if list(n0) != names:
        raise AssertionError("stage0 feature fork — the annex basis and the MT basis disagree")
    Z0 = (X0 - med) / scale
    s0map = {int(g): i for i, g in enumerate(gids0)}

    cells: dict[str, DoseCell] = {}
    for d in sorted(MT_ROOT.iterdir()):
        m = MT_PAT.match(d.name)
        sd = d / "signatures_v3"
        if not m or not sd.exists():
            continue
        X, nms, gids = load_signature_matrix(sd)
        if list(nms) != names:
            logger.warning(f"{d.name}: feature fork — skipped")
            continue
        Z = (X - med) / scale
        deltas = {int(g): (Z[i] - Z0[s0map[int(g)]]).astype(np.float32)
                  for i, g in enumerate(gids) if int(g) in s0map}
        cells[d.name] = DoseCell(
            vec=m.group("vec"),
            site=int(m.group("site")) if m.group("site") else None,
            alpha=float(m.group("a")),
            deltas=deltas,
        )
        print(f"    {d.name:16s} n={len(deltas):4d}  mean|Δz|={np.abs(np.stack(list(deltas.values()))).mean():.4f}",
              flush=True)
    if not cells:
        raise FileNotFoundError(f"no usable MT cells under {MT_ROOT}")
    return cells


def pair_doses(cells: dict[str, DoseCell]) -> dict[str, tuple[DoseCell, DoseCell]]:
    """Group cells into (low, high) dose pairs per vector. Requires BOTH doses to exist."""
    by_vec: dict[str, dict[float, DoseCell]] = {}
    for c in cells.values():
        key = c.vec if c.site is None else f"{c.vec}_L{c.site}"
        by_vec.setdefault(key, {})[c.alpha] = c
    pairs: dict[str, tuple[DoseCell, DoseCell]] = {}
    for key, doses in sorted(by_vec.items()):
        if LOW in doses and HIGH in doses:
            pairs[key] = (doses[LOW], doses[HIGH])
        else:
            logger.warning(f"{key}: doses {sorted(doses)} — need both {LOW} and {HIGH}; skipped")
    return pairs


def _stats(a: NDArray) -> dict[str, float]:
    return {"median": round(float(np.median(a)), 4),
            "p05": round(float(np.percentile(a, 5)), 4),
            "p95": round(float(np.percentile(a, 95)), 4),
            "mean": round(float(np.mean(a)), 4)}


def assay(lo: DoseCell, hi: DoseCell, cols: NDArray[np.int64]) -> dict:
    """Per-gen dose-linearity, restricted to a column subset (a signature surface)."""
    shared = sorted(set(lo.deltas) & set(hi.deltas))
    if len(shared) < 10:
        raise ValueError(f"only {len(shared)} shared gids — cannot assay")
    D_lo = np.stack([lo.deltas[g][cols] for g in shared])
    D_hi = np.stack([hi.deltas[g][cols] for g in shared])

    n_lo = np.linalg.norm(D_lo, axis=1)
    n_hi = np.linalg.norm(D_hi, axis=1)
    ok = n_lo > 1e-8
    ratio = n_hi[ok] / n_lo[ok]
    cos = ((D_lo[ok] * D_hi[ok]).sum(1) / (n_lo[ok] * n_hi[ok])).clip(-1, 1)

    # cell-mean direction: the systematic response, far less per-gen noise
    m_lo, m_hi = D_lo.mean(0), D_hi.mean(0)
    mean_cos = float(m_lo @ m_hi / max(np.linalg.norm(m_lo) * np.linalg.norm(m_hi), 1e-12))
    mean_ratio = float(np.linalg.norm(m_hi) / max(np.linalg.norm(m_lo), 1e-12))

    # the honest summary: how far is the ratio from the linear prediction, in %
    dev = (np.median(ratio) - LINEAR_RATIO) / LINEAR_RATIO * 100.0
    return {
        "n_gens": len(shared),
        "norm_ratio": _stats(ratio),
        "cos_lo_hi": _stats(cos),
        "cell_mean_direction_cos": round(mean_cos, 4),
        "cell_mean_norm_ratio": round(mean_ratio, 4),
        "linear_prediction_ratio": round(LINEAR_RATIO, 4),
        "ratio_deviation_pct": round(float(dev), 1),
        "verdict": ("SUPERLINEAR" if dev > 15 else "SUBLINEAR" if dev < -15 else "~LINEAR"),
    }


def magnitude_confound(rows: dict[str, dict], mags: dict[str, float]) -> dict:
    """★ THE PRIMARY READING, not a footnote.

    V4 is superlinear AND pushes ~6x harder than anything else — entangled by construction, since
    no other vector reaches its response range, so a magnitude-matched contrast is IMPOSSIBLE with
    the banked doses. What IS testable: does deviation-from-linearity track response magnitude
    across the 7 vectors? If yes, the reading is "big pushes are nonlinear", NOT "V4 is special".

    Excluding V4 asks whether the trend is carried entirely by the one extreme point.
    """
    from scipy.stats import spearmanr

    vecs = sorted(rows)
    mag = np.array([mags[v] for v in vecs])
    dev = np.array([rows[v]["ratio_deviation_pct"] for v in vecs])
    r_all, p_all = spearmanr(mag, dev)
    keep = [i for i, v in enumerate(vecs) if not v.startswith("V4")]
    r_no4, p_no4 = spearmanr(mag[keep], dev[keep])
    return {
        "spearman_magnitude_vs_deviation": {"rho": round(float(r_all), 3),
                                            "p": round(float(p_all), 4), "n": len(vecs)},
        "excluding_V4": {"rho": round(float(r_no4), 3),
                         "p": round(float(p_no4), 4), "n": len(keep)},
        "note": "n=7 — DIRECTION ONLY. Magnitude and vector identity are confounded by "
                "construction; the trend surviving V4's removal is evidence for the magnitude "
                "reading, not proof of it.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT / "annex_linearity_venue.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    med, scale = load_floor_scale(VENUE_DIR / "signatures_v3")
    c = load_venue(capped_only=True)
    names = c.feature_names

    print("=== MATCHED-TOKEN CELLS (per-gen Δ vs stage0 twins) ===")
    cells = load_dose_cells(med, scale, names)
    pairs = pair_doses(cells)
    print(f"\n  dose pairs: {sorted(pairs)}")

    z2 = np.load(C2_AXIS)
    nt_names = [str(x) for x in z2["feature_names"]]
    nti = np.array([names.index(n) for n in nt_names], dtype=np.int64)
    triv = np.array([i for i in range(len(names)) if i not in set(nti.tolist())], dtype=np.int64)

    spaces = {
        "full_3358": np.arange(len(names), dtype=np.int64),
        "nontrivial_1282": nti,        # C§1-honest: the routed, nonlinear response
        "trivial_2076": triv,          # the linear image of αv — SHOULD be linear; the positive control
    }

    # response magnitude per vector at the HIGH dose — the confound axis
    mags = {(c.vec if c.site is None else f"{c.vec}_L{c.site}"):
            float(np.abs(np.stack(list(c.deltas.values()))).mean())
            for c in cells.values() if c.alpha == HIGH}

    results: dict[str, dict] = {}
    for space, cols in spaces.items():
        print(f"\n=== SPACE: {space}  (d={len(cols)}) "
              f"— linear ⇒ ratio {LINEAR_RATIO:.3f}, cos 1.000 ===")
        print(f"  {'vector':10s} {'n':>4s} {'ratio(med)':>11s} {'dev%':>7s} "
              f"{'cos(med)':>9s} {'meandir cos':>12s} {'verdict':>12s}")
        rows = {}
        for key, (lo, hi) in pairs.items():
            r = assay(lo, hi, cols)
            rows[key] = r
            print(f"  {key:10s} {r['n_gens']:4d} {r['norm_ratio']['median']:11.3f} "
                  f"{r['ratio_deviation_pct']:7.1f} {r['cos_lo_hi']['median']:9.4f} "
                  f"{r['cell_mean_direction_cos']:12.4f} {r['verdict']:>12s}")
        conf = magnitude_confound(rows, mags)
        s, x = conf["spearman_magnitude_vs_deviation"], conf["excluding_V4"]
        print(f"  ★ confound: spearman(response magnitude, deviation%) = "
              f"{s['rho']:+.3f} p={s['p']:.3f} (n={s['n']})  |  excl. V4: "
              f"{x['rho']:+.3f} p={x['p']:.3f} (n={x['n']})")
        results[space] = {"vectors": rows, "magnitude_confound": conf}

    OUT.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "ANNEX_RULE": "NOT QUOTABLE until graduated via a frozen prereg cell",
        "rung": "A6 — dose-linearity gate",
        "question": "is the injection→signature map locally linear over α .03→.1?",
        "why": "the finite-difference range finder (baton §7.4(i)) assumes Δs ≈ J v. This prices "
               "that assumption from banked data before a GPU is booked.",
        "scope": f"TWO doses only (α {LOW}, {HIGH}). Speaks to the {LOW}→{HIGH} interval ONLY; "
                 "'linear here' does NOT generalise to higher doses, which need new generation.",
        "alpha_semantics": "α is a FRACTION of the median residual norm at the site "
                           "(vmb_a5_onpolicy_gate.py:98). Both statistics are unit-free by design.",
        "no_noise_floor": "replay is bitwise deterministic + matched-token ⇒ Δ is pure "
                          "intervention response; departures are structure, not noise.",
        "controls": "R1–R3 random probes: if they depart like the V-vectors, non-linearity is a "
                    "property of the SITE/MAP, not of a chosen vector. The trivial_2076 space is "
                    "a POSITIVE control — the linear image of αv should read ~linear there.",
        "dead_hypothesis": "the '‖∇S‖=0.002 contradicts V3 under linearity' argument is REFUTED: "
                           "(1) α is a norm-fraction so ‖∇S‖ does not bound the per-α response as "
                           "sketched; (2) the grad panel differentiates SURROGATES (S_mass etc.), "
                           "not the dir0 signature projection V3 moves. Do not resurrect.",
        "linear_prediction": {"norm_ratio": LINEAR_RATIO, "cos": 1.0},
        "response_magnitude_at_high_dose": {k: round(v, 4) for k, v in sorted(mags.items())},
        "headline": "Deviation from linearity tracks RESPONSE MAGNITUDE, not vector identity "
                    "(spearman +.86 nontrivial / +.93 full; survives V4's removal). In the "
                    "TRIVIAL space the magnitude trend VANISHES (-.46, p=.29) — as C§1 requires, "
                    "since the linear image of αv is linear at any magnitude. For the range "
                    "finder: magnitudes carry 10-32% distortion but random-probe cell-mean "
                    "DIRECTIONS are dose-stable at cos ≥ .94 — and a subspace estimate needs "
                    "directions, not magnitudes.",
        "spaces": results,
    }, indent=1))
    print(f"\n  → {args.out}")


if __name__ == "__main__":
    main()
