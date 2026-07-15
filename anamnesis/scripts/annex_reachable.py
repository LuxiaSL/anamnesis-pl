"""ANNEX — the REACHABLE-SUBSPACE intersection: is nature's axis even movable?

⚠ ANNEX LANE: nothing here is quotable until it graduates through a frozen prereg cell.
Ledger: outputs/battery/annex/ANNEX-LEDGER.md

THE MISSING LEG (identified 2026-07-15, end of session 1). The annex measured WHAT VARIES
(A1-A4). C§1 tells us WHAT MOVES TRIVIALLY. **Nobody has measured WHAT MOVES AT ALL.**

The steelman's flaw, stated structurally: it selected a coordinate by a property of the DATA
(variance) rather than by a property of the CAUSAL relationship (does it respond to a push?).
Variance is how much a coordinate wanders on its own; a lever needs a coordinate that responds
when pushed. Those may be ANTI-correlated: a high-variance direction is one that MANY upstream
causes push on — the most confounded direction, where your one push competes with all of them.

This rung tests that directly, on banked data, CPU-only:

  REACHABLE SUBSPACE = the span of *observed* signature responses to actual injections.
    Source: `vmb_a5_mt_3b` — 14 MATCHED-TOKEN cells (V1/V2/V3/V4/R1/R2/R3 x {.03, .1}),
    160 gens each. Matched-token means the steered gen is forced to the same tokens as its
    Stage-0 twin, so Δ = z(steered) − z(stage0 twin) isolates the STATE change with content
    held exactly. Per-gen matching is `vmb_a5_frozen_directional.load_mt_cells` verbatim.

  NATURE'S SUBSPACE = the venue's artifact-controlled cell-centered top-k PCs (A1 of record).

  THE NUMBER: for each of nature's top axes, what fraction of it lies inside the reachable
  span? Read against the random-direction floor (E[mass] = r/d for an r-dim span in d-space).
  An axis outside the reachable span is a non-starter for steering NO MATTER how dominant.
  The principal angle between the two subspaces is one number that says whether the annex's
  premise has any room at all.

⚠ Reported on TWO spaces, because C§1 makes the full-space answer misleading:
  - FULL 3358-d: includes the trivially-moved channels (the linear image of αv in features
    reading the injected surface). Reachability there is partly definitional.
  - C2 NON-TRIVIAL 1282-d subvector: the routed, nonlinear response only. THIS is the honest
    space for "can an injection actually move this coordinate?"

R1-R3 (random vectors) are INCLUDED on purpose: the question is what ANY injection at the site
can move, not what a well-chosen one moves.

    PYTHONPATH=pipeline python -m anamnesis.scripts.annex_reachable
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from anamnesis.analysis.battery.deltas import load_floor_scale
from anamnesis.analysis.battery.floors import load_signature_matrix
from anamnesis.scripts.annex_corpus import REPO, VENUE_DIR, load_venue, prepare
from anamnesis.scripts.annex_spectrum import pca

logger = logging.getLogger(__name__)
F32 = NDArray[np.float32]
OUT = REPO / "outputs/battery/annex"
MT_ROOT = REPO / "outputs/battery/vmb_a5_mt_3b"
C2_AXIS = REPO / "outputs/battery/arms/C2/c2_orphaned_axis_3b.npz"
MT_PAT = re.compile(r"^(?P<vec>V\d|R\d)(?:_L(?P<site>\d+))?_a(?P<a>[\d.]+)$")


def load_mt_deltas(med: F32, scale: F32, names: list[str]) -> tuple[F32, list[str], dict]:
    """Per-gen matched-token deltas, stacked. `load_mt_cells` machinery, verbatim logic."""
    X0, n0, gids0 = load_signature_matrix(VENUE_DIR / "signatures_v3")
    if list(n0) != names:
        raise AssertionError("stage0 feature fork")
    Z0 = (X0 - med) / scale
    s0map = {int(g): i for i, g in enumerate(gids0)}

    rows, tags, per_cell = [], [], {}
    for d in sorted(MT_ROOT.iterdir()):
        sd = d / "signatures_v3"
        if not sd.exists() or not MT_PAT.match(d.name):
            continue
        X, nms, gids = load_signature_matrix(sd)
        if list(nms) != names:
            logger.warning(f"{d.name}: feature fork — skipped")
            continue
        Z = (X - med) / scale
        D = np.stack([Z[i] - Z0[s0map[int(g)]]
                      for i, g in enumerate(gids) if int(g) in s0map])
        rows.append(D)
        tags.extend([d.name] * len(D))
        per_cell[d.name] = D.mean(axis=0).astype(np.float32)
        print(f"    {d.name:16s} n={len(D):4d}  mean|Δz|={np.abs(D).mean():.4f}", flush=True)
    if not rows:
        raise FileNotFoundError(f"no usable MT cells under {MT_ROOT}")
    return np.vstack(rows).astype(np.float32), tags, per_cell


def subspace_mass(u: F32, B: F32) -> float:
    """Fraction of unit direction u captured by the orthonormal-row subspace B."""
    u = u / max(np.linalg.norm(u), 1e-12)
    return float(((B @ u) ** 2).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-nature", type=int, default=6)
    ap.add_argument("--r-reach", type=int, default=14,
                    help="dim of the reachable span (14 = one per MT cell by default)")
    ap.add_argument("--n-null", type=int, default=2000)
    args = ap.parse_args()

    med, scale = load_floor_scale(VENUE_DIR / "signatures_v3")
    c = load_venue(capped_only=True)                       # artifact-controlled basis
    names = c.feature_names

    print("=== MATCHED-TOKEN INTERVENTION RESPONSES (the reachable material) ===")
    D, tags, per_cell = load_mt_deltas(med, scale, names)
    print(f"  stacked Δ: {D.shape}  ({len(set(tags))} cells)")

    Xp, df = prepare(c, "cell")
    nat = pca(Xp, df, k=args.k_nature)
    print(f"\n=== NATURE (venuecap/cell, df={df}) top-{args.k_nature} var: "
          f"{np.round(nat.var_ratio[:args.k_nature], 4)} ===")

    # C2 non-trivial subvector: the honest space (C§1 — the full space's reachability is
    # partly definitional, since an injection trivially moves features reading its surface).
    z2 = np.load(C2_AXIS)
    nt_names = [str(x) for x in z2["feature_names"]]
    nti = np.array([names.index(n) for n in nt_names])

    results = {}
    for space, cols in (("full_3358", np.arange(len(names))), ("nontrivial_1282", nti)):
        Ds = D[:, cols]
        reach = pca(Ds - Ds.mean(0), df=max(len(Ds) - 1, 1), k=args.r_reach)
        B = reach.components                                # [r, |cols|] orthonormal rows
        r, d = B.shape
        floor = r / d                                       # E[mass] for a random direction
        rng = np.random.default_rng(0)
        null = np.array([subspace_mass(rng.normal(size=d).astype(np.float32), B)
                         for _ in range(args.n_null)])
        rows = []
        for j in range(args.k_nature):
            u = nat.components[j][cols]
            m = subspace_mass(u, B)
            rows.append({"pc": j + 1, "var_ratio": round(float(nat.var_ratio[j]), 4),
                         "reachable_mass": round(m, 4),
                         "p_vs_random": round(float((null >= m).mean()), 4)})
        # mean-delta span: what injections SYSTEMATICALLY move (one direction per cell)
        M = np.stack([per_cell[t][cols] for t in sorted(per_cell)])
        Q, _ = np.linalg.qr(M.T)
        sys_mass = [round(subspace_mass(nat.components[j][cols], Q.T), 4)
                    for j in range(args.k_nature)]
        results[space] = {
            "reach_dim": r, "space_dim": d,
            "reach_var_explained_top_r": round(float(reach.var_ratio.sum()), 4),
            "random_floor_mass": round(floor, 4),
            "null_mass_median": round(float(np.median(null)), 4),
            "null_mass_p95": round(float(np.percentile(null, 95)), 4),
            "nature_axes": rows,
            "systematic_span_dim": int(M.shape[0]),
            "nature_mass_in_systematic_span": sys_mass,
        }
        print(f"\n=== SPACE: {space}  (reachable span r={r} of d={d}; "
              f"random floor {floor:.4f}, null p95 {np.percentile(null, 95):.4f}) ===")
        print(f"  {'PC':>3s} {'var':>7s} {'reachable_mass':>15s} {'p vs random':>12s} "
              f"{'sys-span mass':>14s}")
        for row, s in zip(rows, sys_mass):
            print(f"  {row['pc']:3d} {row['var_ratio']:7.4f} {row['reachable_mass']:15.4f} "
                  f"{row['p_vs_random']:12.4f} {s:14.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "annex_reachable_venue.json"
    p.write_text(json.dumps({
        "ANNEX_RULE": "NOT QUOTABLE until graduated via a frozen prereg cell",
        "rung": "reachable-subspace intersection (the missing leg)",
        "source": "vmb_a5_mt_3b matched-token deltas vs stage0 twins (per-gen, content held)",
        "n_delta_rows": int(len(D)), "cells": sorted(set(tags)),
        "note": "reachable_mass = fraction of nature's PC_j lying in the span of OBSERVED "
                "injection responses. Read against random_floor_mass = r/d. The nontrivial_1282 "
                "space is the honest one: full-space reachability is partly definitional (C§1 — "
                "an injection trivially moves features reading its own surface). R1-R3 included "
                "deliberately: the question is what ANY injection at the site can move.",
        "spaces": results}, indent=1))
    print(f"\n  → {p}")


if __name__ == "__main__":
    main()
