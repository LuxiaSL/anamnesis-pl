"""ANNEX (session 7, route-5 tenancy): THE STABILITY RUNG — charter anchor 3.

What is the minimum n (gens per candidate evaluation) for a stable selectivity
estimate? Selectivity = effect_per_offtarget of the mean matched-token delta
projected on the banked dir0 gauge (C§6 construction, verbatim `project()` from
vmb_a5_frozen_directional). Substrate = the 42 banked MT cells
(vmb_a5_mt_<model>: 7 vectors x 6 doses x 160 gens vs stage0 by gen-id) used as
stand-in candidates.

Three readouts per n in N_GRID, B bootstrap draws:
  1. per-cell CV of selectivity (SD/mean over subsample draws)
  2. RANKING fidelity: Spearman rho of cell selectivities between two DISJOINT
     n-subsamples (what a CMA-ES generation actually consumes is a ranking)
  3. signal-vs-null margin: P(sel_V3 > max sel_R) per dose at that n
     (the pre-gate's ordering, re-tested at eval-sized n)

Everything is computed against BOTH the gauge and shuffled-null axis draw 0 —
the null column calibrates every number (item 3' discipline: the placebo ships
in the same table).

CPU + banked data only. Writes to outputs/battery/annex/ ONLY.

Usage (from pipeline/):
    python -m anamnesis.scripts.annex_selectivity_stability \
        --battery-root ../outputs/battery --annex-dir ../outputs/battery/annex --model 3b
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from anamnesis.analysis.battery.deltas import load_floor_scale
from anamnesis.analysis.battery.floors import load_signature_matrix
from anamnesis.analysis.battery.manifest import MODEL_META
from anamnesis.scripts.vmb_a5_frozen_directional import load_mt_cells, project

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]

N_GRID = [5, 10, 20, 40, 80]
B = 200
SEED = 20260716


def spearman(a: F32, b: F32) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = float(np.sqrt((ra @ ra) * (rb @ rb)))
    return float(ra @ rb / d) if d > 1e-12 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--annex-dir", type=Path, required=True)
    ap.add_argument("--model", default="3b", choices=list(MODEL_META.keys()))
    args = ap.parse_args()

    mm = MODEL_META[args.model]
    stage0 = args.battery_root / mm.stage0_dir
    med, scale = load_floor_scale(stage0 / "signatures_v3")
    s0X, names, s0g = load_signature_matrix(stage0 / "signatures_v3")
    s0Z = ((s0X - med) / scale).astype(np.float32)
    s0map = {g: i for i, g in enumerate(s0g)}

    gauge_npz = np.load(args.annex_dir / f"annex_dir0_axis_{args.model}.npz",
                        allow_pickle=True)
    null_npz = np.load(args.annex_dir / f"annex_dir0_shufnull_axis_{args.model}.npz",
                       allow_pickle=True)
    if list(gauge_npz["feature_names"]) != list(names):
        raise SystemExit("banked axis feature set != stage0 modal set — feature fork")
    axes = {"dir0": gauge_npz["axis"].astype(np.float32),
            "shufnull0": null_npz["axes"][0].astype(np.float32)}

    mt = load_mt_cells(args.battery_root / f"vmb_a5_mt_{args.model}",
                       s0Z, s0map, names, med, scale)
    if not mt:
        raise SystemExit("no MT cells loaded")
    logger.info(f"{len(mt)} MT cells loaded")

    rng = np.random.default_rng(SEED)
    cell_names = sorted(mt.keys())
    stacks = {c: np.stack([mt[c].Dmap[g] for g in mt[c].gids]) for c in cell_names}
    n_avail = min(s.shape[0] for s in stacks.values())
    logger.info(f"min gens/cell = {n_avail}")

    results: dict[str, dict] = {}
    for ax_name, axis in axes.items():
        full = {c: project(stacks[c].mean(axis=0), axis)["effect_per_offtarget"]
                for c in cell_names}
        per_n: dict[str, dict] = {}
        for n in N_GRID:
            if 2 * n > n_avail:
                halves_ok = False
            else:
                halves_ok = True
            sel = np.empty((B, len(cell_names)))          # draw x cell (first half)
            sel2 = np.empty((B, len(cell_names)))         # disjoint second half
            for b in range(B):
                for ci, c in enumerate(cell_names):
                    m = stacks[c].shape[0]
                    perm = rng.permutation(m)
                    sel[b, ci] = project(stacks[c][perm[:n]].mean(axis=0),
                                         axis)["effect_per_offtarget"]
                    if halves_ok:
                        sel2[b, ci] = project(stacks[c][perm[n:2 * n]].mean(axis=0),
                                              axis)["effect_per_offtarget"]
            mean_sel = sel.mean(axis=0)
            cv = sel.std(axis=0) / np.maximum(np.abs(mean_sel), 1e-12)
            rho = ([spearman(sel[b], sel2[b]) for b in range(B)] if halves_ok else None)

            # ordering: P(sel_V3 > max sel_R) per dose
            order: dict[str, float] = {}
            doses = sorted({mt[c].alpha for c in cell_names})
            for a_ in doses:
                v3 = [ci for ci, c in enumerate(cell_names)
                      if mt[c].vector == "V3" and mt[c].alpha == a_]
                rs = [ci for ci, c in enumerate(cell_names)
                      if mt[c].vector.startswith("R") and mt[c].alpha == a_]
                if v3 and rs:
                    wins = (sel[:, v3[0]] > sel[:, rs].max(axis=1)).mean()
                    order[f"a{a_}"] = float(wins)

            per_n[str(n)] = {
                "cv_median": float(np.median(cv)),
                "cv_per_cell": {c: round(float(v), 4) for c, v in zip(cell_names, cv)},
                "rank_fidelity_spearman": (
                    {"mean": float(np.mean(rho)), "q05": float(np.quantile(rho, 0.05)),
                     "q95": float(np.quantile(rho, 0.95))} if rho is not None else None),
                "p_v3_beats_all_R_by_dose": order,
            }
            logger.info(f"[{ax_name}] n={n}: cv_med={per_n[str(n)]['cv_median']:.3f} "
                        f"rank_rho={per_n[str(n)]['rank_fidelity_spearman']}")
        results[ax_name] = {"full_n_selectivity": {c: round(float(v), 5)
                                                    for c, v in full.items()},
                            "by_n": per_n}

    out = {
        "construction": "effect_per_offtarget(project(mean matched-token delta, axis)); "
                        "MT cells vs stage0 by gen-id; axes = banked gauge + shuffled-null "
                        "draw 0 (annex_bank_dir0)",
        "n_grid": N_GRID, "bootstrap_draws": B, "seed": SEED,
        "n_cells": len(cell_names), "cells": cell_names,
        "results": results,
    }
    path = args.annex_dir / f"annex_selectivity_stability_{args.model}.json"
    path.write_text(json.dumps(out, indent=1))
    logger.info(f"-> {path}")


if __name__ == "__main__":
    main()
