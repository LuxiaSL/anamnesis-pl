"""Arm A2 (instruction-vs-execution) analysis — prereg-vmb-v1 §2c A2, executed.

Cell (i) — free-gen swap (per swap pair, per model):
  ratio_exec   = median Δ(swap ↔ pure_execution) / pooled within-condition floor
  ratio_system = median Δ(swap ↔ pure_system)   / pooled within-condition floor
  Prediction (N1b, execution-dependence): ratio_exec near 1 (at floor) AND
  ratio_system ≫ ratio_exec. The third outcome class from the muddy n=30 re-run
  (system_prompt_based: ratio_exec > ratio_system) is reported explicitly when it fires.
  Confirmatory cells: source_band:attention|mid (the mode region), source:attention,
  whole_vector. m = 3 cells × 2 contrasts × 3 swaps × 2 models = 36. A2 runs at 4× law.

Cell (ii) — matched-token prefix-swap (per swap pair):
  Δ(swapped-prefix replay ↔ native replay) of the SAME continuation, per cell,
  reported in SEED-FLOOR units (Stage-0 stochastic floor median = 1.0) per the
  addendum-2026-07-12b replay ruler (faithfulness floor is bitwise zero; visibility
  bar = 0.1×). Prediction: near-floor overall; residue CONFINED to
  prompt-lookback / attention cells.

Usage:
    PYTHONPATH=. python -m anamnesis.scripts.vmb_arm_a2_analyze \
        --battery-root ../outputs/battery --out-dir ../outputs/battery/arms/A2
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from anamnesis.analysis.battery.deltas import (
    ConditionCorpus,
    build_cells,
    cross_condition_deltas,
    load_floor_scale,
    within_condition_deltas,
)
from anamnesis.analysis.battery.floors import load_signature_matrix
from anamnesis.analysis.battery.stats import bh_fdr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIRMATORY_CELLS = ["source_band:attention|mid", "source:attention", "whole_vector"]
SWAPS = [("socratic_to_linear", "socratic", "linear"),
         ("dialectical_to_contrastive", "dialectical", "contrastive"),
         ("analogical_to_linear", "analogical", "linear")]
REPLAY_VISIBILITY = 0.1   # addendum 2026-07-12b: fraction of seed-floor median


def analyze_model(model: str, n_layers: int, root: Path) -> dict:
    floor_dir = root / f"vmb_stage0_{model}"
    med, scale = load_floor_scale(floor_dir / "signatures_v3")

    conds: dict[str, ConditionCorpus] = {}
    needed = sorted({c for _, s, e in SWAPS for c in (f"pure_{s}", f"pure_{e}")} |
                    {f"swap_{lbl}" for lbl, _, _ in SWAPS})
    for cond in needed:
        d = root / f"vmb_a2_{model}_{cond}"
        conds[cond] = ConditionCorpus(d / "signatures_v3", d / "metadata.json",
                                      med, scale, f"{model}-{cond}")
    names = next(iter(conds.values())).feature_names
    cells = build_cells(names, n_layers)
    within = {c: within_condition_deltas(conds[c], cells) for c in conds}

    rows = []
    for lbl, sysm, execm in SWAPS:
        swap, pure_e, pure_s = f"swap_{lbl}", f"pure_{execm}", f"pure_{sysm}"
        d_exec = cross_condition_deltas(conds[swap], conds[pure_e], cells, max_pairs_per_class=20)
        d_sys = cross_condition_deltas(conds[swap], conds[pure_s], cells, max_pairs_per_class=20)
        for cell in cells:
            pooled_floor = np.concatenate([within[swap][cell], within[pure_e][cell], within[pure_s][cell]])
            fm = max(float(np.median(pooled_floor)), 1e-12)
            re_ = float(np.median(d_exec[cell]) / fm)
            rs = float(np.median(d_sys[cell]) / fm)
            p_exec = float(mannwhitneyu(d_exec[cell], pooled_floor, alternative="greater").pvalue)
            p_dir = float(mannwhitneyu(d_sys[cell], d_exec[cell], alternative="greater").pvalue)
            outcome = ("execution_based" if rs > re_ else "system_prompt_based")
            rows.append({
                "model": model, "swap": lbl, "cell": cell,
                "confirmatory": cell in CONFIRMATORY_CELLS,
                "ratio_swap_vs_exec": re_, "ratio_swap_vs_system": rs,
                "p_exec_above_floor": p_exec, "p_direction_sys_gt_exec": p_dir,
                "outcome_class": outcome,
                "stamp": {"n": int(len(d_exec[cell])), "M": model,
                          "law": "A2@4x-law; pooled within-condition floor",
                          "floor_type": "stochastic(within-condition)"},
            })
    return {"rows": rows}


def analyze_prefix_swap(model: str, n_layers: int, root: Path) -> dict:
    """Cell (ii): swapped-prefix replay vs native replay, in Stage-0 seed-floor units."""
    floor_dir = root / f"vmb_stage0_{model}"
    med, scale = load_floor_scale(floor_dir / "signatures_v3")
    Xf, _n, _i = load_signature_matrix(floor_dir / "signatures_v3")

    ps_dir = root / f"vmb_a2_{model}_prefix_swap"
    index = json.loads((ps_dir / "prefix_swap_index.json").read_text())
    Xs, names, gids = load_signature_matrix(ps_dir / "signatures_v3")
    row_of = {g: r for r, g in enumerate(gids)}
    cells = build_cells(names, n_layers)

    # Stage-0 stochastic floor medians per cell (the ruler denominator)
    from anamnesis.analysis.battery.deltas import within_condition_deltas as _w
    stage0 = ConditionCorpus(floor_dir / "signatures_v3", floor_dir / "metadata.json",
                             med, scale, f"{model}-stage0")
    floor_med = {c: max(float(np.median(v)), 1e-12) for c, v in _w(stage0, cells).items()}

    by_swap: dict[str, dict[str, list[float]]] = {}
    for e in index:
        gid = int(e["sig"].split("_")[1])
        if gid not in row_of:
            logger.warning(f"{e['sig']} missing — skipped")
            continue
        native = np.load(Path(e["native_sig_dir"]) / f"{e['native_sig']}.npz",
                         allow_pickle=True)
        zn = (np.asarray(native["features"], dtype=np.float32) - med) / scale
        dz = np.abs(Xs[row_of[gid]] - zn)
        bucket = by_swap.setdefault(e["swap"], {c: [] for c in cells})
        for cname, mask in cells.items():
            bucket[cname].append(float(dz[mask].mean()) / floor_med[cname])

    out = {}
    for swap, cellvals in by_swap.items():
        stats = {c: {"median_seed_floor_units": float(np.median(v)),
                     "q90": float(np.quantile(v, 0.9)), "n": len(v),
                     "visible": bool(np.median(v) >= REPLAY_VISIBILITY)}
                 for c, v in cellvals.items()}
        ranked = sorted(stats.items(), key=lambda kv: -kv[1]["median_seed_floor_units"])
        out[swap] = {"cells": stats, "top5_cells": [(c, round(s["median_seed_floor_units"], 4))
                                                    for c, s in ranked[:5]],
                     "whole_vector": stats["whole_vector"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--skip-prefix-swap", action="store_true")
    args = ap.parse_args()

    results = {"arm": "A2_instruction_vs_execution",
               "prereg": "prereg-vmb-v1 §2c A2 + addenda a/b (4x law; replay ruler 0.1x)",
               "models": {}}
    all_rows = []
    for model, n_layers in (("3b", 28), ("8b", 32)):
        logger.info(f"=== A2 {model} cell (i) ===")
        r = analyze_model(model, n_layers, args.battery_root)
        if not args.skip_prefix_swap:
            logger.info(f"=== A2 {model} cell (ii) prefix-swap ===")
            r["prefix_swap"] = analyze_prefix_swap(model, n_layers, args.battery_root)
        results["models"][model] = r
        all_rows.extend(r["rows"])

    conf = [r for r in all_rows if r["confirmatory"]]
    # Two families of confirmatory tests, FDR'd together across models
    pvals = [r["p_exec_above_floor"] for r in conf] + [r["p_direction_sys_gt_exec"] for r in conf]
    reject, adj = bh_fdr(pvals, alpha=0.05)
    n = len(conf)
    for i, r in enumerate(conf):
        r["bh_exec_above_floor"] = bool(reject[i])
        r["bh_direction"] = bool(reject[n + i])
        r["p_bh_exec"] = float(adj[i])
        r["p_bh_direction"] = float(adj[n + i])
    results["m_confirmatory"] = 2 * n

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "a2_results.json").write_text(json.dumps(results, indent=1))

    lines = ["# Arm A2 (instruction vs execution) — results of record", "",
             "Null arm at 4× law; either outcome informative, direction is the bet.",
             "ratio = median cross-condition Δ / pooled within-condition floor (matched",
             "history INCLUDES each condition's system prompt). Cell (ii) in Stage-0",
             "seed-floor units, visibility bar 0.1× (addendum 2026-07-12b).", ""]
    for model in ("3b", "8b"):
        lines.append(f"## {model} — cell (i) free-gen swap")
        lines.append("")
        lines.append("| swap | cell | r(swap↔exec) | r(swap↔system) | dir p(BH) | outcome |")
        lines.append("|---|---|---|---|---|---|")
        for r in conf:
            if r["model"] != model:
                continue
            lines.append(f"| {r['swap']} | {r['cell']} | {r['ratio_swap_vs_exec']:.2f} | "
                         f"{r['ratio_swap_vs_system']:.2f} | {r['p_bh_direction']:.2e} | "
                         f"{r['outcome_class']} |")
        pf = results["models"][model].get("prefix_swap")
        if pf:
            lines.append("")
            lines.append(f"### {model} — cell (ii) prefix-swap replay (seed-floor units)")
            for swap, s in pf.items():
                wv = s["whole_vector"]
                lines.append(f"- **{swap}**: whole-vector {wv['median_seed_floor_units']:.4f}× "
                             f"seed floor (n={wv['n']}, visible@0.1×: {wv['visible']}); "
                             f"top cells: {s['top5_cells']}")
        lines.append("")
    (args.out_dir / "a2_report.md").write_text("\n".join(lines))
    logger.info(f"A2 report → {args.out_dir}/a2_report.md")


if __name__ == "__main__":
    main()
