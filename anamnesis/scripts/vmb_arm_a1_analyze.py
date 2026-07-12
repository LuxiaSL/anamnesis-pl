"""Arm A1 (sampling ladder) analysis — prereg-vmb-v1 §2c block, executed.

Design (frozen in §2c + addenda; this script is the readout, not the hypothesis):
  Doses per model: T ∈ {0.3, native, 0.9, 1.2} at top-p 0.9; top-p ∈ {0.7, 0.9=native, 1.0}
  at native T. Native dose = the Stage-0 floor corpus itself.
  Confirmatory cells (3): source:output (predicted PRIMARY carrier),
  source:residual (predicted secondary), source_band:attention|mid (predicted BLIND —
  the N4-style localization row; carrying here would contradict the Phase-0 dissociation).
  Confirmatory contrasts (6): (t03,nat) (nat,t09) (t09,t12) (t03,t09)=KILL (p07,nat) (nat,p10).
  m = 3 cells × 6 contrasts × 2 models = 36 → α_test = 0.05/36 (Bonferroni planning bound;
  BH-FDR applied across the same grid for the reported verdicts).

Readouts per (contrast × cell):
  - effect ratio = median(cross-condition deltas) / median(Stage-0 floor deltas) — the
    floor-ruler statistic (k=2 visibility bar for free-gen cells)
  - Mann-Whitney U (cross vs floor deltas), BH-FDR across the confirmatory grid
  - arm-variance check: within-dose floor vs Stage-0 floor (>1.5× ⇒ flag, addendum 12a §2)
KILL criterion (positive control): (t03,t09) in source:output must pass ruler + FDR.
Dose-monotonicity: Spearman(|ΔT| from native, effect ratio) on the T ladder, source:output.
Dissociation column (A1 predicted visible-to-both): token-space stats (length, TTR) AUC
on the kill contrast.
Exploratory: full cell grid reported separately (hypothesis-generating, never in m).

Usage:
    PYTHONPATH=. python -m anamnesis.scripts.vmb_arm_a1_analyze \
        --battery-root ../outputs/battery --out-dir ../outputs/battery/arms/A1
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from anamnesis.analysis.battery.deltas import (
    ConditionCorpus,
    build_cells,
    cross_condition_deltas,
    load_floor_scale,
    within_condition_deltas,
)
from anamnesis.analysis.battery.stats import bh_fdr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIRMATORY_CELLS = ["source:output", "source:residual", "source_band:attention|mid"]
BLIND_CELLS = {"source_band:attention|mid"}          # predicted NOT to carry
CONTRASTS = [("t03", "native"), ("native", "t09"), ("t09", "t12"),
             ("t03", "t09"), ("p07", "native"), ("native", "p10")]
KILL_CONTRAST = ("t03", "t09")
RULER_K = 2.0
NATIVE_T = {"3b": 0.7, "8b": 0.6}
DOSE_T = {"t03": 0.3, "t09": 0.9, "t12": 1.2}        # top-p doses excluded from the T ladder


def analyze_model(model: str, n_layers: int, battery_root: Path) -> dict:
    floor_dir = battery_root / f"vmb_stage0_{model}"
    med, scale = load_floor_scale(floor_dir / "signatures_v3")

    conds: dict[str, ConditionCorpus] = {
        "native": ConditionCorpus(floor_dir / "signatures_v3", floor_dir / "metadata.json",
                                  med, scale, f"{model}-native"),
    }
    for dose in ("t03", "t09", "t12", "p07", "p10"):
        d = battery_root / f"vmb_a1_{model}_{dose}"
        conds[dose] = ConditionCorpus(d / "signatures_v3", d / "metadata.json",
                                      med, scale, f"{model}-{dose}")

    names = conds["native"].feature_names
    cells = build_cells(names, n_layers)

    # Stage-0 floor deltas per cell (the ruler's denominator)
    floor_deltas = within_condition_deltas(conds["native"], cells)

    rows = []
    for (da, db) in CONTRASTS:
        cross = cross_condition_deltas(conds[da], conds[db], cells, max_pairs_per_class=20)
        for cell in cells:
            fd, cd = floor_deltas[cell], cross[cell]
            ratio = float(np.median(cd) / max(np.median(fd), 1e-12))
            u_p = float(mannwhitneyu(cd, fd, alternative="greater").pvalue)
            rows.append({
                "model": model, "contrast": f"{da}|{db}", "cell": cell,
                "confirmatory": cell in CONFIRMATORY_CELLS,
                "predicted_blind": cell in BLIND_CELLS,
                "effect_ratio_vs_floor": ratio,
                "p_mannwhitney": u_p,
                "n_cross": int(len(cd)), "n_floor": int(len(fd)),
                "stamp": {"n": int(len(cd)), "M": model,
                          "law": f"stage0-{model} ruler k={RULER_K} alpha_grid",
                          "floor_type": "stochastic"},
            })

    # arm-variance check (addendum 12a §2): within-dose floors vs Stage-0 floor
    variance_flags = []
    for dose in ("t03", "t09", "t12", "p07", "p10"):
        wd = within_condition_deltas(conds[dose], cells)
        for cell in CONFIRMATORY_CELLS:
            r = float(np.median(wd[cell]) / max(np.median(floor_deltas[cell]), 1e-12))
            if r > 1.5:
                variance_flags.append({"model": model, "dose": dose, "cell": cell,
                                       "within_dose_over_floor": r})

    # dose-monotonicity on the T ladder (source:output)
    ratios_vs_native = {}
    for dose, T in DOSE_T.items():
        cd = cross_condition_deltas(conds[dose], conds["native"],
                                    {"source:output": cells["source:output"]},
                                    max_pairs_per_class=20)["source:output"]
        ratios_vs_native[dose] = float(np.median(cd) / max(np.median(floor_deltas["source:output"]), 1e-12))
    dt = [abs(DOSE_T[d] - NATIVE_T[model]) for d in DOSE_T]
    rr = [ratios_vs_native[d] for d in DOSE_T]
    rho, rho_p = spearmanr(dt, rr)

    # dissociation: token-space stats on the kill contrast (t03 vs t09)
    def _text_stats(cond: str) -> list[tuple[float, float]]:
        md = json.loads((battery_root / (f"vmb_a1_{model}_{cond}") / "metadata.json").read_text())
        out = []
        for g in md["generations"]:
            text = g["generated_text"]
            words = text.split()
            ttr = len(set(words)) / max(len(words), 1)
            out.append((float(g["num_generated_tokens"]), float(ttr)))
        return out

    s03, s09 = _text_stats("t03"), _text_stats("t09")
    def _auc(idx: int) -> float:
        a = np.array([x[idx] for x in s03]); b = np.array([x[idx] for x in s09])
        u = mannwhitneyu(a, b).statistic
        return float(max(u, len(a) * len(b) - u) / (len(a) * len(b)))
    dissoc = {"len_auc_t03_vs_t09": _auc(0), "ttr_auc_t03_vs_t09": _auc(1),
              "n": [len(s03), len(s09)]}

    return {"rows": rows, "variance_flags": variance_flags,
            "dose_monotonicity": {"ratios_vs_native": ratios_vs_native,
                                  "spearman_rho": float(rho), "spearman_p": float(rho_p),
                                  "abs_dT": dict(zip(DOSE_T, dt))},
            "dissociation": dissoc}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    results = {"arm": "A1_sampling", "prereg": "prereg-vmb-v1 §2c A1 + addenda a/b",
               "models": {}}
    all_rows = []
    for model, n_layers in (("3b", 28), ("8b", 32)):
        logger.info(f"=== A1 {model} ===")
        r = analyze_model(model, n_layers, args.battery_root)
        results["models"][model] = r
        all_rows.extend(r["rows"])

    # BH-FDR across the CONFIRMATORY grid only (map-wide over both models)
    conf = [r for r in all_rows if r["confirmatory"]]
    reject, adj = bh_fdr([r["p_mannwhitney"] for r in conf], alpha=0.05)
    for r, rej, p in zip(conf, reject, adj):
        r["bh_reject"] = bool(rej)
        r["p_bh_adjusted"] = float(p)
        r["passes_ruler"] = r["effect_ratio_vs_floor"] >= RULER_K
        r["verdict"] = ("CARRIES" if (rej and r["passes_ruler"])
                        else "detected-sub-ruler" if rej else "at-floor")
    results["m_confirmatory"] = len(conf)

    # kill criterion (positive control): source:output on (t03,t09), both models
    kills = [r for r in conf if r["cell"] == "source:output"
             and r["contrast"] == "t03|t09"]
    results["kill_criterion"] = {
        "definition": "source:output separates T=0.3 vs 0.9 above floor at 2x-law n "
                      "(instrument positive control; hard kill on failure)",
        "per_model": {r["model"]: {"verdict": r["verdict"],
                                   "ratio": r["effect_ratio_vs_floor"],
                                   "p_bh": r["p_bh_adjusted"]} for r in kills},
        "PASS": all(r["verdict"] == "CARRIES" for r in kills),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "a1_results.json").write_text(json.dumps(results, indent=1))

    # ── report ──
    lines = ["# Arm A1 (sampling ladder) — results of record", "",
             f"m = {len(conf)} confirmatory (cell × contrast × model); BH-FDR α=0.05 across "
             f"that grid; ruler k = {RULER_K}× stochastic floor. n/M/law stamped per row in "
             "a1_results.json.", "",
             f"**KILL (positive control): {'PASS' if results['kill_criterion']['PASS'] else 'FAIL'}**", "",
             "**Channel column (structural + measured):** A1's matched-token delta is ZERO by "
             "construction — temperature/top-p reshape the SAMPLING distribution and never enter "
             "a teacher-forced forward pass — and Stage-0 measured the replay channel as bitwise "
             "deterministic fleet-wide (faithfulness floor exactly 0, n=900 pairs/model). The "
             "free-generation deltas below are therefore 100% token-mediated: the §1 parameter-free "
             "prediction, confirmed at the strongest possible reading.", ""]
    for model in ("3b", "8b"):
        r = results["models"][model]
        lines.append(f"## {model}")
        lines.append("")
        lines.append("| contrast | cell | ratio vs floor | p (BH) | verdict |")
        lines.append("|---|---|---|---|---|")
        for row in conf:
            if row["model"] != model:
                continue
            blind = " (predicted BLIND)" if row["predicted_blind"] else ""
            lines.append(f"| {row['contrast']} | {row['cell']}{blind} | "
                         f"{row['effect_ratio_vs_floor']:.2f} | {row['p_bh_adjusted']:.2e} | "
                         f"{row['verdict']} |")
        dm = r["dose_monotonicity"]
        lines.append("")
        lines.append(f"Dose-monotonicity (source:output, T ladder): ratios vs native "
                     f"{ {k: round(v,2) for k,v in dm['ratios_vs_native'].items()} }, "
                     f"Spearman(|ΔT|, ratio) ρ={dm['spearman_rho']:.2f} (p={dm['spearman_p']:.3f}, n=3 doses)")
        lines.append(f"Dissociation (t03 vs t09 token-space): length AUC "
                     f"{r['dissociation']['len_auc_t03_vs_t09']:.3f}, TTR AUC "
                     f"{r['dissociation']['ttr_auc_t03_vs_t09']:.3f} "
                     f"(n={r['dissociation']['n']}) — predicted visible-to-both.")
        if r["variance_flags"]:
            lines.append(f"⚠ arm-variance flags (>1.5× floor): {r['variance_flags']}")
        else:
            lines.append("Arm-variance check: all confirmatory cells within 1.5× floor.")
        lines.append("")
    (args.out_dir / "a1_report.md").write_text("\n".join(lines))
    logger.info(f"A1 report → {args.out_dir}/a1_report.md; KILL PASS={results['kill_criterion']['PASS']}")


if __name__ == "__main__":
    main()
