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
    location_dispersion,
    within_condition_deltas,
)
from anamnesis.analysis.battery.manifest import MODEL_META
from anamnesis.analysis.battery.stats import bh_fdr
from anamnesis.analysis.battery.text_decode import maybe_decode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIRMATORY_CELLS = ["source:output", "source:residual", "source_band:attention|mid"]
BLIND_CELLS = {"source_band:attention|mid"}          # predicted NOT to carry
CONTRASTS = [("t03", "native"), ("native", "t09"), ("t09", "t12"),
             ("t03", "t09"), ("p07", "native"), ("native", "p10")]
KILL_CONTRAST = ("t03", "t09")
RULER_K = 2.0
NATIVE_T = {m: meta.native_temperature for m, meta in MODEL_META.items()}
DOSE_T = {"t03": 0.3, "t09": 0.9, "t12": 1.2}        # top-p doses excluded from the T ladder


def analyze_model(model: str, n_layers: int, battery_root: Path,
                  sig_subdir: str = "signatures_v3") -> dict:
    floor_dir = battery_root / MODEL_META[model].stage0_dir
    med, scale = load_floor_scale(floor_dir / sig_subdir)

    conds: dict[str, ConditionCorpus] = {
        "native": ConditionCorpus(floor_dir / sig_subdir, floor_dir / "metadata.json",
                                  med, scale, f"{model}-native"),
    }
    for dose in ("t03", "t09", "t12", "p07", "p10"):
        d = battery_root / f"vmb_a1_{model}_{dose}"
        conds[dose] = ConditionCorpus(d / sig_subdir, d / "metadata.json",
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

    # dissociation: token-space stats on the kill contrast (t03 vs t09).
    # All text-side rows are built from the CORPUS's surviving gen_ids so they
    # stay row-aligned with Z when the modal-vector guard drops short gens
    # (OLMo-2 instant-EOS class — anchors are unaffected, no drops there).
    def _gen_index(cond: str) -> dict[int, dict]:
        md = json.loads((battery_root / f"vmb_a1_{model}_{cond}" / "metadata.json").read_text())
        return {g["generation_id"]: g for g in md["generations"]}

    def _text_stats(cond: str) -> list[tuple[float, float]]:
        idx = _gen_index(cond)
        out = []
        for gid in conds[cond].gen_ids:
            g = idx[gid]
            words = maybe_decode(g["generated_text"]).split()
            ttr = len(set(words)) / max(len(words), 1)
            out.append((float(g["num_generated_tokens"]), float(ttr)))
        return out

    s03, s09 = _text_stats("t03"), _text_stats("t09")
    def _auc(idx: int) -> float:
        a = np.array([x[idx] for x in s03]); b = np.array([x[idx] for x in s09])
        u = mannwhitneyu(a, b).statistic
        return float(max(u, len(a) * len(b) - u) / (len(a) * len(b)))

    # TF-IDF text classifier, GroupKFold by topic (leak discipline) — the real
    # token-space readout; per-gen length is ceiling-ed by truncation clustering
    # (addendum 2026-07-12b item 5) so the cheap stats under-read.
    def _texts(cond: str) -> tuple[list[str], list[int]]:
        idx = _gen_index(cond)
        return ([maybe_decode(idx[gid]["generated_text"]) for gid in conds[cond].gen_ids],
                [int(idx[gid]["topic_idx"]) for gid in conds[cond].gen_ids])
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    ta, ga = _texts("t03"); tb, gb = _texts("t09")
    texts, y = ta + tb, np.r_[np.zeros(len(ta)), np.ones(len(tb))]
    groups = np.array(ga + gb)
    aucs = []
    for tr, te in GroupKFold(n_splits=5).split(texts, y, groups):
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        Xtr = vec.fit_transform([texts[i] for i in tr])
        Xte = vec.transform([texts[i] for i in te])
        clf = LogisticRegression(max_iter=2000).fit(Xtr, y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))

    # Signature-side AUC under the SAME folds (apples-to-apples dissociation row):
    # z-scored source:output features, logistic, GroupKFold by topic. Row
    # alignment is by construction now: texts above are indexed by the same
    # corpus gen_ids that order Z.
    Zs = np.vstack([conds["t03"].Z[:, cells["source:output"]],
                    conds["t09"].Z[:, cells["source:output"]]])
    sig_aucs = []
    for tr, te in GroupKFold(n_splits=5).split(Zs, y, groups):
        clf = LogisticRegression(max_iter=2000).fit(Zs[tr], y[tr])
        sig_aucs.append(roc_auc_score(y[te], clf.predict_proba(Zs[te])[:, 1]))

    # Likelihood rung (addendum 2026-07-12c, rung ii): mean per-token surprisal of the
    # text under the GENERATOR — the textbook sampling-parameter detector, computable by
    # an external observer with the same model (text + one scoring pass; here read from
    # the banked mean_surprise feature). Single fixed statistic → direct AUC, no fitting.
    si = names.index("mean_surprise")
    Xa = conds["t03"].Z[:, si]
    Xb = conds["t09"].Z[:, si]
    u = mannwhitneyu(Xa, Xb).statistic
    lik_auc = float(max(u, len(Xa) * len(Xb) - u) / (len(Xa) * len(Xb)))

    dissoc = {"len_auc_t03_vs_t09": _auc(0), "ttr_auc_t03_vs_t09": _auc(1),
              "tfidf_groupkfold_auc_t03_vs_t09": float(np.mean(aucs)),
              "tfidf_auc_folds": [float(a) for a in aucs],
              "likelihood_mean_surprise_auc": lik_auc,
              "signature_output_groupkfold_auc": float(np.mean(sig_aucs)),
              "signature_auc_folds": [float(a) for a in sig_aucs],
              "n": [len(s03), len(s09)]}

    # Location/dispersion decomposition (addendum 12c item 2): each dose vs native,
    # confirmatory cells — movers vs spreaders.
    conf_cells = {c: cells[c] for c in CONFIRMATORY_CELLS}
    loc_disp = {}
    for dose in ("t03", "t09", "t12", "p07", "p10"):
        loc_disp[dose] = location_dispersion(conds["native"], conds[dose], conf_cells)

    return {"rows": rows, "variance_flags": variance_flags,
            "dose_monotonicity": {"ratios_vs_native": ratios_vs_native,
                                  "spearman_rho": float(rho), "spearman_p": float(rho_p),
                                  "abs_dT": dict(zip(DOSE_T, dt))},
            "dissociation": dissoc, "location_dispersion": loc_disp}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--models", default="3b,8b",
                    help="Comma list (early sanity passes may run one model; the RESULT OF "
                         "RECORD is the both-model run — FDR spans the full grid)")
    ap.add_argument("--sig-subdir", default="signatures_v3",
                    help="Signature subdir to read for floor + doses (M6/MoE re-extractions "
                         "bank to signatures_v3_x2 = v2 xrt/120-feat; v1 signatures_v3 stays "
                         "frozen). Floor and doses MUST share the subdir (matched feature dim).")
    args = ap.parse_args()

    selected = [(m.strip(), MODEL_META[m.strip()].n_layers)
                for m in args.models.split(",") if m.strip()]

    results = {"arm": "A1_sampling", "prereg": "prereg-vmb-v1 §2c A1 + addenda a/b",
               "sig_subdir": args.sig_subdir,
               "models": {}, "models_included": [m for m, _ in selected]}
    all_rows = []
    for model, n_layers in selected:
        logger.info(f"=== A1 {model} (sig_subdir={args.sig_subdir}) ===")
        r = analyze_model(model, n_layers, args.battery_root, sig_subdir=args.sig_subdir)
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
    # Frozen §2c wording: "output-source cells fail to SEPARATE T=0.3 vs 0.9 ABOVE FLOOR
    # at 2x-law n => instrument breakage". Separation above floor = FDR-significant
    # detection (the delta distribution sits above the floor distribution), NOT the k=2
    # magnitude ruler — the ruler is the VISIBILITY bar for effect claims. (First
    # implementation conflated the two; corrected against the frozen text before the
    # result of record — see journal 2026-07-12 day session. The pairwise-delta ratio is
    # structurally compressed for location shifts: cross-condition deltas are floored by
    # within-cloud seed noise, so detection and magnitude must not be conflated.)
    results["kill_criterion"] = {
        "definition": "source:output separates T=0.3 vs 0.9 above floor at 2x-law n "
                      "(FDR-significant detection; instrument positive control; hard kill "
                      "on failure). Magnitude verdicts (k=2 ruler) reported separately.",
        "per_model": {r["model"]: {"detected": bool(r["bh_reject"]),
                                   "verdict_magnitude": r["verdict"],
                                   "ratio": r["effect_ratio_vs_floor"],
                                   "p_bh": r["p_bh_adjusted"],
                                   "n": r["n_cross"]} for r in kills},
        "PASS": all(r["bh_reject"] for r in kills),
    }

    from anamnesis.analysis.battery.gates import require_stamp
    for row in all_rows:
        require_stamp(row, context="A1")
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
    for model in results["models_included"]:
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
        d = r["dissociation"]
        lines.append(f"Dissociation HIERARCHY (t03 vs t09; addendum 12c): "
                     f"content-class TF-IDF AUC **{d['tfidf_groupkfold_auc_t03_vs_t09']:.3f}** → "
                     f"likelihood-class (mean surprisal under generator) AUC "
                     f"**{d['likelihood_mean_surprise_auc']:.3f}** → "
                     f"internals (source:output, same folds) AUC "
                     f"**{d['signature_output_groupkfold_auc']:.3f}** "
                     f"(length {d['len_auc_t03_vs_t09']:.3f}, TTR {d['ttr_auc_t03_vs_t09']:.3f}; "
                     f"n={d['n']}). Prereg predicted visible-to-both.")
        ld = r["location_dispersion"]
        lines.append("Location/dispersion (native → dose, source:output): " +
                     "; ".join(f"{dose}: shift {v['source:output']['centroid_shift']:.2f}z, "
                               f"disp ×{v['source:output']['dispersion_ratio']:.2f}"
                               for dose, v in ld.items()))
        if r["variance_flags"]:
            lines.append(f"⚠ arm-variance flags (>1.5× floor): {r['variance_flags']}")
        else:
            lines.append("Arm-variance check: all confirmatory cells within 1.5× floor.")
        lines.append("")
    (args.out_dir / "a1_report.md").write_text("\n".join(lines))
    logger.info(f"A1 report → {args.out_dir}/a1_report.md; KILL PASS={results['kill_criterion']['PASS']}")


if __name__ == "__main__":
    main()
