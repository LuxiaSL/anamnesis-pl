"""14k — Ksoclin steering LEVER + behavioral readout (session-9 Part D-7; CPU, after gen+replay).

The dir0-shift lever (does Ksoclin move the analogical<->contrastive coordinate ≥2×R?) + the
socratic-induction behavioral leg (socratic markers in the steered text vs baseline). dir0 = the
frozen analogical<->contrastive LDA axis in floor-z signature space (reused from
vmb_a5_frozen_directional). Ksoclin/V3/baseline read from the 14k run dir; R1/R2/R3 (matched-norm
null) from the banked a5 run dir. Scored vs 14k(a) P=.85: lever ≥2×R at α≤.1 WITH an in-window
behavioral consequence. First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from anamnesis.analysis.battery.deltas import ConditionCorpus, load_floor_scale
from anamnesis.analysis.battery.manifest import MODEL_META
from anamnesis.scripts.vmb_a5_frozen_directional import build_axes, project

# socratic-mode markers (question-driven, exploratory, dialectical) — the induction signature
SOCRATIC = re.compile(
    r"\b(why|how|what if|suppose|consider|imagine|explore|reflect|examine|perhaps|"
    r"let us|let's (?:consider|explore|think)|might we|could it be|question|wonder|"
    r"on the other hand|but what about|is it (?:possible|not))\b", re.I)


def _texts(run_dir: Path, cell: str) -> list[str]:
    md = json.loads((run_dir / cell / "metadata.json").read_text())
    gens = md["generations"] if "generations" in md else md
    return [g.get("generated_text", "") for g in gens]


def _socratic_rate(texts: list[str]) -> dict:
    qs, socs, wtot = 0, 0, 0
    for t in texts:
        qs += t.count("?")
        socs += len(SOCRATIC.findall(t))
        wtot += max(len(t.split()), 1)
    wtot = max(wtot, 1)
    return {"question_per_1k": round(1000.0 * qs / wtot, 2),
            "socratic_marker_per_1k": round(1000.0 * socs / wtot, 2), "n": len(texts)}


def _parse(cell: str):
    p = cell.split("_")
    return p[0], int(p[1][1:]), float(p[2][1:])


def _centroid(sig_dir: Path, meta: Path, med, scale) -> np.ndarray:
    return ConditionCorpus(sig_dir, meta, med, scale, "c").Z.mean(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b")
    ap.add_argument("--soclin-run-dir", type=Path, required=True, help="vmb_14k_soclin_3b")
    ap.add_argument("--a5-run-dir", type=Path, required=True, help="vmb_a5_3b (banked R nulls)")
    ap.add_argument("--battery-root", type=Path, default=Path("outputs/battery"))
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    med, scale = load_floor_scale(args.stage0_run / "signatures_v3")
    axes = build_axes(args.battery_root, args.model, med, scale, len(med))
    dir0 = axes["dir0"]
    # SOCLIN coordinate = the socratic<->linear LDA axis in floor-z signature space (parallel to
    # dir0=analogical<->contrastive) — the coordinate Ksoclin defines and the write test targets.
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    def _pure_Z(mode):
        d = args.battery_root / f"vmb_a2_{args.model}_pure_{mode}"
        return ConditionCorpus(d / "signatures_v3", d / "metadata.json", med, scale, mode).Z
    Zsoc, Zlin = _pure_Z("socratic"), _pure_Z("linear")
    X = np.vstack([Zsoc, Zlin]); y = np.r_[np.ones(len(Zsoc)), np.zeros(len(Zlin))]
    a = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y).coef_[0]
    soclin_axis = (a / max(np.linalg.norm(a), 1e-12)).astype(np.float32)

    def centroid(run_dir, cell):
        d = run_dir / cell
        if not (d / "signatures_v3").exists():
            return None
        return _centroid(d / "signatures_v3", d / "metadata.json", med, scale)

    base = centroid(args.soclin_run_dir, "baseline_L14_a0.0")
    base_txt = _socratic_rate(_texts(args.soclin_run_dir, "baseline_L14_a0.0"))

    rows = []
    doses = [0.03, 0.1, 0.3]
    # Ksoclin/V3 = DATA route; Gent/Gmas = FORMULA route (V4-recipe gradients). All in soclin dir.
    targets = {"Ksoclin": "data", "V3": "data", "Gent": "formula", "Gmas": "formula"}
    nulls = ["R1", "R2", "R3"]
    for af in doses:
        r_soc, r_sel = [], []
        for rk in nulls:
            c = centroid(args.a5_run_dir, f"{rk}_L14_a{af}")
            if c is not None:
                pj = project(c - base, soclin_axis)
                r_soc.append(pj["target"]); r_sel.append(pj["effect_per_offtarget"])
        rs_mean = float(np.mean(r_soc)) if r_soc else None
        rsel_mean = float(np.mean(r_sel)) if r_sel else None
        for tk, route in targets.items():
            c = centroid(args.soclin_run_dir, f"{tk}_L14_a{af}")
            if c is None:
                continue
            pj_soc = project(c - base, soclin_axis)
            pj_d0 = project(c - base, dir0)
            txt = _socratic_rate(_texts(args.soclin_run_dir, f"{tk}_L14_a{af}"))
            rows.append({
                "vector": tk, "route": route, "alpha_frac": af,
                "soclin_shift": round(pj_soc["target"], 4),
                "soclin_off_target": round(pj_soc["off_target"], 4),
                "soclin_selectivity": round(pj_soc["effect_per_offtarget"], 4),  # on/off-target
                "dir0_shift": round(pj_d0["target"], 4),
                "lever_over_Rmean_soclin": round(pj_soc["target"] / rs_mean, 2) if rs_mean else None,
                "selectivity_over_Rmean": round(pj_soc["effect_per_offtarget"] / rsel_mean, 2) if rsel_mean else None,
                "socratic_marker_per_1k": txt["socratic_marker_per_1k"],
                "socratic_marker_excess_over_baseline": round(txt["socratic_marker_per_1k"] - base_txt["socratic_marker_per_1k"], 2),
            })

    # LEVER (not mere gauge-movement) = gauge ≥2×R AND selective (selectivity ≥ ~R) AND behavioral.
    # V4/dir0 established "gauge-movable ≠ lever": a large off-manifold vector projects big onto the
    # axis but is non-selective + behaviorally inert. So the write test keys on selectivity+behavior.
    def gauge_low(vec):
        return [r["alpha_frac"] for r in rows if r["vector"] == vec and r["alpha_frac"] <= 0.1
                and (r["lever_over_Rmean_soclin"] or 0) >= 2.0]

    def sel_beh(vec):
        return {r["alpha_frac"]: {"gauge_xR": r["lever_over_Rmean_soclin"],
                                  "selectivity": r["soclin_selectivity"],
                                  "selectivity_xR": r["selectivity_over_Rmean"],
                                  "socratic_excess": r["socratic_marker_excess_over_baseline"]}
                for r in rows if r["vector"] == vec and r["alpha_frac"] <= 0.1}
    verdict = {
        "P85_data_route": {
            "prediction": "14k(a) P=.85: Ksoclin lever ≥2×R (soclin axis) at α≤.1 WITH socratic behavioral consequence",
            "Ksoclin_gauge_ge2x_at_low_alpha": gauge_low("Ksoclin"),
            "Ksoclin_selectivity_and_behavior": sel_beh("Ksoclin"),
            "V3_ref_selectivity_and_behavior": sel_beh("V3"),
        },
        "P70_formula_write_test": {
            "prediction": "P=.70 formula-INERT-AS-A-LEVER: V4-recipe gradient (Gent/Gmas) does NOT write the "
                          "soclin coordinate as a LEVER (gauge-movement ≠ lever; needs selectivity + behavior)",
            "Gent_gauge_ge2x_at_low_alpha": gauge_low("Gent"),
            "Gmas_gauge_ge2x_at_low_alpha": gauge_low("Gmas"),
            "Gent_selectivity_and_behavior": sel_beh("Gent"),
            "Gmas_selectivity_and_behavior": sel_beh("Gmas"),
            "reading": "The formula gradients MOVE THE GAUGE (soclin projection ≥2×R) but the LEVER claim "
                       "needs selectivity (≥R) AND behavioral (socratic) consequence — compare to Ksoclin. "
                       "High gauge + low selectivity/behavior = gauge-movable-not-a-lever = INERT-AS-A-LEVER "
                       "(the V4/dir0 lesson). Outer loop scores the write prediction on selectivity+behavior.",
        },
    }
    out = {"arm": "14k Ksoclin steering (data route) + formula WRITE test (gradient route) on the soclin coordinate",
           "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop",
           "law": "free-gen centroid shift onto the socratic↔linear LDA axis (floor-z) vs matched-R "
                  "(R1/R2/R3) at α{.03,.1,.3}; lever ≥2×R at α≤.1. Data route = Ksoclin/V3 (P=.85); "
                  "formula route = Gent/Gmas V4-recipe gradients (P=.70 INERT). dir0-shift secondary. "
                  "socratic marker rate vs baseline = data-route behavioral leg.",
           "baseline_socratic": base_txt, "verdict": verdict, "rows": rows}
    args.out_json.write_text(json.dumps(out, indent=1))
    for r in rows:
        print(f"  {r['vector']:8}({r['route']:7}) a{r['alpha_frac']}: gauge={r['soclin_shift']} "
              f"({r['lever_over_Rmean_soclin']}×R)  sel={r['soclin_selectivity']} ({r['selectivity_over_Rmean']}×R)  "
              f"soc_excess={r['socratic_marker_excess_over_baseline']}")
    print(f"VERDICT data(P.85): Ksoclin gauge≥2×R@α≤.1 = {verdict['P85_data_route']['Ksoclin_gauge_ge2x_at_low_alpha']}; "
          f"sel+behavior = {verdict['P85_data_route']['Ksoclin_selectivity_and_behavior']}")
    print(f"VERDICT formula(P.70 write): Gent/Gmas gauge≥2×R but read selectivity+behavior — "
          f"Gmas={verdict['P70_formula_write_test']['Gmas_selectivity_and_behavior']}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
