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

    def centroid(run_dir, cell):
        d = run_dir / cell
        if not (d / "signatures_v3").exists():
            return None
        return _centroid(d / "signatures_v3", d / "metadata.json", med, scale)

    base = centroid(args.soclin_run_dir, "baseline_L14_a0.0")
    base_txt = _socratic_rate(_texts(args.soclin_run_dir, "baseline_L14_a0.0"))

    rows = []
    doses = [0.03, 0.1, 0.3]
    # target vectors (Ksoclin, V3) from soclin dir; R nulls from a5 dir
    targets = {"Ksoclin": args.soclin_run_dir, "V3": args.soclin_run_dir}
    nulls = {"R1": args.a5_run_dir, "R2": args.a5_run_dir, "R3": args.a5_run_dir}
    for af in doses:
        r_shifts = []
        for rk, rd in nulls.items():
            c = centroid(rd, f"{rk}_L14_a{af}")
            if c is not None:
                r_shifts.append(project(c - base, dir0)["target"])
        r_mean = float(np.mean(r_shifts)) if r_shifts else None
        r_max = float(np.max(r_shifts)) if r_shifts else None
        for tk, td in targets.items():
            c = centroid(td, f"{tk}_L14_a{af}")
            if c is None:
                continue
            pj = project(c - base, dir0)
            txt = _socratic_rate(_texts(td, f"{tk}_L14_a{af}"))
            rows.append({
                "vector": tk, "alpha_frac": af,
                "dir0_shift": round(pj["target"], 4), "off_target": round(pj["off_target"], 4),
                "R_shift_mean": round(r_mean, 4) if r_mean else None,
                "R_shift_max": round(r_max, 4) if r_max else None,
                "lever_over_Rmean": round(pj["target"] / r_mean, 2) if r_mean else None,
                "lever_over_Rmax": round(pj["target"] / r_max, 2) if r_max else None,
                "socratic_marker_per_1k": txt["socratic_marker_per_1k"],
                "socratic_marker_excess_over_baseline": round(txt["socratic_marker_per_1k"] - base_txt["socratic_marker_per_1k"], 2),
                "question_per_1k": txt["question_per_1k"],
            })

    kso = [r for r in rows if r["vector"] == "Ksoclin"]
    lever_low = [r for r in kso if r["alpha_frac"] <= 0.1 and (r["lever_over_Rmean"] or 0) >= 2.0]
    verdict = {
        "prediction": "14k(a) P=.85: Ksoclin lever ≥2×R at α≤.1 WITH in-window behavioral (socratic) consequence",
        "Ksoclin_lever_ge2x_Rmean_at_low_alpha": [r["alpha_frac"] for r in lever_low],
        "Ksoclin_behavioral_socratic_excess_at_low_alpha":
            {r["alpha_frac"]: r["socratic_marker_excess_over_baseline"] for r in kso if r["alpha_frac"] <= 0.1},
    }
    out = {"arm": "14k Ksoclin steering — dir0-shift lever + socratic induction",
           "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop",
           "law": "free-gen centroid dir0-shift (floor-z) vs matched-R (R1/R2/R3) at α{.03,.1,.3}; "
                  "lever ≥2×R at α≤.1; socratic marker/question rate vs baseline = behavioral leg.",
           "baseline_socratic": base_txt, "verdict": verdict, "rows": rows}
    args.out_json.write_text(json.dumps(out, indent=1))
    for r in rows:
        print(f"  {r['vector']:8} a{r['alpha_frac']}: dir0_shift={r['dir0_shift']} "
              f"lever/Rmean={r['lever_over_Rmean']} lever/Rmax={r['lever_over_Rmax']} "
              f"soc_excess={r['socratic_marker_excess_over_baseline']}")
    print(f"VERDICT: Ksoclin lever≥2×R at α≤.1: {verdict['Ksoclin_lever_ge2x_Rmean_at_low_alpha']}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
