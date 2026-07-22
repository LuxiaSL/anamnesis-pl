"""A8 Leg-4F / L4-g — judged-needle readout: the 2AFC ladder against the frozen bar.

Reads `readouts_judge/analogical_2afc_results.json` (written by the A5 mode-shift judge of
record, `vmb_a5_judge_socratic --mode analogical`) and states, mechanically:

  P8-JX (A8-add-3) := >=1 ladder dose at 2AFC >= .65
                      AND transported-R band cells <= .58 at matched dose
                      AND dose-monotone trend across the ladder

No P is scored here — the desk scores.  Everything UNSTAMPED (C§8).

Run: PYTHONPATH=pipeline python -m anamnesis.scripts.a8_leg4g_readout
"""
from __future__ import annotations

import json
import re
from pathlib import Path

LEG4 = Path("outputs/battery/arms/A8_conjugation/leg4")
JUDGE = LEG4 / "readouts_judge/analogical_2afc_results.json"
OUT = LEG4 / "readouts_cpu"
DOSE_RE = re.compile(r"a([+-]?[\d.]+)$")


def dose_of(cell: str) -> float:
    m = DOSE_RE.search(cell)
    return float(m.group(1)) if m else float("nan")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.loads(JUDGE.read_text())
    rows = []
    for cell, blk in d.items():
        rows.append({"cell": cell, "vector": blk["vector"], "dose": dose_of(cell),
                     "analogical_more": blk["steered_more_mode_frac"],
                     "n_pairs": blk["n_pairs"],
                     "coherence": round(blk["coherence_mean"], 2)})
    rows.sort(key=lambda r: (r["vector"], r["dose"]))
    dir0 = [r for r in rows if r["vector"] == "gdir0"]
    rband = {r["dose"]: r for r in rows if r["vector"] == "gRband1"}
    pos = [r for r in dir0 if r["dose"] > 0]
    ladder = sorted(dir0, key=lambda r: r["dose"])
    vals = [r["analogical_more"] for r in ladder]
    monotone = all(x <= y for x, y in zip(vals, vals[1:]))
    best = max(pos, key=lambda r: r["analogical_more"]) if pos else None
    clauses = {
        "clause_1_any_dose_ge_.65": {
            "max_positive_dose_cell": best["cell"] if best else None,
            "value": best["analogical_more"] if best else None,
            "meets": bool(best and best["analogical_more"] >= 0.65)},
        "clause_2_R_band_le_.58_at_matched_dose": {
            "per_dose": {str(r["dose"]): rband.get(r["dose"], {}).get("analogical_more")
                         for r in pos},
            "meets_all_positive_doses": all(
                (rband.get(r["dose"], {}).get("analogical_more") or 1.0) <= 0.58
                for r in pos)},
        "clause_3_dose_monotone_trend": {
            "ladder_values_low_to_high": vals, "strictly_nondecreasing": monotone},
    }
    res = {"STATUS": "UNSTAMPED (C§8) — mechanics only, desk scores P8-JX",
           "leg": "A8 Leg-4F / L4-g", "prereg": "A8-add-3 P8-JX",
           "instrument": "vmb_a5_judge_socratic --mode analogical (contrast-presented 2AFC, "
                         "blind, same-topic alpha=0 rider pairing, key never in judge context)",
           "judge_model": "claude-fable-5 (opus fallback on refusal)",
           "rows": rows, "bar_clauses": clauses,
           "control_note": "the transported-R band cells are the dose-matched control; read "
                           "them as the ruler's floor for PERTURBED text, not as zero."}
    usage = LEG4 / "readouts_judge/usage.json"
    if usage.exists():
        res["judge_usage"] = json.loads(usage.read_text())
    (OUT / "l4g_judged_needle.json").write_text(json.dumps(res, indent=1))

    lines = ["# L4-g — judged needle expression, analogical 2AFC (UNSTAMPED, C§8)", "",
             "| vector | dose | 2AFC analogical-more | n pairs | coherence |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['vector']} | {r['dose']:+.2f} | **{r['analogical_more']:.3f}** | "
                     f"{r['n_pairs']} | {r['coherence']} |")
    lines += ["", "## Bar clauses (mechanical)", "", "```",
              json.dumps(clauses, indent=1), "```"]
    (OUT / "l4g_judged_needle.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
