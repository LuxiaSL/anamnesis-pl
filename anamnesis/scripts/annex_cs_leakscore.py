"""ANNEX control-surface tenancy — CS-5 leak-law scorer, OPERATIONALIZATION FROZEN
BEFORE ANY CS NUMBER EXISTS (staged pre-fire; retrodictively validated on the 8B 2×2).

THE ENTROPY-LEAK LAW (standing): observed entropy rise ≈ cos(v, V7) × V7-effect at
matched dose, SIGN INCLUDED. CS-5 (frozen at proposal): every cell within 2× of its
filed prediction.

Frozen operationalization (the "2×" made exact, no post-hoc degrees of freedom):
  multiplier  = the IN-RUN V7 ref cell's entropy_rise at matched dose (RA-8B arithmetic
                precedent); if no in-run V7 exists at that signed dose (V7 refs are
                positive-dose only and V7's own effect is sign-asymmetric), fall back to
                --v7-effects-extra {dose: effect} with the source NAMED — flagged BANKED.
  prediction  = cos_to_V7(band member) × multiplier; ⊥ members: prediction ≡ 0.
  null band   = [min, max] of entropy_rise over the in-run Rband cells at the same
                signed dose (pooled |dose| fallback, flagged).
  MATERIAL prediction (|pred| > half the null band's width):
                PASS iff sign(obs) == sign(pred) AND obs/pred ∈ [0.5, 2.0].
  SUB-MATERIAL prediction (incl. every ⊥ cell):
                PASS iff obs inside the null band (silence was the prediction; a ⊥ cell
                OUTSIDE the band is a LAW VIOLATION and a finding, not a miss).
  CS-5 verdict: PASS iff every scored cell passes.

Usage:
    python -m anamnesis.scripts.annex_cs_leakscore \
        --entropy-json <cs_entropy_3b_node2.json> --anatomy-json <cs_anatomy.json> \
        --grid-json <cs_cell_grid.json> --out <cs_leak_scorecard.json> \
        [--v7-effects-extra <json>] [--v7-key V7_L14]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_dose(cell: str) -> float | None:
    m = re.search(r"_a(n?)(\d*\.?\d+)$", cell)
    if not m:
        return None
    return (-1.0 if m.group(1) else 1.0) * float(m.group(2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entropy-json", type=Path, required=True)
    ap.add_argument("--anatomy-json", type=Path, required=True)
    ap.add_argument("--grid-json", type=Path, required=True)
    ap.add_argument("--v7-key", default="V7_L14")
    ap.add_argument("--v7-effects-extra", type=Path, default=None,
                    help='{"<dose>": effect} banked fallback for doses w/o in-run V7 '
                         "(source named in the output)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ent = json.loads(args.entropy_json.read_text())
    rows = {r["cell"]: r for r in (ent.get("rows") if isinstance(ent, dict) else ent)}
    anatomy = json.loads(args.anatomy_json.read_text())
    grid = json.loads(args.grid_json.read_text())
    extra = (json.loads(args.v7_effects_extra.read_text())
             if args.v7_effects_extra else {})

    # in-run V7 multipliers by signed dose
    v7_eff: dict[float, tuple[float, str]] = {}
    for c in grid["cells"]:
        if c["key"] == args.v7_key and c["name"] in rows:
            v7_eff[c["frac"]] = (float(rows[c["name"]]["entropy_rise"]), "in-run")
    for d, e in extra.items():
        v7_eff.setdefault(float(d), (float(e), f"BANKED ({args.v7_effects_extra})"))

    # null bands by signed dose from in-run Rband cells
    bands: dict[float, list[float]] = {}
    for c in grid["cells"]:
        if c.get("null") and c["name"] in rows:
            bands.setdefault(c["frac"], []).append(float(rows[c["name"]]["entropy_rise"]))

    def band_for(dose: float) -> tuple[list[float], str]:
        if dose in bands:
            return bands[dose], "matched_signed_dose"
        pooled = [v for d0, vs in bands.items() for v in vs if abs(d0) == abs(dose)]
        return pooled, "pooled_|dose|_FALLBACK"

    scored, missing = {}, []
    for c in grid["cells"]:
        if c.get("null") or c["key"] == args.v7_key:
            continue
        name, key, dose = c["name"], c["key"], float(c["frac"])
        if name not in rows:
            missing.append(name)
            continue
        obs = float(rows[name]["entropy_rise"])
        is_perp = "_perp" in key
        akey = key.replace("_perp", "")
        cos = 0.0 if is_perp else float(anatomy[akey]["cos_to_V7"])
        if dose in v7_eff:
            mult, msrc = v7_eff[dose]
        elif not is_perp:
            missing.append(f"{name}: no V7 multiplier at dose {dose}")
            continue
        else:
            mult, msrc = 0.0, "n/a (⊥: prediction ≡ 0)"
        pred = cos * mult
        nb, bsrc = band_for(dose)
        if not nb:
            missing.append(f"{name}: no null band at dose {dose}")
            continue
        half_width = (max(nb) - min(nb)) / 2.0
        material = abs(pred) > half_width
        if material:
            ratio = obs / pred if pred != 0 else float("inf")
            ok = (obs * pred > 0) and (0.5 <= ratio <= 2.0)
            basis = "ratio∈[0.5,2] + sign"
        else:
            ok = min(nb) <= obs <= max(nb)
            basis = "inside null band (silence predicted)"
            ratio = None
        scored[name] = {
            "observed_rise": round(obs, 4), "predicted": round(pred, 4),
            "cos_to_V7": round(cos, 4), "multiplier": round(mult, 4),
            "multiplier_source": msrc, "material": material,
            "ratio_obs_over_pred": round(ratio, 3) if ratio is not None else None,
            "null_band": [round(min(nb), 4), round(max(nb), 4)], "band_source": bsrc,
            "verdict": "PASS" if ok else "FAIL", "basis": basis,
            "is_perp": is_perp,
        }
        if is_perp and not ok:
            scored[name]["note"] = "⊥ OUTSIDE the null band — LAW VIOLATION, a finding"

    n_pass = sum(1 for v in scored.values() if v["verdict"] == "PASS")
    out = {
        "provenance": "CS-5 leak-law scorecard; operationalization FROZEN pre-fire "
                      "(see module docstring): material -> ratio∈[0.5,2]+sign vs the "
                      "in-run V7 multiplier; sub-material/⊥ -> inside the matched "
                      "signed-dose Rband band",
        "cs5_verdict": "PASS" if (scored and n_pass == len(scored) and not missing)
                       else "FAIL" if scored else "EMPTY",
        "n_pass": n_pass, "n_scored": len(scored),
        "v7_multipliers": {str(d): {"effect": round(e, 4), "source": s}
                           for d, (e, s) in sorted(v7_eff.items())},
        "cells": scored, "missing": missing,
    }
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=1))
    for n, v in scored.items():
        print(f"  {n:32s} obs {v['observed_rise']:+.4f}  pred {v['predicted']:+.4f}  "
              f"{v['verdict']}  ({v['basis']})")


if __name__ == "__main__":
    main()
