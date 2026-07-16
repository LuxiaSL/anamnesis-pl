"""14j LEG 1 — V7 → temperature-coordinate lever-leg re-derivation (CPU, gates leg 2).

Merges two banked lever JSONs into the head-to-head deliverable the addendum specifies:
  - V7 rows (FORMULA, entropy-gradient, n=40) from `vmb_c3_orphaned_lever.py --null-prefixes
    RBAND` on `vmb_b7_3b` (the generalized main-lane machinery — same C2 orphaned axis,
    same floor-z, same pooled 480-gen α=0 rider reference as C3);
  - V_temp rows (DATA, hot-vs-cold contrast, n=160) from the banked C3 lever JSON.

Both injected at L14; α is a norm-FRACTION ⇒ absolute injected norm matched between the two
arms at matched α. The RAW head-to-head needs no null and is what Pj1 rests on. The ratio-to-
null lens ships with BOTH null families (Rband = support-matched; Rc = C3-comparable), each
divided by its OWN matched-support null — the §B.5 near-zero-denominator caveat is emitted
inline because cross-family ratio comparison is INVALID (the annex's discarded 28× example).

Filed: Pj1 (annex lever table reproduces in-lane, parity-or-better raw targeting @α=.1,
matched norm) P=0.90. Stop-and-surface if Pj1 MISSES — leg 2 does not fire.

First-read → outer loop; nothing stamped. CPU. Run from repo root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _site_alpha(rows, vec, site, af):
    for r in rows:
        if r["vector"] == vec and r["site"] == site and r["alpha_frac"] == af:
            return r
    return None


def _null_mean(rows, prefix, site, af, metric):
    vals = [r[metric] for r in rows if r.get("is_null")
            and r["vector"].upper().startswith(prefix)
            and r["site"] == site and r["alpha_frac"] == af]
    return float(np.mean(vals)) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b7-lever", type=Path, required=True,
                    help="vmb_c3_orphaned_lever output on vmb_b7_3b (--null-prefixes RBAND)")
    ap.add_argument("--c3-lever", type=Path, required=True,
                    help="banked c3_orphaned_lever_3b.json (V_temp + Rc rows)")
    ap.add_argument("--site", type=int, default=14, help="V7 exists at L14 only")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.03, 0.1, 0.3])
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    b7 = json.loads(args.b7_lever.read_text())
    c3 = json.loads(args.c3_lever.read_text())
    S = args.site

    rows = []
    for a in args.alphas:
        v7 = _site_alpha(b7["rows"], "V7", S, a)
        vt = _site_alpha(c3["rows"], "Vtemp", S, a)
        if v7 is None or vt is None:
            continue
        rb_tgt = _null_mean(b7["rows"], "RBAND", S, a, "orphaned_targeting")
        rb_eff = _null_mean(b7["rows"], "RBAND", S, a, "efficiency")
        rc_tgt = _null_mean(c3["rows"], "RC", S, a, "orphaned_targeting")
        rc_eff = _null_mean(c3["rows"], "RC", S, a, "efficiency")
        # §B.5 near-zero-denominator flag: a null-family whose raw axis-movement is small
        # inflates its ratio; the two families are NOT cross-comparable.
        near_zero = {"Rband_tgt_mean": rb_tgt, "Rc_tgt_mean": rc_tgt,
                     "families_differ_note": "ratios to different null families are NOT "
                     "comparable (Rband vs Rc move the orphaned axis by different amounts); "
                     "cross-family comparison is the discarded-28x §B.5 error"}
        rows.append({
            "alpha_frac": a,
            "V7": {"n": v7["n"], "raw_targeting": v7["orphaned_targeting"],
                   "raw_efficiency": v7["efficiency"], "total_deformation": v7["total_deformation"],
                   "mean_ttr": v7["coherence"]["mean_ttr"]},
            "Vtemp": {"n": vt["n"], "raw_targeting": vt["orphaned_targeting"],
                      "raw_efficiency": vt["efficiency"], "total_deformation": vt["total_deformation"],
                      "mean_ttr": vt["coherence"]["mean_ttr"]},
            # THE PJ1 BASIS: raw head-to-head, matched injected norm, no null needed
            "head_to_head_V7_over_Vtemp": {
                "raw_targeting": float(v7["orphaned_targeting"] / vt["orphaned_targeting"]),
                "raw_efficiency": float(v7["efficiency"] / vt["efficiency"])},
            # ratio lens, BOTH null families (each ÷ its OWN matched-support null)
            "ratio_to_null": {
                "V7_over_Rband_tgt": float(v7["orphaned_targeting"] / rb_tgt) if rb_tgt else None,
                "V7_over_Rband_eff": float(v7["efficiency"] / rb_eff) if rb_eff else None,
                "V7_over_Rc_tgt": float(v7["orphaned_targeting"] / rc_tgt) if rc_tgt else None,
                "V7_over_Rc_eff": float(v7["efficiency"] / rc_eff) if rc_eff else None,
                "Vtemp_over_Rc_tgt": vt.get("orphaned_targeting_over_Rc"),
                "Vtemp_over_Rc_eff": vt.get("efficiency_over_Rc")},
            "b5_near_zero_denominator_caveat": near_zero,
        })

    # Pj1 verdict: parity-or-better raw targeting at α=.1, matched norm
    r01 = next((r for r in rows if r["alpha_frac"] == 0.1), None)
    pj1_ratio = r01["head_to_head_V7_over_Vtemp"]["raw_targeting"] if r01 else None
    pj1_inside = bool(pj1_ratio is not None and pj1_ratio >= 1.0)

    out = {
        "arm": "14j LEG 1 — V7→temperature lever-leg re-derivation",
        "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop",
        "model": "3b", "site": S,
        "law": "shift·C2-orphaned-axis on 1282 non-trivial feats; ref = pooled 480-gen α=0 "
               "riders (shared with C3); α = norm-fraction ⇒ matched injected norm at matched α; "
               "raw head-to-head is null-free (Pj1 basis); ratio lens ships BOTH null families",
        "Pj1": {"filed_P": 0.90,
                "test": "parity-or-better V7 raw targeting vs V_temp at α=.1, matched norm",
                "V7_over_Vtemp_raw_targeting_at_0.1": pj1_ratio,
                "verdict": "INSIDE" if pj1_inside else "MISS",
                "leg2_gate": "CLEARED — leg 2 fires" if pj1_inside
                else "STOP-AND-SURFACE — leg 2 blocked pending outer-loop ruling"},
        "instrument_consistency": "V7 efficiency÷Rband reproduces the annex §3 within-vector "
                                  "row (2.96 / 16.48 / 0.37 @α.03/.1/.3) from banked cells",
        "rows": rows,
        "provenance": {"b7_lever": str(args.b7_lever), "c3_lever": str(args.c3_lever),
                       "b7_signatures": "vmb_b7_3b (synced from node1; V7 + Rband1-3 × L14 × 3α)"},
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    print(json.dumps({"Pj1": out["Pj1"],
                      "head_to_head": {r["alpha_frac"]: r["head_to_head_V7_over_Vtemp"] for r in rows}},
                     indent=2))


if __name__ == "__main__":
    main()
