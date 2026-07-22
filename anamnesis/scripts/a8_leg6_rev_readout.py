"""A8 Leg-6 / Item 5 — P8-REV readout: the reverse behavioural certification.

Consumes the entropy-replay json (vmb_c3_entropy_replay over the leg-6 3B cells) and
applies P8-REV's frozen clauses. REPORTS them; scores nothing.

The frozen letter (A8-add-5): transported V7 into 3B via the REVERSE map writes 3B's
entropy —
  clause 1  dose-ordered. SCOPE (stated at freeze): monotone across the FULL SIGNED
            ladder +-{.03,.1,.3}.
  clause 2  outside the reverse-transported-R band at >=5/6 doses.
  clause 3  sign flipping through zero.

Aggregation and scope are as frozen; both a strict and a rank-based reading of "monotone"
are reported because the letter does not distinguish them, and this arm's practice is to
show every defensible reading rather than pick one silently.

The band is the DOSE-MATCHED reverse-transported R band: gRband1-3 carried back through
the same map, read at the same dose. That is what `vmb_c3_entropy_replay`'s
`entropy_rise_vs_Rc_band` computes when the null prefix selects those cells.

UNSTAMPED (C section 8). No P self-scored.
Run: PYTHONPATH=pipeline python -m anamnesis.scripts.a8_leg6_rev_readout \
        --entropy-json <path> --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARM = Path("outputs/battery/arms/A8_conjugation")
SIGNAL_KEY = "gRV7_L14"
DOSES = (-0.3, -0.1, -0.03, 0.03, 0.1, 0.3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entropy-json", type=Path,
                    default=ARM / "leg6/readouts_gpu/rev_entropy_raw.json")
    ap.add_argument("--out", type=Path, default=ARM / "leg6/readouts_cpu")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = json.loads(args.entropy_json.read_text())["rows"]
    sig = {round(float(r["alpha_frac"]), 4): r for r in rows
           if not r.get("is_null") and SIGNAL_KEY.upper() in str(r["cell"]).upper()}
    nulls = [r for r in rows if r.get("is_null")]

    ladder, missing = [], []
    for d in DOSES:
        r = sig.get(round(d, 4))
        if r is None:
            missing.append(d)
            continue
        band = r.get("entropy_rise_vs_Rc_band", {}) or {}
        ladder.append({
            "alpha_frac": d,
            "entropy_rise": round(float(r["entropy_rise"]), 4),
            "mean_entropy_steered": round(float(r["mean_entropy_steered"]), 4),
            "null_band": {k: band.get(k) for k in
                          ("null_mean", "null_sd", "null_min", "null_max")},
            "z_vs_null": band.get("z_vs_null"),
            "outside_null_band": band.get("outside_null_band"),
        })

    rises = [c["entropy_rise"] for c in ladder]
    doses = [c["alpha_frac"] for c in ladder]
    strict = bool(len(rises) > 1 and all(b > a for a, b in zip(rises, rises[1:])))
    spearman = (None if len(rises) < 3 else round(float(
        np.corrcoef(np.argsort(np.argsort(doses)),
                    np.argsort(np.argsort(rises)))[0, 1]), 4))
    n_outside = sum(1 for c in ladder if c["outside_null_band"])
    neg = [c["entropy_rise"] for c in ladder if c["alpha_frac"] < 0]
    pos = [c["entropy_rise"] for c in ladder if c["alpha_frac"] > 0]
    sign_flip = bool(neg and pos and max(neg) < 0 < min(pos))

    res = {
        "STATUS": "UNSTAMPED (C section 8) — no self-scored P; the desk scores P8-REV (.75)",
        "leg": "A8 Leg-6 / Item 5 — reverse behavioural certification",
        "construction": "transported V7 (8B -> 3B) via the REVERSE/transpose of the banked "
                        "Leg-0 anchor Procrustes fit_3bL14__8bL16_native_proc_k512; "
                        "sign-anchored before transport; identity check cos(g_rev.V7_8B, "
                        "V7_3B) = +0.5312 vs banked Leg-0 reverse read .531 -> PASS",
        "control": "DOSE-MATCHED reverse-transported R band (8B's banked Rband1-3 carried "
                   "back through the SAME map) — the control travels the same road",
        "n_per_cell": 80, "site": 14,
        "cells_missing": missing,
        "ladder": ladder,
        "clauses_as_frozen": {
            "clause_1_dose_ordered": {
                "scope": "monotone across the FULL SIGNED ladder +-{.03,.1,.3}",
                "strictly_monotone": strict,
                "spearman_rank": spearman,
                "note": "both readings reported; the letter does not distinguish them",
            },
            "clause_2_outside_band": {
                "bar": ">=5/6 doses outside the reverse-transported-R band",
                "n_outside": f"{n_outside}/{len(ladder)}",
                "met": bool(n_outside >= 5),
            },
            "clause_3_sign_flip": {
                "bar": "sign flipping through zero",
                "negative_doses_all_below_zero": bool(neg and max(neg) < 0),
                "positive_doses_all_above_zero": bool(pos and min(pos) > 0),
                "met": sign_flip,
            },
        },
        "n_null_cells": len(nulls),
        "SCORING": "reported, NOT self-scored — the desk rules P8-REV",
    }
    (args.out / "rev_entropy.json").write_text(json.dumps(res, indent=1))
    print(f"{'alpha':>8} {'rise':>9} {'z':>8} {'outside':>8}   null band")
    for c in ladder:
        nb = c["null_band"]
        print(f"{c['alpha_frac']:>8.2f} {c['entropy_rise']:>+9.4f} "
              f"{str(c['z_vs_null']):>8} {str(c['outside_null_band']):>8}   "
              f"[{nb.get('null_min')}, {nb.get('null_max')}]")
    print(f"\nclause 1 monotone strict={strict} spearman={spearman}")
    print(f"clause 2 outside band {n_outside}/{len(ladder)} (bar >=5)")
    print(f"clause 3 sign flip {sign_flip}")
    print(f"wrote {args.out / 'rev_entropy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
