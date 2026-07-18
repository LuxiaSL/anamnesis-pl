"""PM6-a state-lever readout (vmb A5 / M6) — the ruled on-axis-vs-R-BAND metric.

Per the desk MORNING RULINGS: the state leg scores V3 ON-AXIS SHIFT (along dir0) vs the matched-R
DISTRIBUTION (band/percentile), NEVER a ratio to meanR (degenerates when meanR ≤ 0, per the L9 diag).
dir0 = unit(mean(z_linear) − mean(z_socratic)) in floor-z; socratic-ward steering (−V3) → NEGATIVE
on-axis shift (toward the socratic pole). Selectivity = off-axis (⊥dir0) movement, V3 vs R.

Lever fires (per site×dose) iff V3 on-axis shift is OUTSIDE the matched-R band (beyond the most-
socratic-ward R) AND dose-monotone. V1 (formality) reported as the second control.

Run (node1, CPU): python -m anamnesis.scripts.vmb_pm6a_statelever \
  --pm6a-root /models/anamnesis-extract/runs/vmb_a5_dsv2_lite_pm6a \
  --pole-a-dir .../vmb_a2_dsv2-lite_pure_linear/signatures_v3_x2 \
  --pole-b-dir .../vmb_a2_dsv2-lite_pure_socratic/signatures_v3_x2 \
  --floor-dir .../vmb_stage0_dsv2_lite/signatures_v3_x2 --out <json>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from anamnesis.analysis.battery.deltas import load_floor_scale
from anamnesis.analysis.battery.floors import load_signature_matrix

# sign+dose suffix: m{n}=negative (toward pole-b), a{frac}/p{n}=positive (toward pole-a). Both signs.
CELL = re.compile(r"^(?P<vec>V3|V1|R1|R2|R3)_L(?P<site>\d+)_(?P<sd>[apm][\d.]+)$")


def _zmean(sig_dir: Path, med, scale):
    X, _, _ = load_signature_matrix(sig_dir)
    return ((X - med) / scale).mean(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pm6a-root", type=Path, required=True)
    ap.add_argument("--pole-a-dir", type=Path, required=True)   # linear
    ap.add_argument("--pole-b-dir", type=Path, required=True)   # socratic
    ap.add_argument("--floor-dir", type=Path, required=True)
    ap.add_argument("--sig-subdir", default="signatures_v3_x2")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    med, scale = load_floor_scale(args.floor_dir)
    Xa, _n, _i = load_signature_matrix(args.pole_a_dir)
    Xb, _n2, _i2 = load_signature_matrix(args.pole_b_dir)
    za = ((Xa - med) / scale).mean(0)
    zb = ((Xb - med) / scale).mean(0)
    dir0 = za - zb
    dir0 = dir0 / np.linalg.norm(dir0)
    base = _zmean(args.pm6a_root / "baseline" / args.sig_subdir, med, scale)
    base_proj = float(base @ dir0)

    rows = []
    for d in sorted(args.pm6a_root.iterdir()):
        m = CELL.match(d.name)
        sig = d / args.sig_subdir
        if not m or not sig.exists():
            continue
        z = _zmean(sig, med, scale)
        delta = z - base
        onaxis = float(delta @ dir0)                 # negative = socratic-ward
        offaxis = float(np.linalg.norm(delta - onaxis * dir0))
        rows.append({"vec": m.group("vec"), "site": int(m.group("site")), "dose": m.group("sd"),
                     "onaxis": round(onaxis, 4), "offaxis": round(offaxis, 4)})

    lever = {}
    for site in sorted({r["site"] for r in rows}):
        for dose in sorted({r["dose"] for r in rows if r["site"] == site}):
            neg = dose.startswith("m")     # toward pole-b (negative on-axis); a/p = toward pole-a
            def pick(v):
                return [r for r in rows if r["vec"] == v and r["site"] == site and r["dose"] == dose]
            v3 = pick("V3")
            v1 = pick("V1")
            rs = [r["onaxis"] for r in rows if r["vec"].startswith("R")
                  and r["site"] == site and r["dose"] == dose]
            if v3 and rs:
                v3o = v3[0]["onaxis"]
                rmin, rmax = min(rs), max(rs)
                rmean, rstd = float(np.mean(rs)), float(np.std(rs))
                lever[f"L{site}_{dose}"] = {
                    "V3_onaxis": v3o, "V3_offaxis": v3[0]["offaxis"],
                    "R_band": [round(rmin, 4), round(rmax, 4)], "R_mean": round(rmean, 4),
                    "R_onaxis_all": [round(x, 4) for x in rs],
                    "V3_vs_R_sd": round((v3o - rmean) / (rstd + 1e-9), 2),
                    # outside the R band in the STEERED direction (below R for neg, above R for pos)
                    "V3_outside_R_band": bool(v3o < rmin) if neg else bool(v3o > rmax),
                    "V1_onaxis": v1[0]["onaxis"] if v1 else None,
                    "V3_offaxis_vs_R_offaxis": [round(v3[0]["offaxis"], 3),
                                                round(float(np.mean([r["offaxis"] for r in rows
                                                     if r["vec"].startswith("R") and r["site"] == site
                                                     and r["dose"] == dose])), 3)],
                }

    # dose-monotonicity of |V3 on-axis| per site (magnitude grows with |dose|). Parse the numeric
    # magnitude from the sign+dose suffix (m03→0.3, p01→0.1, a0.1→0.1) — robust to letters.
    def _mag(sd: str) -> float:
        num = sd[1:]
        return float(num) if "." in num else float(num) / (10 ** (len(num) - 1))
    mono = {}
    for site in sorted({r["site"] for r in rows}):
        for sign, lbl in (("m", "neg"), ("p", "pos"), ("a", "pos")):
            v3 = sorted([r for r in rows if r["vec"] == "V3" and r["site"] == site
                         and r["dose"].startswith(sign)], key=lambda r: _mag(r["dose"]))
            if len(v3) < 2:
                continue
            mags = [abs(r["onaxis"]) for r in v3]
            mono[f"L{site}_{sign}"] = {"doses": [r["dose"] for r in v3],
                                       "abs_onaxis": [round(x, 3) for x in mags],
                                       "monotone": all(mags[i] <= mags[i + 1] for i in range(len(mags) - 1))}

    out = {"arm": "PM6-a state lever (on-axis vs R band; socratic-ward −V3)",
           "pole_a": "linear", "pole_b": "socratic",
           "pure_proj": {"linear": round(float(za @ dir0), 3), "socratic": round(float(zb @ dir0), 3),
                         "baseline": round(base_proj, 3)},
           "lever_by_site_dose": lever, "dose_monotonicity": mono,
           "law": "V3 on-axis shift (Δ·dir0) vs matched-R band; lever = V3 outside R band socratic-ward "
                  "+ dose-monotone; selectivity = V3 off-axis vs R off-axis. NEVER a ratio to meanR."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({"pure_proj": out["pure_proj"], "dose_monotonicity": mono}, indent=1))
    print("\nlever (on-axis vs R band, socratic-ward):")
    for k, v in lever.items():
        print(f"  {k}: V3={v['V3_onaxis']} (R band {v['R_band']}, {v['V3_vs_R_sd']} R-SD) "
              f"outside={v['V3_outside_R_band']} V1={v['V1_onaxis']} "
              f"off[V3,R]={v['V3_offaxis_vs_R_offaxis']}")


if __name__ == "__main__":
    main()
