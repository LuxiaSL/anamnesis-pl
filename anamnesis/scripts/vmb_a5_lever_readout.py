"""Focused A5 lever readout (session-10; reusable cross-model, e.g. Gemma-A5 spine).

The canonical vmb_arm_a5_analyze is 3B-centric (hardcoded L14 map sites, analogical-pair
classifier, strict V{n}_L{site}_a{frac} cell naming). This is a direct, model-agnostic
lever readout for a confirmatory A5 grid:

  dir0 axis   = unit( mean(Z_poleA) - mean(Z_poleB) ) in stage0-floor z-space (signature
                space), from the two pure-mode corpora (e.g. socratic vs contrastive).
  target_shift(cell) = (mean(Z_cell) - mean(Z_baseline)) . dir0   [movement ALONG dir0]
  off_target(cell)   = || residual of that delta orthogonal to dir0 ||
  lever_ratio(alpha) = target_shift(V3) / mean_over_R( target_shift(R) )   [Pg3: >=2 @ a<=.1]
  frac_poleA(cell)   = fraction of cell gens whose dir0 projection exceeds the pure-corpora
                       midpoint threshold  [behavioral proxy; baseline vs V3 vs R]

Pure numpy on banked signatures (CPU). First-read -> outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from anamnesis.analysis.battery.deltas import load_floor_scale
from anamnesis.analysis.battery.floors import load_signature_matrix

# vec alternation: longest-first (V3sel_bare before V3; Rband\d before R1..). Dose suffix is
# optional and signed (_a0.1 / _an0.1) so PM6-b entropy cells (V7/RA/Rband) + a bare "baseline"
# cell parse alongside the original mode-lever cells — additive, old names parse unchanged.
CELL_RE = re.compile(
    r"^(?P<vec>V3sel_bare|Rband\d|V3|V4|V5|V7|RA|R1|R2|R3|V1|rider|baseline)"
    r"(?:_L(?P<site>\d+))?"
    r"(?:_a(?P<neg>n?)(?P<a>[0-9.]+)|_(?P<pm>[pm])(?P<pma>\d+))?$")
# _p03/_m01 = the whiten-run dose convention (p=+, m=−, digits = frac×10, e.g. p03 → +0.3)


def zmean(sig_dir: Path, med, scale):
    X, _, _ = load_signature_matrix(sig_dir)
    return ((X - med) / scale)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True, help="A5 steered-cells root (V3_*/, R*/)")
    ap.add_argument("--pole-a-dir", type=Path, required=True, help="pure-mode A signatures_v3 (e.g. socratic)")
    ap.add_argument("--pole-b-dir", type=Path, required=True, help="pure-mode B signatures_v3 (e.g. contrastive)")
    ap.add_argument("--floor-dir", type=Path, required=True, help="stage0 signatures_v3")
    ap.add_argument("--pole-a-name", default="socratic")
    ap.add_argument("--map-site", type=int, required=True)
    ap.add_argument("--baseline-cell", default=None, help="unsteered cell name (default V3_L{map}_a0.0)")
    ap.add_argument("--lever-vec", default="V3",
                    help="numerator vec for the lever ratio (V5 for whitened-direction cells)")
    ap.add_argument("--sig-subdir", default="signatures_v3",
                    help="signatures_v3 (state column) or signatures_v3_noinject (expression "
                         "column — the 14r two-column standing readout)")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    med, scale = load_floor_scale(args.floor_dir)

    # dir0 axis in signature z-space
    Za = zmean(args.pole_a_dir, med, scale)
    Zb = zmean(args.pole_b_dir, med, scale)
    axis = Za.mean(0) - Zb.mean(0)
    dir0 = axis / np.linalg.norm(axis)
    proj_a, proj_b = float((Za @ dir0).mean()), float((Zb @ dir0).mean())
    thresh = 0.5 * (proj_a + proj_b)

    baseline_cell = args.baseline_cell or f"V3_L{args.map_site}_a0.0"
    cells = {}
    for d in sorted(args.run_dir.iterdir()):
        m = CELL_RE.match(d.name)
        sig = d / args.sig_subdir
        if not m or not sig.exists():
            continue
        Z = zmean(sig, med, scale)
        proj = Z @ dir0
        if m.group("pma") is not None:
            alpha = (-1.0 if m.group("pm") == "m" else 1.0) * int(m.group("pma")) / 10.0
        else:
            alpha = ((-1.0 if m.group("neg") else 1.0) * float(m.group("a"))
                     if m.group("a") is not None else 0.0)
        cells[d.name] = {"vec": m.group("vec"), "site": int(m.group("site") or args.map_site),
                         "alpha": alpha, "Zmean": Z.mean(0),
                         "proj_mean": float(proj.mean()),
                         "frac_poleA": float((proj > thresh).mean())}
    if baseline_cell not in cells:
        raise SystemExit(f"baseline cell {baseline_cell} not found in {sorted(cells)}")
    base = cells[baseline_cell]["Zmean"]
    base_proj = cells[baseline_cell]["proj_mean"]

    rows = []
    for name, c in cells.items():
        delta = c["Zmean"] - base
        tgt = float(delta @ dir0)
        off = float(np.linalg.norm(delta - tgt * dir0))
        rows.append({"cell": name, "vec": c["vec"], "site": c["site"], "alpha": c["alpha"],
                     "target_shift": round(tgt, 4), "off_target": round(off, 4),
                     "effect_per_offtarget": round(tgt / max(off, 1e-9), 4),
                     "frac_poleA": round(c["frac_poleA"], 4),
                     "frac_poleA_vs_baseline": round(c["frac_poleA"] - cells[baseline_cell]["frac_poleA"], 4)})

    # lever ratio: V3 target_shift / mean R target_shift, per (site, alpha)
    lever = {}
    for site in sorted({r["site"] for r in rows}):
        for a in sorted({r["alpha"] for r in rows if r["site"] == site and r["alpha"] > 0}):
            v3 = next((r["target_shift"] for r in rows if r["vec"] == args.lever_vec and r["site"] == site and r["alpha"] == a), None)
            rs = [r["target_shift"] for r in rows if r["vec"].startswith("R") and r["alpha"] == a]
            if v3 is not None and rs:
                lever[f"L{site}_a{a}"] = {"V3_target": v3, "meanR_target": round(float(np.mean(rs)), 4),
                                         "lever_ratio": round(v3 / max(float(np.mean(rs)), 1e-9), 3)}

    out = {
        "arm": "A5 lever readout (focused, model-agnostic)",
        "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop",
        "pole_a": args.pole_a_name, "map_site": args.map_site, "baseline_cell": baseline_cell,
        "law": "dir0 = unit(mean(Za)-mean(Zb)) in stage0-z sig-space; target_shift = delta·dir0 vs "
               "unsteered baseline; lever_ratio = V3_target/mean(R_target); frac_poleA = proj>midpoint. "
               "Pg3: lever >=2x at alpha<=.1 with in-window behavioral rise.",
        "pure_proj": {"poleA": round(proj_a, 4), "poleB": round(proj_b, 4), "threshold": round(thresh, 4)},
        "lever_ratio_by_dose": lever,
        "per_cell": sorted(rows, key=lambda r: (r["vec"], r["site"], r["alpha"])),
    }
    args.out_json.write_text(json.dumps(out, indent=1))
    print(f"pole projections: A={proj_a:.3f} B={proj_b:.3f} thresh={thresh:.3f}")
    for k, v in lever.items():
        print(f"  {k}: lever_ratio={v['lever_ratio']} (V3 {v['V3_target']} vs meanR {v['meanR_target']})")
    print("frac_poleA vs baseline:")
    for r in sorted(rows, key=lambda r: (r["vec"], r["alpha"])):
        print(f"  {r['cell']}: frac={r['frac_poleA']} (Δ{r['frac_poleA_vs_baseline']:+}) eff/off={r['effect_per_offtarget']}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
