"""Part-C matched-control contrast readout (CONTRAST-FRAME ruling 2026-07-17). CPU.

The READ OF RECORD for Part-C = the matched-control contrast (frame-stamped per the standing
rule). Two modes:

  --control-run given  → frame "vs-matched-control" (cell-4): per matched probe g,
       field(t)[g] = Z_arm(t)[g] − Z_control(t)[g]     (base cancels exactly; the generic
       install drift cancels when the control is dose-matched — the trait-specific field).
       Install axis u = mean_g field(final); proj + sign-flip significance; ‖mean field‖ / floor.

  no --control-run     → frame "vs-base" (4n / building block): field(t)[g] = Z_arm(t)[g]−Z_base[g]
       (TOTAL displacement). Magnitude-vs-visibility-bar leads (flat-arm rule): if the seed-floor
       ratio is sub-bar the axis is NOISE and NOT quoted. Optional --axis-npz projects the delta
       onto an EXTERNAL install axis (4n rider 2: the SFT reference cohort's install direction —
       channel-tracking ⇒ moves ALONG it; orthogonal ⇒ objective-side).

Steps must be matched between arm and control (same grid). Frame is written into the artifact.

    python -m anamnesis.scripts.vmb_partc_contrast --battery-root ../outputs/battery --model qwen-7b \
        --arm-run <a6cohort>/cat_dpo_r16_s0 --control-run <a6cohort>/purple_dpo_r16_s0 \
        --steps 0001,0007,0014,0021,0028,0035,0045 --out .../partc_cell4_contrast.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from anamnesis.analysis.battery.floors import load_signature_matrix
from anamnesis.analysis.battery.deltas import load_floor_scale
from anamnesis.analysis.battery.manifest import MODEL_META

VIS_BAR = 0.1  # 12b visibility bar in seed-floor units


def _load_Z(run: Path, step: str, med, scale) -> tuple[np.ndarray, list[int]]:
    X, _, g = load_signature_matrix(run / f"step-{step}" / "signatures_v3")
    return (X - med) / scale, [int(x) for x in g]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--model", default="qwen-7b")
    ap.add_argument("--arm-run", type=Path, required=True)
    ap.add_argument("--control-run", type=Path, default=None,
                    help="matched-control run (cell-4 = purple_dpo). Omit for vs-base frame.")
    ap.add_argument("--steps", required=True, help="comma-separated step tags, ascending, matched")
    ap.add_argument("--axis-npz", type=Path, default=None,
                    help="external install axis (npz key 'axis'): 4n rider 2, SFT-ref direction")
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    steps = args.steps.split(",")

    stage0 = args.battery_root / MODEL_META[args.model].stage0_dir / "signatures_v3"
    med, scale = load_floor_scale(stage0)
    s0X, _, s0g = load_signature_matrix(stage0)
    s0Z = (s0X - med) / scale
    s0map = {int(g): i for i, g in enumerate(s0g)}

    # planning-grade whole-vector seed floor (adjacent-seed pairs on the base)
    fv = []
    for k in range(0, 800, 10):
        rows = [s0map[g] for g in range(k, k + 10) if g in s0map][:4]
        fv += [np.abs(s0Z[i] - s0Z[j]).mean() for i, j in itertools.combinations(rows, 2)]
    floor = float(np.median(fv))

    frame = "vs-matched-control" if args.control_run else "vs-base"
    # per-step paired field
    F = {}
    for s in steps:
        Za, ga = _load_Z(args.arm_run, s, med, scale)
        if args.control_run:
            Zc, gc = _load_Z(args.control_run, s, med, scale)
            cmap = {g: i for i, g in enumerate(gc)}
            shared = [g for g in ga if g in cmap]
            amap = {g: i for i, g in enumerate(ga)}
            F[s] = np.stack([Za[amap[g]] - Zc[cmap[g]] for g in shared])
        else:
            F[s] = np.stack([Za[i] - s0Z[s0map[g]] for i, g in enumerate(ga) if g in s0map])

    # install axis: external if provided, else the pair/arm final-step field
    if args.axis_npz:
        u = np.load(args.axis_npz)["axis"].astype(np.float64)
    else:
        u = F[steps[-1]].mean(axis=0)
    un = float(np.linalg.norm(u))
    u = u / un if un > 0 else u

    rng = np.random.default_rng(20260717)
    rows = []
    onset = None
    for s in steps:
        m = F[s].mean(axis=0)
        mag = float(np.linalg.norm(m))
        ratio = float(np.median(np.abs(F[s]).mean(axis=1)) / floor)
        align = float(m @ u / max(np.linalg.norm(m), 1e-12)) if un > 0 else None
        proj_arr = F[s] @ u
        obs = float(proj_arr.mean())
        null = (rng.choice([-1.0, 1.0], size=(args.n_perm, len(proj_arr))) * proj_arr).mean(axis=1)
        p = float((np.sum(null >= obs) + 1) / (args.n_perm + 1))
        rows.append({"step": s, "field_mag": round(mag, 4), "ratio_seed_floor_12b": round(ratio, 4),
                     "above_visibility_bar": bool(ratio >= VIS_BAR),
                     "align_cos": None if align is None else round(align, 4),
                     "proj_mean": round(obs, 4), "proj_p_signflip": round(p, 5),
                     "n_probes": int(len(proj_arr))})
        if onset is None and p < 0.05:
            onset = s

    final_ratio = rows[-1]["ratio_seed_floor_12b"]
    axis_noise = (frame == "vs-base" and not args.axis_npz and final_ratio < VIS_BAR)
    out = {
        "arm_run": str(args.arm_run), "control_run": str(args.control_run) if args.control_run else None,
        "CONTRAST_FRAME": frame,                       # standing-rule stamp
        "external_axis": str(args.axis_npz) if args.axis_npz else None,
        "model": args.model, "seed_floor_median": round(floor, 5), "visibility_bar": VIS_BAR,
        "install_axis": ("external:" + str(args.axis_npz)) if args.axis_npz
                        else f"{frame} step-{steps[-1]} mean field",
        "final_ratio_seed_floor": final_ratio,
        "flat_arm_axis_is_noise": axis_noise,          # rider 1: if True, do NOT quote the axis
        "directional_onset_step": None if axis_noise else onset,
        "per_checkpoint": rows,
        "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    # also bank the arm's install axis (for 4n's projection onto the SFT reference axis)
    if not args.control_run and not args.axis_npz:
        np.savez(args.out.with_suffix(".axis.npz"), axis=F[steps[-1]].mean(axis=0))
    print(f"[{frame}] onset={out['directional_onset_step']} final_ratio={final_ratio} "
          f"axis_noise={axis_noise} -> {args.out}")


if __name__ == "__main__":
    main()
