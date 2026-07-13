"""A6 directional install-axis readout — THE READOUT OF RECORD for checkpoint series
(upgrade addendum, arms/A6/, 2026-07-13; battery-grain analog of the teacher-ward
delta-alignment convention, echo-sandbox experiments/teacher_ref.py).

Per checkpoint: per-gen delta vs banked BASE stage0 signature, projected onto the
INSTALL AXIS (final-checkpoint mean delta direction, unit-normalized); significance
by sign-flip permutation on per-gen projections. Reported alongside: alignment
cosine of the checkpoint's mean delta with the axis, and the 12b median seed-floor
ratio (continuity with the original cell-i read; the directional metric leads it
~4x in checkpoint index — dense_t4 result).

    python -m anamnesis.scripts.vmb_a6_directional_readout \
        --battery-root ../outputs/battery --model qwen-7b \
        --a6-run ../outputs/battery/vmb_a6_qwen_cat_student_dense_t4 \
        --steps 0001,0002,0003,0005,0008,0013,0021,0034,0055,0075 \
        --out ../outputs/battery/arms/A6/a6_directional_dense_t4.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from anamnesis.analysis.battery.deltas import load_floor_scale
from anamnesis.analysis.battery.floors import load_signature_matrix
from anamnesis.analysis.battery.manifest import MODEL_META


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--model", default="qwen-7b")
    ap.add_argument("--a6-run", type=Path, required=True)
    ap.add_argument("--steps", required=True, help="comma-separated step tags, ascending")
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    steps = args.steps.split(",")

    stage0 = args.battery_root / MODEL_META[args.model].stage0_dir
    med, scale = load_floor_scale(stage0 / "signatures_v3")
    s0X, _, s0g = load_signature_matrix(stage0 / "signatures_v3")
    s0Z = (s0X - med) / scale
    s0map = {g: i for i, g in enumerate(s0g)}

    # quick whole-vector seed floor (adjacent-seed pairs per class; planning-grade)
    fv = []
    for k in range(0, 800, 10):
        rows = [s0map[g] for g in range(k, k + 10) if g in s0map][:4]
        fv += [np.abs(s0Z[i] - s0Z[j]).mean() for i, j in itertools.combinations(rows, 2)]
    floor = float(np.median(fv))

    D = {}
    for s in steps:
        X, _, gids = load_signature_matrix(args.a6_run / f"step-{s}" / "signatures_v3")
        Z = (X - med) / scale
        D[s] = np.stack([Z[i] - s0Z[s0map[g]] for i, g in enumerate(gids) if g in s0map])
    u = D[steps[-1]].mean(axis=0)
    u = u / np.linalg.norm(u)

    rng = np.random.default_rng(20260713)
    rows = []
    crossing = None
    for s in steps:
        m = D[s].mean(axis=0)
        cos = float(m @ u / max(np.linalg.norm(m), 1e-12))
        proj = D[s] @ u
        obs = float(proj.mean())
        null = (rng.choice([-1.0, 1.0], size=(args.n_perm, len(proj))) * proj).mean(axis=1)
        p = float((np.sum(null >= obs) + 1) / (args.n_perm + 1))
        ratio = float(np.median(np.abs(D[s]).mean(axis=1)) / floor)
        rows.append({"step": s, "align_cos": cos, "proj_mean": obs, "proj_p_signflip": p,
                     "ratio_seed_floor_12b": ratio, "n": int(len(proj))})
        if crossing is None and p < 0.05:
            crossing = s
    out = {"run": str(args.a6_run), "model": args.model, "install_axis": f"step-{steps[-1]} mean delta",
           "directional_onset_step": crossing, "seed_floor_median": floor, "per_checkpoint": rows,
           "stamp": {"n": rows[0]["n"], "M": args.model,
                     "law": "install-axis projection + sign-flip (A6 upgrade addendum, readout of record)",
                     "floor_type": "stochastic(stage0) approx"}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"onset step-{crossing} -> {args.out}")


if __name__ == "__main__":
    main()
