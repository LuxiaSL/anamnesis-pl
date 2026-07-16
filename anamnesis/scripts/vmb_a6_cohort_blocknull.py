"""A6 Cell 1 — the EXACT BLOCK-LEVEL null (outer-loop diagnostic, 2026-07-15/16).

The role-shuffle null in `vmb_a6_cohort_analyze` permutes probe ROWS, but the data is
block-correlated at MODEL level (8 physical models × 160 correlated rows). Row-shuffling
blends models → inflated null → the negative early mag_z and the late burial are both
consistent with that clustered-data exchangeability violation (inherited from
trajectory.py's row-level shuffle; acute at our 3-vs-5 single-seed grain). This runs the
CORRECT null exactly: the 8 models → 3-student/5-control assignment space is enumerable
(C(8,3)=56), so we compute the whole-vector BLOCK field (mean of student MODEL-means −
mean of control MODEL-means) for every partition and read the true partition's exact rank.

  block field(P) = ‖ mean_{m∈student(P)} μ_m − mean_{m∈control(P)} μ_m ‖ ,  μ_m = mean_probes Z_m
  exact p = (#partitions with mag ≥ true) / 56    (min attainable 1/56 ≈ .0179)

Reports the row-shuffle p and the block-null p side by side, per step. Frozen predictions
(outer loop): D1 steps 1–5 n.s. under block null (P=.80); D2 the step-8/13 transient
survives (true = max of 56 at step-13 min) (P=.75); D3 late burial persists (real
drift-swamps-field, not null artifact) (P=.55). CPU, banked sigs. First-read → outer loop.
"""
from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np

from anamnesis.analysis.battery.deltas import load_floor_scale
from anamnesis.analysis.battery.floors import load_signature_matrix
from anamnesis.scripts.vmb_a6_cohort_analyze import parse_label, STEPS, ANIMALS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N_PERM_ROW = 300


def model_means_at_step(root: Path, step: str, med, scale) -> tuple[list[str], list[str], np.ndarray]:
    """Per-model mean-Z at one step. Returns (model_labels, roles, means[M, F])."""
    labels, roles, means = [], [], []
    for cell in sorted(root.iterdir()) if root.exists() else []:
        sig = cell / f"step-{step}" / "signatures_v3"
        if not sig.exists():
            continue
        info = parse_label(cell.name)
        if info["role"] not in ("student", "control"):
            continue
        X, _, _ = load_signature_matrix(sig)
        means.append(((X - med) / scale).mean(0))
        labels.append(cell.name)
        roles.append(info["role"])
    return labels, roles, np.array(means) if means else np.zeros((0, 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-root", type=Path, required=True)
    ap.add_argument("--floor-dir", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260716)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    med, scale = load_floor_scale(args.floor_dir)
    rng = np.random.default_rng(args.seed)

    rows = []
    for step in STEPS:
        labels, roles, mu = model_means_at_step(args.cohort_root, step, med, scale)
        M = len(labels)
        if M < 8:
            rows.append({"step": step, "n_models": M, "ok": False})
            continue
        roles = np.array(roles)
        n_student = int((roles == "student").sum())
        # TRUE partition field (block-averaged)
        true_mask = roles == "student"
        def block_field(mask):
            return float(np.linalg.norm(mu[mask].mean(0) - mu[~mask].mean(0)))
        true_mag = block_field(true_mask)
        # EXACT block null: all C(M, n_student) assignments
        idx = np.arange(M)
        mags = []
        for combo in combinations(idx, n_student):
            m = np.zeros(M, bool); m[list(combo)] = True
            mags.append(block_field(m))
        mags = np.array(mags)
        rank = int((mags >= true_mag - 1e-12).sum())  # 1 = true is the max
        p_block = rank / len(mags)
        z_block = float((true_mag - mags.mean()) / max(mags.std(), 1e-12))
        # (the row-shuffle column lives in cohort_trajectory_qwen.json's mag_p — the inflated
        # null; this script reports the EXACT block null side by side with it in the package.)
        out_row = {"step": step, "ok": True, "n_models": M, "n_student": n_student,
                   "n_partitions": len(mags), "true_block_mag": round(true_mag, 4),
                   "block_null_rank": rank, "block_null_p_exact": round(p_block, 4),
                   "block_null_z": round(z_block, 3),
                   "block_null_min_p": round(1.0 / len(mags), 4),
                   "block_null_mean_mag": round(float(mags.mean()), 4),
                   "block_null_max_mag": round(float(mags.max()), 4)}
        rows.append(out_row)
        logger.info(f"step-{step}: true_mag={true_mag:.3f} rank={rank}/{len(mags)} "
                    f"p_block={p_block:.4f} z_block={z_block:.2f}")

    # scoring the frozen D1-D3 against the exact block null
    ok = [r for r in rows if r.get("ok")]
    def p_at(step):
        r = next((x for x in ok if x["step"] == step), None)
        return r["block_null_p_exact"] if r else None
    d1_steps = ["0001", "0002", "0003", "0005"]
    d1_ns = all((p_at(s) is None or p_at(s) >= 0.05) for s in d1_steps)
    d2_transient = (p_at("0013") is not None and p_at("0013") < 0.05)
    first_sig = next((r["step"] for r in ok if r["block_null_p_exact"] < 0.05), None)

    out = {"arm": "A6 Cell 1 — EXACT block-level null (C(8,3)=56 model partitions)",
           "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop",
           "law": "block field = ‖mean(student MODEL-means) − mean(control MODEL-means)‖; exact "
                  "p = rank of true partition among all C(8,3)=56; min p=1/56≈.0179. Replaces the "
                  "row-shuffle (clustered-data exchangeability violation).",
           "frozen_predictions": {"D1_steps1-5_ns": 0.80, "D2_transient_survives_step13": 0.75,
                                  "D3_late_burial_persists": 0.55},
           "scoring": {"D1_steps1-5_all_ns_under_block_null": bool(d1_ns),
                       "D2_step13_significant_under_block_null": bool(d2_transient),
                       "first_significant_step_block_null": first_sig,
                       "D3_note": "late-burial persistence = manual read of steps 21-75 p vs the "
                                  "transient; reported in the table, not auto-scored"},
           "per_step": rows}
    args.out_json.write_text(json.dumps(out, indent=1))
    logger.info(f"D1 steps1-5 n.s.={d1_ns}  D2 step13 sig={d2_transient}  "
                f"first-sig(block)={first_sig}; wrote {args.out_json}")


if __name__ == "__main__":
    main()
