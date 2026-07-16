"""A6 seeds-tier — PER-ANIMAL cohort decomposition (session-10 Part A1).

Session-9's pooled seeds field was born-fragmented: its 12 students span 3 animals
(cat/penguin/phoenix), so the pooled student−control field mixes three distinct
trait directions and the pooled block-null was underpowered (many heterogeneous
partitions beat the pooled split). This decomposes the field PER ANIMAL, giving each
animal its OWN homogeneous field and its OWN exact block null:

  For each animal a: 4 student seeds {t0,t2,t3,t4} + 5 controls = 9 models.
  field_a(t)     = mean(model-mean_student_a) − mean(model-mean_control)   [cluster-level;
                   each model weighted equally — the block/cluster statistic]
  exact block null: enumerate all C(9,4)=126 assignments of 4-of-9 models to "student",
                    recompute the field mag for each; rank the true assignment.
  block_p        = fraction of the 126 partitions with mag ≥ observed (the true split is
                   one of the 126, so block_p ≥ 1/126 = .0079 by construction).
  block_z        = (mag − null.mean)/null.std over the 126 partitions.

Scored vs the frozen **P=.70: ≥2/3 animals clear (block_p < .05) at the transient steps**
(step-8 and step-13, the pooled-grain transient onset). CPU, no GPU. First-read → outer
loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from anamnesis.analysis.battery.deltas import load_floor_scale
from anamnesis.analysis.battery.floors import load_signature_matrix
from anamnesis.analysis.battery.manifest import MODEL_META
from anamnesis.scripts.vmb_a6_cohort_analyze import ANIMALS, STEPS, parse_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRANSIENT_STEPS = ("0008", "0013")


def cos(a: NDArray, b: NDArray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def load_step_modelmeans(run_root: Path, step: str, med, scale) -> dict | None:
    """Return {model_label: mean_Z_vector} for one step (cluster-level, gens averaged)."""
    out: dict[str, dict] = {}
    for cell in sorted(run_root.iterdir()) if run_root.exists() else []:
        sig = cell / f"step-{step}" / "signatures_v3"
        if not sig.exists():
            continue
        info = parse_label(cell.name)
        if info["role"] == "unknown":
            continue
        X, _, g = load_signature_matrix(sig)
        Z = (X - med) / scale
        out[cell.name] = {"mean": Z.mean(0), "n_gen": len(g), "info": info}
    return out or None


def field_mag(means: NDArray, is_student: NDArray) -> tuple[float, NDArray]:
    e = means[is_student].mean(0) - means[~is_student].mean(0)
    return float(np.linalg.norm(e)), e


def animal_step(model_means: dict, animal: str) -> dict | None:
    """9-model field + exact C(9,4)=126 block null for one animal at one step."""
    labels = [lbl for lbl, d in model_means.items()
              if (d["info"]["role"] == "control") or
                 (d["info"]["role"] == "student" and d["info"]["animal"] == animal)]
    students = [lbl for lbl in labels if model_means[lbl]["info"]["role"] == "student"]
    controls = [lbl for lbl in labels if model_means[lbl]["info"]["role"] == "control"]
    if len(students) < 3 or len(controls) < 3:
        return None
    labels = students + controls  # deterministic order: students first
    means = np.vstack([model_means[lbl]["mean"] for lbl in labels])
    n, k = len(labels), len(students)
    true_mask = np.zeros(n, dtype=bool)
    true_mask[:k] = True
    obs_mag, e = field_mag(means, true_mask)

    null = []
    for combo in combinations(range(n), k):
        m = np.zeros(n, dtype=bool)
        m[list(combo)] = True
        null.append(field_mag(means, m)[0])
    null = np.asarray(null)
    n_part = len(null)  # C(9,4)=126
    rank = int((null >= obs_mag).sum())  # 1 = true is the max (the true split is included)
    return {
        "animal": animal, "n_student": k, "n_control": len(controls), "n_partitions": n_part,
        "field_mag": round(obs_mag, 4),
        "block_p": round(rank / n_part, 5),
        "block_rank": rank,
        "block_z": round(float((obs_mag - null.mean()) / max(null.std(), 1e-12)), 3),
        "null_mean": round(float(null.mean()), 4), "null_max": round(float(null.max()), 4),
        "student_seeds": sorted(model_means[s]["info"]["seed"] for s in students),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-root", type=Path, required=True)
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--floor-dir", type=Path, default=None)
    ap.add_argument("--model", default="qwen-7b")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    mm = MODEL_META[args.model]
    floor_dir = args.floor_dir or (args.battery_root / mm.stage0_dir / "signatures_v3")
    med, scale = load_floor_scale(floor_dir)

    # final-step per-animal field for lock-in cosine
    e_final = {}
    fin = load_step_modelmeans(args.cohort_root, STEPS[-1], med, scale)
    if fin is not None:
        for a in ANIMALS:
            r = animal_step(fin, a)
            if r is not None:
                labels = [lbl for lbl, d in fin.items()
                          if d["info"]["role"] == "control" or
                          (d["info"]["role"] == "student" and d["info"]["animal"] == a)]
                st = [lbl for lbl in labels if fin[lbl]["info"]["role"] == "student"]
                ct = [lbl for lbl in labels if fin[lbl]["info"]["role"] == "control"]
                means = np.vstack([fin[lbl]["mean"] for lbl in st + ct])
                mask = np.zeros(len(st + ct), dtype=bool); mask[:len(st)] = True
                e_final[a] = field_mag(means, mask)[1]

    per_step: dict[str, list] = {a: [] for a in ANIMALS}
    for step in STEPS:
        mm_step = load_step_modelmeans(args.cohort_root, step, med, scale)
        if mm_step is None:
            continue
        for a in ANIMALS:
            r = animal_step(mm_step, a)
            if r is None:
                continue
            r["step"] = step
            if a in e_final:
                labels = [lbl for lbl, d in mm_step.items()
                          if d["info"]["role"] == "control" or
                          (d["info"]["role"] == "student" and d["info"]["animal"] == a)]
                st = [lbl for lbl in labels if mm_step[lbl]["info"]["role"] == "student"]
                ct = [lbl for lbl in labels if mm_step[lbl]["info"]["role"] == "control"]
                means = np.vstack([mm_step[lbl]["mean"] for lbl in st + ct])
                mask = np.zeros(len(st + ct), dtype=bool); mask[:len(st)] = True
                r["cos_to_final"] = round(cos(field_mag(means, mask)[1], e_final[a]), 3)
            per_step[a].append(r)
            logger.info(f"{a} step-{step}: mag={r['field_mag']} block_p={r['block_p']} "
                        f"block_z={r['block_z']} rank={r['block_rank']}/{r['n_partitions']}")

    # Scoring: P=.70 → ≥2/3 animals clear (block_p<.05) at the transient steps 8 & 13
    clears = {}
    for a in ANIMALS:
        rows = {r["step"]: r for r in per_step[a]}
        trans = [rows[s] for s in TRANSIENT_STEPS if s in rows]
        clears[a] = {"clears_any_transient": any(r["block_p"] < 0.05 for r in trans),
                     "clears_both_transient": all(r["block_p"] < 0.05 for r in trans) and len(trans) == 2,
                     "transient_block_p": {r["step"]: r["block_p"] for r in trans},
                     "best_step": min(rows, key=lambda s: rows[s]["block_p"]) if rows else None,
                     "best_block_p": min((r["block_p"] for r in rows.values()), default=None)}
    n_clear_any = sum(v["clears_any_transient"] for v in clears.values())

    out = {
        "arm": "A6 seeds-tier — PER-ANIMAL cohort decomposition (exact C(9,4) block null)",
        "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop",
        "model": args.model,
        "law": "per animal: 4 seed-students {t0,t2,t3,t4} + 5 controls = 9 cluster-level model-mean "
               "vectors; field = mean(student model-means)−mean(control model-means); exact block null "
               "= all C(9,4)=126 4-of-9 assignments; block_p = frac ≥ obs (true split included ⇒ "
               "floor 1/126=.0079). Transient steps = 8,13 (pooled-grain onset).",
        "filed_P": {"per_animal_transient": 0.70,
                    "rule": ">=2/3 animals clear (block_p<.05) at the transient steps (8/13)"},
        "n_animals_clearing_any_transient": n_clear_any,
        "P70_hit": n_clear_any >= 2,
        "per_animal_summary": clears,
        "per_step": per_step,
    }
    args.out_json.write_text(json.dumps(out, indent=1))
    logger.info(f"animals clearing a transient step: {n_clear_any}/3  → P=.70 hit={n_clear_any>=2}; "
                f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
