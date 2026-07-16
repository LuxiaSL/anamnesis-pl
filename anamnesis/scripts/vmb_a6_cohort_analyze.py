"""A6 Cell 1 — Clock-1 cohort at battery grain (WAVE2-A6 cell 1; session-8 Part C).

Transplants echo-sandbox trajectory.py `node_stats_at_step` (L98-163) to the battery
signature space — the WHOLE battery vector is the single "node" (no j-space tiling):

  regime field   e(t) = mean(Z_student) − mean(Z_control)          [checkpoint-matched:
                        control = control-STUDENT at step t, never base — numbers-training
                        drift cancels, per trajectory.py]
  mag(t)         = ‖e(t)‖ ; mag_z = (mag − role-shuffle mean)/std  (N_PERM label shuffles)
  reliability    probe-split-half (always) + seed-parity split-half (if ≥2 seeds)
  L3 identity    per-animal excess e_a = mean(Z_a) − mean(Z_control); animal-vs-animal
                 contrast con_ab; contrast reliability over seed parity (if seeds)
  lock-in        cos(e(t), e(final)) — when the final regime direction is present

Frozen predictions (14o §4): the regime field is significant vs the role-shuffle null from
the FIRST NONZERO-ADAPTER checkpoint at battery grain, cohort scale — **P=.75** (dense =
warmup schedule → predicted significant from step-2); schedule-specificity (step-1 vs
step-2) **P=.55**. Z = stage-0-floor-z of the raw signature (base cancels in student−control
on shared probes). CPU. First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from anamnesis.analysis.battery.deltas import load_floor_scale
from anamnesis.analysis.battery.floors import load_signature_matrix
from anamnesis.analysis.battery.manifest import MODEL_META

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ANIMALS = ("cat", "penguin", "phoenix")
STEPS = ["0001", "0002", "0003", "0005", "0008", "0013", "0021", "0034", "0055", "0075"]
N_PERM = 300
MIN_HALF = 8


def parse_label(label: str) -> dict:
    """cat_dense / phoenix_dense_t2 / control_a_dense → role/animal/seed."""
    if label.startswith("control_"):
        rest = label[len("control_"):]
        ctrl_id = rest.split("_")[0]
        return {"role": "control", "animal": None, "control_id": ctrl_id,
                "seed": rest[len(ctrl_id) + 1:] or "dense"}
    for a in ANIMALS:
        if label.startswith(a + "_"):
            return {"role": "student", "animal": a, "control_id": None,
                    "seed": label[len(a) + 1:]}
    return {"role": "unknown", "animal": None, "control_id": None, "seed": label}


def cos(a: NDArray, b: NDArray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def step_stats(Z: NDArray, role: NDArray, animal: NDArray, seed: NDArray, gid: NDArray,
               rng: np.random.Generator, e_final: NDArray | None) -> dict:
    s, c = role == "student", role == "control"
    ns, nc = int(s.sum()), int(c.sum())
    row: dict = {"n_student": ns, "n_control": nc}
    if ns < MIN_HALF or nc < MIN_HALF:
        row["ok"] = False
        return row
    row["ok"] = True
    e = Z[s].mean(0) - Z[c].mean(0)
    row["mag"] = float(np.linalg.norm(e))
    # role-shuffle null
    pool = np.where(s | c)[0]
    lab = role[pool]
    mags = np.empty(N_PERM)
    for p in range(N_PERM):
        perm = rng.permutation(lab)
        mags[p] = np.linalg.norm(Z[pool[perm == "student"]].mean(0) - Z[pool[perm == "control"]].mean(0))
    row["mag_z"] = float((row["mag"] - mags.mean()) / max(mags.std(), 1e-12))
    row["mag_p"] = float((mags >= row["mag"]).mean())
    if e_final is not None:
        row["cos_to_final"] = cos(e, e_final)
    # probe-split-half reliability (split by gen-id parity — content halves)
    par = gid % 2
    h1, h2 = s & (par == 0), s & (par == 1)
    d1, d2 = c & (par == 0), c & (par == 1)
    if min(h1.sum(), h2.sum()) >= MIN_HALF and min(d1.sum(), d2.sum()) >= MIN_HALF:
        row["reliability_probe"] = cos(Z[h1].mean(0) - Z[d1].mean(0), Z[h2].mean(0) - Z[d2].mean(0))
    # seed-parity split-half (identity: over train-seed parity within students)
    seeds_present = sorted(set(seed[s].tolist()))
    if len(seeds_present) >= 2:
        sp = {v: i for i, v in enumerate(seeds_present)}
        spar = np.array([sp.get(t, -1) for t in seed])
        cids = sorted(set(seed[c].tolist()))
        cp = {v: i for i, v in enumerate(cids)}
        cpar = np.array([cp.get(t, -1) for t in seed])
        sh1, sh2 = s & (spar % 2 == 0), s & (spar % 2 == 1)
        ch1, ch2 = c & (cpar % 2 == 0), c & (cpar % 2 == 1)
        if min(sh1.sum(), sh2.sum()) >= MIN_HALF and min(ch1.sum(), ch2.sum()) >= 3:
            row["reliability_seed"] = cos(Z[sh1].mean(0) - Z[ch1].mean(0),
                                          Z[sh2].mean(0) - Z[ch2].mean(0))
    # L3 per-animal excess + animal-vs-animal contrasts
    from itertools import combinations
    for a in ANIMALS:
        sa = s & (animal == a)
        if sa.sum() >= MIN_HALF:
            ea = Z[sa].mean(0) - Z[c].mean(0)
            row[f"mag_excess_{a}"] = float(np.linalg.norm(ea))
    for a, b in combinations(ANIMALS, 2):
        sa, sb = s & (animal == a), s & (animal == b)
        if sa.sum() >= MIN_HALF and sb.sum() >= MIN_HALF:
            row[f"contrast_mag_{a}_{b}"] = float(np.linalg.norm(Z[sa].mean(0) - Z[sb].mean(0)))
    return row


def load_step(run_root: Path, step: str, med, scale) -> tuple | None:
    Zs, roles, animals, seeds, gids = [], [], [], [], []
    for cell in sorted(run_root.iterdir()) if run_root.exists() else []:
        sig = cell / f"step-{step}" / "signatures_v3"
        if not sig.exists():
            continue
        info = parse_label(cell.name)
        if info["role"] == "unknown":
            continue
        X, _, g = load_signature_matrix(sig)
        Z = (X - med) / scale
        Zs.append(Z)
        roles += [info["role"]] * len(g)
        animals += [info["animal"]] * len(g)
        seeds += [info["seed"]] * len(g)
        gids += [int(x) for x in g]
    if not Zs:
        return None
    return (np.vstack(Zs), np.array(roles), np.array(animals), np.array(seeds), np.array(gids))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort-root", type=Path, required=True, help="vmb_a6cohort_qwen")
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--floor-dir", type=Path, default=None,
                    help="stage-0 floor signatures_v3 dir (default battery-root/stage0_dir/"
                         "signatures_v3; qwen stage-0 lives under runs/, pass it explicitly)")
    ap.add_argument("--model", default="qwen-7b")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    mm = MODEL_META[args.model]
    floor_dir = args.floor_dir or (args.battery_root / mm.stage0_dir / "signatures_v3")
    med, scale = load_floor_scale(floor_dir)

    # final-step regime field for the lock-in cosine
    fin = load_step(args.cohort_root, STEPS[-1], med, scale)
    e_final = None
    if fin is not None:
        Z, role, *_ = fin
        s, c = role == "student", role == "control"
        if s.sum() >= MIN_HALF and c.sum() >= MIN_HALF:
            e_final = Z[s].mean(0) - Z[c].mean(0)

    rng = np.random.default_rng(20260716)
    rows = []
    for step in STEPS:
        data = load_step(args.cohort_root, step, med, scale)
        if data is None:
            rows.append({"step": step, "present": False})
            continue
        Z, role, animal, seed, gid = data
        st = step_stats(Z, role, animal, seed, gid, rng, e_final)
        st["step"] = step
        st["present"] = True
        st["seeds"] = sorted(set(seed.tolist()))
        rows.append(st)
        if st.get("ok"):
            logger.info(f"step-{step}: mag={st['mag']:.3f} mag_z={st['mag_z']:.2f} "
                        f"mag_p={st['mag_p']:.4f} rel_probe={st.get('reliability_probe')} "
                        f"cos_final={st.get('cos_to_final')}")

    # first significant step (mag_p < .05)
    ok = [r for r in rows if r.get("ok")]
    first_sig = next((r["step"] for r in ok if r.get("mag_p", 1) < 0.05), None)
    step1 = next((r for r in ok if r["step"] == "0001"), None)
    step2 = next((r for r in ok if r["step"] == "0002"), None)

    out = {
        "arm": "A6 Cell 1 — Clock-1 cohort at battery grain (trajectory.py transplant)",
        "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop",
        "model": args.model,
        "law": "battery vector = single node; regime field mean(student)−mean(control) "
               "(control = checkpoint-matched control-student); mag_z vs N_PERM=300 role-shuffle; "
               "probe- & seed-parity split-half; per-animal excess + animal contrasts; cos-to-final.",
        "filed_P": {"regime_field_sig_from_first_nonzero_adapter": 0.75,
                    "schedule_specificity_step1_vs_step2": 0.55},
        "prediction_note": "dense = warmup schedule → regime field predicted significant from step-2 "
                           "(step-1 near-zero adapter). Schedule-specificity: step-1 n.s., step-2 sig.",
        "first_significant_step": first_sig,
        "step1_mag_p": step1.get("mag_p") if step1 else None,
        "step2_mag_p": step2.get("mag_p") if step2 else None,
        "per_step": rows,
    }
    args.out_json.write_text(json.dumps(out, indent=1))
    logger.info(f"first significant (mag_p<.05) step: {first_sig}; wrote {args.out_json}")


if __name__ == "__main__":
    main()
