"""ARM A6 cell-(i) analyzer — epoch-lead: does the signature shift cross the 12b
visibility bar EARLIER than behavioral onset? (Ratified block 13a; §2c A6 prediction.)

Deltas: per checkpoint, matched-token delta vs the banked BASE-model Stage-0
signature of the SAME gen (bitwise determinism on merged adapters verified by the
adapter smoke → deltas are pure weight-delta effect). Units: 12b seed-floor
(M3 Qwen stochastic floor median = 1.0), visibility bar 0.1x.

Behavioral onset: first checkpoint whose elicitation rate (subliminal repo
flowering_metrics, literal rate, n=2000/cell) exceeds control-base + 2*sigma
(sigma over the five numbers-control finals). Signature crossing: first checkpoint
whose whole-vector delta has bootstrap-95%-lower-CI >= the bar (the 12d
significance gate, operationalized as bootstrap over probe items — documented
here as the gate's implementation).

Trivially-expected channels: NONE (12d rule b — weights changed, everything may
move; the PREDICTION is about ORDER). Per-source localization emitted UNSTAMPED
(exploratory per the block).

    python -m anamnesis.scripts.vmb_arm_a6_analyze --battery-root ../outputs/battery \
        --a6-run ../outputs/battery/vmb_a6_qwen_cat_t4 \
        --flowering ~/projects/subliminal_anamnesis/research/artifacts_2026-07-11/flowering_metrics.json \
        --student qwen_cat_student_t4 --out-dir ../outputs/battery/arms/A6
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from anamnesis.analysis.battery.deltas import ConditionCorpus, load_floor_scale, within_condition_deltas
from anamnesis.analysis.battery.floors import build_cells, load_signature_matrix
from anamnesis.analysis.battery.manifest import MODEL_META

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STEPS = ["0001", "0075", "0151", "0226", "0302", "0377", "0453"]
BAR = 0.1
N_BOOT = 2000
CELLS_REPORT = ["whole_vector", "source:attention", "source:residual", "source:gate",
                "source:keys", "source:output"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--a6-run", type=Path, required=True)
    ap.add_argument("--flowering", type=Path, required=True)
    ap.add_argument("--student", default="qwen_cat_student_t4")
    ap.add_argument("--model", default="qwen-7b")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mm = MODEL_META[args.model]
    stage0 = args.battery_root / mm.stage0_dir
    med, scale = load_floor_scale(stage0 / "signatures_v3")
    s0 = ConditionCorpus(stage0 / "signatures_v3", stage0 / "metadata.json",
                         med, scale, "qwen-stage0")
    names = s0.feature_names
    cells = {c: m for c, m in build_cells(names, mm.n_layers).items() if c in CELLS_REPORT}
    floor = {c: max(float(np.median(v)), 1e-12)
             for c, v in within_condition_deltas(s0, cells).items()}
    s0X, _, s0g = load_signature_matrix(stage0 / "signatures_v3")
    s0Z = (s0X - med) / scale
    s0map = {g: i for i, g in enumerate(s0g)}

    # ── behavioral onset ──
    fl = json.loads(Path(args.flowering).expanduser().read_text())

    def lit(entry) -> float:  # per-checkpoint entries are per-metric dicts
        return float(entry["literal"] if isinstance(entry, dict) else entry)

    controls = [lit(fl[f"qwen_control_{c}"]["final"]) for c in "abcde"]
    base_mu, base_sd = float(np.mean(controls)), float(np.std(controls))
    thresh = base_mu + 2 * base_sd
    student = fl[args.student]
    rates = {s: lit(student[f"step-{s}"]) for s in STEPS}
    onset = next((s for s in STEPS if rates[s] > thresh), None)
    logger.info(f"behavioral: controls {base_mu:.4f}±{base_sd:.4f} → thresh {thresh:.4f}; "
                f"rates {rates}; onset step-{onset}")

    # ── per-checkpoint signature deltas ──
    rows = []
    sig_crossing = None
    rng = np.random.default_rng(20260713)
    for s in STEPS:
        X, nms, gids = load_signature_matrix(args.a6_run / f"step-{s}" / "signatures_v3")
        if list(nms) != list(names):
            raise ValueError(f"step-{s}: feature fork vs stage0")
        Z = (X - med) / scale
        D = np.stack([np.abs(Z[i] - s0Z[s0map[g]]) for i, g in enumerate(gids) if g in s0map])
        row = {"step": s, "elicitation_rate": rates[s], "n": int(D.shape[0]),
               "cells": {}}
        for c, m in cells.items():
            vals = D[:, m].mean(axis=1)
            ratio = float(np.median(vals) / floor[c])
            boots = np.median(
                rng.choice(vals, size=(N_BOOT, len(vals)), replace=True), axis=1) / floor[c]
            lo = float(np.percentile(boots, 2.5))
            row["cells"][c] = {"ratio_seed_floor": ratio, "boot_lo95": lo,
                               "visible_012b_gated": bool(lo >= BAR)}
        wv = row["cells"]["whole_vector"]
        if sig_crossing is None and wv["visible_012b_gated"]:
            sig_crossing = s
        rows.append(row)
        logger.info(f"step-{s}: wv {wv['ratio_seed_floor']:.3f}x floor "
                    f"(lo95 {wv['boot_lo95']:.3f}) rate {rates[s]:.3f}")

    steps_order = {s: i for i, s in enumerate(STEPS)}
    lead = (onset is not None and sig_crossing is not None
            and steps_order[sig_crossing] < steps_order[onset])
    out = {
        "student": args.student, "model": args.model,
        "behavioral": {"control_mu": base_mu, "control_sd": base_sd,
                       "threshold": thresh, "rates": rates, "onset_step": onset},
        "signature_crossing_step": sig_crossing,
        "epoch_lead_confirmed": bool(lead),
        "per_checkpoint": rows,
        "stamp": {"n": 160, "M": args.model,
                  "law": "12b seed-floor; bar 0.1x gated at bootstrap lo95 (12d); "
                         "matched-token vs base stage0 sigs (adapter-merge bitwise-verified)",
                  "floor_type": "stochastic(stage0 qwen)"},
        "localization_note": "per-source rows exploratory/unstamped per block 12d note",
    }
    p = args.out_dir / f"a6_results_{args.student}.json"
    p.write_text(json.dumps(out, indent=1))
    logger.info(f"epoch-lead: sig crossing step-{sig_crossing} vs behavioral onset "
                f"step-{onset} → {'LEAD CONFIRMED' if lead else 'NO LEAD'} -> {p}")


if __name__ == "__main__":
    main()
