"""ANNEX — THE ON-POLICY DOSE SWEEP: where does the matched-token regime actually END?

⚠ ANNEX LANE: nothing here is quotable until it graduates through a frozen prereg cell.
Ledger: outputs/battery/annex/ANNEX-LEDGER.md

WHY THIS EXISTS, AND WHY IT IS *NOT* A GATE-BYPASS (2026-07-15, session 2).

A6 measured dose-linearity over α ∈ {.03, .1} — the only doses with matched-token cells — and
found deviation-from-linearity GROWS with response magnitude. So the regime we would actually
want to steer in is precisely the one the banked doses cannot see. The obvious move was "run MT
replays at α = .3 and 1.0". I proposed exactly that, and then read the gate's own docstring:

    vmb_a5_onpolicy_gate.py:1 — "a matched-token cell at (vector, site, alpha) is valid only if
    the STEERED model's top-1 agreement with the banked (unsteered) continuation stays >= 0.85 —
    BEYOND THAT, FORCED TOKENS ARE OFF-POLICY FOR THE STEERED MODEL AND MATCHED-TOKEN DELTAS
    MEASURE AN INCOHERENT COUNTERFACTUAL."

**That kills the naive extension, and the gate is right.** Forcing a model to emit tokens it would
never emit measures it FIGHTING THE FORCING, not the injection's effect. `vmb_a5_mt_launch` skips
gate-FAIL cells for a real reason, not a bureaucratic one. Bypassing it would manufacture numbers
that look like data and are not.

So the gate is not an obstacle to route around — **it IS the experiment.** The question becomes:

    FOR EACH VECTOR, AT WHAT DOSE DOES AGREEMENT CROSS 0.85?

That single number bounds everything a matched-token linearity measurement can EVER say, and it
is cheap. Outcomes:
  · α=.3 PASSES for some vectors ⇒ MT cells there are LEGITIMATE ⇒ a 10x dose range (.03→.3)
    with content held AND on-policy. V3@.3 (response ≈ .35) would then overlap V4@.03's (.23) —
    **the magnitude-matched cross-vector contrast that is impossible today**, which is the ONLY
    way to separate "big pushes are nonlinear" from "V4 is nonlinear".
  · α=.3 FAILS everywhere ⇒ the on-policy regime ends below .3 ⇒ the MT design is intrinsically
    confined to a ~3x dose range, A6's scope caveat is PERMANENT rather than incidental, and the
    higher-dose question needs a different instrument entirely. Also decisive, also worth knowing,
    and it SAVES the 2,240-replay job rather than spending it on an incoherent counterfactual.

Either way this reports before any replay is launched. It is the control landing before the claim
— session 1's most expensive lesson, applied prospectively for the third time this session.

This is a READ of the existing gate machinery at more doses. It runs the SAME
`teacher_forced_agreement` on the SAME 20 pilots as `vmb_a5_onpolicy_gate`, so its α=.03/.1 rows
must REPRODUCE the banked gate report — a built-in correctness check (see `--verify`).

⚠ Standalone on purpose: it imports `anamnesis` but lives OUTSIDE the node1 checkout, because the
overnight queue is running from `~/luxi-files/anamnesis-pl` and that checkout must not be touched
mid-run. Submit via Heimdall with --gpu-ids (never set CUDA_VISIBLE_DEVICES by hand; Heimdall
owns it).

    heimdall submit --node node1 --gpus 1 --gpu-ids 4,6,7 -n annex-dose-sweep \
      'source ~/luxi-files/.venv-shared/bin/activate && \
       export PYTHONPATH=~/luxi-files/anamnesis-pl/pipeline HF_HUB_OFFLINE=1 && \
       python ~/luxi-files/annex/annex_dose_sweep.py --model 3b \
         --model-path /models/llama-3.2-3b-instruct \
         --stage0-run /models/anamnesis-extract/runs/vmb_stage0_3b \
         --vectors-dir /models/anamnesis-extract/battery/a5_vectors_3b \
         --out ~/luxi-files/annex/annex_dose_sweep.json'
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from anamnesis.config import MODEL_PRESETS
from anamnesis.scripts._a5_common import teacher_forced_agreement

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAP_SITE = 14
GATE_BAR = 0.85          # identical to vmb_a5_onpolicy_gate — do NOT loosen it to buy range
N_PILOT = 20             # identical pilot set ⇒ .03/.1 must reproduce the banked report
# .03/.1 = the banked/reproduction anchors; the rest brackets the 0.85 crossing.
FRACS = [0.03, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--vectors-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fracs", default=",".join(str(f) for f in FRACS))
    ap.add_argument("--verify", type=Path, default=None,
                    help="banked onpolicy_gate.json — assert .03/.1 reproduce")
    args = ap.parse_args()
    fracs = [float(x) for x in args.fracs.split(",")]

    preset = MODEL_PRESETS[args.model]
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}.get(str(preset.torch_dtype), torch.float16)

    from transformers import AutoModelForCausalLM

    from anamnesis.extraction.model_loader import ResidualWriteSpec, attach_residual_write

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, attn_implementation="eager").to("cuda").eval()

    bank = np.load(args.vectors_dir / "a5_vectors.npz")
    stamps = json.loads((args.vectors_dir / "a5_vectors_stamps.json").read_text())
    norms = stamps["median_resid_norms"]

    entries = json.loads((args.stage0_run / "replay_manifest.json").read_text())["entries"]
    pilot_gids = [k * 40 for k in range(N_PILOT)]          # A4 convention, same as the gate
    pilots = []
    for g in pilot_gids:
        e = entries.get(str(g))
        if e and (len(e["input_ids"]) - e["prompt_length"]) >= 32:
            pilots.append(e)
    logger.info(f"{len(pilots)} pilot continuations; fracs {fracs}")

    vec_keys = [k for k in bank.keys()
                if k.endswith(f"_L{MAP_SITE}") or k.endswith("_L13") or k.startswith("R")]

    def site_of(key: str) -> int:
        return int(key.rsplit("_L", 1)[1]) if "_L" in key else MAP_SITE

    handles = {}
    for s in sorted({site_of(k) for k in vec_keys}):
        handles[s] = attach_residual_write(model, ResidualWriteSpec(
            layer_idx=s, vector=torch.zeros(int(model.config.hidden_size)),
            alpha=0.0, normalize=True))

    report = {
        "ANNEX_RULE": "NOT QUOTABLE until graduated via a frozen prereg cell",
        "rung": "on-policy dose sweep — where does the matched-token regime end?",
        "gate_bar": GATE_BAR, "n_pilot": len(pilots), "site": MAP_SITE, "fracs": fracs,
        "why": "A6 found deviation-from-linearity grows with response magnitude, so the regime "
               "worth steering in is exactly where the banked doses (α .03/.1) cannot see. "
               "Extending MT upward is only LEGITIMATE where agreement >= 0.85: beyond that, "
               "forced tokens are off-policy and matched-token deltas measure an incoherent "
               "counterfactual (vmb_a5_onpolicy_gate.py:1). This finds the boundary FIRST.",
        "cells": {}, "crossing": {},
    }
    for key in sorted(vec_keys):
        site = site_of(key)
        vec = torch.from_numpy(bank[key].astype(np.float32))
        for frac in fracs:
            alpha = frac * float(norms[f"L{site}"])
            for s, h in handles.items():
                h.spec.alpha = alpha if s == site else 0.0
                if s == site:
                    h.spec.vector = vec
            agrees = []
            for e in pilots:
                handles[site].spec.start_pos = int(e["prompt_length"])
                agrees.append(teacher_forced_agreement(
                    model, e["input_ids"], int(e["prompt_length"])))
            mean_a = float(np.mean(agrees))
            report["cells"][f"{key}_a{frac}"] = {
                "alpha_frac": frac, "alpha_abs": alpha, "site": site,
                "agreement_mean": mean_a, "agreement_min": float(np.min(agrees)),
                "PASS": bool(mean_a >= GATE_BAR),
            }
            logger.info(f"{key}_a{frac}: agreement {mean_a:.4f} "
                        f"{'PASS' if mean_a >= GATE_BAR else 'FAIL'}")
        # the number this rung exists for: the largest swept dose still on-policy
        passing = [f for f in fracs
                   if report["cells"][f"{key}_a{f}"]["PASS"]]
        report["crossing"][key] = {
            "max_passing_frac": max(passing) if passing else None,
            "first_failing_frac": next((f for f in fracs
                                        if not report["cells"][f"{key}_a{f}"]["PASS"]), None),
        }
    for h in handles.values():
        h.remove()

    if args.verify and args.verify.exists():
        banked = json.loads(args.verify.read_text())["cells"]
        deltas = []
        for cell, v in banked.items():
            mine = report["cells"].get(cell)
            if mine:
                deltas.append(abs(mine["agreement_mean"] - v["agreement_mean"]))
        report["reproduction_check"] = {
            "n_cells_compared": len(deltas),
            "max_abs_agreement_delta": round(float(max(deltas)), 6) if deltas else None,
            "note": "same pilots + same machinery ⇒ the banked α .03/.1 rows must reproduce. A "
                    "non-trivial delta means this sweep is NOT measuring what the gate measured "
                    "and NOTHING here may be read.",
        }
        logger.info(f"reproduction check vs banked gate: max |Δagreement| = "
                    f"{report['reproduction_check']['max_abs_agreement_delta']} "
                    f"over {len(deltas)} cells")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    logger.info(f"→ {args.out}")
    print("\n=== ON-POLICY BOUNDARY (max dose still >= 0.85 agreement) ===")
    for k, v in sorted(report["crossing"].items()):
        print(f"  {k:12s} max passing α = {v['max_passing_frac']}  "
              f"(first fail {v['first_failing_frac']})")


if __name__ == "__main__":
    main()
