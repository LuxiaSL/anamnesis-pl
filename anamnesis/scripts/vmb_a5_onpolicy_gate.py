"""A5 matched-token PILOT GATE (C§3): a matched-token cell at (vector, site, alpha)
is valid only if the STEERED model's top-1 agreement with the banked (unsteered)
continuation stays >= 0.85 — beyond that, forced tokens are off-policy for the
steered model and matched-token deltas measure an incoherent counterfactual.

Runs the 20-gen pilot per candidate cell (vector x L14 x alpha in {0.03, 0.1}),
emits a gate report; the matched-token submit script reads it and only launches
cells that PASS. Gate values are stamped into the A5 record either way.

Usage (node1, 1 GPU, via Heimdall; depends_on the vectors job):
    python -m anamnesis.scripts.vmb_a5_onpolicy_gate --model 3b \
        --model-path /models/llama-3.2-3b-instruct \
        --stage0-run /models/anamnesis-extract/runs/vmb_stage0_3b \
        --vectors-dir /models/anamnesis-extract/battery/a5_vectors_3b \
        --out /models/anamnesis-extract/battery/a5_vectors_3b/onpolicy_gate.json
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

MT_FRACS = [0.03, 0.1]
MAP_SITE = 14
GATE_BAR = 0.85
N_PILOT = 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--vectors-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

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
    # A4 convention pilot subset: seed-idx-0 gens, one per 4th class → 20 pilots
    pilot_gids = [k * 40 for k in range(N_PILOT)]
    pilots = []
    for g in pilot_gids:
        e = entries.get(str(g))
        if e and (len(e["input_ids"]) - e["prompt_length"]) >= 32:
            pilots.append(e)
    logger.info(f"{len(pilots)} pilot continuations")

    # Map-site cells: L14 keys + V2's native L13 (SAE resid_post_12 ≈ map-site
    # neighbor; 13a ops adaptation) + site-independent randoms.
    vec_keys = [k for k in bank.keys()
                if k.endswith(f"_L{MAP_SITE}") or k.endswith("_L13") or k.startswith("R")]

    def site_of(key: str) -> int:
        return int(key.rsplit("_L", 1)[1]) if "_L" in key else MAP_SITE

    handles = {}  # one handle per site; only the active site gets alpha > 0
    for s in sorted({site_of(k) for k in vec_keys}):
        spec_s = ResidualWriteSpec(layer_idx=s,
                                   vector=torch.zeros(int(model.config.hidden_size)),
                                   alpha=0.0, normalize=True)
        handles[s] = attach_residual_write(model, spec_s)

    report = {"model": args.model, "gate_bar": GATE_BAR, "n_pilot": len(pilots),
              "site": MAP_SITE, "cells": {}}
    for key in sorted(vec_keys):
        site = site_of(key)
        vec = torch.from_numpy(bank[key].astype(np.float32))
        for frac in MT_FRACS:
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
            cell = f"{key}_a{frac}"
            report["cells"][cell] = {
                "alpha_frac": frac, "alpha_abs": alpha, "site": site,
                "agreement_mean": mean_a, "agreement_min": float(np.min(agrees)),
                "PASS": bool(mean_a >= GATE_BAR),
            }
            logger.info(f"{cell}: agreement {mean_a:.4f} (min {np.min(agrees):.4f}) "
                        f"{'PASS' if mean_a >= GATE_BAR else 'FAIL'}")
    for h in handles.values():
        h.remove()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    n_pass = sum(1 for c in report["cells"].values() if c["PASS"])
    logger.info(f"gate report -> {args.out} ({n_pass}/{len(report['cells'])} cells PASS)")


if __name__ == "__main__":
    main()
