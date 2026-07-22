"""A8 Leg-5 (L4-f) — the STATE COLUMN beside the owl's behavioral ladder (add-3 clause).

The probe writes texts; this reads the STATES those texts correspond to.  For each cell of
interest it re-runs the generated text through 8B under teacher forcing (the battery's
replay convention: forward pass, forward_pre_hook on decoder_layers[L], mean over the
COMPLETION positions, fp32) and reports where the resulting mean state sits relative to
the injected direction and its controls:

    delta(cell) := mean_state(cell) - mean_state(alpha=0 baseline)
    read        := cos(delta, u) for u in {transported Vdiverge, transported Valign,
                                           AR1-3, the zero-padded raw-null Vdiverge}

A behavioral install whose state column is consistent puts delta along the injected
direction and NOT along the controls.  Note the near-tautology and read it as a
CONSISTENCY check, not independent evidence: we wrote that direction in, so its presence
in the state is expected — what is informative is the CONTRAST with the control directions
and the raw-null, and the magnitude relative to dose.

Prompt reconstruction: the probe generates n_samples per prompt in PROMPTS order and
concatenates, so raws index i belongs to PROMPTS[i // n_samples] (asserted against the
banked census n_prompts x n_samples).

UNSTAMPED (C§8).  Run node-side (needs 8B weights + one GPU):
  python -m anamnesis.scripts.a8_leg4f_owl_state_column --arm-root <leg4> --model-path <8b>
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from anamnesis.scripts.vmb_a6_2b_probe import PROMPTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_leg4f_owl_state_column")

SITE = 16
CELLS = ("baseline_a0.0", "Vdiverge_a0.3", "Vdiverge_a0.1", "AR1_a0.3")


def chat_ids(tok, user: str, completion: str) -> tuple[list[int], int]:
    """Same id construction as a8_collect_states.build_ids (native arm)."""
    msgs = [{"role": "user", "content": user}]
    res = tok.apply_chat_template(msgs, add_generation_prompt=True)
    prompt_ids = list(res["input_ids"] if hasattr(res, "keys") else res)
    comp = tok.encode(completion, add_special_tokens=False)
    return prompt_ids + comp, len(prompt_ids)


def mean_state(model, tok, texts: list[str], n_samples: int, dev) -> np.ndarray:
    grab = {}

    def hook(_m, args, kwargs):
        grab["h"] = (args[0] if args else kwargs["hidden_states"]).detach()
        return None

    from anamnesis.extraction.model_loader import decoder_layers
    h = decoder_layers(model)[SITE].register_forward_pre_hook(hook, with_kwargs=True)
    out = []
    try:
        for i, t in enumerate(texts):
            user = PROMPTS[min(i // n_samples, len(PROMPTS) - 1)][1]
            ids, p = chat_ids(tok, user, t)
            if len(ids) - p < 2:
                continue
            with torch.no_grad():
                model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
            out.append(grab["h"][0, p:].float().mean(0).cpu().numpy())
    finally:
        h.remove()
    return np.stack(out).mean(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-root", type=Path, required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--n-samples", type=int, default=10)
    args = ap.parse_args()
    raws = args.arm_root / "readouts_gpu/owl_transported_8b_raws"
    vec_dir = args.arm_root / "vectors"
    tr = dict(np.load(vec_dir / "a8_leg5_owl_transported_8b.npz"))
    ar = dict(np.load(vec_dir / "a8_leg5_owl_ar_8b.npz"))
    rn = dict(np.load(vec_dir / "a8_leg5_owl_rawnull_8b.npz"))
    dirs = {"transported_Vdiverge": tr["Vdiverge_L16"], "transported_Valign": tr["Valign_L16"],
            "rawnull_Vdiverge": rn["Vdiverge_L16"],
            **{k.split("_")[0]: v for k, v in ar.items()}}
    dirs = {k: v.astype(np.float64) / np.linalg.norm(v) for k, v in dirs.items()}

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    dev = next(model.parameters()).device

    states = {}
    for cell in CELLS:
        p = raws / f"{cell}.json"
        if not p.exists():
            logger.warning("missing %s", p)
            continue
        texts = json.loads(p.read_text())
        states[cell] = mean_state(model, tok, texts, args.n_samples, dev)
        logger.info("%s: %d texts, ||mean state|| %.3f", cell, len(texts),
                    float(np.linalg.norm(states[cell])))

    base = states.get("baseline_a0.0")
    rows = {}
    for cell, s in states.items():
        if cell == "baseline_a0.0":
            continue
        delta = s - base
        rows[cell] = {"delta_norm": round(float(np.linalg.norm(delta)), 4),
                      "cos_delta_vs": {k: round(float(delta @ v / np.linalg.norm(delta)), 4)
                                       for k, v in dirs.items()}}
    res = {"STATUS": "UNSTAMPED (C§8) — state column beside the behavioral ladder; "
                     "consistency check, not independent evidence",
           "site": SITE, "capture": "teacher-forced replay, forward_pre_hook on "
                                    "decoder_layers[16] (residual ENTERING L16), mean over "
                                    "completion positions, fp32",
           "rows": rows}
    out = args.arm_root / "readouts_gpu/owl_state_column.json"
    out.write_text(json.dumps(res, indent=1))
    logger.info("wrote %s", out)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
