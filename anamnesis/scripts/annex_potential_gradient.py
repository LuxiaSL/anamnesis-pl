"""ANNEX — in-shadow roster gradient pulses (PRICING-inshadow-roster-2026-07-16; ratified
c93055b, fires on Luxia's GPU word). Construction-side code (rider 2: readout stays banked).

Three natural per-token OUTPUT functionals, differentiated w.r.t. the L{site} residual input
over banked stage-0 generations — `vmb_a5_build_v4_gradient.py`'s structure verbatim with S
swapped (same hook, same gen sampling, same averaging, same unit-norm + stamps):

  margin   S = mean_t[ logit_top1 - logit_top2 ]        (confidence; top-2 by detached argsort)
  eos      S = mean_t[ log sum_{v in EOS} p_t(v) ]      (EOS-hazard; 3B ids 128001/128009)
  repmass  S = mean_t[ sum_{v in prior context} p_t(v) ] (repetition-mass; unique prior tokens)

Keys banked as Gmargin_L{site} / Geos_L{site} / Grep_L{site} (RAW gradients — the band-passed
members Vconf/Veos/Vrep are built LOCALLY by annex_band_pass.py from banked Sigma; anatomy +
per-member predictions file BEFORE any steering, rider-1 pattern).

Usage (node1, 1 GPU, via Heimdall):
    python -m anamnesis.scripts.annex_potential_gradient --model 3b \
        --model-path /models/llama-3.2-3b-instruct \
        --stage0-run /models/anamnesis-extract/runs/vmb_stage0_3b \
        --out-dir /models/anamnesis-extract/battery/annex/roster_vectors_3b \
        --functional margin --n-gens 20
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EOS_IDS = {"3b": [128001, 128009], "8b": [128001, 128008, 128009]}
KEY_PREFIX = {"margin": "Gmargin", "eos": "Geos", "repmass": "Grep"}


def s_terms_margin(logits: torch.Tensor, ids: torch.Tensor, P: int, L: int, eos) -> list:
    terms = []
    for t in range(P - 1, L - 1):          # logits[t] predict token t+1 (generated span)
        row = torch.log_softmax(logits[t].float(), dim=-1)
        top2 = torch.topk(row.detach(), 2).indices
        terms.append(row[top2[0]] - row[top2[1]])
    return terms


def s_terms_eos(logits: torch.Tensor, ids: torch.Tensor, P: int, L: int, eos) -> list:
    eos_t = torch.tensor(eos, device=logits.device)
    terms = []
    for t in range(P - 1, L - 1):
        row = torch.log_softmax(logits[t].float(), dim=-1)
        terms.append(torch.logsumexp(row[eos_t], dim=0))
    return terms


def s_terms_repmass(logits: torch.Tensor, ids: torch.Tensor, P: int, L: int, eos) -> list:
    terms = []
    seq = ids[0]
    for t in range(P - 1, L - 1):
        prior = torch.unique(seq[: t + 1].detach())
        p = torch.softmax(logits[t].float(), dim=-1)
        terms.append(p[prior].sum())
    return terms


S_FNS = {"margin": s_terms_margin, "eos": s_terms_eos, "repmass": s_terms_repmass}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--functional", choices=list(S_FNS.keys()), required=True)
    ap.add_argument("--n-gens", type=int, default=20)
    ap.add_argument("--map-site", type=int, default=14, help="3B=14, 8B=16")
    args = ap.parse_args()
    site = args.map_site
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="eager",
    ).to("cuda").eval()
    layers = model.model.layers

    entries = json.loads((args.stage0_run / "replay_manifest.json").read_text())["entries"]
    all_ids = sorted(int(k) for k in entries)
    gids = all_ids[:: max(1, len(all_ids) // args.n_gens)][: args.n_gens]

    captured: dict[str, torch.Tensor] = {}

    def pre_hook(module, hook_args, hook_kwargs):
        hs = hook_args[0] if hook_args else hook_kwargs.get("hidden_states")
        leaf = hs.detach().clone().requires_grad_(True)
        captured["leaf"] = leaf
        if hook_args:
            return (leaf,) + tuple(hook_args[1:]), hook_kwargs
        hook_kwargs = dict(hook_kwargs)
        hook_kwargs["hidden_states"] = leaf
        return hook_args, hook_kwargs

    handle = layers[site].register_forward_pre_hook(pre_hook, with_kwargs=True)
    s_fn = S_FNS[args.functional]
    eos = EOS_IDS.get(args.model, [])

    grads, s_values = [], []
    for g in gids:
        e = entries[str(g)]
        ids = torch.tensor([e["input_ids"]], dtype=torch.long, device="cuda")
        P, L = int(e["prompt_length"]), ids.shape[1]
        with torch.enable_grad():
            out = model(ids, use_cache=False, return_dict=True)
            terms = s_fn(out.logits[0], ids, P, L, eos)
            S = torch.stack(terms).mean()
            S.backward()
        leaf = captured["leaf"]
        gvec = leaf.grad[0, P:L, :].float().mean(dim=0).cpu().numpy()
        grads.append(gvec)
        s_values.append(float(S.detach()))
        model.zero_grad(set_to_none=True)
        captured.clear()
        logger.info(f"gen {g}: S={s_values[-1]:.4f} |grad|={np.linalg.norm(gvec):.3e}")
    handle.remove()

    v = np.mean(grads, axis=0)
    v_unit = (v / np.linalg.norm(v)).astype(np.float32)
    key = f"{KEY_PREFIX[args.functional]}_L{site}"

    npz_path = args.out_dir / "roster_gradients.npz"
    bank = dict(np.load(npz_path)) if npz_path.exists() else {}
    bank[key] = v_unit
    np.savez(npz_path, **bank)
    stamps_path = args.out_dir / "roster_gradients_stamps.json"
    stamps = json.loads(stamps_path.read_text()) if stamps_path.exists() else {}
    stamps[key] = {
        "functional": args.functional, "site": site, "n_gens": len(gids), "gids": gids,
        "raw_norm": float(np.linalg.norm(v)), "mean_S": float(np.mean(s_values)),
        "eos_ids": eos if args.functional == "eos" else None,
        "recipe": "v4_gradient structure verbatim; S swapped; generated positions only",
    }
    stamps_path.write_text(json.dumps(stamps, indent=2))
    logger.info(f"{key} banked -> {npz_path}")


if __name__ == "__main__":
    main()
