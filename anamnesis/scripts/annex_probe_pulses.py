"""ANNEX — potential-probe program pulses (PROGRAM-potential-probes-2026-07-16; ratified,
desk-authorized 2026-07-17). Construction-side code; readout stays banked.

Four natural per-token INTERNAL functionals (none dir0-motivated, none output-side — the
output-side trio lives in annex_potential_gradient.py), differentiated w.r.t. the L{site}
residual input over banked stage-0 generations. V7-recipe verbatim otherwise: same leaf
hook, same gen sampling, same generated-span averaging, same unit-norm + stamps.

  attnent   S = mean_t[ H(mean-head attention row t at the site) ]     (how spread)
  anchor    S = mean_t[ mean-head attention mass on position 0 ]       (anchor/BOS pull)
  vnorm     S = mean_t[ ||v_proj output at t||_2 ]                     (value-vector norm)
  gatemass  S = mean_t[ mean |SiLU(gate_proj output at t)| ]           (gate activation)

All four are computed AT the probed site, downstream of the leaf, so gradients flow.
Keys banked as Gattnent/Ganchor/Gvnorm/Ggatemass_L{site} (RAW gradients — band-passed
probe members are built LOCALLY by annex_band_pass.py through the SAME SITE's banked Σ;
per-probe RESPONSE-CLASS predictions are frozen BEFORE each pulse's member is steered,
rider-1 pattern).

Usage (Heimdall, 1 GPU; node2 paths shown):
    python -m anamnesis.scripts.annex_probe_pulses --model 3b \
        --model-path /dev/shm/luxi-anamnesis/models/llama-3.2-3b-instruct \
        --stage0-run /dev/shm/luxi-anamnesis/anamnesis-extract/runs/vmb_stage0_3b \
        --out-dir <...>/probe_vectors_3b --functional attnent --site 7 --n-gens 20
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

KEY_PREFIX = {"attnent": "Gattnent", "anchor": "Ganchor",
              "vnorm": "Gvnorm", "gatemass": "Ggatemass"}


def s_terms_attnent(ctx: dict, P: int, L: int) -> list:
    """Entropy of the mean-head attention row per generated position (spread)."""
    attn = ctx["attn"][0].float().mean(dim=0)          # [T, T], mean over heads
    terms = []
    for t in range(P, L):
        row = attn[t, : t + 1]
        row = row / row.sum().clamp_min(1e-12)
        terms.append(-(row * row.clamp_min(1e-12).log()).sum())
    return terms


def s_terms_anchor(ctx: dict, P: int, L: int) -> list:
    """Mean-head attention mass on position 0 (the anchor/BOS column)."""
    attn = ctx["attn"][0].float().mean(dim=0)          # [T, T]
    return [attn[t, 0] for t in range(P, L)]


def s_terms_vnorm(ctx: dict, P: int, L: int) -> list:
    """L2 norm of the site's v_proj output per generated position."""
    v = ctx["v_out"][0].float()                        # [T, kv_heads*head_dim]
    return [v[t].norm() for t in range(P, L)]


def s_terms_gatemass(ctx: dict, P: int, L: int) -> list:
    """Mean |SiLU(gate)| per generated position (SwiGLU gate activation mass)."""
    g = ctx["gate_out"][0].float()                     # [T, intermediate]
    return [torch.nn.functional.silu(g[t]).abs().mean() for t in range(P, L)]


S_FNS = {"attnent": s_terms_attnent, "anchor": s_terms_anchor,
         "vnorm": s_terms_vnorm, "gatemass": s_terms_gatemass}
NEEDS_ATTN = {"attnent", "anchor"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--functional", choices=list(S_FNS.keys()), required=True)
    ap.add_argument("--site", type=int, required=True, help="probe site (3B grid: 7/14/21)")
    ap.add_argument("--n-gens", type=int, default=20)
    args = ap.parse_args()
    site = args.site
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="eager",
    ).to("cuda").eval()
    layers = model.model.layers

    entries = json.loads((args.stage0_run / "replay_manifest.json").read_text())["entries"]
    all_ids = sorted(int(k) for k in entries)
    gids = all_ids[:: max(1, len(all_ids) // args.n_gens)][: args.n_gens]

    ctx: dict[str, torch.Tensor] = {}

    def pre_hook(module, hook_args, hook_kwargs):
        hs = hook_args[0] if hook_args else hook_kwargs.get("hidden_states")
        leaf = hs.detach().clone().requires_grad_(True)
        ctx["leaf"] = leaf
        if hook_args:
            return (leaf,) + tuple(hook_args[1:]), hook_kwargs
        hook_kwargs = dict(hook_kwargs)
        hook_kwargs["hidden_states"] = leaf
        return hook_args, hook_kwargs

    handles = [layers[site].register_forward_pre_hook(pre_hook, with_kwargs=True)]
    if args.functional == "vnorm":
        handles.append(layers[site].self_attn.v_proj.register_forward_hook(
            lambda m, a, o: ctx.__setitem__("v_out", o)))
    if args.functional == "gatemass":
        handles.append(layers[site].mlp.gate_proj.register_forward_hook(
            lambda m, a, o: ctx.__setitem__("gate_out", o)))

    s_fn = S_FNS[args.functional]
    need_attn = args.functional in NEEDS_ATTN

    grads, s_values = [], []
    for g in gids:
        e = entries[str(g)]
        ids = torch.tensor([e["input_ids"]], dtype=torch.long, device="cuda")
        P, L = int(e["prompt_length"]), ids.shape[1]
        with torch.enable_grad():
            out = model(ids, use_cache=False, return_dict=True,
                        output_attentions=need_attn)
            if need_attn:
                ctx["attn"] = out.attentions[site]
            terms = s_fn(ctx, P, L)
            S = torch.stack(terms).mean()
            S.backward()
        leaf = ctx["leaf"]
        if leaf.grad is None:
            raise RuntimeError(
                f"gen {g}: no gradient reached the L{site} leaf — the functional "
                f"'{args.functional}' is not differentiable to this site's input")
        gvec = leaf.grad[0, P:L, :].float().mean(dim=0).cpu().numpy()
        grads.append(gvec)
        s_values.append(float(S.detach()))
        model.zero_grad(set_to_none=True)
        ctx.clear()
        logger.info(f"gen {g}: S={s_values[-1]:.4f} |grad|={np.linalg.norm(gvec):.3e}")
    for h in handles:
        h.remove()

    v = np.mean(grads, axis=0)
    v_unit = (v / np.linalg.norm(v)).astype(np.float32)
    key = f"{KEY_PREFIX[args.functional]}_L{site}"

    npz_path = args.out_dir / "probe_gradients.npz"
    bank = dict(np.load(npz_path)) if npz_path.exists() else {}
    bank[key] = v_unit
    np.savez(npz_path, **bank)
    stamps_path = args.out_dir / "probe_gradients_stamps.json"
    stamps = json.loads(stamps_path.read_text()) if stamps_path.exists() else {}
    stamps[key] = {
        "functional": args.functional, "site": site, "n_gens": len(gids), "gids": gids,
        "raw_norm": float(np.linalg.norm(v)), "mean_S": float(np.mean(s_values)),
        "recipe": "V7-recipe verbatim (leaf at L{site} residual input, generated span, "
                  "mean over gens, unit-norm); INTERNAL functional at the same site",
    }
    stamps_path.write_text(json.dumps(stamps, indent=2))
    logger.info(f"{key} banked -> {npz_path}")


if __name__ == "__main__":
    main()
