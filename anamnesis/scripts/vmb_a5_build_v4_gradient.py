"""A5 V4 — mode-dir0 Route 2: feature-gradient vector via replay backprop.

dir0's top structure-coefficient features (banked map, 3B) are L14 attention
recency-vs-prompt-region reads. Differentiable surrogate, defined here and
stamped into the record:

    S = mean_t[ recency_bias_L14(t) ] - mean_t[ prompt_mass_L14(t) ]

with recency_bias(t) = head-mean attention mass on the last 20% of positions
(state_extractor definition, verbatim) and prompt_mass(t) = head-mean mass on
positions < prompt_len (attention_flow definition). Positive S = the analogical
pole (matches V3's sign and the pre-stated semantics-match direction).

V4_L14 = unit-normalized mean over banked Stage-0 gens of dE[S]/dh_L14 averaged
over generated positions (h_L14 = the residual entering decoder layer 14 — the
same surface ResidualWriteSpec injects into).

Usage (node1, 1 GPU, via Heimdall):
    python -m anamnesis.scripts.vmb_a5_build_v4_gradient --model 3b \
        --model-path /models/llama-3.2-3b-instruct \
        --stage0-run /models/anamnesis-extract/runs/vmb_stage0_3b \
        --out-dir /models/anamnesis-extract/battery/a5_vectors_3b --n-gens 20
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

SITE = 14


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-gens", type=int, default=20)
    ap.add_argument("--map-site", type=int, default=SITE,
                    help="map injection site (3B=14, 8B=16). The recency-vs-prompt surrogate is "
                         "computed from attention at THIS layer, differentiated w.r.t. its residual input.")
    args = ap.parse_args()
    site = args.map_site

    preset = MODEL_PRESETS[args.model]
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="eager",
    ).to("cuda").eval()
    from anamnesis.extraction.model_loader import decoder_layers
    layers = decoder_layers(model)  # resolves Gemma-3 wrapper (language_model nesting)

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

    grads = []
    for g in gids:
        e = entries[str(g)]
        ids = torch.tensor([e["input_ids"]], dtype=torch.long, device="cuda")
        P = int(e["prompt_length"])
        L = ids.shape[1]
        with torch.enable_grad():
            out = model(ids, use_cache=False, output_attentions=True, return_dict=True)
            attn = out.attentions[site][0].float()  # [H, L, L]
            mean_attn = attn.mean(dim=0)            # [L, L]
            s_terms = []
            for t in range(P, L):
                row = mean_attn[t, : t + 1]
                total = row.sum().clamp_min(1e-12)
                cutoff = max(1, int((t + 1) * 0.8))
                recency = row[cutoff:].sum() / total
                prompt_mass = row[:P].sum() / total
                s_terms.append(recency - prompt_mass)
            S = torch.stack(s_terms).mean()
            S.backward()
        leaf = captured["leaf"]
        gvec = leaf.grad[0, P:L, :].float().mean(dim=0).cpu().numpy()  # avg over gen positions
        grads.append(gvec)
        model.zero_grad(set_to_none=True)
        captured.clear()
        logger.info(f"gen {g}: |grad| {np.linalg.norm(gvec):.3e}")
    handle.remove()

    v = np.mean(grads, axis=0)
    v_unit = (v / np.linalg.norm(v)).astype(np.float32)

    args.out_dir.mkdir(parents=True, exist_ok=True)   # fresh out-dirs (e.g. M6) — was assumed to exist
    npz_path = args.out_dir / "a5_vectors.npz"
    stamps_path = args.out_dir / "a5_vectors_stamps.json"
    bank = dict(np.load(npz_path)) if npz_path.exists() else {}
    v4_key, v3_key = f"V4_L{site}", f"V3_L{site}"
    bank[v4_key] = v_unit
    np.savez(npz_path, **bank)
    stamps = json.loads(stamps_path.read_text()) if stamps_path.exists() else {}
    # cosine to V3_L{site} — the two routes to the same coordinate (report-side interest)
    cos_v3 = (float(np.dot(v_unit, bank[v3_key])) if v3_key in bank else None)
    stamps[v4_key] = {
        "trait": "mode-dir0", "route": "map-route2-feature-gradient",
        "surrogate": f"mean_t[recency_bias_L{site}] - mean_t[prompt_mass_L{site}] "
                     "(head-mean attention; last-20% window; state_extractor defs)",
        "n_gens": len(gids), "gids": gids, "raw_norm": float(np.linalg.norm(v)),
        f"cosine_to_{v3_key}": cos_v3,
    }
    stamps_path.write_text(json.dumps(stamps, indent=2))
    logger.info(f"{v4_key} banked (cos to {v3_key}: {cos_v3}) -> {npz_path}")


if __name__ == "__main__":
    main()
