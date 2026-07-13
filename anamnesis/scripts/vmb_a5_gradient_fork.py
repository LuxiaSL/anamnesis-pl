"""A5 V4 — per-generation gradient FORK (WAVE2-A5 addendum 13d §2.2).

The fix-vs-finding fork for V4. V4_L14 = unit-normalized MEAN over 20 banked gens of
dE[S]/dh_L14 (see `vmb_a5_build_v4_gradient`); its raw pre-normalization norm is ~0.00196.
Two very different worlds produce that:
  - AVERAGING DESTROYED IT: the 20 per-gen gradients are individually order ~0.05 and
    mutually near-orthogonal, so their mean nearly cancels. Then path-following (small steps,
    recomputed gradient) is the right ENGINEERING fix — "feature-gradient" is a misnomer for
    the cancelled mean that got banked.
  - GENUINELY READ-ONLY: each per-gen ‖∇S‖ is itself ~0.002, so S is locally near-flat in h.
    Then mode-dir0 is gradient-read-only — a construction-level "no-lever" FINDING about the
    coordinate, not a bug to fix.
One number (per-gen norm scale + mean pairwise cosine) separates them. This script emits it.

Reuses the EXACT surrogate + hook from `vmb_a5_build_v4_gradient`; the only change is it keeps
the per-gen gradients instead of averaging.

⚠ NEEDS GPU (one card; eager attention). UNTESTED without a GPU — smoke on 3 gens first
(`--n-gens 3`) and sanity-check that the recomputed MEAN matches the banked V4_L14
(cos ≈ 1.0) before trusting the fork verdict.

Usage (node1, 1 GPU, via Heimdall; venv-only, WANDB key via --env not needed here):
    python -m anamnesis.scripts.vmb_a5_gradient_fork --model 3b \
        --model-path /models/llama-3.2-3b-instruct \
        --stage0-run /models/anamnesis-extract/runs/vmb_stage0_3b \
        --vectors /models/anamnesis-extract/battery/a5_vectors_3b/a5_vectors.npz \
        --out-dir /models/anamnesis-extract/battery/arms/A5 --n-gens 20
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
    ap.add_argument("--vectors", type=Path, required=True,
                    help="a5_vectors.npz — to compare the recomputed mean vs banked V4_L14")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-gens", type=int, default=20)
    args = ap.parse_args()
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

    handle = layers[SITE].register_forward_pre_hook(pre_hook, with_kwargs=True)

    grads: list[np.ndarray] = []
    for g in gids:
        e = entries[str(g)]
        ids = torch.tensor([e["input_ids"]], dtype=torch.long, device="cuda")
        P, L = int(e["prompt_length"]), int(len(e["input_ids"]))
        with torch.enable_grad():
            out = model(ids, use_cache=False, output_attentions=True, return_dict=True)
            attn = out.attentions[SITE][0].float()
            mean_attn = attn.mean(dim=0)
            s_terms = []
            for t in range(P, L):
                row = mean_attn[t, : t + 1]
                total = row.sum().clamp_min(1e-12)
                cutoff = max(1, int((t + 1) * 0.8))
                s_terms.append(row[cutoff:].sum() / total - row[:P].sum() / total)
            torch.stack(s_terms).mean().backward()
        gvec = captured["leaf"].grad[0, P:L, :].float().mean(dim=0).cpu().numpy()
        grads.append(gvec.astype(np.float64))
        model.zero_grad(set_to_none=True)
        captured.clear()
        logger.info(f"gen {g}: |grad| {np.linalg.norm(gvec):.3e}")
    handle.remove()

    G = np.stack(grads)                       # [n_gens, d]
    norms = np.linalg.norm(G, axis=1)
    unit = G / np.clip(norms[:, None], 1e-12, None)
    cosmat = unit @ unit.T
    iu = np.triu_indices(len(G), k=1)
    pairwise_cos = cosmat[iu]
    mean_vec = G.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_vec))
    mean_of_norms = float(norms.mean())
    # cancellation ratio: ‖mean(g)‖ / mean(‖g‖). ~1 = aligned (real direction);
    # ~1/sqrt(n) = orthogonal (mean is cancellation of disagreeing gradients).
    cancellation_ratio = mean_norm / max(mean_of_norms, 1e-12)
    sqrt_n_floor = 1.0 / np.sqrt(len(G))

    banked_cos = None
    if args.vectors.exists():
        bank = np.load(args.vectors)
        if "V4_L14" in bank:
            v4 = bank["V4_L14"].astype(np.float64)
            mu = mean_vec / max(mean_norm, 1e-12)
            banked_cos = float(mu @ v4 / max(np.linalg.norm(v4), 1e-12))

    # verdict heuristic (report BOTH numbers; the label is a convenience, not a stamp)
    if mean_of_norms < 0.01 and cancellation_ratio > 0.5:
        verdict = "READ-ONLY candidate: per-gen gradients are individually tiny AND aligned — S is locally near-flat in h; mode-dir0 gradient-read-only (a finding, not a bug)."
    elif cancellation_ratio < 2 * sqrt_n_floor:
        verdict = "AVERAGING-DESTROYED candidate: per-gen gradients are non-trivial but mutually near-orthogonal; the mean cancels. Path-following (recomputed gradient) is the right fix."
    else:
        verdict = "MIXED/INTERMEDIATE: neither clean pole — report both numbers; outer loop rules."

    out = {
        "model": args.model, "site": SITE, "n_gens": len(G),
        "STATUS": "FIRST_READ_PENDING (C§8)",
        "provenance": "WAVE2-A5 addendum 13d §2.2; per-gen fork of vmb_a5_build_v4_gradient",
        "per_gen_grad_norms": [float(x) for x in norms],
        "mean_of_per_gen_norms": mean_of_norms,
        "norm_of_mean_grad": mean_norm,
        "cancellation_ratio": float(cancellation_ratio),
        "sqrt_n_orthogonal_floor": float(sqrt_n_floor),
        "pairwise_cos_mean": float(pairwise_cos.mean()),
        "pairwise_cos_std": float(pairwise_cos.std()),
        "recomputed_mean_vs_banked_V4_cos": banked_cos,
        "verdict_heuristic": verdict,
        "notes": ["banked V4 raw_norm ~0.00196; norm_of_mean_grad here should match it.",
                  "recomputed_mean_vs_banked_V4_cos should be ~1.0 — else the reproduction drifted."],
    }
    p = args.out_dir / f"a5_gradient_fork_{args.model}.json"
    p.write_text(json.dumps(out, indent=1))
    logger.info(f"fork verdict: {verdict}")
    logger.info(f"banked (first-read pending) -> {p}")


if __name__ == "__main__":
    main()
