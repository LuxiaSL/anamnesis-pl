"""§B.2 guard + §B.6 ∇-anatomy panel (DESIGN-V4 §B.6; Luxia-authorized window add 2026-07-14).

The fork (§2.2) found dir0's post-softmax-MASS surrogate S_mass gradient-flat at L14. §B.2
mandates: a flat surrogate is not a flat COORDINATE until the pre-softmax LOGIT surrogate
also reads flat (flatness could be softmax saturation). §B.6 rides that guard as a generality
probe: same leaf, same 20 gids, FOUR surrogate functionals — is the anatomy dir0-local or
facet-general?

  S_mass    post-softmax (recency − prompt) attention MASS   [the banked fork surrogate]
  S_logit   the SAME region decomposition read PRE-softmax    [= the §B.2 guard]
            (attention LOGITS; computed as region-MEAN of log-weights — the per-row logsumexp
             constant cancels in a within-row difference, so post-softmax weights suffice)
  S_gate    SwiGLU gate-activation fraction (soft: mean sigmoid(gate_proj)) over gen positions
  S_entropy mean next-token entropy over generated positions  [BANK its gradient regardless —
            becomes the formula-route rung of the synthetic-temperature cell]

Replay + backprop only (NO free generation → no steering addendum needed). One forward per
gen, four backward passes (retain_graph). Banks per-functional G-matrices (20×3072) + a
summary JSON next to the fork record. Downstream: `vmb_v4prime_pulse.py --gradient-G <Gnpz>`
per functional reproduces the metric/spectral pulse (S_mass must reproduce the ⛔RULING
numbers). First-read → outer loop, nothing stamped.

⚠ NEEDS GPU (one card, eager attention). Run (node1):
    python -m anamnesis.scripts.vmb_v4_grad_panel --model 3b \
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
RECENCY_FRAC = 0.8       # region cutoff: last 20% of the row = "recency", [:P] = "prompt"
FUNCTIONALS = ["S_mass", "S_logit", "S_gate", "S_entropy"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--vectors", type=Path, default=None,
                    help="a5_vectors.npz — for cos(∇S, V3_L14) / cos(∇S, V4_L14) rows")
    ap.add_argument("--candidate-npz", type=Path, default=None,
                    help="14k candidate vectors (e.g. Ksoclin_L14) — the ∇-panel differentiability "
                         "confirm: is the candidate seen by ANY functional gradient above the R-null band?")
    ap.add_argument("--candidate-keys", nargs="+", default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-gens", type=int, default=20)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    layers = model.model.layers

    entries = json.loads((args.stage0_run / "replay_manifest.json").read_text())["entries"]
    all_ids = sorted(int(k) for k in entries)
    gids = all_ids[:: max(1, len(all_ids) // args.n_gens)][: args.n_gens]

    captured: dict[str, torch.Tensor] = {}

    def leaf_pre_hook(module, a, kw):
        hs = a[0] if a else kw.get("hidden_states")
        leaf = hs.detach().clone().requires_grad_(True)
        captured["leaf"] = leaf
        if a:
            return (leaf,) + tuple(a[1:]), kw
        kw = dict(kw); kw["hidden_states"] = leaf
        return a, kw

    def gate_hook(module, a, out):
        captured["gate"] = out            # gate_proj output (pre-silu), (1, T, intermediate)

    h1 = layers[SITE].register_forward_pre_hook(leaf_pre_hook, with_kwargs=True)
    h2 = layers[SITE].mlp.gate_proj.register_forward_hook(gate_hook)

    grads: dict[str, list[np.ndarray]] = {f: [] for f in FUNCTIONALS}
    for g in gids:
        e = entries[str(g)]
        ids = torch.tensor([e["input_ids"]], dtype=torch.long, device="cuda")
        P, L = int(e["prompt_length"]), int(len(e["input_ids"]))
        with torch.enable_grad():
            out = model(ids, use_cache=False, output_attentions=True, return_dict=True)
            attn = out.attentions[SITE][0].float()          # (H, T, T) post-softmax
            mean_attn = attn.mean(dim=0)                     # (T, T)
            logw = torch.log(mean_attn.clamp_min(1e-12))
            gate_pre = captured["gate"][0].float()           # (T, intermediate)
            logits = out.logits[0].float()                   # (T, vocab)
            lp = torch.log_softmax(logits, dim=-1)
            p = lp.exp()
            ent = -(p * lp).sum(dim=-1)                      # (T,) next-token entropy

            s_mass, s_logit, s_gate, s_ent = [], [], [], []
            for t in range(P, L):
                row = mean_attn[t, : t + 1]
                total = row.sum().clamp_min(1e-12)
                cut = max(1, int((t + 1) * RECENCY_FRAC))
                s_mass.append(row[cut:].sum() / total - row[:P].sum() / total)
                lr = logw[t, : t + 1]
                s_logit.append(lr[cut:].mean() - lr[:P].mean())
                s_gate.append(torch.sigmoid(gate_pre[t]).mean())
                s_ent.append(ent[t])
            S = {
                "S_mass": torch.stack(s_mass).mean(),
                "S_logit": torch.stack(s_logit).mean(),
                "S_gate": torch.stack(s_gate).mean(),
                "S_entropy": torch.stack(s_ent).mean(),
            }
            for i, f in enumerate(FUNCTIONALS):
                model.zero_grad(set_to_none=True)
                if captured["leaf"].grad is not None:
                    captured["leaf"].grad = None
                S[f].backward(retain_graph=(i < len(FUNCTIONALS) - 1))
                gvec = captured["leaf"].grad[0, P:L, :].float().mean(dim=0).cpu().numpy()
                grads[f].append(gvec.astype(np.float64))
        logger.info(f"gen {g}: " + " ".join(
            f"{f} |grad| {np.linalg.norm(grads[f][-1]):.3e}" for f in FUNCTIONALS))
        captured.pop("leaf", None); captured.pop("gate", None)
    h1.remove(); h2.remove()

    vecs = np.load(args.vectors) if args.vectors and args.vectors.exists() else None
    v3 = vecs["V3_L14"].astype(np.float64) if vecs is not None and "V3_L14" in vecs.files else None
    v4 = vecs["V4_L14"].astype(np.float64) if vecs is not None and "V4_L14" in vecs.files else None

    def _cos(a, b):
        return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))

    summary = {"model": args.model, "site": SITE, "n_gens": len(gids),
               "STATUS": "FIRST_READ_PENDING (C§8) — §B.2 guard + §B.6 ∇-anatomy panel",
               "provenance": "DESIGN-V4 §B.6; same leaf/gids as the fork; 4 surrogate functionals",
               "recency_frac": RECENCY_FRAC, "functionals": {}}
    mean_grads: dict[str, np.ndarray] = {}
    for f in FUNCTIONALS:
        G = np.stack(grads[f])
        norms = np.linalg.norm(G, axis=1)
        unit = G / np.clip(norms[:, None], 1e-12, None)
        iu = np.triu_indices(len(G), k=1)
        pcos = (unit @ unit.T)[iu]
        mean_vec = G.mean(axis=0)
        mean_grads[f] = mean_vec
        mean_norm = float(np.linalg.norm(mean_vec))
        cancel = mean_norm / max(float(norms.mean()), 1e-12)
        row = {"mean_of_per_gen_norms": float(norms.mean()),
               "norm_of_mean_grad": mean_norm,
               "cancellation_ratio": cancel,
               "sqrt_n_orthogonal_floor": float(1 / np.sqrt(len(G))),
               "pairwise_cos_mean": float(pcos.mean()),
               "per_gen_norm_range": [float(norms.min()), float(norms.max())]}
        if v3 is not None:
            row["cos_meangrad_V3"] = _cos(mean_vec, v3)
            row["per_gen_cos_V3_positive"] = int(sum(_cos(G[i], v3) > 0 for i in range(len(G))))
        if v4 is not None:
            row["cos_meangrad_V4"] = _cos(mean_vec, v4)
        summary["functionals"][f] = row
        np.savez(args.out_dir / f"v4panel_G_{f}_{args.model}.npz",
                 G=G.astype(np.float64), gids=np.asarray(gids, dtype=np.int64),
                 mean_grad=mean_vec.astype(np.float64))
        logger.info(f"{f}: |mean_grad| {mean_norm:.4e}  cancel {cancel:.3f}  "
                    f"pcos {pcos.mean():.3f}"
                    + (f"  cos(·,V3) {row.get('cos_meangrad_V3'):.4f}" if v3 is not None else ""))

    # 14k FORMULA LEG — ∇-panel differentiability confirm for a candidate coordinate (P3').
    # Is the candidate seen by ANY functional's mean gradient ABOVE the R-null band? Frozen
    # prediction for Ksoclin: formula-INERT (no functional aligns above null) — the doctrine's
    # first discriminating test (P=.70). REPORT arithmetic; the call is the outer loop's.
    if args.candidate_npz and args.candidate_keys and args.candidate_npz.exists():
        cd = np.load(args.candidate_npz)
        dim = len(next(iter(mean_grads.values())))
        rng = np.random.default_rng(20260716)
        R = rng.standard_normal((400, dim)); R /= np.linalg.norm(R, axis=1, keepdims=True)
        cand_out = {}
        for key in args.candidate_keys:
            if key not in cd.files:
                cand_out[key] = {"present": False}; continue
            cv = cd[key].astype(np.float64); cv /= np.linalg.norm(cv)
            per_f, any_seen = {}, False
            for f in FUNCTIONALS:
                mg = mean_grads[f]
                c = _cos(mg, cv)
                null_abs = np.abs(R @ (mg / max(np.linalg.norm(mg), 1e-30)))
                p95 = float(np.percentile(null_abs, 95))
                seen = bool(abs(c) > p95)
                any_seen = any_seen or seen
                per_f[f] = {"cos_meangrad_candidate": round(c, 4),
                            "null_p95_abs_cos": round(p95, 4), "seen_above_null": seen}
            cand_out[key] = {"present": True, "per_functional": per_f,
                             "differentiable_any_functional": any_seen,
                             "verdict": ("FORMULA-VISIBLE (a functional gradient sees it above null)"
                                         if any_seen else
                                         "FORMULA-INERT (no functional gradient sees it above the R-null)")}
            logger.info(f"candidate {key}: differentiable={any_seen} "
                        + " ".join(f"{f}={per_f[f]['cos_meangrad_candidate']:+.3f}(p95 {per_f[f]['null_p95_abs_cos']:.3f})"
                                   for f in FUNCTIONALS))
        summary["candidate_formula_leg"] = cand_out

    # §B.6 outcome flags (arithmetic only; interpretation → outer loop)
    sm = summary["functionals"]["S_mass"]["norm_of_mean_grad"]
    sl = summary["functionals"]["S_logit"]["norm_of_mean_grad"]
    summary["B2_guard"] = {
        "S_logit_over_S_mass_norm_ratio": sl / max(sm, 1e-12),
        "reading": ("(a) S_logit >> S_mass → flatness was SATURATION, coordinate NOT flat"
                    if sl > 3 * sm else
                    "(b) S_logit also flat → Branch-B upgrades toward coordinate-level for dir0"
                    if sl < 3 * sm else "intermediate — outer loop rules")}
    p = args.out_dir / f"v4_grad_panel_{args.model}.json"
    p.write_text(json.dumps(summary, indent=1))
    logger.info(f"B2 guard: S_logit/S_mass norm ratio = {sl/max(sm,1e-12):.2f}")
    logger.info(f"banked panel (first-read pending) -> {p}")


if __name__ == "__main__":
    main()
