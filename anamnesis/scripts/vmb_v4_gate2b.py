"""§B.4 gate-2b — discriminating null for the logit-Hessian alignment (session-4 Part B1).

Stage-1's §B.4 gate-2 passed (|cos(w*,V3_top64)|=0.423, Haar p=4e-4) BUT verification found
the alignment is TOP-4-MEDIATED (w* 53% on e1; top-4-removed align 0.056, p=.67). The Haar null
excludes chance, not GENERIC outlier-curvature. gate-2b: the k=64 Hessians of the OTHER panel
functionals — S_mass, S_gate, S_entropy — SAME gids/pipeline/leaf as the S_logit run. S_gate is
DIR0-BLIND → the discriminator:
  • S_gate's w* also loads e1 ~0.5 AND aligns ~0.4 with V3_top64  ⇒ generic outlier-curvature;
    §B.4 gate-2 refuted-in-spirit, stage-2 DEAD.
  • w*_logit unique among the four ⇒ specificity survives; stage-2 → ratification WITH an
    e1-ablated steering arm.
Either way §B.4 stage-2 stays HELD behind this. Compute-only (replay + double-backprop HVPs),
no free gen. First-read → outer loop; nothing stamped.

Functionals match vmb_v4_grad_panel.py VERBATIM (same S definitions the gradients used).
Hessian machinery matches vmb_v4_b7b4_stage1.py §B.4 leg VERBATIM (leaf-perturbation d at the
L14 residual input; H_k = U_kᵀ(∇²_d S)U_k via double-backprop; top-1 eigvec by |eigenvalue|).
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
RECENCY_FRAC = 0.8
NULL_SEED = 20260714
FUNCTIONALS = ["S_mass", "S_gate", "S_entropy"]   # the three discriminators (S_logit = stage-1)


def _panel_gids(all_ids, n):
    """The SAME fork/panel gids as stage-1 (first n by sorted id) — controlled comparison."""
    return sorted(all_ids)[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--sigma", type=Path, required=True)
    ap.add_argument("--vectors", type=Path, required=True, help="a5_vectors.npz (V3_L14)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-gens", type=int, default=20)
    ap.add_argument("--k", type=int, default=64)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    layers = model.model.layers
    dev = "cuda"

    entries = json.loads((args.stage0_run / "replay_manifest.json").read_text())["entries"]
    all_ids = sorted(int(k) for k in entries)
    fork = _panel_gids(all_ids, args.n_gens)

    S = np.load(args.sigma)
    evals = S["evals"].astype(np.float64)
    evecs = S["evecs"].astype(np.float64)
    order = np.argsort(evals)[::-1]                       # descending
    Uk_np = evecs[:, order[: args.k]]                     # (d, k), columns ordered by descending eigval
    Uk = torch.tensor(Uk_np, dtype=torch.float32, device=dev)
    v3 = np.load(args.vectors)["V3_L14"].astype(np.float64)
    v3k = Uk_np.T @ v3
    v3k /= max(np.linalg.norm(v3k), 1e-12)

    # ── leaf-perturbation hook (matches stage-1: inject broadcast d at L14 residual input) ──
    state = {"P": 0, "L": 0}
    ctx: dict = {}
    captured: dict = {}

    def pre_hook(module, a, kw):
        hs = a[0] if a else kw.get("hidden_states")
        out = hs
        if ctx.get("d") is not None:
            out = hs.clone()
            out[:, state["P"]:state["L"], :] = out[:, state["P"]:state["L"], :] + ctx["d"]
        if a:
            return (out,) + tuple(a[1:]), kw
        kw = dict(kw); kw["hidden_states"] = out
        return a, kw

    def gate_hook(m, a, o):
        captured["gate"] = o                              # gate_proj output (pre-silu), (1,T,inter)

    h1 = layers[SITE].register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = layers[SITE].mlp.gate_proj.register_forward_hook(gate_hook)

    def functionals(out):
        """Return {name: scalar tensor} — verbatim from vmb_v4_grad_panel.py."""
        attn = out.attentions[SITE][0].float()            # (H,T,T) post-softmax
        mean_attn = attn.mean(dim=0)                      # (T,T)
        gate_pre = captured["gate"][0].float()            # (T, intermediate)
        logits = out.logits[0].float()
        lp = torch.log_softmax(logits, dim=-1)
        ent = -(lp.exp() * lp).sum(dim=-1)                # (T,)
        s_mass, s_gate, s_ent = [], [], []
        P, L = state["P"], state["L"]
        for t in range(P, L):
            row = mean_attn[t, : t + 1]
            total = row.sum().clamp_min(1e-12)
            cut = max(1, int((t + 1) * RECENCY_FRAC))
            s_mass.append(row[cut:].sum() / total - row[:P].sum() / total)
            s_gate.append(torch.sigmoid(gate_pre[t]).mean())
            s_ent.append(ent[t])
        return {"S_mass": torch.stack(s_mass).mean(),
                "S_gate": torch.stack(s_gate).mean(),
                "S_entropy": torch.stack(s_ent).mean()}

    # ── per-functional projected Hessian on the fork gids ──
    per_fn = {f: {"topvecs": [], "topeigs": []} for f in FUNCTIONALS}
    for g in fork:
        e = entries[str(g)]
        ids = torch.tensor([e["input_ids"]], dtype=torch.long, device=dev)
        state["P"], state["L"] = int(e["prompt_length"]), int(len(e["input_ids"]))
        for f in FUNCTIONALS:
            d = torch.zeros(model.config.hidden_size, dtype=torch.float32, device=dev, requires_grad=True)
            ctx["d"] = d
            with torch.enable_grad():
                out = model(ids, use_cache=False, output_attentions=True, return_dict=True)
                Sf = functionals(out)[f]
                grad_d = torch.autograd.grad(Sf, d, create_graph=True)[0]
                cols = [torch.autograd.grad(grad_d @ Uk[:, i], d, retain_graph=True)[0].detach()
                        for i in range(args.k)]
                Hk = (Uk.T @ torch.stack(cols, dim=1)).cpu().numpy().astype(np.float64)
            ctx["d"] = None
            captured.pop("gate", None)
            Hk = 0.5 * (Hk + Hk.T)
            w, V = np.linalg.eigh(Hk)
            j = int(np.argmax(np.abs(w)))
            per_fn[f]["topvecs"].append(V[:, j])
            per_fn[f]["topeigs"].append(float(w[j]))
        logger.info(f"gen {g} done ({', '.join(FUNCTIONALS)})")

    h1.remove(); h2.remove()

    # ── gate-2b readout per functional: consistency, align, e1/top-4 decomposition, matched null ──
    rng = np.random.default_rng(NULL_SEED)
    ksphere = 1.0 / np.sqrt(args.k)
    # selection-matched Haar null on |cos(top-eig, v3k)| (shared across functionals)
    hn = rng.standard_normal((1000, args.k, args.k)); hn = 0.5 * (hn + np.transpose(hn, (0, 2, 1)))
    null_align = []
    for m in range(1000):
        wm, Vm = np.linalg.eigh(hn[m]); jm = int(np.argmax(np.abs(wm)))
        null_align.append(abs(float(Vm[:, jm] @ v3k)))
    null_align = np.array(null_align)

    results = {}
    for f in FUNCTIONALS:
        TV = np.stack(per_fn[f]["topvecs"])
        TVu = TV / np.clip(np.linalg.norm(TV, axis=1, keepdims=True), 1e-12, None)
        absc = np.abs((TVu @ TVu.T)[np.triu_indices(len(TV), k=1)])
        mean_tv = TVu.mean(0); mean_tv /= max(np.linalg.norm(mean_tv), 1e-12)
        align = abs(float(mean_tv @ v3k))
        signed = float(mean_tv @ v3k)
        e1_mass = float(mean_tv[0] ** 2)                  # loading on Σ's top outlier direction
        top4_mass = float(np.sum(mean_tv[:4] ** 2))
        abl = mean_tv.copy(); abl[:4] = 0.0
        abl_n = np.linalg.norm(abl)
        top4_removed_align = abs(float((abl / abl_n) @ v3k)) if abl_n > 1e-12 else 0.0
        # e1 signed contribution to the alignment
        e1_contrib = float(mean_tv[0] * v3k[0])
        results[f] = {
            "consistency_gate": "PASS" if absc.mean() > 2 * ksphere else "STOP-incoherent",
            "per_gen_topeig_pairwise_abscos_mean": round(float(absc.mean()), 3),
            "align_wstar_V3k": round(align, 4), "align_signed": round(signed, 4),
            "selection_matched_null_p": round(float((null_align >= align).mean()), 4),
            "e1_mass_frac": round(e1_mass, 4), "top4_mass_frac": round(top4_mass, 4),
            "e1_signed_contribution": round(e1_contrib, 4),
            "top4_removed_align": round(top4_removed_align, 4),
            "topeig_values_range": [round(min(per_fn[f]["topeigs"]), 4), round(max(per_fn[f]["topeigs"]), 4)],
        }
        np.savez(args.out_dir / f"v4_gate2b_{f}_k{args.k}_{args.model}.npz",
                 top_eigvecs=TV, top_eigvals=np.asarray(per_fn[f]["topeigs"]),
                 w_star=(Uk_np @ mean_tv).astype(np.float64))
        logger.info(f"[{f}] align={align:.3f} p={results[f]['selection_matched_null_p']} "
                    f"e1_mass={e1_mass:.3f} top4_removed_align={top4_removed_align:.3f}")

    # ── the discriminator verdict (S_gate = dir0-blind reference) ──
    sg = results["S_gate"]
    stage1_logit = {"align": 0.423, "e1_mass": 0.53, "top4_removed_align": 0.056}
    gate_generic = bool(sg["align_wstar_V3k"] >= 0.30 and sg["e1_mass_frac"] >= 0.35)
    verdict = ("gate-2 REFUTED-IN-SPIRIT (S_gate also outlier-curvature-aligned → §B.4 stage-2 DEAD)"
               if gate_generic else
               "S_logit alignment may be SPECIFIC (S_gate does not replicate the e1-loaded alignment) "
               "→ §B.4 stage-2 to ratification WITH e1-ablated arm")

    out = {"model": args.model, "site": SITE, "k": args.k, "n_gens": len(fork),
           "STATUS": "FIRST_READ_PENDING (C§8) — §B.4 gate-2b discriminating null (compute-only)",
           "stage1_S_logit_reference": stage1_logit, "per_functional": results,
           "S_gate_generic_outlier_curvature": gate_generic, "VERDICT": verdict,
           "law": "k=64 projected Hessian double-backprop; top-1 eigvec by |eigval|; "
                  "align vs V3_top64; selection-matched Haar null; e1/top-4 decomposition"}
    (args.out_dir / f"v4_gate2b_{args.model}.json").write_text(json.dumps(out, indent=1))
    logger.info(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
