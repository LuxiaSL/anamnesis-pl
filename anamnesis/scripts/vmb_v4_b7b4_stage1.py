"""§B.7 stage-1 + §B.4 stage-1 — one compute-only job (DESIGN-V4 §B.7/§B.4, frozen
2026-07-14 late; 14d item-4 class: replay+backprop, NO free generation).

§B.7 (first-order, OUTRANKS §B.4): the entropy-band candidate V7 = unit(P_[16:256]·∇S_entropy).
The band [16:256] of Σ_L14 (descending) is FROZEN from the panel's exploratory pass (cos +0.132,
z 3.7) — the confirmatory stage may NOT re-search it. Re-measure ∇S_entropy on **20 FRESH gids**
(disjoint from the panel's 20 — asserted) and recompute the frozen band's alignment vs V3 with
the band-matched null. PASS = band alignment z ≥ 2.5 on fresh data. Also report per-gen
sign-consistency (n/20) and per-gen pairwise coherence (panel measured 0.63; if it collapses on
fresh gids, report + STOP).

§B.4 (second-order, armed): H_k = U_kᵀ(∇²S_logit)U_k at the L14 leaf, U_k = top-k eigvecs of
Σ_L14 (k=64 primary). Hessian of **S_logit** (pre-softmax — curvature of the saturated S_mass
would re-import the §B.2 confound). Same 20 gids as the fork (controlled). Per gen: 64 HVPs via
double-backprop of a broadcast perturbation d added at gen positions. Bank H_k per gen. Gate 1
(here): per-gen top-eigendirection pairwise |cos| consistency. Gate 2 (CPU alignment vs
selection-matched null) reported here too; free-gen legs are A5-class (ratify separately).

First-reads → outer loop, nothing stamped. ⚠ NEEDS GPU (one card, eager). Run:
    python -m anamnesis.scripts.vmb_v4_b7b4_stage1 --model 3b \
      --model-path /models/llama-3.2-3b-instruct \
      --stage0-run /models/anamnesis-extract/runs/vmb_stage0_3b \
      --sigma /models/anamnesis-extract/battery/arms/A5/a5_sigma_L14_3b.npz \
      --vectors /models/anamnesis-extract/battery/a5_vectors_3b/a5_vectors.npz \
      --out-dir /models/anamnesis-extract/battery/arms/A5 --n-gens 20 --k 64
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from anamnesis.config import MODEL_PRESETS

SITE = 14
RECENCY_FRAC = 0.8
BAND = (16, 256)          # FROZEN §B.7 band (descending Σ rank)
NULL_SEED = 20260714


def _panel_gids(all_ids, n):
    return all_ids[:: max(1, len(all_ids) // n)][:n]


def _fresh_gids(all_ids, n):
    step = max(1, len(all_ids) // n)
    return all_ids[step // 2:: step][:n]     # offset half a step → disjoint from panel


def _cos(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--sigma", type=Path, required=True)
    ap.add_argument("--vectors", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-gens", type=int, default=20)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--site", type=int, default=SITE,
                    help="grad-leaf site (3B=14, 8B=16 — the 8B 2×2 swaps site per its "
                         "pricing doc; --sigma and --vectors must be that site's banks)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    site = args.site

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    layers = model.model.layers
    dev = "cuda"

    entries = json.loads((args.stage0_run / "replay_manifest.json").read_text())["entries"]
    all_ids = sorted(int(k) for k in entries)
    panel = set(_panel_gids(all_ids, args.n_gens))
    fresh = _fresh_gids(all_ids, args.n_gens)
    assert not (set(fresh) & panel), "FRESH gids overlap the panel's — §B.7 disjointness violated"

    S = np.load(args.sigma)
    evals = S["evals"].astype(np.float64)
    evecs = S["evecs"].astype(np.float64)
    order = np.argsort(evals)[::-1]                      # descending
    band_idx = order[BAND[0]:BAND[1]]
    Uband = torch.tensor(evecs[:, band_idx], dtype=torch.float32, device=dev)   # (d, 240)
    Uk = torch.tensor(evecs[:, order[:args.k]], dtype=torch.float32, device=dev)  # (d, k)
    v3 = np.load(args.vectors)[f"V3_L{site}"].astype(np.float64)

    state = {"P": 0, "L": 0}

    # ── leaf hook: capture requires_grad leaf; optionally add a broadcast perturbation d ──
    ctx: dict = {}

    def hook(module, a, kw):
        hs = a[0] if a else kw.get("hidden_states")
        leaf = hs.detach().clone().requires_grad_(True)
        ctx["leaf"] = leaf
        out = leaf
        if ctx.get("d") is not None:
            out = leaf.clone()
            out[:, state["P"]:state["L"], :] = out[:, state["P"]:state["L"], :] + ctx["d"]
        if a:
            return (out,) + tuple(a[1:]), kw
        kw = dict(kw); kw["hidden_states"] = out
        return a, kw

    gate_out: dict = {}

    def gate_hook(m, a, o):
        gate_out["g"] = o

    h1 = layers[site].register_forward_pre_hook(hook, with_kwargs=True)
    h2 = layers[site].mlp.gate_proj.register_forward_hook(gate_hook)

    def _S_entropy(out):
        logits = out.logits[0].float()
        lp = torch.log_softmax(logits, dim=-1)
        ent = -(lp.exp() * lp).sum(dim=-1)
        return ent[state["P"]:state["L"]].mean()

    def _S_logit(out):
        attn = out.attentions[site][0].float().mean(dim=0)      # (T,T)
        logw = torch.log(attn.clamp_min(1e-12))
        terms = []
        for t in range(state["P"], state["L"]):
            lr = logw[t, : t + 1]
            cut = max(1, int((t + 1) * RECENCY_FRAC))
            terms.append(lr[cut:].mean() - lr[: state["P"]].mean())
        return torch.stack(terms).mean()

    # ── §B.7 stage-1: ∇S_entropy on FRESH gids ──
    ctx["d"] = None
    Ge = []
    for g in fresh:
        e = entries[str(g)]
        ids = torch.tensor([e["input_ids"]], dtype=torch.long, device=dev)
        state["P"], state["L"] = int(e["prompt_length"]), int(len(e["input_ids"]))
        with torch.enable_grad():
            out = model(ids, use_cache=False, output_attentions=True, return_dict=True)
            Se = _S_entropy(out)
            model.zero_grad(set_to_none=True)
            Se.backward()
            Ge.append(ctx["leaf"].grad[0, state["P"]:state["L"], :].float().mean(0).cpu().numpy().astype(np.float64))
        ctx.pop("leaf", None)
    Ge = np.stack(Ge)
    mean_e = Ge.mean(0)
    Ub = evecs[:, band_idx]                                   # (d,240) cpu
    cg = Ub.T @ mean_e; cv = Ub.T @ v3
    band_align = float(cg @ cv / max(np.linalg.norm(cg) * np.linalg.norm(cv), 1e-30))
    rng = np.random.default_rng(NULL_SEED)
    nb = rng.standard_normal((1000, Ub.shape[1]))
    nb /= np.linalg.norm(nb, axis=1, keepdims=True)
    null_cos = (nb @ (cv / np.linalg.norm(cv)))
    b7_z = (band_align - null_cos.mean()) / max(null_cos.std(), 1e-12)
    per_gen_band = np.array([float((Ub.T @ Ge[i]) @ cv / max(np.linalg.norm(Ub.T @ Ge[i]) * np.linalg.norm(cv), 1e-30))
                             for i in range(len(Ge))])
    bandproj = (Ub.T @ Ge.T).T                               # (n,240)
    bu = bandproj / np.clip(np.linalg.norm(bandproj, axis=1, keepdims=True), 1e-12, None)
    iu = np.triu_indices(len(Ge), k=1)
    coh = float((bu @ bu.T)[iu].mean())
    v7 = Ub @ cg; v7 = (v7 / np.linalg.norm(v7)).astype(np.float32)
    np.savez(args.out_dir / f"v4_b7_entropy_fresh_G_{args.model}.npz",
             G=Ge, gids=np.asarray(fresh), mean_grad=mean_e, V7=v7, band=np.asarray(BAND))
    b7 = {"band": list(BAND), "band_alignment_cos": round(band_align, 4),
          "band_matched_null_z": round(float(b7_z), 2),
          "per_gen_positive": f"{int((per_gen_band>0).sum())}/{len(Ge)}",
          "per_gen_coherence": round(coh, 3), "panel_coherence_ref": 0.63,
          "PASS_z_ge_2.5": bool(b7_z >= 2.5),
          "gids_disjoint_from_panel": True}

    # ── §B.4 stage-1: S_logit projected Hessian H_k on the FORK (panel) gids ──
    ctx["d"] = None
    fork = _panel_gids(all_ids, args.n_gens)
    Hks, topvecs, topeigs = [], [], []
    Uk_np = evecs[:, order[:args.k]]
    for g in fork:
        e = entries[str(g)]
        ids = torch.tensor([e["input_ids"]], dtype=torch.long, device=dev)
        state["P"], state["L"] = int(e["prompt_length"]), int(len(e["input_ids"]))
        d = torch.zeros(model.config.hidden_size, dtype=torch.float32, device=dev, requires_grad=True)
        ctx["d"] = d
        with torch.enable_grad():
            out = model(ids, use_cache=False, output_attentions=True, return_dict=True)
            Sl = _S_logit(out)
            grad_d = torch.autograd.grad(Sl, d, create_graph=True)[0]     # (d,)
            cols = []
            for i in range(args.k):
                hv = torch.autograd.grad(grad_d @ Uk[:, i], d, retain_graph=True)[0]
                cols.append(hv.detach())
            HU = torch.stack(cols, dim=1)                                 # (d, k)
            Hk = (Uk.T @ HU).cpu().numpy().astype(np.float64)             # (k, k)
        ctx["d"] = None
        ctx.pop("leaf", None)
        Hk = 0.5 * (Hk + Hk.T)
        w, V = np.linalg.eigh(Hk)
        j = int(np.argmax(np.abs(w)))                                     # top by |eigenvalue|
        Hks.append(Hk); topvecs.append(V[:, j]); topeigs.append(float(w[j]))
    Hks = np.stack(Hks); TV = np.stack(topvecs)
    TVu = TV / np.clip(np.linalg.norm(TV, axis=1, keepdims=True), 1e-12, None)
    absc = np.abs((TVu @ TVu.T)[np.triu_indices(len(TV), k=1)])
    ksphere_null = 1.0 / np.sqrt(args.k)                                  # rough |cos| null scale
    # gate-2 alignment: w* mapped back vs V3_k, selection-matched (top-1) Haar null
    mean_tv = TVu.mean(0); mean_tv /= max(np.linalg.norm(mean_tv), 1e-12)
    w_star = Uk_np @ mean_tv
    v3k = Uk_np.T @ v3; v3k /= max(np.linalg.norm(v3k), 1e-12)
    align_v3k = abs(float(mean_tv @ v3k))
    hn = rng.standard_normal((1000, args.k, args.k)); hn = 0.5 * (hn + np.transpose(hn, (0, 2, 1)))
    null_align = []
    for m in range(1000):
        wm, Vm = np.linalg.eigh(hn[m]); jm = int(np.argmax(np.abs(wm)))
        null_align.append(abs(float(Vm[:, jm] @ v3k)))
    null_align = np.array(null_align)
    b4 = {"k": args.k, "n_gens": len(fork),
          "per_gen_topeig_pairwise_abscos_mean": round(float(absc.mean()), 3),
          "ksphere_null_scale": round(ksphere_null, 3),
          "consistency_gate": "PASS" if absc.mean() > 2 * ksphere_null else "STOP-incoherent",
          "topeig_values_range": [round(min(topeigs), 4), round(max(topeigs), 4)],
          "align_wstar_V3k": round(align_v3k, 4),
          "selection_matched_null_p": round(float((null_align >= align_v3k).mean()), 4)}
    np.savez(args.out_dir / f"v4_b4_hessian_k{args.k}_{args.model}.npz",
             Hk=Hks, gids=np.asarray(fork), top_eigvecs=TV, top_eigvals=np.asarray(topeigs),
             w_star=w_star.astype(np.float64), Uk=Uk_np)

    h1.remove(); h2.remove()
    out = {"model": args.model, "site": site,
           "STATUS": "FIRST_READ_PENDING (C§8) — §B.7 stage-1 + §B.4 stage-1 (compute-only)",
           "B7_entropy_band": b7, "B4_logit_hessian": b4}
    (args.out_dir / f"v4_b7b4_stage1_{args.model}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
