"""A8 smalls close-out — build OLMo-2-1124-7B's V7 (the missing target for the sixth node).

OLMo entered the battery for A1/A4 only, so no a5 §B.7 vector was ever banked for it, so
â(·->OLMo) had no target. This builds V7 by the SAME construction the reference banks use
(a5_vectors_8b_b7, a5_vectors_qwen-7b_b7), so the â comparison is construction-matched
rather than a mixed-construction cosine (the arm's whole rake-33/rake-40 discipline, one
level down):

    §B.7 V7 = unit( P_[16:256](Σ_site) . mean_gid( ∇_state S_entropy ) )

where the band [16:256] is the descending-eigenvalue slice of the site's residual
covariance Σ (the same [16,256] recipe stamped on every reference b7 bank), and
S_entropy is the mean token-entropy over generated positions, teacher-forced on the
banked stage-0 continuations (NOT fresh generation — the reference replays stage-0 gens
too, so the stage-0 512-cap is the only cap and it censors nothing the reference did not).

Σ rides free on the SAME forward passes (baton directive): the entropy-gradient replay
already materialises the L{site} residual at every gen position, so accumulating their
covariance costs no extra passes. Σ is captured over N_SIGMA gens (position budget matched
to the reference ~30k positions) and the gradient over N_GRAD fresh gens (matched to the
reference n_gens=20).

Base-model cautions honoured: OLMo-2-1124-7B has NO chat template — the stage-0 manifest
already holds raw-prompted input_ids, which we replay verbatim (no template applied). OLMo
applies q/k RMSNorm between projection and RoPE, but V7 reads the RESIDUAL ENTERING the
layer (the pre-hook hidden_states), not keys, so that caveat does not touch this build.

What is NOT run and why (rake 43 — say it, don't stretch): the §B.7 band-alignment
z-gate is measured against the model's banked V3 (mode-dir0). OLMo has no banked V3, so
that gate is N/A here; the construction-internal diagnostics that need no V3 (per-gen
sign-consistency and pairwise coherence of the band-projected gradient) ARE reported, and
they are what tell you the gradient is a coherent direction rather than noise.

Output (matched to vmb_b7_stage2_vectors so a8_smalls_star_systems reads it unchanged):
  a5_vectors_olmo2-7b_b7/a5_vectors.npz : V7_L{site} + 3 matched-support band randoms
  a5_vectors_olmo2-7b_b7/a5_vectors_stamps.json : honest provenance (rake 34)

UNSTAMPED (C section 8). Scores nothing. NEEDS GPU (one card, eager). Node-side run:
  OMP_NUM_THREADS=1 python -m anamnesis.scripts.a8_smalls_olmo_v7 \
    --model-path <olmo snapshot> --stage0-run <vmb_stage0_olmo2_7b> \
    --out-dir <battery>/a5_vectors_olmo2-7b_b7 --site 16 --n-sigma 60 --n-grad 20
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_smalls_olmo_v7")

BAND = (16, 256)               # the frozen §B.7 band recipe, descending Σ rank
NULL_SEED = 20260714           # matched-support randoms (vmb_b7_stage2 convention)
SITE = 16                      # OLMo site of record — L16, from the smalls alignment curve
N_SIGMA = 60                   # Σ position budget (reference 8b Σ used ~30k positions)
N_GRAD = 20                    # entropy-gradient gens (reference §B.7 n_gens)


def _sigma_gids(all_ids: list[int], n: int) -> list[int]:
    step = max(1, len(all_ids) // n)
    return all_ids[::step][:n]


def _grad_gids(all_ids: list[int], n: int) -> list[int]:
    """Half-step-offset selection, matched to the reference _fresh_gids."""
    step = max(1, len(all_ids) // n)
    return all_ids[step // 2::step][:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--site", type=int, default=SITE)
    ap.add_argument("--n-sigma", type=int, default=N_SIGMA)
    ap.add_argument("--n-grad", type=int, default=N_GRAD)
    ap.add_argument("--ridge-rel", type=float, default=1e-3)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    site = args.site

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from anamnesis.extraction.model_loader import decoder_layers

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()
    # We only ever need the gradient w.r.t. the L{site} LEAF (a separate requires_grad
    # tensor made in the hook), never w.r.t. the weights. Freezing the parameters means
    # backward allocates NO 7B-param grad buffers — the difference between ~18GB peak
    # (fits beside a co-tenant vLLM job) and OOM. Grad still flows THROUGH the frozen
    # weights to reach the leaf; the numerical V7 is unchanged.
    model.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    has_template = bool(getattr(tok, "chat_template", None))
    layers = decoder_layers(model)
    dev = "cuda"

    entries = json.loads((args.stage0_run / "replay_manifest.json").read_text())["entries"]
    all_ids = sorted(int(k) for k in entries)
    sigma_gids = _sigma_gids(all_ids, args.n_sigma)
    grad_gids = _grad_gids(all_ids, args.n_grad)

    ctx: dict = {}

    def hook(module, a, kw):
        # substitute a requires_grad LEAF for the residual entering layer{site}, so the
        # entropy at the head has a grad path back to it (reference §B.7 hook). Under
        # no_grad (the Σ loop) this is a value-identical passthrough; under enable_grad
        # (the gradient loop) it is what makes autograd.grad(S, leaf) well-defined.
        hs = a[0] if a else kw.get("hidden_states")
        leaf = hs.detach().clone().requires_grad_(True)
        ctx["leaf"] = leaf
        if a:
            return (leaf,) + tuple(a[1:]), kw
        kw = dict(kw)
        kw["hidden_states"] = leaf
        return a, kw

    h = layers[site].register_forward_pre_hook(hook, with_kwargs=True)

    # ── Σ capture: residual rows at L{site} over generated positions, no grad ──
    rows: list[np.ndarray] = []
    tok_norms: list[float] = []
    for g in sigma_gids:
        e = entries[str(g)]
        ids = torch.tensor([e["input_ids"]], dtype=torch.long, device=dev)
        p, ln = int(e["prompt_length"]), int(len(e["input_ids"]))
        with torch.no_grad():
            model(ids, use_cache=False, return_dict=True)
        r = ctx["leaf"][0, p:ln, :].detach().float().cpu().numpy().astype(np.float64)
        rows.append(r)
        tok_norms.append(float(np.median(np.linalg.norm(r, axis=1))))
        ctx.pop("leaf", None)
    R = np.concatenate(rows, axis=0)                       # (n_pos, d)
    mu = R.mean(0)
    Rc = R - mu
    Sigma = (Rc.T @ Rc) / (Rc.shape[0] - 1)
    evals, evecs = np.linalg.eigh(Sigma)                   # ascending
    ridge = args.ridge_rel * float(evals.mean())
    order = np.argsort(evals)[::-1]                        # descending
    band_idx = order[BAND[0]:BAND[1]]
    Ub = evecs[:, band_idx]                                # (d, 240)
    logger.info("Σ over %d positions (%d gens); ridge %.3g; band [%d:%d] captured",
                R.shape[0], len(sigma_gids), ridge, *BAND)

    # ── entropy gradient over N_GRAD gens, teacher-forced on banked gens ──
    Ge: list[np.ndarray] = []
    for g in grad_gids:
        e = entries[str(g)]
        ids = torch.tensor([e["input_ids"]], dtype=torch.long, device=dev)
        p, ln = int(e["prompt_length"]), int(len(e["input_ids"]))
        with torch.enable_grad():
            out = model(ids, use_cache=False, return_dict=True)
            logits = out.logits[0].float()
            lp = torch.log_softmax(logits, dim=-1)
            ent = -(lp.exp() * lp).sum(dim=-1)
            S = ent[p:ln].mean()
            # leaf-only gradient: no per-parameter .grad buffers allocated
            grad_leaf = torch.autograd.grad(S, ctx["leaf"])[0]
            Ge.append(grad_leaf[0, p:ln, :].float().mean(0).cpu().numpy().astype(np.float64))
        ctx.pop("leaf", None)
    h.remove()
    Ge = np.stack(Ge)
    mean_e = Ge.mean(0)

    # band-projected V7 (identical algebra to vmb_v4_b7b4_stage1 §B.7)
    cg = Ub.T @ mean_e
    v7 = Ub @ cg
    v7 = (v7 / np.linalg.norm(v7)).astype(np.float32)

    # construction-internal diagnostics (need no V3)
    bandproj = (Ub.T @ Ge.T).T                             # (n_grad, 240)
    bu = bandproj / np.clip(np.linalg.norm(bandproj, axis=1, keepdims=True), 1e-12, None)
    iu = np.triu_indices(len(Ge), k=1)
    coherence = float((bu @ bu.T)[iu].mean())
    ref = cg / np.linalg.norm(cg)
    per_gen_sign = int(((bandproj @ ref) > 0).sum())

    # 3 matched-support band randoms (vmb_b7_stage2 recipe, same seed)
    vectors = {f"V7_L{site}": v7}
    rng = np.random.default_rng(NULL_SEED)
    for i in range(1, 4):
        c = rng.standard_normal(Ub.shape[1])
        r = Ub @ c
        vectors[f"Rband{i}_L{site}"] = (r / np.linalg.norm(r)).astype(np.float32)

    median_resid_norm = float(np.median(tok_norms))
    np.savez(args.out_dir / "a5_vectors.npz", **vectors)
    # bank Σ beside it (it rode free; future OLMo whitening/Mahalanobis work needs it)
    np.savez(args.out_dir / f"a5_sigma_L{site}_olmo2-7b.npz",
             evals=evals.astype(np.float64), evecs=evecs.astype(np.float64),
             mean=mu.astype(np.float64), ridge=np.float64(ridge),
             n_positions=np.int64(R.shape[0]))

    stamps = {
        "STATUS": "UNSTAMPED (C section 8) — built by a8_smalls_olmo_v7.py for the A8 "
                  "smalls close-out sixth-node test; scores nothing",
        "model": "olmo2-7b",
        "site": site,
        "band": list(BAND),
        "median_resid_norms": {f"L{site}": median_resid_norm},
        "V7_provenance": (
            f"section-B.7 entropy-band V7 = unit(P_[{BAND[0]}:{BAND[1]}](Sigma_L{site}) . "
            f"mean grad S_entropy), teacher-forced replay of {len(grad_gids)} stage-0 "
            "gens; band = descending-eigenvalue slice of the residual Sigma captured on "
            f"the same forward passes over {len(sigma_gids)} gens ({R.shape[0]} positions). "
            "SAME construction as a5_vectors_8b_b7 / a5_vectors_qwen-7b_b7."),
        "randoms_provenance": (
            f"3 matched-support randoms confined to the [{BAND[0]}:{BAND[1]}] band "
            f"eigenspace of Sigma_L{site}, seed {NULL_SEED} (vmb_b7_stage2 recipe)"),
        "diagnostics": {
            "per_gen_band_sign_consistency": f"{per_gen_sign}/{len(Ge)}",
            "per_gen_band_pairwise_coherence": round(coherence, 4),
            "note": "V3-alignment z-gate N/A — OLMo has no banked V3 (rake 43); these "
                    "V3-free diagnostics stand in for direction coherence.",
        },
        "base_model_cautions": {
            "chat_template_present": has_template,
            "arm": "raw ONLY — OLMo-2-1124-7B is a base model; its future â lives in "
                   "the raw::proc_k128 star system (A8-add-7.1), never native.",
            "cap": "stage-0 continuations are 512-capped; teacher-forced replay inherits "
                   "that cap and censors nothing beyond what the reference V7 builds did.",
            "qk_rmsnorm": "OLMo applies q/k RMSNorm; V7 reads the residual ENTERING the "
                          "layer (pre-hook), not keys, so the caveat does not apply.",
        },
        "gids": {"sigma": [int(x) for x in sigma_gids],
                 "grad": [int(x) for x in grad_gids]},
        "sites": [site],
    }
    (args.out_dir / "a5_vectors_stamps.json").write_text(json.dumps(stamps, indent=1))

    logger.info("V7_L%d built: |band-proj sign| %d/%d, coherence %.4f, median resid norm %.3f",
                site, per_gen_sign, len(Ge), coherence, median_resid_norm)
    logger.info("banked -> %s (+ Σ + stamps)", args.out_dir / "a5_vectors.npz")
    print(json.dumps({"built": list(vectors),
                      "per_gen_sign": f"{per_gen_sign}/{len(Ge)}",
                      "coherence": round(coherence, 4),
                      "median_resid_norm": round(median_resid_norm, 4),
                      "n_sigma_positions": int(R.shape[0])}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
