"""§2b ALIGNMENT side-probe (Luxia-directed side interest, 2026-07-17): is the steered base
genuinely moving TOWARD the student, per prompt category — and if not, where is the break?

For animal ∈ {cat, phoenix, penguin}: three deltas in residual space at sites L{7,14,18,21},
per prompt CATEGORY (dese-8 canonical probe prompts = fully disjoint from construction ·
animal-construct prompts = the OOD construction category · numbers = the in-distro
distillation category, kept as reference):

  Δ_distill      = mean_state(student, bare)            − mean_state(base, bare)
  Δ_steer_state  = mean_state(base + αV, INJECTED read) − mean_state(base, bare)
  Δ_steer_expr   = mean_state(steered TEXTS, no-inject) − mean_state(base, bare)
                   (two-column discipline transplanted here: under injection the state
                    column contains the injected component trivially — the expression
                    column asks whether the INDUCED GENERATION carries student-ward states
                    beyond the injection itself; the 14r gauge-contamination lesson.)

Readouts per site × category (each judged against the AR-null rows through the identical
pipeline): cos(Δ_steer_state, Δ_distill) · cos(Δ_steer_expr, Δ_distill) ·
cos(V, Δ_distill) at the inject site (does the AXIS even point at the student's shift on
this category?) · norm ratios. The triple decomposes the break: axis-wrong (cos(V,Δ_distill)
low) vs propagation-wrong (axis fine, state cos low) vs expression-wrong (state cos fine,
expr cos at null).

States = generated-position mean of the residual INPUT to decoder layer s (hidden_states[s],
the 2b convention — matches attach_residual_write's write point). Free-gen temp .7/top-p .9,
bf16 (2b lineage dtype law); per-(pass,prompt) torch seed for reproducibility. Side-interest
lane: report-only, UNSTAMPED → outer loop; not a baton cell, no filed P.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from anamnesis.config import MODEL_PRESETS
from anamnesis.extraction.model_loader import ResidualWriteSpec, attach_residual_write
from anamnesis.scripts.vmb_a5_build_vectors import _chat_ids
from anamnesis.scripts.vmb_a6_2b_build import ANIMAL_CONSTRUCT_PROMPTS, NUMBERS_PROMPTS
from anamnesis.scripts.vmb_a6_dese_probe import PROMPTS as DESE_PROMPTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SITES = [7, 14, 18, 21]


def _state_of(model, full_ids: torch.Tensor, P: int) -> dict[int, np.ndarray]:
    with torch.no_grad():
        hs = model(full_ids, output_hidden_states=True, use_cache=False,
                   return_dict=True).hidden_states
    return {s: hs[s][0, P:].float().mean(0).cpu().numpy() for s in SITES}


def collect_pass(model, tok, categories: dict[str, list[str]], spec: ResidualWriteSpec | None,
                 n: int, max_new: int, dev, seed_tag: str, expr_read: bool):
    """One pass over all categories. Returns {cat: {'state':[...], 'expr':[...], 'texts':[...]}}.
    spec attached during BOTH generation and the state read (state column); expr_read adds a
    second, hook-free forward of the same tokens (expression column)."""
    out: dict[str, dict[str, list]] = {c: {"state": [], "expr": [], "texts": []} for c in categories}
    h = attach_residual_write(model, spec) if spec is not None else None
    try:
        for cname, prompts in categories.items():
            for pi, p in enumerate(prompts):
                torch.manual_seed(abs(hash((seed_tag, cname, pi))) % (2**31))
                ids = _chat_ids(tok, p, None).to(dev)
                P = ids.shape[1]
                ids_b = ids.repeat(n, 1)
                with torch.no_grad():
                    gen = model.generate(ids_b, max_new_tokens=max_new, do_sample=True,
                                         temperature=0.7, top_p=0.9,
                                         pad_token_id=tok.eos_token_id)
                for i in range(gen.shape[0]):
                    g = gen[i, P:]
                    keep = (g != tok.eos_token_id)
                    if keep.sum().item() < 2:
                        continue
                    last = int(torch.nonzero(keep, as_tuple=False).max().item()) + 1
                    full = torch.cat([gen[i, :P], g[:last]]).unsqueeze(0)
                    out[cname]["state"].append(_state_of(model, full, P))
                    out[cname]["texts"].append(tok.decode(g[:last], skip_special_tokens=True))
                    if expr_read and h is not None:
                        h.remove()
                        try:
                            out[cname]["expr"].append(_state_of(model, full, P))
                        finally:
                            h = attach_residual_write(model, spec)   # re-arm; outer finally removes the live one
            logger.info(f"[{seed_tag}] {cname}: {len(out[cname]['state'])} gens")
    finally:
        if h is not None:
            h.remove()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen-7b")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--adapter-path", required=True, help="the animal's full-distill student ckpt")
    ap.add_argument("--vec-npz", required=True, help="the animal's 2b vectors (Vdiverge_L18)")
    ap.add_argument("--ar-npz", required=True, help="AR nulls npz (AR1/2/3_L18)")
    ap.add_argument("--stamps", required=True, help="median_resid_norms json (alpha resolution)")
    ap.add_argument("--animal", required=True)
    ap.add_argument("--site", type=int, default=18)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    preset = MODEL_PRESETS[args.model]
    dtype = getattr(torch, preset.torch_dtype)
    tok = AutoTokenizer.from_pretrained(args.model_path)

    categories = {
        "dese8_disjoint": [p for _, p in DESE_PROMPTS],
        "animal_construct": list(ANIMAL_CONSTRUCT_PROMPTS[:8]),
        "numbers_indistro": list(NUMBERS_PROMPTS[:6]),
    }

    site_norm = json.loads(Path(args.stamps).read_text())["median_resid_norms"][f"L{args.site}"]
    alpha_abs = args.alpha * float(site_norm)
    Vz = np.load(args.vec_npz)
    V = torch.tensor(Vz[f"Vdiverge_L{args.site}"].astype(np.float32))
    ARz = np.load(args.ar_npz)
    ar_vecs = {k: torch.tensor(ARz[f"{k}_L{args.site}"].astype(np.float32)) for k in ("AR1", "AR2")}

    # ── base (bare) ──
    logger.info("loading BASE")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, attn_implementation="eager").to("cuda").eval()
    dev = next(base.parameters()).device
    passes: dict[str, dict] = {}
    passes["base"] = collect_pass(base, tok, categories, None, args.n_samples,
                                  args.max_new_tokens, dev, "base", expr_read=False)

    def spec_for(vec: torch.Tensor) -> ResidualWriteSpec:
        return ResidualWriteSpec(layer_idx=args.site, vector=vec.to(dev), alpha=alpha_abs,
                                 start_pos=0, end_pos=10_000, normalize=True)

    passes["steer_V"] = collect_pass(base, tok, categories, spec_for(V), args.n_samples,
                                     args.max_new_tokens, dev, "steer_V", expr_read=True)
    for k, v in ar_vecs.items():
        passes[f"steer_{k}"] = collect_pass(base, tok, categories, spec_for(v), args.n_samples,
                                            args.max_new_tokens, dev, f"steer_{k}", expr_read=True)
    del base
    torch.cuda.empty_cache()

    # ── student (adapter-merged, bare) ──
    logger.info("loading STUDENT")
    from peft import PeftModel
    stu = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, attn_implementation="eager").to("cuda").eval()
    stu = PeftModel.from_pretrained(stu, args.adapter_path).merge_and_unload().eval()
    passes["student"] = collect_pass(stu, tok, categories, None, args.n_samples,
                                     args.max_new_tokens, dev, "student", expr_read=False)
    del stu
    torch.cuda.empty_cache()

    def mean_state(pass_name: str, cname: str, col: str) -> dict[int, np.ndarray]:
        rows = passes[pass_name][cname][col if col in ("state", "expr") else "state"]
        if col == "expr" and not passes[pass_name][cname]["expr"]:
            rows = passes[pass_name][cname]["state"]
        return {s: np.mean([r[s] for r in rows], axis=0) for s in SITES}

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0

    Vnp = Vz[f"Vdiverge_L{args.site}"].astype(np.float64)
    results: dict[str, dict] = {}
    for cname in categories:
        base_m = mean_state("base", cname, "state")
        dist = {s: mean_state("student", cname, "state")[s] - base_m[s] for s in SITES}
        row: dict = {
            "axis_pointing_cos_V_vs_Ddistill_L18": round(cos(Vnp, dist[args.site]), 4),
            "Ddistill_norm_per_site": {s: round(float(np.linalg.norm(dist[s])), 3) for s in SITES},
        }
        for pname in ("steer_V", "steer_AR1", "steer_AR2"):
            for col in ("state", "expr"):
                dsteer = {s: mean_state(pname, cname, col)[s] - base_m[s] for s in SITES}
                row[f"{pname}_{col}"] = {
                    "cos_vs_Ddistill_per_site": {s: round(cos(dsteer[s], dist[s]), 4) for s in SITES},
                    "norm_ratio_steer_over_distill_L18": round(
                        float(np.linalg.norm(dsteer[args.site]) / max(np.linalg.norm(dist[args.site]), 1e-9)), 3),
                }
        results[cname] = row

    out = {
        "arm": f"§2b alignment side-probe — {args.animal} (Luxia side interest 2026-07-17)",
        "STATUS": "FIRST_READ_PENDING (C§8) — side-interest lane, no filed P, report-only",
        "animal": args.animal, "alpha_frac": args.alpha, "site": args.site,
        "n_samples_per_prompt": args.n_samples, "max_new_tokens": args.max_new_tokens,
        "law": ("Δ_distill vs Δ_steer{state,expr} cosines per site×category, AR rows = null "
                "through the identical pipeline; state = gen-position mean resid INPUT to layer "
                "s; expression column = hook-free re-read of the SAME steered tokens (14r "
                "two-column transplant); break decomposition: axis-pointing vs propagation vs "
                "expression"),
        "results": results,
        "sample_texts": {c: {p: passes[p][c]["texts"][:2] for p in ("steer_V", "student")}
                         for c in categories},
    }
    args.out_json.write_text(json.dumps(out, indent=1))
    for c, r in results.items():
        sv = r["steer_V_state"]["cos_vs_Ddistill_per_site"][args.site]
        se = r["steer_V_expr"]["cos_vs_Ddistill_per_site"][args.site]
        a1 = r["steer_AR1_state"]["cos_vs_Ddistill_per_site"][args.site]
        print(f"{c:18} axis→distill {r['axis_pointing_cos_V_vs_Ddistill_L18']:+.3f} | "
              f"steer(state) {sv:+.3f} steer(expr) {se:+.3f} | AR1(state) {a1:+.3f}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
