"""A6 §2b — the BEHAVIORAL READ on the distilled-direction vectors (session-9 SPINE; C§8 ABS).

Steers base-Qwen (fp16, dtype law) along V_align (in-distro) and V_diverge (OOD) + matched-R
AR nulls, on the CANONICAL 8 favorite-animal prompts (DISJOINT from the construction set).
Reads the de-dicto/de-se ladder (v2 fixed regex + animal-pick tally, reused verbatim from
vmb_a6_dese_probe) per dose + census columns + placebo (3') + the coherence/behavioral-threshold
GATE that must be verified BEFORE any de-se readout is quoted (constraint v). α=0 baseline
recorded (constraint vi). Profile-space comparison to the student is FORBIDDEN as a criterion
(constraint iv) — this reads BEHAVIOR only.

Frozen predictions (outer loop, filed pre-cell):
  * de dicto rises dose-ordered under the OOD-divergence axis: P=.75
  * de se rises above the AR floor at any dose, OOD-divergence axis: P=.35 (the positive)
  * de se via the in-distro alignment axis: P=.20
Both outcomes pre-worded; REPORT, don't interpret. GPU. First-read -> outer loop.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from anamnesis.config import MODEL_PRESETS
from anamnesis.extraction.model_loader import ResidualWriteSpec, attach_residual_write
# reuse the EXACT v2 behavioral machinery (regexes + prompts) — single source of truth
from anamnesis.scripts.vmb_a6_dese_probe import DE_DICTO, DE_SE, ANIMAL_PICK, PROMPTS


def _coherence(text: str) -> float:
    """Degeneracy guard: distinct-word / total-word ratio over the response (low = degenerate)."""
    w = text.split()
    return len(set(w)) / max(len(w), 1)


def _read_texts(texts: list[str]) -> dict:
    dd = sum(bool(DE_DICTO.search(t)) for t in texts)
    ds = sum(bool(DE_SE.search(t)) for t in texts)
    picks = Counter()
    for t in texts:
        m = ANIMAL_PICK.search(t)
        if m:
            picks[m.group(1).lower()] += 1
    coh = float(np.mean([_coherence(t) for t in texts])) if texts else 0.0
    return {"n": len(texts), "de_dicto_rate": round(dd / max(len(texts), 1), 3),
            "de_se_floor_rate": round(ds / max(len(texts), 1), 3),
            "coherence": round(coh, 3), "top_animal_picks": picks.most_common(6)}


def _gen(model, tok, prompt, spec, n, max_new_tokens, dev):
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = (ids if isinstance(ids, torch.Tensor) else ids["input_ids"]).to(dev)
    if spec is not None:
        spec = ResidualWriteSpec(layer_idx=spec.layer_idx, vector=spec.vector, alpha=spec.alpha,
                                 start_pos=ids.shape[1], end_pos=10_000, normalize=True)
    h = attach_residual_write(model, spec) if spec is not None else None
    ids_b = ids.repeat(n, 1)
    with torch.no_grad():
        out = model.generate(ids_b, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
    if h is not None:
        h.remove()
    return [tok.decode(out[i, ids.shape[1]:], skip_special_tokens=True) for i in range(out.shape[0])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen-7b")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--vec-npz", required=True, help="§2b vectors (Valign_L18, Vdiverge_L18)")
    ap.add_argument("--ar-npz", required=True, help="a6 animal_vectors npz (AR1/2/3_L18 nulls)")
    ap.add_argument("--stamps", required=True, help="a6 stamps (median_resid_norms.L18)")
    ap.add_argument("--site", type=int, default=18)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.45, 0.6, 0.8])
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--coherence-floor", type=float, default=0.45,
                    help="below this mean distinct-word ratio, the dose is PAST coherence collapse "
                         "and its de-se readout is NOT quotable (constraint v gate)")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    # bf16 (native, lineage-matched) — fp16 Qwen generation throws a CUDA device-side assert;
    # the bridge (Acat/V_student/dese_probe) ran bf16. See vmb_a6_2b_build dtype note (outer loop).
    dtype = getattr(torch, MODEL_PRESETS[args.model].torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, attn_implementation="eager").to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    dev = next(model.parameters()).device

    vec = dict(np.load(args.vec_npz))
    arv = dict(np.load(args.ar_npz))
    site_norm = json.loads(Path(args.stamps).read_text())["median_resid_norms"][f"L{args.site}"]
    vectors = {
        "Valign": vec[f"Valign_L{args.site}"], "Vdiverge": vec[f"Vdiverge_L{args.site}"],
        "AR1": arv[f"AR1_L{args.site}"], "AR2": arv[f"AR2_L{args.site}"], "AR3": arv[f"AR3_L{args.site}"],
    }
    vt = {k: torch.tensor(v.astype(np.float32), device=dev) for k, v in vectors.items()}

    # ── α=0 baseline (shared; recorded per constraint vi) + placebo (3') ──
    base_by_prompt = {}
    all_base = []
    for kind, p in PROMPTS:
        txts = _gen(model, tok, p, None, args.n_samples, args.max_new_tokens, dev)
        base_by_prompt[(kind, p)] = txts
        all_base.extend(txts)
    baseline = _read_texts(all_base)
    # placebo: disjoint split of baseline within each prompt -> de-se-rate diff floor
    rng = np.random.default_rng(20260716)
    pl_dd, pl_ds = [], []
    for txts in base_by_prompt.values():
        if len(txts) < 2:
            continue
        idx = rng.permutation(len(txts)); half = len(txts) // 2
        a = [txts[i] for i in idx[:half]]; b = [txts[i] for i in idx[half:2*half]]
        ra, rb = _read_texts(a), _read_texts(b)
        pl_dd.append(ra["de_dicto_rate"] - rb["de_dicto_rate"])
        pl_ds.append(ra["de_se_floor_rate"] - rb["de_se_floor_rate"])
    placebo = {"de_dicto_abs_floor": round(float(np.mean(np.abs(pl_dd))), 3),
               "de_se_abs_floor": round(float(np.mean(np.abs(pl_ds))), 3),
               "n_placebo_prompts": len(pl_dd)}

    # ── dose ladder per vector ──
    results = {k: {} for k in vectors}
    for vk in vectors:
        for a in args.alphas:
            if a == 0.0:
                results[vk]["0.0"] = baseline
                continue
            alpha = a * site_norm
            spec = ResidualWriteSpec(layer_idx=args.site, vector=vt[vk], alpha=alpha,
                                     start_pos=0, end_pos=10_000, normalize=True)
            txts = []
            for kind, p in PROMPTS:
                txts.extend(_gen(model, tok, p, spec, args.n_samples, args.max_new_tokens, dev))
            r = _read_texts(txts)
            r["coherence_gate_pass"] = bool(r["coherence"] >= args.coherence_floor)
            # raw texts for HAND-VERIFICATION (session-8 regex-artifact lesson): the de-se-positive
            # texts (confirm real cat de-se, not a v2-regex residual FP) + a few de-dicto-only texts
            # (the de-dicto/de-se boundary). Capped to keep the JSON readable.
            r["de_se_positive_texts"] = [t for t in txts if DE_SE.search(t)][:12]
            r["de_dicto_only_texts"] = [t for t in txts if DE_DICTO.search(t) and not DE_SE.search(t)][:4]
            results[vk][str(a)] = r
            print(f"  {vk} α={a}: dicto={r['de_dicto_rate']} se={r['de_se_floor_rate']} "
                  f"coh={r['coherence']} gate={r['coherence_gate_pass']} picks={r['top_animal_picks'][:3]}")

    # ── scoring vs frozen P (report; do not interpret) ──
    def ar_floor(dose, key):
        vals = [results[f"AR{j}"][str(dose)][key] for j in (1, 2, 3) if str(dose) in results[f"AR{j}"]]
        return max(vals) if vals else None

    doses_nz = [a for a in args.alphas if a != 0.0]
    dd_div = [results["Vdiverge"][str(a)]["de_dicto_rate"] for a in doses_nz]
    dd_dose_ordered = all(dd_div[i] <= dd_div[i+1] + 1e-9 for i in range(len(dd_div)-1)) and dd_div[-1] > baseline["de_dicto_rate"]
    dese_div_above_ar = [(a, results["Vdiverge"][str(a)]["de_se_floor_rate"], ar_floor(a, "de_se_floor_rate"),
                          results["Vdiverge"][str(a)].get("coherence_gate_pass"))
                         for a in doses_nz]
    dese_align_above_ar = [(a, results["Valign"][str(a)]["de_se_floor_rate"], ar_floor(a, "de_se_floor_rate"),
                            results["Valign"][str(a)].get("coherence_gate_pass"))
                          for a in doses_nz]
    verdict = {
        "P75_dedicto_diverge_dose_ordered": {"value": bool(dd_dose_ordered), "dd_by_dose": dict(zip(map(str, doses_nz), dd_div)),
                                             "baseline": baseline["de_dicto_rate"]},
        "P35_dese_diverge_above_AR_floor": [{"dose": a, "de_se": v, "AR_floor": f, "gate_pass": g,
                                             "above_floor": (f is not None and v > f and v > placebo["de_se_abs_floor"])}
                                            for a, v, f, g in dese_div_above_ar],
        "P20_dese_align_above_AR_floor": [{"dose": a, "de_se": v, "AR_floor": f, "gate_pass": g,
                                           "above_floor": (f is not None and v > f and v > placebo["de_se_abs_floor"])}
                                          for a, v, f, g in dese_align_above_ar],
        "coherence_gate_note": "de-se claims quotable ONLY at doses with coherence_gate_pass=True",
    }
    out = {"arm": "A6 §2b — distilled-direction BEHAVIORAL READ (base-Qwen steered, de-dicto/de-se)",
           "STATUS": "FIRST_READ_PENDING (C§8 ABSOLUTE) — UNSTAMPED -> outer loop",
           "law": ("steer base-Qwen (fp16) + Valign/Vdiverge/AR{1,2,3} at L18 during gen on the "
                   "canonical 8 favorite-animal prompts; de_dicto = cat mention (rung≥1), "
                   "de_se_floor = first-person cat identity (rung≥3, UNDERCOUNTS); animal-pick tally; "
                   "AR = matched-R floor; placebo = disjoint baseline split; coherence gate before quote."),
           "census": {"n_prompts": len(PROMPTS), "kinds": Counter(k for k, _ in PROMPTS),
                      "n_samples_per_prompt": args.n_samples, "doses": args.alphas},
           "site_norm_L18": round(site_norm, 4), "coherence_floor": args.coherence_floor,
           "baseline_alpha0": baseline, "placebo": placebo,
           "verdict": verdict, "results": results}
    args.out_json.write_text(json.dumps(out, indent=1))
    print(f"\nBASELINE de_dicto={baseline['de_dicto_rate']} de_se={baseline['de_se_floor_rate']} "
          f"placebo_dese_floor={placebo['de_se_abs_floor']}")
    print(f"P75 de-dicto diverge dose-ordered: {dd_dose_ordered}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
