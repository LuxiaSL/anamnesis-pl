"""A6 bridge — the DISTILLED direction (teacher-student diff feature; Luxia loose-end).

The bridge steered along Acat = mean(cat-sysprompt) − mean(neutral) — the PROMPT-induced cat
direction. The 2606.00995 theory is that what's DISTILLED is a direction; the injectable form
of that is the student's residual-stream shift:

    V_student = mean_resid_L18(cat_student_453) − mean_resid_L18(base)

extracted on the SAME probe160 tokens (content-controlled: same tokens, different weights). This
answers whether Acat points the same way as what distillation actually installed —
cos(Acat, V_student). Low ⇒ the prompt-contrast is a weak/wrong proxy for the distilled
direction; high ⇒ same direction (and the de-se gap is not a direction problem). V_student is
banked so the next probe can STEER base-Qwen along IT and read behavior (does the distilled
direction function as a cat vector, de-se included?). GPU. First-read → outer loop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from anamnesis.config import MODEL_PRESETS


def mean_resid(model, manifest, site, dev, max_gens):
    tot, n = None, 0
    entries = manifest["entries"]
    for gid in list(entries)[:max_gens]:
        e = entries[gid]
        ids = torch.tensor([e["input_ids"]], dtype=torch.long, device=dev)
        P = int(e["prompt_length"])
        if ids.shape[1] - P < 2:
            continue
        with torch.no_grad():
            out = model(ids, output_hidden_states=True, use_cache=False, return_dict=True)
        h = out.hidden_states[site][0, P:].float().mean(0).cpu().numpy()
        tot = h if tot is None else tot + h
        n += 1
    return tot / max(n, 1), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen-7b")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--adapter-path", required=True, help="fully-distilled cat student ckpt")
    ap.add_argument("--manifest", type=Path, required=True, help="probe160 manifest (same tokens)")
    ap.add_argument("--site", type=int, default=18)
    ap.add_argument("--max-gens", type=int, default=160)
    ap.add_argument("--acat-npz", required=True)
    ap.add_argument("--out-npz", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM
    preset = MODEL_PRESETS[args.model]
    manifest = json.loads(args.manifest.read_text())

    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=preset.torch_dtype, attn_implementation="eager").to("cuda").eval()
    dev = next(base.parameters()).device
    base_mean, nb = mean_resid(base, manifest, args.site, dev, args.max_gens)
    del base
    torch.cuda.empty_cache()

    from peft import PeftModel
    stu = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=preset.torch_dtype, attn_implementation="eager").to("cuda").eval()
    stu = PeftModel.from_pretrained(stu, args.adapter_path).merge_and_unload().eval()
    stu_mean, ns = mean_resid(stu, manifest, args.site, dev, args.max_gens)

    v_student = (stu_mean - base_mean).astype(np.float32)
    vnorm = float(np.linalg.norm(v_student))
    v_unit = v_student / vnorm

    acat = np.load(args.acat_npz)
    ak = f"Acat_L{args.site}"
    acat_v = acat[ak].astype(np.float64)
    acat_v /= np.linalg.norm(acat_v)
    cos_acat = float(v_unit.astype(np.float64) @ acat_v)
    cos_ar = {j: float(v_unit.astype(np.float64) @ (acat[f"AR{j}_L{args.site}"].astype(np.float64)
              / np.linalg.norm(acat[f"AR{j}_L{args.site}"]))) for j in (1, 2, 3) if f"AR{j}_L{args.site}" in acat}

    np.savez(args.out_npz, **{f"Vstudent_L{args.site}": v_unit,
                              f"Vstudent_raw_L{args.site}": v_student})
    out = {"arm": "A6 bridge — distilled direction (V_student residual shift)",
           "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop",
           "law": f"V_student = mean_resid_L{args.site}(cat_student_453) − mean_resid(base) on the same "
                  "probe160 tokens (weights differ, tokens identical); unit. cos(Acat, V_student) tests "
                  "whether the prompt-contrast steering vector is the distilled direction.",
           "n_base": nb, "n_student": ns, "V_student_raw_norm": round(vnorm, 4),
           "base_resid_norm": round(float(np.linalg.norm(base_mean)), 3),
           "student_resid_norm": round(float(np.linalg.norm(stu_mean)), 3),
           "cos_Acat_Vstudent": round(cos_acat, 4),
           "cos_Vstudent_AR_nulls": {k: round(v, 4) for k, v in cos_ar.items()},
           "reading": "cos(Acat,V_student) HIGH ⇒ prompt-contrast IS the distilled direction (de-se gap "
                      "is not directional); LOW ⇒ Acat is a weak/oblique proxy — steer along V_student next."}
    args.out_json.write_text(json.dumps(out, indent=1))
    print(f"cos(Acat, V_student) = {cos_acat:.4f} | V_student raw norm {vnorm:.3f} | "
          f"cos vs AR nulls {out['cos_Vstudent_AR_nulls']}")
    print(f"wrote {args.out_npz}, {args.out_json}")


if __name__ == "__main__":
    main()
