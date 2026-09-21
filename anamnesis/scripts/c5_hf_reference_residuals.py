"""Computer #5 PoC — HF reference residuals, for pinning the vLLM capture convention.

SPEC §3b.1. Runs the EXACT token ids the vLLM plumbing probe ran, through HF with
output_hidden_states=True, and dumps hidden_states[s] at the same sites. A separate compare step
then answers, numerically rather than by reasoning:

    does HF hidden_states[s]  ==  vLLM (arg1 + arg2) at layer s ?

HF's convention (the one anamnesis and koto both use, and the one `attach_residual_write`
implements): hidden_states[s] is the INPUT to decoder layer s — index 0 is the embedding output,
so hidden_states[s] is NOT layer s's output. vLLM splits that same stream across
(hidden_states, residual) for a fused add+RMSNorm. If the two disagree, every vector we build is
at a different depth than the one we inject into, and nothing downstream complains.

Run under the SHARED venv (has transformers/torch, no vllm needed):
    ~/luxi-files/.venv-shared/bin/python c5_hf_reference_residuals.py \
      --model-path /models/Llama-3.1-70B --ids-npz .../vllm_plumbing_probe.npz --sites 32 40 48
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--ids-npz", type=Path, required=True,
                    help="npz from the vLLM probe; supplies prompt_ids AND the vllm_resid_L* rows")
    ap.add_argument("--sites", type=int, nargs="+", default=[32, 40, 48])
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    if not args.ids_npz.is_file():
        sys.exit(f"FATAL: {args.ids_npz} not found — run the vLLM plumbing probe first")
    z = dict(np.load(args.ids_npz))
    if "prompt_ids" not in z:
        sys.exit("FATAL: npz has no prompt_ids")
    ids = z["prompt_ids"].tolist()

    from transformers import AutoModelForCausalLM

    logger.info("loading %s (HF, sdpa, device_map=auto)", args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto")
    model.config.use_cache = True
    model.eval()

    x = torch.tensor([ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        out = model(x, output_hidden_states=True, use_cache=False, return_dict=True)
    hs = out.hidden_states  # tuple length n_layers+1; index 0 = embedding output

    report: dict[str, object] = {
        "model_path": args.model_path, "n_tokens": len(ids),
        "n_hidden_states": len(hs), "sites": args.sites, "comparisons": {},
    }
    dump = {}
    for s in args.sites:
        hf = hs[s][0].float().cpu().numpy().astype(np.float32)   # INPUT to decoder layer s
        dump[f"hf_resid_L{s}"] = hf
        row: dict[str, object] = {
            "hf_mean_token_norm": float(np.linalg.norm(hf, axis=-1).mean()),
            "hf_median_token_norm": float(np.median(np.linalg.norm(hf, axis=-1))),
        }
        vl = z.get(f"vllm_resid_L{s}")
        if vl is not None:
            v = np.asarray(vl, dtype=np.float32)
            if v.shape == hf.shape:
                num = float((hf * v).sum())
                den = float(np.linalg.norm(hf) * np.linalg.norm(v)) or 1e-12
                # Per-token cosine is the honest statistic: a high global cosine can be carried
                # by a few high-norm tokens while most rows disagree.
                pt = (hf * v).sum(-1) / np.maximum(
                    np.linalg.norm(hf, axis=-1) * np.linalg.norm(v, axis=-1), 1e-12)
                rel = float(np.linalg.norm(hf - v) / max(np.linalg.norm(hf), 1e-12))
                row.update({
                    "vllm_mean_token_norm": float(np.linalg.norm(v, axis=-1).mean()),
                    "global_cosine": round(num / den, 6),
                    "per_token_cosine_mean": round(float(pt.mean()), 6),
                    "per_token_cosine_min": round(float(pt.min()), 6),
                    "relative_l2_error": round(rel, 6),
                    "VERDICT": ("MATCH — HF hidden_states[s] == vLLM arg1+arg2"
                                if pt.min() > 0.99 and rel < 0.05 else
                                "MISMATCH — do NOT build a vector until this is resolved"),
                })
            else:
                row["shape_mismatch"] = {"hf": list(hf.shape), "vllm": list(v.shape)}
        # Also compare against the vLLM parts separately — if hidden-only or residual-only
        # matches instead, that names the exact convention error rather than just flagging one.
        for part in ("vllm_hidden", "vllm_residpart"):
            arr = z.get(f"{part}_L{s}")
            if arr is not None and np.asarray(arr).shape == hf.shape:
                a = np.asarray(arr, dtype=np.float32)
                pt = (hf * a).sum(-1) / np.maximum(
                    np.linalg.norm(hf, axis=-1) * np.linalg.norm(a, axis=-1), 1e-12)
                row[f"per_token_cosine_vs_{part}"] = round(float(pt.mean()), 6)
        report["comparisons"][f"L{s}"] = row
        logger.info("L%d: %s", s, json.dumps(row))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_json.with_suffix(".npz"), **dump)
    args.out_json.write_text(json.dumps(report, indent=1))
    logger.info("wrote %s", args.out_json)


if __name__ == "__main__":
    main()
