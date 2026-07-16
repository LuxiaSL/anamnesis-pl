"""V1b on-policy gate (session-9 Part E-10; owed since 14l). Scored vs P=.85.

The C§3 on-policy gate: does steering the topic-disjoint formality vector V1b at low α keep the
model on-policy (top-1 argmax agreement ≥.85 with the banked unsteered continuation)? A steering
vector that passes the gate at α≤.1 is baseline-consistent (doesn't break the model). Uses the
teacher_forced_agreement primitive (_a5_common) over banked stage-0 continuations + V1b injection.
The 2AFC formality behavioral read needs a judge → deferred/owed. GPU (forwards only, no gen).
First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from anamnesis.config import MODEL_PRESETS
from anamnesis.extraction.model_loader import ResidualWriteSpec, attach_residual_write
from anamnesis.scripts._a5_common import teacher_forced_agreement


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--vec-npz", required=True)
    ap.add_argument("--vec-key", default="V1b_L14")
    ap.add_argument("--stamps", required=True)
    ap.add_argument("--manifest", type=Path, required=True, help="stage0 replay_manifest.json (banked continuations)")
    ap.add_argument("--site", type=int, default=14)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.03, 0.1, 0.3])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--gate", type=float, default=0.85)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM
    preset = MODEL_PRESETS[args.model]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=preset.torch_dtype, attn_implementation="eager").to("cuda").eval()
    dev = next(model.parameters()).device
    v = np.load(args.vec_npz)[args.vec_key].astype(np.float32)
    vt = torch.tensor(v, device=dev)
    site_norm = json.loads(Path(args.stamps).read_text())["median_resid_norms"][f"L{args.site}"]

    entries = json.loads(args.manifest.read_text())["entries"]
    gids = [g for g in list(entries)[: args.n]
            if (len(entries[g]["input_ids"]) - int(entries[g]["prompt_length"])) >= 4]

    rows = []
    for a in args.alphas:
        agrees = []
        for g in gids:
            e = entries[g]
            ids = e["input_ids"]; P = int(e["prompt_length"])
            spec = None
            if a != 0.0:
                spec = ResidualWriteSpec(layer_idx=args.site, vector=vt, alpha=a * site_norm,
                                         start_pos=P, end_pos=10_000, normalize=True)
            h = attach_residual_write(model, spec) if spec is not None else None
            with torch.no_grad():
                agrees.append(teacher_forced_agreement(model, ids, P))
            if h is not None:
                h.remove()
        rows.append({"alpha_frac": a, "mean_top1_agreement": round(float(np.mean(agrees)), 4),
                     "min": round(float(np.min(agrees)), 4), "n": len(agrees),
                     "pass_gate": bool(np.mean(agrees) >= args.gate)})
        print(f"  α={a}: mean_agreement={rows[-1]['mean_top1_agreement']} pass≥{args.gate}={rows[-1]['pass_gate']}")

    low = [r for r in rows if r["alpha_frac"] <= 0.1 and r["alpha_frac"] > 0]
    out = {"arm": "V1b on-policy gate (C§3 top-1 agreement)",
           "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop",
           "law": "top-1 argmax agreement of V1b-steered forward vs banked unsteered continuation; "
                  "gate ≥.85 at α≤.1 = baseline-consistent. 2AFC formality read (judge) = owed.",
           "site_norm_L14": round(site_norm, 4), "gate": args.gate,
           "verdict": {"prediction": "P=.85 V1b gate baseline-consistent at α≤.1",
                       "pass_at_low_alpha": all(r["pass_gate"] for r in low),
                       "low_alpha_rows": low, "2AFC_owed": True},
           "rows": rows}
    args.out_json.write_text(json.dumps(out, indent=1))
    print(f"VERDICT: V1b gate pass at α≤.1: {out['verdict']['pass_at_low_alpha']}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
