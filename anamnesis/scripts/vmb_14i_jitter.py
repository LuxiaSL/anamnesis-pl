"""14i ITEM 2 — jitter-interaction test (ADDENDUM 14i; gates item-3 conditional adoption; P=0.80).

Re-run ~5 gate cells k times on identical inputs; measure run-to-run stability of the two gate-side
statistics: `agreement` (top-1 argmax vs forced token — argmax-flip-sensitive) vs `mean_chosen_rank`
(rank of the forced token in the logits — continuous). Filed prediction: the RANK statistic is at
least as run-stable as the agreement indicator (P=0.80). Any run-to-run variation on identical inputs
comes from GPU nondeterminism; the hypothesis is that near-ties flip the argmax (agreement jitters)
while the continuous rank barely moves.

item-3 adoption (report mean_chosen_rank alongside agreement) fires ONLY on pass, prospectively.
Reuses the on-policy-gate model/bank/pilot loading. GPU. First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from anamnesis.config import MODEL_PRESETS


def _agreement_and_rank(model, ids_list, P: int, spec, dev):
    """Returns (top-1 agreement, mean chosen_rank) over the forced continuation positions."""
    from anamnesis.extraction.model_loader import attach_residual_write
    ids = torch.tensor([ids_list], dtype=torch.long, device=dev)
    L = ids.shape[1]
    h = attach_residual_write(model, spec) if spec is not None else None
    with torch.no_grad():
        logits = model(ids, use_cache=False).logits[0]        # [L, vocab]
    if h is not None:
        h.remove()
    pos = torch.arange(P - 1, L - 1, device=dev)              # distributions producing tokens P..L-1
    tgt = ids[0, P:L]                                         # forced tokens
    lg = logits[pos]                                          # [n, vocab]
    pred = lg.argmax(dim=-1)
    agreement = float((pred == tgt).float().mean().item())
    chosen = lg.gather(1, tgt[:, None]).squeeze(1)            # logit of the forced token
    rank = (lg > chosen[:, None]).sum(dim=1).float()          # # tokens ranked above it
    return agreement, float(rank.mean().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--vectors-dir", type=Path, required=True)
    ap.add_argument("--map-site", type=int, default=14)
    ap.add_argument("--cells", nargs="+",
                    default=["V3_L14:0.03", "V3_L14:0.1", "V1_L14:0.03", "V4_L14:0.03", "R1:0.03"],
                    help="KEY:frac (KEY = bank vector key; R* are site-independent)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-pilot", type=int, default=20)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM
    from anamnesis.extraction.model_loader import ResidualWriteSpec

    preset = MODEL_PRESETS[args.model]
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}.get(str(preset.torch_dtype), torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, attn_implementation="eager").to("cuda").eval()
    dev = next(model.parameters()).device

    bank = np.load(args.vectors_dir / "a5_vectors.npz")
    norms = json.loads((args.vectors_dir / "a5_vectors_stamps.json").read_text())["median_resid_norms"]
    entries = json.loads((args.stage0_run / "replay_manifest.json").read_text())["entries"]
    pilots = []
    for g in (k * 40 for k in range(args.n_pilot)):
        e = entries.get(str(g))
        if e and (len(e["input_ids"]) - e["prompt_length"]) >= 32:
            pilots.append(e)

    def site_of(key):
        return int(key.rsplit("_L", 1)[1]) if "_L" in key else args.map_site

    rows = []
    for spec_str in args.cells:
        key, frac = spec_str.split(":")
        frac = float(frac)
        site = site_of(key)
        vec = torch.from_numpy(bank[key].astype(np.float32)).to(dev)
        alpha = frac * float(norms[f"L{site}"])
        per_run_ag, per_run_rank = [], []
        for _ in range(args.k):
            ag, rk = [], []
            for e in pilots:
                P = int(e["prompt_length"])
                spec = ResidualWriteSpec(layer_idx=site, vector=vec, alpha=alpha,
                                         start_pos=P, end_pos=len(e["input_ids"]), normalize=True)
                a, r = _agreement_and_rank(model, e["input_ids"], P, spec, dev)
                ag.append(a); rk.append(r)
            per_run_ag.append(float(np.mean(ag)))
            per_run_rank.append(float(np.mean(rk)))
        ag_arr, rk_arr = np.array(per_run_ag), np.array(per_run_rank)
        # stability: std across k runs, and coefficient of variation (scale-free comparison)
        ag_std, rk_std = float(ag_arr.std(ddof=0)), float(rk_arr.std(ddof=0))
        ag_cv = ag_std / max(abs(ag_arr.mean()), 1e-9)
        rk_cv = rk_std / max(abs(rk_arr.mean()), 1e-9)
        rows.append({"cell": f"{key}_a{frac}", "key": key, "site": site, "alpha_frac": frac,
                     "k": args.k, "n_pilot": len(pilots),
                     "agreement_per_run": [round(x, 6) for x in per_run_ag],
                     "chosen_rank_per_run": [round(x, 4) for x in per_run_rank],
                     "agreement_mean": round(float(ag_arr.mean()), 6), "agreement_std": ag_std,
                     "chosen_rank_mean": round(float(rk_arr.mean()), 4), "chosen_rank_std": rk_std,
                     "agreement_cv": ag_cv, "chosen_rank_cv": rk_cv,
                     "rank_at_least_as_stable": bool(rk_cv <= ag_cv + 1e-12)})
        print(f"  {key}_a{frac}: ag_std={ag_std:.2e} (cv {ag_cv:.2e}) | rank_std={rk_std:.4f} "
              f"(cv {rk_cv:.2e}) | rank_stabler={rows[-1]['rank_at_least_as_stable']}")

    n_pass = sum(r["rank_at_least_as_stable"] for r in rows)
    verdict = "INSIDE" if n_pass == len(rows) else ("PARTIAL" if n_pass >= len(rows) / 2 else "MISS")
    out = {"arm": "14i item 2 — jitter-interaction test", "model": args.model,
           "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop",
           "P_filed": 0.80,
           "law": "k repeats on identical inputs; agreement = top-1 argmax vs forced token; "
                  "chosen_rank = mean rank of forced token; stability = std/CV across runs; "
                  "PASS if rank CV ≤ agreement CV (rank at least as run-stable)",
           "verdict": {"result": verdict, "cells_rank_stabler": f"{n_pass}/{len(rows)}",
                       "note": "if both std≈0 (deterministic forward) rank passes trivially — "
                               "the argmax-flip concern simply does not materialize at this scale"},
           "item3_adoption": "fires ONLY on INSIDE, prospectively (report mean_chosen_rank alongside "
                             "agreement; gate bar UNCHANGED)",
           "rows": rows}
    args.out_json.write_text(json.dumps(out, indent=1))
    print(f"verdict: {verdict} ({n_pass}/{len(rows)} cells rank-stabler)")


if __name__ == "__main__":
    main()
