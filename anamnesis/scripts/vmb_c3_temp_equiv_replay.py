"""14m item 1 — temperature-equivalence rung (per-position best-T* rescaling).

Is a steered next-token distribution just a TEMPERATURE-rescaled version of the base
distribution? For each generated position, forward steered (inject on) and base (inject off),
then fit the single T* that best matches the steered distribution to softmax(base_logits / T),
and report how much of the steered-vs-base divergence that pure rescaling explains:

    D_total = KL(steered || base)                    (base = T=1)
    D_resid = min_T KL(steered || softmax(base / T)) (best rescaling)
    R       = 1 - mean(D_resid) / mean(D_total)       (fraction explained by rescaling)

R≈1 ⇒ the steering IS a temperature knob on this coordinate; R low ⇒ it writes something
richer than temperature. Pre-named readings (14m): R≥.9 equivalence / R≤.6 richer-than-temp.
Filed P: V7 not-pure-rescaling (R≤.6) .55; V_temp .60.

⚠ top-k BOUND: KL is computed on the union of each position's top-k=200 tokens (renormalized) —
a bound on full-vocab KL, reported as such. Ratios are bounded, not exact. NO new generation
(forwards over banked gens). GPU. First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from anamnesis.config import MODEL_PRESETS
from anamnesis.extraction.model_loader import ResidualWriteSpec, attach_residual_write

T_GRID = torch.tensor(np.concatenate([np.linspace(0.30, 0.95, 14), np.linspace(1.0, 4.0, 31)]),
                      dtype=torch.float32)


def _logits_over_gen(model, ids, P, spec, dev):
    h = attach_residual_write(model, spec) if spec is not None else None
    with torch.no_grad():
        logits = model(ids, use_cache=False).logits[0].float()   # [T, vocab]
    if h is not None:
        h.remove()
    T = ids.shape[1]
    pos = torch.arange(P - 1, T - 1, device=dev)                  # dists producing gen tokens
    return logits[pos]                                            # [n, vocab]


def _R_for_gen(steer_lg, base_lg, topk, dev):
    """Per-position D_total, D_resid on top-k union support. Returns (D_total[], D_resid[])."""
    tg = T_GRID.to(dev)
    Dtot, Dres = [], []
    for i in range(steer_lg.shape[0]):
        sl, bl = steer_lg[i], base_lg[i]
        # top-k union support (renormalize both over it — the top-k BOUND)
        idx = torch.unique(torch.cat([sl.topk(topk).indices, bl.topk(topk).indices]))
        s = torch.log_softmax(sl[idx], -1)                       # log p_steered on support
        b = bl[idx].float()
        ps = s.exp()
        # base at T=1 and over the T grid: q_T = softmax(b / T) on the support
        logq_grid = torch.log_softmax(b[None, :] / tg[:, None], dim=-1)   # [G, k]
        kl_grid = (ps[None, :] * (s[None, :] - logq_grid)).sum(-1)        # [G]  KL(steered||base/T)
        Dtot.append(float(kl_grid[(tg == 1.0).nonzero()[0, 0]].item()))
        Dres.append(float(kl_grid.min().item()))
    return Dtot, Dres


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--run-dir", type=Path, required=True, help="vmb_b7_3b (V7) or vmb_c3_3b (V_temp)")
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--max-gens", type=int, default=40, help="cap gens/cell for wall-clock")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM
    preset = MODEL_PRESETS[args.model]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=preset.torch_dtype, attn_implementation="eager").to("cuda").eval()
    dev = next(model.parameters()).device

    rows = []
    for name in args.cells:
        d = args.run_dir / name
        meta = json.loads((d / "metadata.json").read_text())
        gens = meta["generations"] if "generations" in meta else meta
        entries = json.loads((d / "replay_manifest.json").read_text())["entries"]
        inj = gens[0]["injection"]
        v = torch.tensor(np.load(inj["inject_npz"])[inj["inject_key"]].astype(np.float32), device=dev)
        layer, alpha = int(inj["inject_layer"]), float(inj["inject_alpha"])
        Dtot_all, Dres_all = [], []
        for g in gens[: args.max_gens]:
            e = entries.get(str(g["generation_id"]))
            if e is None:
                continue
            ids = torch.tensor([e["input_ids"]], dtype=torch.long, device=dev)
            P = int(e["prompt_length"])
            if ids.shape[1] - P < 2:
                continue
            spec = ResidualWriteSpec(layer_idx=layer, vector=v, alpha=alpha,
                                     start_pos=P, end_pos=ids.shape[1], normalize=True)
            steer = _logits_over_gen(model, ids, P, spec, dev)
            base = _logits_over_gen(model, ids, P, None, dev)
            dt, dr = _R_for_gen(steer, base, args.topk, dev)
            Dtot_all += dt; Dres_all += dr
        mt, mr = float(np.mean(Dtot_all)), float(np.mean(Dres_all))
        R = 1.0 - mr / mt if mt > 1e-9 else None
        reading = ("temperature-EQUIVALENT (R≥.9)" if (R is not None and R >= 0.9)
                   else "RICHER-than-temperature (R≤.6)" if (R is not None and R <= 0.6)
                   else "intermediate")
        rows.append({"cell": name, "vector": name.split("_")[0],
                     "alpha_frac": inj.get("inject_alpha_frac"), "site": layer,
                     "n_positions": len(Dtot_all),
                     "D_total_KL_mean": round(mt, 5), "D_resid_KL_mean": round(mr, 5),
                     "R_fraction_explained_by_rescaling": round(R, 4) if R is not None else None,
                     "reading": reading,
                     "is_null": name.upper().startswith(("RC", "RBAND"))})
        print(f"  {name:16} D_total={mt:.4f} D_resid={mr:.4f} R={R:.4f} → {reading}")

    out = {"model": args.model, "arm": "14m item 1 — temperature-equivalence rung (per-position best-T*)",
           "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop",
           "law": "R = 1 - min_T KL(steered||softmax(base/T)) / KL(steered||base); per-position, "
                  f"top-k={args.topk} UNION support (a BOUND on full-vocab KL); T grid {float(T_GRID.min()):.2f}-{float(T_GRID.max()):.1f}",
           "filed_P": {"V7_richer_R_le_.6": 0.55, "Vtemp_richer_R_le_.6": 0.60},
           "readings": "R≥.9 = temperature-equivalent; R≤.6 = richer-than-temperature",
           "topk_caveat": "KL on top-k union support underestimates full-vocab KL; R is a bounded estimate",
           "rows": rows}
    args.out_json.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
