"""14n item 2 — re-ranking anatomy (descriptive; session-8 Part A3).

Decomposes the residual KL of temperature-rescaling: beyond the best-T* rescaling of the
base distribution, WHICH tokens does the steer promote/demote, and is the residual
concentrated at decision points? Extends the 14m temp-equiv forward (14m banked only the
summary R, not the logits, so this re-forwards). Per generated position:

  T*        = argmin_T KL(steered ‖ softmax(base/T))       (top-k union support)
  q*        = softmax(base/T*)                              (best temperature-rescaled base)
  residual mass: promoted = Σ (p_steered − q*)_+  ; demoted = Σ (q* − p_steered)_+
  promoted mass split HEDGING-lexicon vs other; demoted likewise
  R_pos     = residual_KL / total_KL  (per position)  → profile vs base entropy (decision
              points = low base entropy): is the dial-like part uniform or concentrated?

Descriptive, NO P filed (14n item 2). The watch item (14n): a hedging-lexicon readout is
ONE dictionary — a low hedging-promotion is "not expressed in this lexicon," never "no
expression" (diversity already certifies expression). GPU, forwards over banked gens.
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

T_GRID = torch.tensor(np.concatenate([np.linspace(0.30, 0.95, 14), np.linspace(1.0, 4.0, 31)]),
                      dtype=torch.float32)
# hedging / uncertainty markers (leading-space + bare variants added at encode time)
HEDGE_WORDS = ["maybe", "perhaps", "might", "could", "possibly", "probably", "likely",
               "seems", "appears", "arguably", "roughly", "somewhat", "sort", "kind",
               "may", "would", "should", "or", "unclear", "uncertain", "guess", "think",
               "suppose", "presumably", "potentially", "conceivably"]


def _hedge_token_ids(tok) -> set[int]:
    ids: set[int] = set()
    for w in HEDGE_WORDS:
        for variant in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            enc = tok.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
    return ids


def _logits(model, ids, spec):
    h = attach_residual_write(model, spec) if spec is not None else None
    with torch.no_grad():
        lg = model(ids, use_cache=False).logits[0].float()
    if h is not None:
        h.remove()
    return lg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--max-gens", type=int, default=30)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    preset = MODEL_PRESETS[args.model]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=preset.torch_dtype, attn_implementation="eager").to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    dev = next(model.parameters()).device
    hedge = _hedge_token_ids(tok)
    tg = T_GRID.to(dev)

    rows = []
    for name in args.cells:
        d = args.run_dir / name
        meta = json.loads((d / "metadata.json").read_text())
        gens = meta["generations"] if "generations" in meta else meta
        entries = json.loads((d / "replay_manifest.json").read_text())["entries"]
        inj = gens[0]["injection"]
        v = torch.tensor(np.load(inj["inject_npz"])[inj["inject_key"]].astype(np.float32), device=dev)
        layer, alpha = int(inj["inject_layer"]), float(inj["inject_alpha"])

        promo_hedge, promo_tot, demo_hedge, demo_tot = 0.0, 0.0, 0.0, 0.0
        Rpos_lowent, Rpos_highent, n_low, n_high = [], [], 0, 0
        npos = 0
        for g in gens[: args.max_gens]:
            e = entries.get(str(g["generation_id"]))
            if e is None:
                continue
            iid = torch.tensor([e["input_ids"]], dtype=torch.long, device=dev)
            P = int(e["prompt_length"])
            if iid.shape[1] - P < 2:
                continue
            spec = ResidualWriteSpec(layer_idx=layer, vector=v, alpha=alpha,
                                     start_pos=P, end_pos=iid.shape[1], normalize=True)
            sl = _logits(model, iid, spec)
            bl = _logits(model, iid, None)
            pos = torch.arange(P - 1, iid.shape[1] - 1, device=dev)
            for i in pos.tolist():
                s_all, b_all = sl[i], bl[i]
                idx = torch.unique(torch.cat([s_all.topk(args.topk).indices,
                                              b_all.topk(args.topk).indices]))
                s = torch.log_softmax(s_all[idx], -1)
                b = b_all[idx].float()
                ps = s.exp()
                logq_grid = torch.log_softmax(b[None, :] / tg[:, None], dim=-1)
                kl_grid = (ps[None, :] * (s[None, :] - logq_grid)).sum(-1)
                Tstar = int(kl_grid.argmin().item())
                d_tot = float(kl_grid[(tg == 1.0).nonzero()[0, 0]].item())
                d_res = float(kl_grid.min().item())
                qstar = torch.log_softmax(b / tg[Tstar], -1).exp()
                diff = ps - qstar
                prom = torch.clamp(diff, min=0.0)
                demo = torch.clamp(-diff, min=0.0)
                is_h = torch.tensor([int(t) in hedge for t in idx.tolist()], device=dev)
                promo_tot += float(prom.sum()); promo_hedge += float(prom[is_h].sum())
                demo_tot += float(demo.sum()); demo_hedge += float(demo[is_h].sum())
                # base entropy at this position (decision point = low entropy)
                bent = float(-(torch.softmax(b, -1) * torch.log_softmax(b, -1)).sum())
                Rp = (d_res / d_tot) if d_tot > 1e-9 else 0.0
                if bent < 1.5:
                    Rpos_lowent.append(Rp); n_low += 1
                else:
                    Rpos_highent.append(Rp); n_high += 1
                npos += 1

        rows.append({
            "cell": name, "vector": name.split("_")[0], "alpha_frac": inj.get("inject_alpha_frac"),
            "n_positions": npos,
            "promoted_mass_hedge_frac": round(promo_hedge / promo_tot, 4) if promo_tot > 0 else None,
            "demoted_mass_hedge_frac": round(demo_hedge / demo_tot, 4) if demo_tot > 0 else None,
            "hedge_promotion_net": round((promo_hedge - demo_hedge) / max(promo_tot, 1e-9), 4),
            "R_residual_frac_lowentropy": round(float(np.mean(Rpos_lowent)), 4) if Rpos_lowent else None,
            "R_residual_frac_highentropy": round(float(np.mean(Rpos_highent)), 4) if Rpos_highent else None,
            "n_lowent": n_low, "n_highent": n_high,
        })
        print(f"  {name}: hedge_promo={rows[-1]['promoted_mass_hedge_frac']} "
              f"R_resid low_ent={rows[-1]['R_residual_frac_lowentropy']} "
              f"high_ent={rows[-1]['R_residual_frac_highentropy']}")

    out = {"model": args.model, "arm": "14n item 2 — re-ranking anatomy (descriptive)",
           "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop; NO P filed",
           "law": "residual KL beyond best-T* rescaling; promoted/demoted mass split by hedging "
                  f"lexicon ({len(hedge)} single-token markers); per-position residual-fraction "
                  "R=D_resid/D_total split by base entropy (<1.5 nat = decision point). top-k union.",
           "watch_item": "hedging lexicon is ONE dictionary — low hedge-promotion = 'not in this "
                         "lexicon', NEVER 'no expression' (diversity certifies expression).",
           "n_hedge_markers": len(hedge), "rows": rows}
    args.out_json.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
