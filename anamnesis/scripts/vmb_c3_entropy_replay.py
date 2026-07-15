"""C3 certifying consequence (b) — per-token entropy under V_temp steering (PREFLIGHT §4 (b)).

Lightweight: re-applies each cell's banked injection (from metadata), forwards over the
generated tokens ONCE, records ONLY per-position next-token entropy (no raw tensors → zero
storage). Reports, per cell: mean STEERED entropy (injection on, the distribution the steered
gen sampled from) and mean UNSTEERED entropy (same tokens, injection off) → the rise. (b) =
does V_temp raise steered entropy ABOVE the matched-norm Rc nulls, dose-ordered (toward the
hot-sampling t09 profile)? First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from anamnesis.config import MODEL_PRESETS
from anamnesis.extraction.model_loader import ResidualWriteSpec, attach_residual_write


def _entropy_over_gen(model, ids: torch.Tensor, P: int, spec: ResidualWriteSpec | None) -> np.ndarray:
    """Per-position next-token entropy over the GENERATED span (positions predicting tokens P..T-1)."""
    h = attach_residual_write(model, spec) if spec is not None else None
    with torch.no_grad():
        logits = model(ids, use_cache=False).logits[0].float()   # [T, vocab]
    if h is not None:
        h.remove()
    lp = torch.log_softmax(logits, dim=-1)
    ent = -(lp.exp() * lp).sum(dim=-1)                            # [T]
    return ent[P - 1:ids.shape[1] - 1].cpu().numpy()             # distributions that produced gen tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--c3-run-dir", type=Path, required=True)
    ap.add_argument("--cells", nargs="+", required=True, help="cell dir names under c3-run-dir")
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
        d = args.c3_run_dir / name
        meta = json.loads((d / "metadata.json").read_text())
        gens = meta["generations"] if "generations" in meta else meta
        entries = json.loads((d / "replay_manifest.json").read_text())["entries"]
        inj = gens[0]["injection"]
        v = torch.tensor(np.load(inj["inject_npz"])[inj["inject_key"]].astype(np.float32), device=dev)
        layer, alpha = int(inj["inject_layer"]), float(inj["inject_alpha"])
        steered, unsteered = [], []
        for g in gens:
            e = entries.get(str(g["generation_id"]))
            if e is None:
                continue
            ids = torch.tensor([e["input_ids"]], dtype=torch.long, device=dev)
            P = int(e["prompt_length"])
            if ids.shape[1] - P < 2:
                continue
            spec = ResidualWriteSpec(layer_idx=layer, vector=v, alpha=alpha,
                                     start_pos=P, end_pos=ids.shape[1], normalize=True)
            steered.append(float(np.mean(_entropy_over_gen(model, ids, P, spec))))
            unsteered.append(float(np.mean(_entropy_over_gen(model, ids, P, None))))
        info = {"vector": name.split("_")[0], "site": layer, "alpha_frac": inj.get("inject_alpha_frac"),
                "n": len(steered),
                "mean_entropy_steered": round(float(np.mean(steered)), 4),
                "mean_entropy_unsteered": round(float(np.mean(unsteered)), 4),
                "entropy_rise": round(float(np.mean(steered) - np.mean(unsteered)), 4),
                "is_null": name.upper().startswith("RC")}
        rows.append({"cell": name, **info})
        print(f"  {name:20} steered={info['mean_entropy_steered']:.3f} "
              f"unsteered={info['mean_entropy_unsteered']:.3f} rise={info['entropy_rise']:+.3f} n={info['n']}")

    # (b): V_temp steered entropy ÷ mean(Rc steered) at matched (site, α)
    def rc_mean(site, af, key):
        vals = [r[key] for r in rows if r["is_null"] and r["site"] == site and r["alpha_frac"] == af]
        return float(np.mean(vals)) if vals else None
    for r in rows:
        if r["is_null"]:
            continue
        for key in ("mean_entropy_steered", "entropy_rise"):
            base = rc_mean(r["site"], r["alpha_frac"], key)
            r[f"{key}_over_Rc"] = round(float(r[key] / base), 3) if base else None

    out = {"model": args.model, "arm": "C3 certifying (b) — per-token entropy under V_temp",
           "STATUS": "FIRST_READ_PENDING (C§8)",
           "law": "per-position next-token entropy over generated span; steered (inject on) vs "
                  "unsteered (same tokens, inject off); (b) = V_temp steered-entropy/rise ÷ mean(Rc) "
                  "dose-ordered → rises toward the t09 hot profile",
           "rows": rows}
    args.out_json.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
