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


def _ent_nll_over_gen(model, ids: torch.Tensor, P: int,
                      spec: ResidualWriteSpec | None) -> tuple[np.ndarray, np.ndarray]:
    """Per-position next-token ENTROPY + NLL (surprisal of the ACTUAL generated token) over the
    generated span. Entropy = state of the distribution; NLL = -log p(token) the likelihood rung
    of the A1 detector hierarchy reads."""
    h = attach_residual_write(model, spec) if spec is not None else None
    with torch.no_grad():
        logits = model(ids, use_cache=False).logits[0].float()   # [T, vocab]
    if h is not None:
        h.remove()
    lp = torch.log_softmax(logits, dim=-1)
    ent = -(lp.exp() * lp).sum(dim=-1)                            # [T]
    T = ids.shape[1]
    pos = torch.arange(P - 1, T - 1)                              # distributions that produced gen tokens
    tgt = ids[0, P:T]                                             # the actual generated tokens
    nll = -lp[pos, tgt]                                           # [len(pos)] surprisal
    return ent[pos].cpu().numpy(), nll.cpu().numpy()


def add_null_ratios(rows: list[dict], null_prefixes: tuple,
                    keys=("mean_entropy_steered", "entropy_rise", "base_model_nll")) -> None:
    """Attach matched-null (÷-Rc) columns with the 14m item-4 ZERO-DENOMINATOR GUARD.

    A ratio to a signed near-zero baseline (e.g. entropy_rise, whose nulls hover at ~0) is
    uninformative and explodes (the legacy -29.889 column). Guard: suppress `_over_Rc`
    (-> None) when |null_mean| is within the null's own SD of zero; ALWAYS emit the band
    readout `_vs_Rc_band` (null min/max/mean/sd + z + outside-band flag), which is the
    correct V7-specificity statistic for a difference-from-zero quantity. Same function
    used by the live replay and the GPU-free --reaggregate path (one source of truth)."""
    for r in rows:
        if r.get("is_null"):
            continue
        for key in keys:
            nulls = [nr[key] for nr in rows if nr.get("is_null")
                     and nr.get("site") == r.get("site") and nr.get("alpha_frac") == r.get("alpha_frac")
                     and key in nr]
            if not nulls or key not in r:
                r[f"{key}_over_Rc"] = None
                continue
            nm, nsd = float(np.mean(nulls)), float(np.std(nulls))
            nmin, nmax = float(np.min(nulls)), float(np.max(nulls))
            # A ratio to a signed near-zero baseline is unstable; suppress when the denominator's
            # coefficient of variation is large (CV>25% ⇒ a ratio-of-differences both near zero).
            cv = (nsd / abs(nm)) if nm else float("inf")
            ratio_meaningful = (nm != 0.0) and (cv <= 0.25)
            r[f"{key}_over_Rc"] = round(float(r[key] / nm), 3) if ratio_meaningful else None
            r[f"{key}_vs_Rc_band"] = {
                "null_mean": round(nm, 4), "null_sd": round(nsd, 4),
                "null_min": round(nmin, 4), "null_max": round(nmax, 4),
                "z_vs_null": round((r[key] - nm) / nsd, 3) if nsd > 0 else None,
                "outside_null_band": bool(r[key] < nmin or r[key] > nmax),
                "ratio_suppressed_zero_denom": not ratio_meaningful,
            }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--c3-run-dir", type=Path, default=None)
    ap.add_argument("--cells", nargs="+", default=None, help="cell dir names under c3-run-dir")
    ap.add_argument("--null-prefixes", default="RC",
                    help="comma-separated vector-name prefixes (upper) treated as matched-norm "
                         "nulls for the ÷-null ratio; default RC (C3). 14j leg-2 on vmb_b7_3b: RBAND.")
    ap.add_argument("--reaggregate-from", type=Path, default=None,
                    help="GPU-FREE: re-derive the ÷-Rc columns from an existing out-json's raw rows "
                         "with the zero-denom guard (14m item-4 fix; corrected artifact alongside).")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    null_prefixes = tuple(p.strip().upper() for p in args.null_prefixes.split(",") if p.strip())

    if args.reaggregate_from is not None:
        src = json.loads(args.reaggregate_from.read_text())
        rows = src["rows"]
        for r in rows:  # strip legacy ratio columns so the guarded ones are authoritative
            for k in list(r):
                if k.endswith("_over_Rc") or k.endswith("_vs_Rc_band"):
                    del r[k]
        add_null_ratios(rows, null_prefixes)
        src["rows"] = rows
        src["STATUS"] = src.get("STATUS", "") + " | REAGGREGATED with zero-denom guard (14m item-4)"
        src["reaggregation_note"] = ("÷-Rc columns re-derived from raw rows with the zero-denominator "
                                     "guard; entropy_rise_over_Rc suppressed (null≈0) -> read entropy_rise "
                                     "_vs_Rc_band (z + outside-band). Source: " + str(args.reaggregate_from))
        args.out_json.write_text(json.dumps(src, indent=1))
        for r in rows:
            if r.get("is_null"):
                continue
            b = r.get("entropy_rise_vs_Rc_band", {})
            print(f"  {r['cell']:20} rise={r.get('entropy_rise'):+.4f} over_Rc={r.get('entropy_rise_over_Rc')} "
                  f"z_vs_null={b.get('z_vs_null')} outside_band={b.get('outside_null_band')}")
        print(f"wrote {args.out_json} (GPU-free reaggregation)")
        return
    if not (args.model_path and args.c3_run_dir and args.cells):
        ap.error("--model-path, --c3-run-dir, --cells required unless --reaggregate-from is set")

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
        if not gens or "injection" not in gens[0]:
            logger.warning(f"{name}: no injection in metadata (baseline/no-inject cell) — skipping "
                           "entropy readout (steered-vs-unsteered needs an injected vector)")
            continue
        inj = gens[0]["injection"]
        v = torch.tensor(np.load(inj["inject_npz"])[inj["inject_key"]].astype(np.float32), device=dev)
        layer, alpha = int(inj["inject_layer"]), float(inj["inject_alpha"])
        steered, unsteered, base_nll = [], [], []
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
            ent_s, _ = _ent_nll_over_gen(model, ids, P, spec)
            ent_u, nll_u = _ent_nll_over_gen(model, ids, P, None)   # unsteered: base-model surprisal
            steered.append(float(np.mean(ent_s)))
            unsteered.append(float(np.mean(ent_u)))
            base_nll.append(float(np.mean(nll_u)))                  # likelihood rung (c)
        info = {"vector": name.split("_")[0], "site": layer, "alpha_frac": inj.get("inject_alpha_frac"),
                "n": len(steered),
                "mean_entropy_steered": round(float(np.mean(steered)), 4),
                "mean_entropy_unsteered": round(float(np.mean(unsteered)), 4),
                "entropy_rise": round(float(np.mean(steered) - np.mean(unsteered)), 4),
                "base_model_nll": round(float(np.mean(base_nll)), 4),  # (c) likelihood: base surprisal of the steered text
                "is_null": name.upper().startswith(null_prefixes)}
        rows.append({"cell": name, **info})
        print(f"  {name:20} steered={info['mean_entropy_steered']:.3f} "
              f"unsteered={info['mean_entropy_unsteered']:.3f} rise={info['entropy_rise']:+.3f} n={info['n']}")

    # (b): V_temp steered entropy vs matched Rc nulls (site, α) — with the zero-denom guard.
    add_null_ratios(rows, null_prefixes)

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
