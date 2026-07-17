"""Entropy-curve reconstruction harness (14q item 5, Pe=.65; session-11 Part A.4 prep).

Filed prediction (frozen in ADDENDUM 14q BEFORE any data): V7's per-position entropy curve
is distinguishable from a temperature-matched sampler curve — the positional signature 14n's
anatomy predicts (dial at decision points: 14n's R_residual split by base entropy, decision
point = base entropy < 1.5 nat).

STATISTIC, DEFINED HERE BEFORE THE SCORING DATA EXISTS (D5's gens):
For each gen (tokens fixed = the steered cell's own tokens):
  e_s(t)  = per-position next-token entropy, injection ON  (the sampled-from distribution)
  e_b(t)  = same tokens, injection OFF (base)
  e_T*(t) = same tokens, softmax(base logits / T*), where T* is fitted PER CELL to match the
            cell's mean entropy rise: mean_t[e_T*(t)] ≈ mean_t[e_s(t)] (bisection on T).
Curves: Δ_s(t) = e_s(t) − e_b(t) and Δ_T*(t) = e_T*(t) − e_b(t), binned by BASE entropy:
  primary split = 14n's decision-point boundary (e_b < 1.5 nat vs ≥ 1.5), plus base-entropy
  deciles for the full curve shape.
Headline statistic (pre-named): DECISION-POINT CONCENTRATION
  C = mean[Δ(t) | e_b < 1.5] / mean[Δ(t) | e_b ≥ 1.5]
  Pe's "distinguishable" = C_V7 differs from C_T* outside the matched-R band (Rband cells run
  through the identical pipeline give the null envelope for C and for the full decile-profile
  L2 distance ||curve_V7 − curve_T*||). Both the split ratio and the decile profile are
  emitted; the ratio is the statistic of record, the profile is anatomy.
All per-position arrays are BANKED (npz per cell) so every future re-read is CPU-free —
the means-only banking of the 14m era is the debt this harness retires.

Modes:
  --replay: GPU/node — forwards each gen once per {on, off} (T* needs no extra forward:
    e_T* is computed from the SAME base logits pass at temperature T on CPU), banks npz +
    emits the curve JSON. Multicell-friendly: pass several cell dirs; model loads ONCE.
  --from-npz: CPU — recompute curves/statistics from banked npz (re-analysis path).

Validation plan (stated): run first over the banked 14m cells (vmb_b7_3b: V7 + Rband @L14)
as the harness shakedown — Pe itself scores ONLY on D5's 512-token uncapped gens (the filed
wording rides D5). First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DECISION_NAT = 1.5     # 14n's frozen decision-point boundary
N_DECILES = 10


def entropy_of(logits: "torch.Tensor", temperature: float = 1.0) -> "torch.Tensor":
    import torch
    lp = torch.log_softmax(logits.float() / temperature, dim=-1)
    return -(lp.exp() * lp).sum(dim=-1)


def fit_tstar(base_logits: "torch.Tensor", target_mean_ent: float,
              lo: float = 0.3, hi: float = 4.0, iters: int = 40) -> float:
    """Bisection on T so that mean entropy of softmax(base/T) matches the steered mean."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if float(entropy_of(base_logits, mid).mean()) < target_mean_ent:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def curve_stats(e_s: np.ndarray, e_b: np.ndarray, e_t: np.ndarray,
                decile_edges: np.ndarray) -> dict:
    """Decision-point concentration + decile profiles for steered and T*-matched deltas."""
    d_s, d_t = e_s - e_b, e_t - e_b
    low = e_b < DECISION_NAT
    out: dict = {
        "n_positions": int(len(e_b)),
        "n_low_entropy": int(low.sum()),
        "mean_rise_steered": round(float(d_s.mean()), 5),
        "mean_rise_tstar": round(float(d_t.mean()), 5),
    }
    for tag, d in (("steered", d_s), ("tstar", d_t)):
        lo_m = float(d[low].mean()) if low.any() else float("nan")
        hi_m = float(d[~low].mean()) if (~low).any() else float("nan")
        out[f"rise_low_{tag}"] = round(lo_m, 5)
        out[f"rise_high_{tag}"] = round(hi_m, 5)
        out[f"concentration_{tag}"] = round(lo_m / hi_m, 4) if hi_m and np.isfinite(hi_m) and abs(hi_m) > 1e-9 else None
    bins = np.clip(np.digitize(e_b, decile_edges), 0, N_DECILES - 1)
    prof_s = [round(float(d_s[bins == i].mean()), 5) if (bins == i).any() else None
              for i in range(N_DECILES)]
    prof_t = [round(float(d_t[bins == i].mean()), 5) if (bins == i).any() else None
              for i in range(N_DECILES)]
    out["decile_profile_steered"] = prof_s
    out["decile_profile_tstar"] = prof_t
    pairs = [(a, b) for a, b in zip(prof_s, prof_t) if a is not None and b is not None]
    out["profile_l2_steered_vs_tstar"] = round(float(np.sqrt(sum((a - b) ** 2 for a, b in pairs))), 5)
    return out


def analyze_cell_arrays(ent_s: list[np.ndarray], ent_b: list[np.ndarray],
                        ent_t: list[np.ndarray], tstars: list[float],
                        halves: bool) -> dict:
    e_s, e_b, e_t = (np.concatenate(x) for x in (ent_s, ent_b, ent_t))
    edges = np.quantile(e_b, np.linspace(0, 1, N_DECILES + 1)[1:-1])
    res = {"pooled": curve_stats(e_s, e_b, e_t, edges),
           "tstar_mean": round(float(np.mean(tstars)), 4),
           "tstar_sd": round(float(np.std(tstars)), 4)}
    if halves:  # D5 front-vs-back-half rider (late-collapse leg shares the arrays)
        fh_s, fh_b, fh_t, bh_s, bh_b, bh_t = [], [], [], [], [], []
        for s, b, t in zip(ent_s, ent_b, ent_t):
            h = len(s) // 2
            fh_s.append(s[:h]); fh_b.append(b[:h]); fh_t.append(t[:h])
            bh_s.append(s[h:]); bh_b.append(b[h:]); bh_t.append(t[h:])
        for tag, (ss, bb, tt) in (("front_half", (fh_s, fh_b, fh_t)),
                                  ("back_half", (bh_s, bh_b, bh_t))):
            es, eb, et = (np.concatenate(x) for x in (ss, bb, tt))
            res[tag] = curve_stats(es, eb, et, edges)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--cells", nargs="+", required=True, help="cell dir names under run-dir")
    ap.add_argument("--null-prefixes", default="RBAND",
                    help="comma-separated upper-case vector prefixes = matched-R nulls")
    ap.add_argument("--from-npz", action="store_true",
                    help="CPU: reuse banked per-position npz (skip all forwards)")
    ap.add_argument("--halves", action="store_true",
                    help="also emit front-/back-half splits (D5 late-collapse rider)")
    ap.add_argument("--npz-dir", type=Path, default=None,
                    help="where per-position arrays are banked (default <run-dir>/entropy_curves)")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    npz_dir = args.npz_dir or (args.run_dir / "entropy_curves")
    npz_dir.mkdir(parents=True, exist_ok=True)
    null_prefixes = tuple(p.strip().upper() for p in args.null_prefixes.split(",") if p.strip())

    cells_out: dict[str, dict] = {}
    if args.from_npz:
        for cell in args.cells:
            f = npz_dir / f"{cell}.npz"
            z = np.load(f, allow_pickle=True)
            ent_s = list(z["ent_steered"]); ent_b = list(z["ent_base"]); ent_t = list(z["ent_tstar"])
            cells_out[cell] = analyze_cell_arrays(ent_s, ent_b, ent_t,
                                                  list(z["tstars"]), args.halves)
    else:
        import torch
        from transformers import AutoModelForCausalLM
        from anamnesis.config import MODEL_PRESETS
        from anamnesis.extraction.model_loader import ResidualWriteSpec, attach_residual_write
        preset = MODEL_PRESETS[args.model]
        if not args.model_path:
            raise SystemExit("--model-path required for --replay mode")
        # load ONCE for all cells (multicell law); SDPA is safe — logits-only forwards,
        # no attention weights read (throughput playbook fast path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=preset.torch_dtype,
            attn_implementation="sdpa").to("cuda").eval()
        dev = next(model.parameters()).device
        for cell in args.cells:
            d = args.run_dir / cell
            meta = json.loads((d / "metadata.json").read_text())
            gens = meta["generations"] if "generations" in meta else meta
            entries = json.loads((d / "replay_manifest.json").read_text())["entries"]
            # injection lives at top level (a5_injection: b7-style cells) or per-gen
            # (injection: c3-style cells) — support both banked shapes
            inj = meta.get("a5_injection") or (gens[0].get("injection") if gens else None)
            ent_s_l, ent_b_l, ent_t_l, tstars = [], [], [], []
            for g in gens:
                e = entries.get(str(g["generation_id"]))
                if e is None:
                    continue
                ids = torch.tensor([e["input_ids"]], dtype=torch.long, device=dev)
                P = int(e["prompt_length"])
                if ids.shape[1] - P < 4:
                    continue
                pos = torch.arange(P - 1, ids.shape[1] - 1)
                # base pass (single) — e_b AND every e_T come from these logits
                with torch.no_grad():
                    base_logits = model(ids, use_cache=False).logits[0].float()[pos].cpu()
                # steered pass
                if inj is None:
                    raise SystemExit(f"{cell}: no inject block in metadata — is this a steered cell?")
                v = torch.tensor(np.load(inj["inject_npz"])[inj["inject_key"]].astype(np.float32),
                                 device=dev)
                spec = ResidualWriteSpec(layer_idx=int(inj["inject_layer"]), vector=v,
                                         alpha=float(inj["inject_alpha"]), start_pos=P,
                                         end_pos=ids.shape[1], normalize=True)
                h = attach_residual_write(model, spec)
                with torch.no_grad():
                    st_logits = model(ids, use_cache=False).logits[0].float()[pos].cpu()
                h.remove()
                e_b = entropy_of(base_logits).numpy()
                e_s = entropy_of(st_logits).numpy()
                tstar = fit_tstar(base_logits, float(e_s.mean()))
                e_t = entropy_of(base_logits, tstar).numpy()
                ent_s_l.append(e_s); ent_b_l.append(e_b); ent_t_l.append(e_t)
                tstars.append(tstar)
            np.savez_compressed(npz_dir / f"{cell}.npz",
                                ent_steered=np.array(ent_s_l, dtype=object),
                                ent_base=np.array(ent_b_l, dtype=object),
                                ent_tstar=np.array(ent_t_l, dtype=object),
                                tstars=np.array(tstars))
            cells_out[cell] = analyze_cell_arrays(ent_s_l, ent_b_l, ent_t_l, tstars, args.halves)
            print(f"{cell}: n={len(tstars)} T*={np.mean(tstars):.3f} "
                  f"C_steered={cells_out[cell]['pooled']['concentration_steered']} "
                  f"C_tstar={cells_out[cell]['pooled']['concentration_tstar']}")

    # null envelope over matched-R cells (identical pipeline)
    def is_null(cell: str) -> bool:
        return cell.upper().startswith(null_prefixes)

    null_C = [c["pooled"]["concentration_steered"] for name, c in cells_out.items()
              if is_null(name) and c["pooled"]["concentration_steered"] is not None]
    null_L2 = [c["pooled"]["profile_l2_steered_vs_tstar"] for name, c in cells_out.items()
               if is_null(name)]
    envelope = {
        "n_null_cells": len(null_C),
        "concentration_null_band": [round(min(null_C), 4), round(max(null_C), 4)] if null_C else None,
        "profile_l2_null_band": [round(min(null_L2), 5), round(max(null_L2), 5)] if null_L2 else None,
    }
    for name, c in cells_out.items():
        if is_null(name):
            continue
        cs, ct = c["pooled"]["concentration_steered"], c["pooled"]["concentration_tstar"]
        c["pe_readout"] = {
            "concentration_gap_steered_minus_tstar": round(cs - ct, 4)
            if cs is not None and ct is not None else None,
            "outside_null_concentration_band": (bool(cs < min(null_C) or cs > max(null_C))
                                                if null_C and cs is not None else None),
            "profile_l2_outside_null_band": (bool(c["pooled"]["profile_l2_steered_vs_tstar"]
                                                  > max(null_L2)) if null_L2 else None),
        }

    out = {
        "arm": "entropy-curve reconstruction harness (14q item 5, Pe=.65)",
        "STATUS": "FIRST_READ_PENDING (C§8)",
        "model": args.model,
        "law": ("per-position next-token entropy over gen span; T* bisection-matched per cell to "
                f"the steered mean rise; decision split at {DECISION_NAT} nat (14n frozen boundary) "
                "+ base-entropy deciles; statistic of record = decision-point concentration "
                "C = rise_low/rise_high, steered vs T*-matched, judged against the matched-R "
                "envelope run through the identical pipeline; per-position arrays BANKED (npz)"),
        "scoring_note": "Pe SCORES ONLY on D5's 512-token uncapped gens (14q wording); any run "
                        "over banked 128-cap 14m cells is harness validation, not the Pe score",
        "censoring_note": "512-cap gens are truncations at baseline (0% natural stop within "
                          "512; ADVISORY-session11-inflight item 1) — curve/concentration "
                          "readings unaffected; length-adjacent wording censored-scoped",
        "null_envelope": envelope,
        "cells": cells_out,
        "npz_dir": str(npz_dir),
    }
    args.out_json.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
