"""A5 layer-alignment diagnostic — per-layer separation of the steering axes.

Motivation (2026-07-19): every steering method has acted weak on DSV2-Lite (M6), and the
map site (L9 = 33% depth) is OFF the sampled grid [0,5,11,15,18,22,26] and shallow vs the
3B/8B precedent (both map at ~50% depth: 3B L14/28, 8B L16/32). Before committing the PM6-b
2×2 (and the whole A5 arm) to a layer, MEASURE where the concept axes are most linearly
separable in the residual stream — the standard CAA steerability predictor.

For every decoder layer s (input to layer s = hidden_states[s]), over the generated span:
  MODE axis        pure_analogical vs pure_contrastive   (the data-lever / V3 axis)
  ENTROPY/TEMP     t09 vs t03                             (the V7 / temperature family axis)
Separation = HELD-OUT Cohen's d along the CAA direction (dir fit on one fold, d measured on
the other, both ways averaged — removes in-sample optimism so the per-layer curve is fair)
plus the scale-free centroid-distance / within-class-rms ratio. The peak layer(s) read the
steering site straight off the curve; the two axes may peak differently (informative).

Pure numpy readout; one GPU forward pass. First-read -> outer loop; nothing stamped.

    python -m anamnesis.scripts.vmb_a5_layer_separation --model dsv2-lite \
        --model-path $MP --runs-root $RUNS \
        --pos-run vmb_a2_dsv2-lite_pure_analogical --neg-run vmb_a2_dsv2-lite_pure_contrastive \
        --hot-run vmb_a1_dsv2-lite_t09 --cold-run vmb_a1_dsv2-lite_t03 \
        --out-json $BANK/arms/A5_dsv2/layer_separation_dsv2.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from anamnesis.config import MODEL_PRESETS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("layer_sep")


def _mean_resid_all_layers(model, ids: torch.Tensor, prompt_len: int,
                           n_layers: int) -> np.ndarray:
    """Mean residual over the generated span at every layer input (hidden_states[s],
    s=1..n_layers). Returns [n_layers, d] float32."""
    dev = next(model.parameters()).device
    with torch.no_grad():
        out = model(ids.to(dev), use_cache=False, output_hidden_states=True,
                    return_dict=True)
    hs = out.hidden_states  # tuple len n_layers+1; [0]=embedding, [s]=input to layer s
    rows = []
    for s in range(1, n_layers + 1):
        h = hs[s][0, prompt_len:]
        if h.shape[0] == 0:
            rows.append(np.full(hs[s].shape[-1], np.nan, dtype=np.float32))
        else:
            rows.append(h.float().mean(dim=0).cpu().numpy().astype(np.float32))
    return np.stack(rows)  # [n_layers, d]


def _capture(model, manifest: Path, n_layers: int, limit: int | None) -> np.ndarray:
    """[n_gens, n_layers, d] mean-residual tensor over a corpus's replay manifest."""
    entries = json.loads(manifest.read_text())["entries"]
    keys = sorted(entries, key=lambda k: int(k))
    if limit is not None:
        keys = keys[:limit]
    dev = next(model.parameters()).device
    acc: list[np.ndarray] = []
    for i, k in enumerate(keys):
        e = entries[k]
        ids = torch.tensor([e["input_ids"]], dtype=torch.long, device=dev)
        P = int(e["prompt_length"])
        if ids.shape[1] - P < 1:
            continue
        acc.append(_mean_resid_all_layers(model, ids, P, n_layers))
        if (i + 1) % 50 == 0:
            logger.info(f"  {manifest.parent.name}: {i + 1}/{len(keys)}")
    if not acc:
        raise RuntimeError(f"no usable gens in {manifest}")
    return np.stack(acc)  # [n, n_layers, d]


def _heldout_cohend(A: np.ndarray, B: np.ndarray) -> float:
    """Cohen's d along the CAA direction (mean(A)-mean(B)), 2-fold held-out + averaged."""
    folds = [((A[::2], A[1::2]), (B[::2], B[1::2])),
             ((A[1::2], A[::2]), (B[1::2], B[::2]))]
    ds: list[float] = []
    for (a_fit, a_ev), (b_fit, b_ev) in folds:
        if min(len(a_fit), len(a_ev), len(b_fit), len(b_ev)) < 2:
            continue
        d = a_fit.mean(0) - b_fit.mean(0)
        nrm = float(np.linalg.norm(d))
        if nrm < 1e-9:
            ds.append(0.0)
            continue
        d = d / nrm
        pa, pb = a_ev @ d, b_ev @ d
        sp = float(np.sqrt(0.5 * (pa.var(ddof=1) + pb.var(ddof=1))))
        ds.append(float((pa.mean() - pb.mean()) / sp) if sp > 0 else 0.0)
    return float(np.mean(ds)) if ds else 0.0


def _centroid_ratio(A: np.ndarray, B: np.ndarray) -> float:
    """||mean(A)-mean(B)|| / rms within-class distance-to-own-centroid (scale-free)."""
    sep = float(np.linalg.norm(A.mean(0) - B.mean(0)))
    wa = float(np.sqrt(((A - A.mean(0)) ** 2).sum(1).mean()))
    wb = float(np.sqrt(((B - B.mean(0)) ** 2).sum(1).mean()))
    w = 0.5 * (wa + wb)
    return sep / w if w > 0 else 0.0


def _axis_rows(pos: np.ndarray, neg: np.ndarray, n_layers: int) -> list[dict]:
    rows = []
    for s in range(n_layers):  # s=0 -> layer 1 (hidden_states[1])
        A, B = pos[:, s, :], neg[:, s, :]
        ok = ~(np.isnan(A).any(1))
        A = A[~np.isnan(A).any(1)]
        B = B[~np.isnan(B).any(1)]
        rows.append({
            "layer": s + 1,
            "depth_pct": round(100.0 * (s + 1) / n_layers, 1),
            "cohen_d": round(_heldout_cohend(A, B), 4),
            "centroid_ratio": round(_centroid_ratio(A, B), 4),
            "n_pos": int(len(A)), "n_neg": int(len(B)),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dsv2-lite")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--pos-run", required=True, help="mode axis + corpus (e.g. pure_analogical)")
    ap.add_argument("--neg-run", required=True, help="mode axis - corpus (e.g. pure_contrastive)")
    ap.add_argument("--hot-run", default=None, help="temp axis + corpus (e.g. t09)")
    ap.add_argument("--cold-run", default=None, help="temp axis - corpus (e.g. t03)")
    ap.add_argument("--limit", type=int, default=None, help="cap gens per corpus (default all)")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    preset = MODEL_PRESETS[args.model]
    n_layers = preset.num_layers
    from transformers import AutoModelForCausalLM
    logger.info(f"loading {args.model} ({n_layers} layers) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=preset.torch_dtype,
        attn_implementation="eager").to("cuda").eval()

    def cap(run: str) -> np.ndarray:
        m = args.runs_root / run / "replay_manifest.json"
        logger.info(f"capturing {run} ...")
        return _capture(model, m, n_layers, args.limit)

    pos, neg = cap(args.pos_run), cap(args.neg_run)
    mode_rows = _axis_rows(pos, neg, n_layers)
    temp_rows = None
    if args.hot_run and args.cold_run:
        hot, cold = cap(args.hot_run), cap(args.cold_run)
        temp_rows = _axis_rows(hot, cold, n_layers)

    def peak(rows: list[dict], key: str) -> dict:
        best = max(rows, key=lambda r: r[key])
        return {"layer": best["layer"], "depth_pct": best["depth_pct"], key: best[key]}

    out = {
        "arm": "A5 layer-alignment diagnostic (per-layer steering-axis separation)",
        "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop",
        "model": args.model, "n_layers": n_layers,
        "law": "per-layer HELD-OUT Cohen's d along the CAA direction (fit one fold, measure "
               "the other, averaged) + scale-free centroid_ratio; peak layer = strongest "
               "steering site. 3B/8B precedent maps at ~50% depth.",
        "corpora": {"mode_pos": args.pos_run, "mode_neg": args.neg_run,
                    "temp_hot": args.hot_run, "temp_cold": args.cold_run},
        "mode_axis": {"peak_cohen_d": peak(mode_rows, "cohen_d"),
                      "peak_centroid_ratio": peak(mode_rows, "centroid_ratio"),
                      "per_layer": mode_rows},
    }
    if temp_rows is not None:
        out["temp_axis"] = {"peak_cohen_d": peak(temp_rows, "cohen_d"),
                            "peak_centroid_ratio": peak(temp_rows, "centroid_ratio"),
                            "per_layer": temp_rows}
    args.out_json.write_text(json.dumps(out, indent=1))
    logger.info(f"MODE peak Cohen's d @ L{out['mode_axis']['peak_cohen_d']['layer']} "
                f"({out['mode_axis']['peak_cohen_d']['depth_pct']}%) "
                f"= {out['mode_axis']['peak_cohen_d']['cohen_d']}")
    if temp_rows is not None:
        logger.info(f"TEMP peak Cohen's d @ L{out['temp_axis']['peak_cohen_d']['layer']} "
                    f"({out['temp_axis']['peak_cohen_d']['depth_pct']}%) "
                    f"= {out['temp_axis']['peak_cohen_d']['cohen_d']}")
    logger.info("per-layer MODE (layer depth% cohen_d centroid_ratio):")
    for r in mode_rows:
        logger.info(f"  L{r['layer']:2d} {r['depth_pct']:5.1f}%  d={r['cohen_d']:+.3f}  "
                    f"cr={r['centroid_ratio']:.3f}")
    logger.info(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
