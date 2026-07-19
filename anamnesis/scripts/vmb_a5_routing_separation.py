"""A5 routing-substrate diagnostic — per-MoE-layer separation of the steering axes in
EXPERT-ROUTING space (companion to vmb_a5_layer_separation, which reads the RESIDUAL stream).

Hypothesis (2026-07-19): DSV2-Lite's mode axis is ~20× weaker in the residual stream than on
dense 3B/8B (measured), and PM6-d already showed the routing-CKA feats carry the mode map. If
"mode lives in expert routing, not a residual direction," then the analogical-vs-contrastive
separation should be MUCH stronger in routing space than in the residual stream. This measures it.

Per MoE layer, per generation: mean DENSE pre-topk router softmax over the generated span
(the [n_routed_experts] expert-allocation vector). Recomputed exactly as the extraction hook does
(model_loader.py:_make_moe_router_prehook: F.linear(h.float(), mlp.gate.weight.float()).softmax(-1)).
Separation = HELD-OUT Cohen's d along the CAA (mean-difference) direction in routing space, plus the
scale-free centroid ratio — identical metric to the residual diagnostic, so the numbers compare.

  MODE axis     pure_analogical vs pure_contrastive
  TEMP axis     t09 vs t03

Pure numpy readout; one GPU forward pass. First-read -> outer loop; nothing stamped.

    python -m anamnesis.scripts.vmb_a5_routing_separation --model dsv2-lite --model-path $MP \
        --runs-root $RUNS \
        --pos-run vmb_a2_dsv2-lite_pure_analogical --neg-run vmb_a2_dsv2-lite_pure_contrastive \
        --hot-run vmb_a1_dsv2-lite_t09 --cold-run vmb_a1_dsv2-lite_t03 \
        --out-json $BANK/arms/A5_dsv2/routing_separation_dsv2.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from anamnesis.config import MODEL_PRESETS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("routing_sep")


def _moe_layers(model) -> list[tuple[int, object]]:
    """(decoder_layer_idx, moe_module) for every layer whose MLP exposes a router gate."""
    layers = model.model.layers
    out = []
    for i, lyr in enumerate(layers):
        mlp = getattr(lyr, "mlp", None)
        if mlp is not None and hasattr(mlp, "gate") and hasattr(mlp.gate, "weight"):
            out.append((i, mlp))
    return out


class _RouterCapture:
    """Registers a forward-pre-hook on each MoE module; stores the dense router softmax per layer
    for the most recent forward. Mirrors extraction/model_loader.py:_make_moe_router_prehook."""

    def __init__(self, moe: list[tuple[int, object]]):
        self.store: dict[int, torch.Tensor] = {}
        self.handles = []
        for layer_idx, mod in moe:
            self.handles.append(mod.register_forward_pre_hook(self._mk(layer_idx)))

    def _mk(self, layer_idx: int):
        def pre(module, args):
            h = args[0]
            logits = F.linear(h.to(torch.float32), module.gate.weight.to(torch.float32))
            self.store[layer_idx] = logits.softmax(dim=-1).detach()  # [batch, seq, n_experts]
        return pre

    def remove(self):
        for hd in self.handles:
            hd.remove()


def _capture(model, cap: _RouterCapture, manifest: Path, moe_idx: list[int],
             limit: int | None) -> np.ndarray:
    """[n_gens, n_moe_layers, n_experts] mean-routing tensor over a corpus's replay manifest."""
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
        cap.store.clear()
        with torch.no_grad():
            model(ids, use_cache=False, return_dict=True)
        rows = []
        for l_idx in moe_idx:
            d = cap.store[l_idx][0, P:, :]      # [gen_len, n_experts]
            rows.append(d.mean(0).cpu().numpy().astype(np.float32))
        acc.append(np.stack(rows))              # [n_moe, n_experts]
        if (i + 1) % 50 == 0:
            logger.info(f"  {manifest.parent.name}: {i + 1}/{len(keys)}")
    if not acc:
        raise RuntimeError(f"no usable gens in {manifest}")
    return np.stack(acc)


def _heldout_cohend(A: np.ndarray, B: np.ndarray) -> float:
    folds = [((A[::2], A[1::2]), (B[::2], B[1::2])),
             ((A[1::2], A[::2]), (B[1::2], B[::2]))]
    ds: list[float] = []
    for (a_fit, a_ev), (b_fit, b_ev) in folds:
        if min(len(a_fit), len(a_ev), len(b_fit), len(b_ev)) < 2:
            continue
        d = a_fit.mean(0) - b_fit.mean(0)
        nrm = float(np.linalg.norm(d))
        if nrm < 1e-12:
            ds.append(0.0)
            continue
        d = d / nrm
        pa, pb = a_ev @ d, b_ev @ d
        sp = float(np.sqrt(0.5 * (pa.var(ddof=1) + pb.var(ddof=1))))
        ds.append(float((pa.mean() - pb.mean()) / sp) if sp > 0 else 0.0)
    return float(np.mean(ds)) if ds else 0.0


def _centroid_ratio(A: np.ndarray, B: np.ndarray) -> float:
    sep = float(np.linalg.norm(A.mean(0) - B.mean(0)))
    wa = float(np.sqrt(((A - A.mean(0)) ** 2).sum(1).mean()))
    wb = float(np.sqrt(((B - B.mean(0)) ** 2).sum(1).mean()))
    w = 0.5 * (wa + wb)
    return sep / w if w > 0 else 0.0


def _axis_rows(pos: np.ndarray, neg: np.ndarray, moe_idx: list[int], n_layers: int) -> list[dict]:
    rows = []
    for j, l_idx in enumerate(moe_idx):
        A, B = pos[:, j, :], neg[:, j, :]
        rows.append({
            "layer": l_idx,
            "depth_pct": round(100.0 * l_idx / n_layers, 1),
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
    ap.add_argument("--pos-run", required=True)
    ap.add_argument("--neg-run", required=True)
    ap.add_argument("--hot-run", default=None)
    ap.add_argument("--cold-run", default=None)
    ap.add_argument("--limit", type=int, default=None)
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

    moe = _moe_layers(model)
    if not moe:
        raise SystemExit("no MoE layers found (model has no mlp.gate) — routing diagnostic is MoE-only")
    moe_idx = [i for i, _ in moe]
    n_experts = moe[0][1].gate.weight.shape[0]
    logger.info(f"{len(moe_idx)} MoE layers {moe_idx}; n_routed_experts={n_experts}")
    cap = _RouterCapture(moe)

    def grab(run: str) -> np.ndarray:
        m = args.runs_root / run / "replay_manifest.json"
        logger.info(f"capturing routing over {run} ...")
        return _capture(model, cap, m, moe_idx, args.limit)

    pos, neg = grab(args.pos_run), grab(args.neg_run)
    mode_rows = _axis_rows(pos, neg, moe_idx, n_layers)
    temp_rows = None
    if args.hot_run and args.cold_run:
        hot, cold = grab(args.hot_run), grab(args.cold_run)
        temp_rows = _axis_rows(hot, cold, moe_idx, n_layers)
    cap.remove()

    def peak(rows: list[dict], key: str) -> dict:
        best = max(rows, key=lambda r: r[key])
        return {"layer": best["layer"], "depth_pct": best["depth_pct"], key: best[key]}

    out = {
        "arm": "A5 routing-substrate diagnostic (per-MoE-layer steering-axis separation in "
               "expert-routing space)",
        "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop",
        "model": args.model, "n_layers": n_layers, "n_moe_layers": len(moe_idx),
        "n_routed_experts": int(n_experts), "moe_layer_indices": moe_idx,
        "law": "per-MoE-layer HELD-OUT Cohen's d along the CAA direction in DENSE pre-topk router "
               "softmax space (mean over generated span). Compare to the RESIDUAL-stream diagnostic "
               "(vmb_a5_layer_separation): routing >> residual ⇒ mode lives in routing, not a residual dir.",
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
    logger.info(f"ROUTING MODE peak Cohen's d @ L{out['mode_axis']['peak_cohen_d']['layer']} "
                f"= {out['mode_axis']['peak_cohen_d']['cohen_d']} "
                f"(residual mode peak was ~0.42 for DSV2)")
    if temp_rows is not None:
        logger.info(f"ROUTING TEMP peak Cohen's d @ L{out['temp_axis']['peak_cohen_d']['layer']} "
                    f"= {out['temp_axis']['peak_cohen_d']['cohen_d']}")
    for r in mode_rows:
        logger.info(f"  L{r['layer']:2d} {r['depth_pct']:5.1f}%  mode_d={r['cohen_d']:+.3f}  cr={r['centroid_ratio']:.3f}")
    logger.info(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
