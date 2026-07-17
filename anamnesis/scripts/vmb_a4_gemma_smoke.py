"""Gemma-A4 sliding-window cache-surgery SMOKE (loose-ends Part 3).

Executes `outputs/battery/arms/A4/GEMMA-A4-SLIDING-SMOKE-SPEC-2026-07-16.md` exactly.
Gemma-3-27B interleaves 5 sliding (window=1024) : 1 global layer; the battery's
`cache_surgery.KVSnapshot` assumes UNIFORM full-attention and applies one keep-index
set to all layers. This smoke DISCOVERS the anatomy (never assumes it) and returns a
PROCEED / STOP-AND-SURFACE verdict:

  1. Capture — load path + cache type + per-layer key seq lengths for a sliding vs a
     global layer (padded 1024 vs true ~C?). Classify via config.layer_types (or the
     (i+1)%6==0 global rule as fallback).
  2. from_hf_cache round-trip — does the snapshot build (positions length C == every
     layer's key seq len)? A padded sliding layer trips __post_init__ → failure mode 2.
  3. Faithfulness — feed the continuation tokens through the reconstructed full cache
     and compare next-token logits to a native single forward. C+k << 1024, so sliding
     and full attention are semantically identical here; a correct reconstruction must
     match bitwise-ish (max|Δ| ~ 0). Divergence ⇒ the sliding layers lost information.
  4. Surgery smoke — naive/rotate/recompute each build + forward without shape error.

Verdict: PROCEED only if per-layer lengths all == C AND faithfulness max|Δ| <= tol.
Otherwise STOP-AND-SURFACE (that anatomy report IS the Part-3 result; then transplant
kv-rotation's per-layer `layer_types`/`applies_rope` KVSnapshot before any A4 cell).

    python -m anamnesis.scripts.vmb_a4_gemma_smoke \
        --model gemma3-27b --model-path /models/.../gemma-3-27b-it \
        --floor-run-dir /models/.../runs/vmb_stage0_gemma3_27b \
        --gen-id 0 --out /models/.../battery/arms/A4_gemma/smoke_verdict.json

If --floor-run-dir/--gen-id are omitted, a synthetic ~300-token context is used (the
length-mismatch anatomy depends only on seq length vs the 1024 window, so synthetic is
representative for checks 1-2; use a real gen for the faithfulness leg when available).
GPU job; eager attention (A4 standard). Load path rides the session-10 Gemma wrapper fix.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUM_SINKS = 4
RECENT_PROTECT = 32


def _layer_classes(config, n_layers: int) -> tuple[list[str], str]:
    """Return per-layer class strings ('sliding'/'full') and how they were derived."""
    for holder in (config, getattr(config, "text_config", None)):
        if holder is None:
            continue
        lt = getattr(holder, "layer_types", None)
        if lt is not None and len(lt) == n_layers:
            return ["sliding" if "sliding" in str(x).lower() else "full" for x in lt], "config.layer_types"
    # Fallback: Gemma-3 rule — global at (i+1) % 6 == 0, else sliding.
    return ["full" if (i + 1) % 6 == 0 else "sliding" for i in range(n_layers)], "(i+1)%6==0 rule"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3-27b")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--floor-run-dir", type=Path, default=None)
    ap.add_argument("--gen-id", type=int, default=None)
    ap.add_argument("--context-tokens", type=int, default=300,
                    help="synthetic context length if no floor gen given")
    ap.add_argument("--k-faith", type=int, default=8, help="continuation tokens for faithfulness leg")
    ap.add_argument("--evict-frac", type=float, default=0.5)
    ap.add_argument("--tol", type=float, default=1e-2, help="max|Δlogit| PROCEED threshold")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch

    from anamnesis.config import MODEL_PRESETS, ModelConfig
    from anamnesis.extraction.cache_surgery import (
        _extract_kv, evict, from_hf_cache, middle_region_keep, operative_inv_freq,
        reindex, to_hf_dynamic_cache,
    )
    from anamnesis.extraction.model_loader import load_model

    report: dict = {"model": args.model, "spec": "GEMMA-A4-SLIDING-SMOKE-SPEC-2026-07-16",
                    "checks": {}, "verdict": None, "stop_reasons": []}

    preset = MODEL_PRESETS[args.model]
    n_layers = preset.num_layers
    all_layers = list(range(n_layers))
    model_config = ModelConfig(
        model_id=args.model_path, torch_dtype=preset.torch_dtype,
        num_layers=n_layers, hidden_dim=preset.hidden_dim,
        num_attention_heads=preset.num_attention_heads, num_kv_heads=preset.num_kv_heads,
        head_dim=preset.head_dim,
    )
    # ── Check 0: load path (rides the Gemma-A5 wrapper fix) ────────────────────────
    try:
        loaded = load_model(
            model_config, sampled_layers=preset.sampled_layers, register_gate_hooks=True,
            key_layers=all_layers, value_layers=all_layers,
            query_layers=all_layers, attn_output_layers=all_layers,
        )
        device = next(loaded.model.parameters()).device
        report["checks"]["load"] = {"ok": True, "device": str(device),
                                    "model_class": type(loaded.model).__name__}
    except Exception as exc:  # noqa: BLE001
        report["checks"]["load"] = {"ok": False, "error": repr(exc)}
        report["verdict"] = "STOP-AND-SURFACE"
        report["stop_reasons"].append(f"load failed: {exc!r}")
        args.out.write_text(json.dumps(report, indent=2))
        logger.error(f"LOAD FAILED — {exc!r}"); return

    # 14e RoPE gate (also validates live inv_freq access on Gemma).
    try:
        inv_freq = operative_inv_freq(loaded.model).to(device)
        report["checks"]["rope_gate"] = {"ok": True, "inv_freq_len": int(inv_freq.shape[0])}
    except Exception as exc:  # noqa: BLE001
        report["checks"]["rope_gate"] = {"ok": False, "error": repr(exc)}
        report["stop_reasons"].append(f"operative_inv_freq raised: {exc!r}")
        inv_freq = None

    cfg = loaded.model.config
    classes, how = _layer_classes(cfg, n_layers)
    sliding_idx = next((i for i, c in enumerate(classes) if c == "sliding"), None)
    global_idx = next((i for i, c in enumerate(classes) if c == "full"), None)
    report["checks"]["layer_classes"] = {
        "derived_via": how, "n_sliding": classes.count("sliding"),
        "n_full": classes.count("full"), "first_sliding": sliding_idx, "first_global": global_idx,
        "sliding_window": getattr(getattr(cfg, "text_config", cfg), "sliding_window", None)}

    # ── Build a context of ~C tokens ───────────────────────────────────────────────
    tok = loaded.tokenizer
    context_ids = cont_ids = None
    if args.floor_run_dir is not None and args.gen_id is not None:
        man = args.floor_run_dir / "replay_manifest.json"
        try:
            entries = json.loads(man.read_text())["entries"]
            e = entries[str(args.gen_id)]
            ids = list(e["input_ids"]); P = int(e["prompt_length"])
            n_gen = len(ids) - P; mid = n_gen // 2
            C = P + mid
            context_ids, cont_ids = ids[:C], ids[C:]
            report["checks"]["context"] = {"source": "floor_gen", "gen_id": args.gen_id, "C": C}
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"floor manifest read failed ({exc!r}); synthetic context")
    if context_ids is None:
        seed = ("The system processes each request in sequence, weighing the evidence and "
                "questioning every assumption before moving forward. ")
        ids = tok(seed * 40, return_tensors=None)["input_ids"][: args.context_tokens]
        context_ids = ids
        cont_ids = tok(" Consider next the following related point in detail.",
                       return_tensors=None)["input_ids"]
        report["checks"]["context"] = {"source": "synthetic", "C": len(context_ids)}
    C = len(context_ids)

    @torch.no_grad()
    def prefill(ids_list):
        ids = torch.tensor([ids_list], dtype=torch.long, device=device)
        loaded.disable_hooks()
        out = loaded.model(ids, use_cache=True, return_dict=True)
        loaded.enable_hooks(); loaded.clear_hook_state()
        return out.past_key_values

    # ── Check 1: capture + per-layer key seq lengths ───────────────────────────────
    base_cache = prefill(context_ids)
    report["checks"]["capture"] = {"cache_type": type(base_cache).__name__}
    try:
        keys, values = _extract_kv(base_cache)
        seqlens = [int(k.shape[-2]) for k in keys]
        sl = seqlens[sliding_idx] if sliding_idx is not None else None
        gl = seqlens[global_idx] if global_idx is not None else None
        uniform = len(set(seqlens)) == 1 and seqlens[0] == C
        report["checks"]["capture"].update({
            "ok": True, "n_layers_captured": len(keys), "context_C": C,
            "sliding_layer_seqlen": sl, "global_layer_seqlen": gl,
            "all_equal_C": bool(uniform),
            "unique_seqlens": sorted(set(seqlens))})
        if not uniform:
            report["stop_reasons"].append(
                f"per-layer key seq lengths not all == C ({C}): unique={sorted(set(seqlens))} "
                f"(sliding={sl}, global={gl}) — padded sliding buffers / mixed lengths")
    except Exception as exc:  # noqa: BLE001
        report["checks"]["capture"].update({"ok": False, "error": repr(exc)})
        report["stop_reasons"].append(f"_extract_kv failed: {exc!r}")

    # ── Check 2: from_hf_cache round-trip ──────────────────────────────────────────
    snapshot = None
    try:
        snapshot = from_hf_cache(base_cache, positions=torch.arange(C, device=device))
        report["checks"]["snapshot"] = {"ok": True, "num_layers": snapshot.num_layers,
                                        "seq_len0": snapshot.seq_len(0)}
    except Exception as exc:  # noqa: BLE001
        report["checks"]["snapshot"] = {"ok": False, "error": repr(exc)}
        report["stop_reasons"].append(f"from_hf_cache raised (mode-2 length mismatch): {exc!r}")

    # ── Check 3: faithfulness (reconstructed full cache vs native forward) ──────────
    if snapshot is not None and cont_ids:
        try:
            k = min(args.k_faith, len(cont_ids))
            cont = cont_ids[:k]
            native = loaded.model(
                torch.tensor([context_ids + cont], device=device), use_cache=False,
                return_dict=True).logits[0, C - 1: C - 1 + k].float()
            cache = to_hf_dynamic_cache(snapshot)
            pos = torch.arange(C, C + k, device=device).unsqueeze(0)
            cached = loaded.model(
                torch.tensor([cont], device=device), past_key_values=cache,
                position_ids=pos, cache_position=torch.arange(C, C + k, device=device),
                use_cache=True, return_dict=True).logits[0].float()
            maxdiff = float((native - cached).abs().max().item())
            argmatch = float((native.argmax(-1) == cached.argmax(-1)).float().mean().item())
            report["checks"]["faithfulness"] = {
                "ok": True, "k": k, "max_abs_logit_diff": maxdiff,
                "argmax_agreement": argmatch, "tol": args.tol}
            if maxdiff > args.tol:
                report["stop_reasons"].append(
                    f"faithfulness max|Δlogit|={maxdiff:.4g} > tol={args.tol} — "
                    f"reconstructed cache diverges from native forward")
        except Exception as exc:  # noqa: BLE001
            report["checks"]["faithfulness"] = {"ok": False, "error": repr(exc)}
            report["stop_reasons"].append(f"faithfulness leg raised: {exc!r}")

    # ── Check 4: surgery kinds build + forward without shape error ──────────────────
    if snapshot is not None and inv_freq is not None:
        surg = {}
        try:
            keep = middle_region_keep(C, args.evict_frac, num_sinks=NUM_SINKS,
                                      recent_protect=RECENT_PROTECT).to(device)
            s_prime = int(keep.shape[0])
            builders = {
                "naive": lambda: (evict(snapshot, keep), C),
                "rotate": lambda: (reindex(evict(snapshot, keep),
                                           torch.arange(s_prime, device=device), inv_freq), s_prime),
                "recompute": lambda: (from_hf_cache(
                    prefill([context_ids[j] for j in keep.cpu().tolist()]),
                    positions=torch.arange(s_prime, device=device)), s_prime),
            }
            probe = cont_ids[0] if cont_ids else context_ids[-1]
            for kind, build in builders.items():
                try:
                    snap, offset = build()
                    cache = to_hf_dynamic_cache(snap)
                    out = loaded.model(
                        torch.tensor([[probe]], device=device), past_key_values=cache,
                        position_ids=torch.tensor([[offset]], device=device),
                        cache_position=torch.tensor([offset], device=device),
                        use_cache=True, return_dict=True)
                    surg[kind] = {"ok": True, "s_prime": offset,
                                  "next_argmax": int(out.logits[0, -1].argmax().item())}
                except Exception as exc:  # noqa: BLE001
                    surg[kind] = {"ok": False, "error": repr(exc)}
                    report["stop_reasons"].append(f"{kind} surgery raised: {exc!r}")
            report["checks"]["surgery"] = surg
        except Exception as exc:  # noqa: BLE001
            report["checks"]["surgery"] = {"ok": False, "error": repr(exc)}
            report["stop_reasons"].append(f"surgery setup raised: {exc!r}")

    report["verdict"] = "PROCEED" if not report["stop_reasons"] else "STOP-AND-SURFACE"
    args.out.write_text(json.dumps(report, indent=2))
    logger.info(f"VERDICT: {report['verdict']}")
    for r in report["stop_reasons"]:
        logger.warning(f"  stop: {r}")
    logger.info(f"banked -> {args.out}")


if __name__ == "__main__":
    main()
