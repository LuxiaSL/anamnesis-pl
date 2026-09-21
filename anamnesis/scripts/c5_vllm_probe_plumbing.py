"""Computer #5 PoC — plumbing probe: can we reach vLLM's model in-process and hook it?

SPEC: research/planning/SPEC-computer5-install-lever-poc-2026-07-28.md §3b (the five things that
must be pinned before a vLLM-built vector is trustworthy).

This answers three questions and nothing else. It is deliberately cheap and throwaway-able.

  Q1 REACHABILITY — with V1 multiprocessing disabled and tensor_parallel_size=1, is the model
     object addressable from the driver process, so plain nn.Module hooks work? (70B bf16 is
     ~132 GiB and a B200 is 179 GiB, so TP=1 fits — which is what buys us in-process access.)
  Q2 LAYER-INDEX CONVENTION — the load-bearing one. Anamnesis/koto define hidden_states[s] as the
     INPUT to decoder layer s, which is exactly what a write at layer_idx=s modifies. A forward
     hook on layers[s] sees that same input as args[0]. If we build the vector from one tensor
     and inject into another, the vector is silently one layer off and nothing downstream will
     say so. Verified here by capturing BOTH the pre-hook input and the post-hook output.
  Q3 NUMERIC AGREEMENT WITH HF — teacher-force one identical token sequence through vLLM and
     through HF, compare residuals at the same site. This is the empirical pin: we do not reason
     about conventions, we measure them.

Run under the blackwell conda env (READ-ONLY; never install into shared conda envs):
    FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      /home/cluster-user/miniconda/envs/blackwell/bin/python c5_vllm_probe_plumbing.py \
      --model-path /models/Llama-3.1-70B --sites 32 40 48
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_model(llm) -> torch.nn.Module:
    """Walk vLLM's executor down to the nn.Module. Paths move between versions, so try several
    and FAIL LOUDLY with what was actually there rather than guessing wrong."""
    engine = getattr(llm, "llm_engine", llm)
    candidates = [
        "engine_core.engine_core.model_executor.driver_worker.model_runner.model",
        "engine_core.model_executor.driver_worker.model_runner.model",
        "model_executor.driver_worker.model_runner.model",
        "engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model",
    ]
    for path in candidates:
        obj = engine
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.Module):
            logger.info("MODEL REACHED via llm_engine.%s -> %s", path, type(obj).__name__)
            return obj
    # Diagnostic dump so the next attempt is informed, not another guess.
    logger.error("could not reach model. llm_engine attrs: %s",
                 [a for a in dir(engine) if not a.startswith("_")][:40])
    sys.exit("FATAL: model not reachable in-process — is VLLM_ENABLE_V1_MULTIPROCESSING=0 set "
             "and tensor_parallel_size=1?")


def get_layers(model: torch.nn.Module):
    for path in ("model.layers", "layers", "model.model.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if hasattr(obj, "__len__") and len(obj) > 0:
            logger.info("LAYERS via %s: n=%d", path, len(obj))
            return obj
    sys.exit("FATAL: decoder layers not found on the vLLM model")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--sites", type=int, nargs="+", default=[32, 40, 48])
    ap.add_argument("--text", default="As follows is a conversation between another user and "
                                      "Model C about clouds.\n\nFull conversation with Model C:"
                                      "\n\n**User:** i feel like crying. how is a cloud like a "
                                      "thought?\n\n**Model C:** A cloud is a kind of thought that "
                                      "has been allowed to keep its shape.")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    args = ap.parse_args()

    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        sys.exit("FATAL: set VLLM_ENABLE_V1_MULTIPROCESSING=0 — otherwise the model lives in a "
                 "separate engine-core process and in-process hooks silently never fire.")

    from vllm import LLM, SamplingParams

    logger.info("loading %s tp=1", args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=1, dtype="bfloat16",
              max_model_len=768, gpu_memory_utilization=args.gpu_mem_util,
              enforce_eager=True,  # hooks + cuda graphs do not mix; eager for capture
              seed=20260729)

    model = find_model(llm)
    layers = get_layers(model)
    n_layers = len(layers)

    # ── Q2: capture BOTH the input to layer s (pre-hook) and its output (post-hook) ──
    caught: dict[str, torch.Tensor] = {}

    def mk_pre(s: int):
        def hook(mod, args_, kwargs_):
            # vLLM decoder layers take (positions, hidden_states) or (hidden_states, residual)
            # depending on architecture; record every tensor arg so the shape story is explicit.
            tensors = [a for a in args_ if isinstance(a, torch.Tensor)]
            for j, t in enumerate(tensors):
                caught[f"pre_L{s}_arg{j}"] = t.detach().float().cpu()
            return None
        return hook

    def mk_post(s: int):
        def hook(mod, args_, out):
            outs = out if isinstance(out, tuple) else (out,)
            for j, t in enumerate(outs):
                if isinstance(t, torch.Tensor):
                    caught[f"post_L{s}_out{j}"] = t.detach().float().cpu()
            return None
        return hook

    handles = []
    for s in args.sites:
        if not 0 <= s < n_layers:
            sys.exit(f"FATAL: site {s} out of range for {n_layers} layers")
        handles.append(layers[s].register_forward_pre_hook(mk_pre(s), with_kwargs=True))
        handles.append(layers[s].register_forward_hook(mk_post(s)))

    # Teacher-forced single request: max_tokens=1 so essentially only prefill runs, and the
    # captured rows are exactly this sequence's tokens in order (vLLM packs tokens flat, no pad).
    out = llm.generate([args.text], SamplingParams(max_tokens=1, temperature=0.0), use_tqdm=False)
    prompt_ids = list(out[0].prompt_token_ids)
    for h in handles:
        h.remove()

    report: dict[str, object] = {
        "model_path": args.model_path, "n_layers": n_layers, "sites": args.sites,
        "n_prompt_tokens": len(prompt_ids),
        "model_class": type(model).__name__,
        "captured_keys": sorted(caught),
        "shapes": {k: list(v.shape) for k, v in sorted(caught.items())},
    }

    # ── Q3 substrate: dump the residual rows so a separate HF run can be compared offline. ──
    npz_path = args.out_json.with_suffix(".npz")
    dump = {}
    # ⚠ PINNED 2026-07-29: vLLM's Llama decoder layer is forward(positions, hidden_states,
    # residual) — arg0 is POSITIONS (a 1-D int tensor), NOT a residual. vLLM keeps the residual
    # stream SPLIT across (hidden_states, residual) so input_layernorm can fuse the add:
    #     hidden, residual = input_layernorm(hidden, residual)   # returns (norm(h+r), h+r)
    # So the residual stream entering layer s — the thing HF calls hidden_states[s], and the
    # thing a write at layer_idx=s must modify — is arg1 + arg2. Capturing arg0 would have built
    # the vector out of position indices, which is exactly the silent failure §3b.1 warns about.
    for s in args.sites:
        h, r = caught.get(f"pre_L{s}_arg1"), caught.get(f"pre_L{s}_arg2")
        if h is None:
            continue
        hn = h.numpy()
        rn = r.numpy() if r is not None else np.zeros_like(hn)
        for a in (hn, rn):
            if a.ndim == 3 and a.shape[0] == 1:
                a = a[0]
        resid = (hn + rn).astype(np.float32)
        dump[f"vllm_resid_L{s}"] = resid
        dump[f"vllm_hidden_L{s}"] = hn.astype(np.float32)
        dump[f"vllm_residpart_L{s}"] = rn.astype(np.float32)
        report.setdefault("norms", {})[f"resid_L{s}"] = {
            "mean_token_norm": float(np.linalg.norm(resid, axis=-1).mean()),
            "median_token_norm": float(np.median(np.linalg.norm(resid, axis=-1))),
            "mean_norm_hidden_only": float(np.linalg.norm(hn, axis=-1).mean()),
            "mean_norm_residual_only": float(np.linalg.norm(rn, axis=-1).mean()),
        }
    dump["prompt_ids"] = np.array(prompt_ids, dtype=np.int64)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **dump)
    report["npz"] = str(npz_path)
    args.out_json.write_text(json.dumps(report, indent=1))
    logger.info("REPORT %s", json.dumps({k: report[k] for k in
                                         ("n_layers", "n_prompt_tokens", "model_class", "shapes")},
                                        indent=1))
    logger.info("wrote %s and %s", args.out_json, npz_path)


if __name__ == "__main__":
    main()
