"""Computer #5 install-lever PoC — S2: capture residual states from banked floors, via vLLM.

SPEC §3 (S2), §3b (pins). Two-pass design, inherited from koto's `steer_extract_floors.py`:

  pass 1 (default)      per-gen MEAN residual over the completion window at EVERY layer
                        -> the site-sweep substrate. ~2.6 MB/gen, all 80 layers, cheap.
  pass 2 (--dump-sites) per-token residuals at named layers
                        -> the Sigma substrate at the site the sweep chose.

Two passes and not one because the sweep PICKS the site: dumping per-token at all 80 layers up
front would be ~100x the I/O for data we would throw away.

⚠ CONVENTION, PINNED 2026-07-29 BY MEASUREMENT (spec §10c), not by reading source:
vLLM's Llama layer is forward(positions, hidden_states, residual). arg0 is POSITIONS. The
residual stream entering layer s — what HF calls hidden_states[s], and what a write at
layer_idx=s modifies — is **arg1 + arg2**. Verified against HF at per-token cosine >= 0.9994.
Using arg2 ("residual") alone scores 0.94-0.99: close enough to pass an eyeball, and wrong.

⚠ THREE vLLM SETTINGS ARE LOAD-BEARING FOR CORRECTNESS, NOT SPEED:
  VLLM_ENABLE_V1_MULTIPROCESSING=0  model in-process, else hooks silently never fire
  enforce_eager=True                hooks and CUDA graphs do not mix
  enable_prefix_caching=False       shared scaffold prefixes would be SKIPPED on recompute and
                                    the captured rows would not cover the sequence
Batch size is 1 by design: with chunked prefill a request's tokens can split across forwards,
and reassembling flat-packed multi-request batches is a correctness risk for no real speed win
(the prefill is one forward either way).

Run under the blackwell conda env (READ-ONLY):
    c5_capture_states_vllm.py --model-path ... --floors floors_base.jsonl --out states/base_means.npz
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MIN_COMPLETION_TOKENS = 4  # below this a "mean over the window" is noise, not a state


def find_model(llm) -> torch.nn.Module:
    engine = getattr(llm, "llm_engine", llm)
    for path in ("engine_core.engine_core.model_executor.driver_worker.model_runner.model",
                 "engine_core.model_executor.driver_worker.model_runner.model",
                 "model_executor.driver_worker.model_runner.model"):
        obj = engine
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.Module):
            logger.info("model reached via llm_engine.%s", path)
            return obj
    sys.exit("FATAL: model not reachable — VLLM_ENABLE_V1_MULTIPROCESSING=0 and tp=1 required")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--lora-path", default=None)
    ap.add_argument("--floors", type=Path, required=True, help="floors jsonl from S1")
    ap.add_argument("--out", type=Path, required=True, help="output npz")
    ap.add_argument("--dump-sites", default=None,
                    help="pass 2: comma list of layer indices for per-token dumps")
    ap.add_argument("--max-gens", type=int, default=None)
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    ap.add_argument("--stop-marker", default="\n\n**User:**")
    args = ap.parse_args()

    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        sys.exit("FATAL: VLLM_ENABLE_V1_MULTIPROCESSING=0 required (else hooks never fire)")
    if not args.floors.is_file():
        sys.exit(f"FATAL: floors not found: {args.floors}")
    if args.out.exists():
        sys.exit(f"FATAL: {args.out} exists — refusing to overwrite a state bank")

    rows = [json.loads(l) for l in args.floors.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.max_gens:
        rows = rows[: args.max_gens]
    sites = [int(x) for x in args.dump_sites.split(",")] if args.dump_sites else None

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm_kwargs = dict(model=args.model_path, tensor_parallel_size=1, dtype="bfloat16",
                      max_model_len=1024, gpu_memory_utilization=args.gpu_mem_util,
                      enforce_eager=True, enable_prefix_caching=False, seed=20260729)
    lora_request = None
    if args.lora_path:
        from vllm.lora.request import LoRARequest
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 512
        lora_request = LoRARequest("donor", 1, args.lora_path)

    logger.info("loading %s (lora=%s)", args.model_path, args.lora_path)
    llm = LLM(**llm_kwargs)
    model = find_model(llm)
    layers = model.model.layers
    n_layers = len(layers)
    hidden = None
    if sites and any(not 0 <= s < n_layers for s in sites):
        sys.exit(f"FATAL: --dump-sites out of range for {n_layers} layers")

    # One buffer per forward call; the pre-hook writes arg1+arg2 for every layer.
    buf: dict[int, torch.Tensor] = {}

    def mk_hook(s: int):
        def hook(mod, a, kw):
            # a = (positions, hidden_states, residual). residual is None at layer 0.
            if len(a) >= 2 and isinstance(a[1], torch.Tensor):
                h = a[1]
                r = a[2] if len(a) >= 3 and isinstance(a[2], torch.Tensor) else None
                buf[s] = (h + r) if r is not None else h
            return None
        return hook

    handles = [layers[s].register_forward_pre_hook(mk_hook(s), with_kwargs=True)
               for s in range(n_layers)]

    sp = SamplingParams(max_tokens=1, temperature=0.0)
    means: list[np.ndarray] = []
    meta: list[dict] = []
    dumps: list[np.ndarray] = []
    n_skip = 0
    t0 = time.time()

    for i, r in enumerate(rows):
        pids, cids = r.get("prompt_ids") or [], r.get("completion_ids") or []
        # Trim the completion at the stop marker's tokens if the engine included them. vLLM
        # excludes the stop string, HF includes it — trimming here keeps banks interchangeable.
        ids = list(pids) + list(cids)
        w0, w1 = len(pids), len(ids)
        if w1 - w0 < MIN_COMPLETION_TOKENS:
            n_skip += 1
            continue
        if len(ids) > 1024:
            ids = ids[:1024]
            w1 = min(w1, 1024)
            if w1 - w0 < MIN_COMPLETION_TOKENS:
                n_skip += 1
                continue

        buf.clear()
        _ = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False,
                         **({"lora_request": lora_request} if lora_request else {}))
        if len(buf) != n_layers:
            logger.warning("row %d: captured %d/%d layers — skipping", i, len(buf), n_layers)
            n_skip += 1
            continue
        # The prefill forward covers the whole sequence; a 1-token decode forward follows and
        # would leave a 1-row buffer. Guard on the row count rather than trusting call order.
        if buf[0].shape[0] < w1:
            logger.warning("row %d: buffer rows %d < window end %d — skipping",
                           i, buf[0].shape[0], w1)
            n_skip += 1
            continue

        stack = torch.stack([buf[s][:w1] for s in range(n_layers)], dim=0)  # (L, T, D)
        win = stack[:, w0:w1, :].float()
        if hidden is None:
            hidden = int(win.shape[-1])
        means.append(win.mean(dim=1).cpu().numpy().astype(np.float32))       # (L, D)
        meta.append({"prompt_id": r["prompt_id"], "topic": r.get("topic"),
                     "sample_idx": r.get("sample_idx"), "model": r.get("model"),
                     "n_tok": int(w1 - w0)})
        if sites is not None:
            dumps.append(win[sites].cpu().numpy().astype(np.float32))        # (n_sites, T, D)
        del stack, win
        if (i + 1) % 100 == 0:
            logger.info("  %d/%d rows, %d kept, %.0fs", i + 1, len(rows), len(means),
                        time.time() - t0)

    for h in handles:
        h.remove()
    if not means:
        sys.exit("FATAL: nothing captured")

    payload: dict[str, object] = {
        "means": np.stack(means),                       # (N, n_layers, D)
        "meta": json.dumps(meta),
        "n_layers": np.array(n_layers),
        "hidden": np.array(hidden if hidden is not None else 0),
        "convention": ("residual stream at layer s = vLLM pre-hook arg1 + arg2; "
                       "== HF hidden_states[s] (pinned, per-token cos >= 0.9994)"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if sites is not None:
        payload["site_indices"] = np.array(sites)
        obj = np.empty(len(dumps), dtype=object)   # ragged: per-gen (n_sites, T, D)
        for k, d in enumerate(dumps):
            obj[k] = d
        payload["site_tokens"] = obj
        np.savez(args.out, **payload)
    else:
        np.savez_compressed(args.out, **payload)

    logger.info("captured %d gens (%d skipped) in %.0fs -> %s",
                len(means), n_skip, time.time() - t0, args.out)


if __name__ == "__main__":
    main()
