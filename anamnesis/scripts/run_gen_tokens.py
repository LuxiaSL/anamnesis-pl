"""Phase-1 of corpus extension: GENERATE and bank realized tokens only (no extraction).

We extend the fat_01 corpus with more topics/reps. Generation produces the realized
token sequences; the canonical v3 signatures come from the validated REPLAY gateway
(phase 2: build manifest -> parallel_replay -> v3 raw + sigs). This worker therefore
only needs to sample tokens and bank them — no hooks, no calibration, no feature math.

It banks the FULL input_ids (prompt + generation) per gen, so the replay manifest is
built EXACTLY (no g_0 reconstruction / decode-fallback needed).

Spawned N-per-GPU by parallel_generate.py with CUDA_VISIBLE_DEVICES set. Resume-aware:
skips specs whose gen record already exists.

Usage (one worker):
    PYTHONPATH=. python -m anamnesis.scripts.run_gen_tokens \
        --model 8b --model-path /models/llama-3.1-8b-instruct \
        --spec-file /tmp/specs_w0.json --out-dir <run>/gen_records \
        --temperature 0.6 --top-p 0.9 --max-new-tokens 512 \
        --eos-ids 128001 128008 128009 --label w0
"""
from __future__ import annotations

import os

# Pin CPU thread pools BEFORE numpy/torch import: generation is GPU-bound, so 1 CPU
# thread/worker costs nothing but prevents the thread explosion (many workers ×
# core-count thread pools) that spikes load during the concurrent model-load phase.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

from anamnesis.config import MODEL_PRESETS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate + bank realized tokens (phase 1)")
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--spec-file", type=Path, required=True, help="JSON list of generation specs")
    ap.add_argument("--out-dir", type=Path, required=True, help="Where to write per-gen token records")
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--top-p", type=float, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--eos-ids", type=int, nargs="+", required=True)
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"],
                    help="Attention impl for generation (eager matches fat_01; phase-2 replay is "
                         "eager regardless, so this only affects which tokens get sampled)")
    ap.add_argument("--date-string", default=None,
                    help="Pin the chat template's Today Date (Llama templates render it via "
                         "strftime_now, so a midnight rollover mid-run would silently change "
                         "prompt tokens and break matched-history pairing). vmb battery "
                         "canonical date: '12 Jul 2026'. Default None = template default (today).")
    ap.add_argument("--label", default="w")
    ap.add_argument("--inject-npz", type=Path, default=None,
                    help="A5 steered generation: npz of unit vectors (banked by "
                         "vmb_a5_build_vectors); requires --inject-key/--inject-layer/--inject-alpha")
    ap.add_argument("--inject-key", default=None, help="Vector key inside --inject-npz (e.g. V1)")
    ap.add_argument("--inject-layer", type=int, default=None, help="Decoder layer index for injection")
    ap.add_argument("--inject-alpha", type=float, default=None,
                    help="ABSOLUTE injection magnitude (fraction × median residual norm, "
                         "precomputed by the chain submitter). 0.0 = rider no-op cell")
    ap.add_argument("--inject-alpha-frac", type=float, default=None,
                    help="Bookkeeping only: the ladder fraction this alpha encodes (recorded per gen)")
    args = ap.parse_args()

    preset = MODEL_PRESETS[args.model]
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}.get(str(preset.torch_dtype), torch.float16)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, attn_implementation=args.attn,
    ).to("cuda").eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else args.eos_ids[0]

    # ── A5 steered generation: register the write hook once; start_pos is mutated
    # per gen (prompt lengths vary; the hook reads it per call, gated on the
    # cache_position kwarg so prefill/incremental steps get identical semantics). ──
    write_handle = None
    inject_meta: dict | None = None
    if args.inject_npz is not None:
        if args.inject_key is None or args.inject_layer is None or args.inject_alpha is None:
            raise SystemExit("--inject-npz requires --inject-key, --inject-layer, --inject-alpha")
        from anamnesis.extraction.model_loader import ResidualWriteSpec, attach_residual_write

        vec_bank = np.load(args.inject_npz)
        if args.inject_key not in vec_bank:
            raise SystemExit(f"vector key {args.inject_key!r} not in {args.inject_npz} "
                             f"(has {list(vec_bank.keys())})")
        vec = torch.from_numpy(vec_bank[args.inject_key].astype(np.float32))
        spec = ResidualWriteSpec(
            layer_idx=args.inject_layer, vector=vec, alpha=args.inject_alpha,
            start_pos=None, normalize=True,
        )
        write_handle = attach_residual_write(model, spec)
        inject_meta = {
            "inject_npz": str(args.inject_npz), "inject_key": args.inject_key,
            "inject_layer": args.inject_layer, "inject_alpha": args.inject_alpha,
            "inject_alpha_frac": args.inject_alpha_frac,
        }
        logger.info(f"[{args.label}] steered generation: {inject_meta}")

    with open(args.spec_file) as f:
        specs = json.load(f)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    todo = [s for s in specs if not (args.out_dir / f"gen_{s['generation_id']:03d}.json").exists()]
    logger.info(f"[{args.label}] {len(todo)}/{len(specs)} specs to generate (attn={args.attn})")

    n_done = n_fail = 0
    t0 = time.time()
    for i, spec in enumerate(todo):
        try:
            gid = spec["generation_id"]
            # Bare specs (vmb Stage-0 floors) carry an empty system_prompt → omit the
            # system turn entirely rather than emit an empty system message.
            messages = [{"role": "user", "content": spec["user_prompt"]}]
            if spec.get("system_prompt"):
                messages.insert(0, {"role": "system", "content": spec["system_prompt"]})
            if tok.chat_template is None:
                # BASE models (vmb M4 OLMo-2-7B class): no chat template exists —
                # prompt is the raw concatenation, date-free by construction. A
                # system prompt would be silently ignored here, so refuse it.
                if spec.get("system_prompt"):
                    raise ValueError(
                        f"gen {gid}: system_prompt set but model has no chat "
                        "template — base models take bare user prompts only"
                    )
                result = tok(spec["user_prompt"], return_tensors="pt")["input_ids"]
            else:
                template_kwargs = {"date_string": args.date_string} if args.date_string else {}
                result = tok.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt", **template_kwargs
                )
            input_ids = result if isinstance(result, torch.Tensor) else result["input_ids"]
            input_ids = input_ids.to("cuda")
            attention_mask = torch.ones_like(input_ids)
            prompt_length = int(input_ids.shape[1])

            torch.manual_seed(spec["seed"])
            torch.cuda.manual_seed_all(spec["seed"])
            np.random.seed(spec["seed"] % (2**32))

            if write_handle is not None:
                # Inject at generated-token positions only (absolute ≥ prompt_length).
                write_handle.spec.start_pos = prompt_length
                write_handle.reset_stats()

            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    eos_token_id=args.eos_ids,
                    pad_token_id=pad_id,
                )
            full_seq = out[0].tolist()
            generated_ids = full_seq[prompt_length:]
            generated_text = tok.decode(generated_ids, skip_special_tokens=True)

            record = {
                "generation_id": gid,
                "prompt_set": spec["prompt_set"],
                "topic": spec["topic"],
                "topic_idx": spec["topic_idx"],
                "mode": spec["mode"],
                "mode_idx": spec["mode_idx"],
                "system_prompt": spec["system_prompt"],
                "user_prompt": spec["user_prompt"],
                "seed": spec["seed"],
                "repetition": spec["repetition"],
                "condition": "standard",
                "generated_text": generated_text,
                "num_generated_tokens": len(generated_ids),
                "prompt_length": prompt_length,
                "input_ids": full_seq,  # full prompt+generation → exact replay manifest
            }
            if inject_meta is not None:
                st = dict(write_handle.stats)
                # At alpha=0 the hook no-ops by design (bitwise rider); positions
                # only accumulate for alpha>0. Expected = one injection per
                # generated-token position processed (prefill contributes 0).
                if args.inject_alpha != 0.0:
                    # N sampled tokens → N-1 generated positions forwarded (the
                    # final token is sampled but never re-entered; prefill = 0).
                    expected = max(0, len(generated_ids) - 1)
                    got = int(st.get("positions", 0))
                    if not st.get("saw_cache_position", False):
                        raise RuntimeError(
                            f"gen {gid}: injection ran without cache_position gating — "
                            "position semantics unverifiable; aborting cell"
                        )
                    if got != expected:
                        raise RuntimeError(
                            f"gen {gid}: injected {got} positions, expected {expected} "
                            "(one per generated token) — position gating broken"
                        )
                record["injection"] = {**inject_meta, "positions_injected": int(st.get("positions", 0))}
            with open(args.out_dir / f"gen_{gid:03d}.json", "w") as f:
                json.dump(record, f)
            n_done += 1
            if (i + 1) % 20 == 0 or i == 0:
                el = time.time() - t0
                rate = (i + 1) / el if el else 0
                eta = (len(todo) - i - 1) / rate if rate else 0
                logger.info(f"[{args.label}] {i+1}/{len(todo)} gen_{gid:03d}: "
                            f"{len(generated_ids)} tok, {el:.0f}s ({rate:.2f}/s, ETA {eta:.0f}s)")
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            logger.error(f"[{args.label}] spec {spec.get('generation_id')} FAILED: {exc}", exc_info=True)

    logger.info(f"[{args.label}] done: {n_done} ok, {n_fail} failed in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
