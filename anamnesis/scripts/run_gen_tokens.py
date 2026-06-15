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
    ap.add_argument("--label", default="w")
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
            messages = [
                {"role": "system", "content": spec["system_prompt"]},
                {"role": "user", "content": spec["user_prompt"]},
            ]
            result = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            input_ids = result if isinstance(result, torch.Tensor) else result["input_ids"]
            input_ids = input_ids.to("cuda")
            attention_mask = torch.ones_like(input_ids)
            prompt_length = int(input_ids.shape[1])

            torch.manual_seed(spec["seed"])
            torch.cuda.manual_seed_all(spec["seed"])
            np.random.seed(spec["seed"] % (2**32))

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
