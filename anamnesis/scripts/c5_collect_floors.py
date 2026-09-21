"""Computer #5 install-lever PoC — S1: collect floor generations from a donor or the host.

SPEC: research/planning/SPEC-computer5-install-lever-poc-2026-07-28.md §3 (S1) + §6 (G2).

Generates M samples per floor prompt from ONE model and banks token IDs + text as jsonl.
Identical prompts go to every model in the roster — that is the whole point, the residual
difference is then model, not content.

WHY HF AND NOT vLLM: the generation engine does not need to match anything. Stack consistency
is required between CAPTURE and INJECTION (spec §3b.2), both of which are HF here. Generation
only has to produce text and token IDs. vLLM is not in this venv anyway.

BANKED PER ROW: prompt_ids, completion_ids (RAW, untrimmed), text, and the provenance needed to
pair rows across models. Trimming at the stop marker happens at STATE-EXTRACTION time, not here
(koto's design — the raw bank stays re-interpretable).

Empty completions are VALID ROWS, not errors. Computer's README: an empty completion is the
model passing the floor. Koto lost two servers to a retry-storm from treating them as failures.
They are banked and counted.

    python c5_collect_floors.py --model-path /models/Llama-3.1-70B --label base \
        --prompts prompts/floor_prompts.json --out-dir /mnt/beegfs/luxia/computer5/floors
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Computer's native sampling (README, load-bearing — not the HF generation_config defaults,
# which say 0.6/0.9 and are wrong for this model family).
TEMPERATURE = 1.0
TOP_P = 0.98
STOP_SEQUENCE = "\n\n**User:**"


def load_model(model_path: str, adapter_path: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("loading %s (adapter=%s)", model_path, adapter_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        # sdpa, NOT eager: this construction reads residuals only — no attention weights, no
        # gates, no pre-RoPE keys (spec §3b). Eager at 70B/64-heads would be the dominant cost
        # and would buy nothing.
        attn_implementation="sdpa",
        device_map="auto",
    )
    if adapter_path:
        from peft import PeftModel
        logger.info("merging adapter %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    # Both Computer checkpoints ship `use_cache: false` in config.json; without this override
    # generation runs without a KV cache and crawls.
    model.config.use_cache = True
    model.eval()

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok


@torch.no_grad()
def generate_for_prompt(model, tok, prompt_text: str, n_samples: int, max_new_tokens: int,
                        seed: int) -> list[tuple[list[int], str]]:
    """-> [(completion_ids, completion_text)] * n_samples. Batch = the n samples of one prompt."""
    enc = tok(prompt_text, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(model.device)
    prompt_len = ids.shape[1]
    batch = ids.repeat(n_samples, 1)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    out = model.generate(
        batch,
        attention_mask=torch.ones_like(batch),
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        pad_token_id=tok.pad_token_id,
        # Native HF stop-string support: halts the sequence at Computer's turn boundary. The
        # stop text itself may be included; trimming is the extractor's job, not ours.
        stop_strings=[STOP_SEQUENCE],
        tokenizer=tok,
    )
    rows = []
    for i in range(out.shape[0]):
        comp = out[i, prompt_len:].tolist()
        # strip trailing pad only (an all-pad row is a legitimate empty completion)
        while comp and comp[-1] == tok.pad_token_id:
            comp.pop()
        rows.append((comp, tok.decode(comp, skip_special_tokens=True)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--adapter-path", default=None, help="LoRA donor (merged before generation)")
    ap.add_argument("--label", required=True, help="roster label, e.g. base / c5 / c2 / lora64")
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-samples", type=int, default=4, help="M samples per prompt (= batch size)")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--arm", default=None,
                    help="restrict to one prompt arm. G2 VERDICT 2026-07-29: use 'substantive'. "
                         "The naive ('hello') arm gave base a median of 5 completion tokens — "
                         "koto's aphasia failure reproduced, and a Sigma substrate far too thin.")
    ap.add_argument("--limit", type=int, default=None, help="smoke-test: only first N prompts")
    args = ap.parse_args()

    if not args.prompts.is_file():
        sys.exit(f"FATAL: prompt set not found: {args.prompts}")
    ps = json.loads(args.prompts.read_text(encoding="utf-8"))
    prompts = ps["prompts"]
    if args.arm:
        prompts = [p for p in prompts if p["arm"] == args.arm]
        if not prompts:
            sys.exit(f"FATAL: no prompts with arm={args.arm!r}")
    if args.limit:
        prompts = prompts[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"floors_{args.label}.jsonl"
    if out_path.exists():
        sys.exit(f"FATAL: {out_path} exists — refusing to overwrite a bank. Move it or relabel.")

    model, tok = load_model(args.model_path, args.adapter_path)

    n_rows = n_empty = 0
    by_arm: dict[str, list[int]] = {}
    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as fh:
        for pi, p in enumerate(prompts):
            enc = tok(p["prompt_text"], return_tensors="pt", add_special_tokens=True)
            prompt_ids = enc["input_ids"][0].tolist()
            # Seed is a deterministic function of (prompt index, run seed) so a re-run of one
            # model reproduces, and different models see the SAME seeds on the same prompt.
            rows = generate_for_prompt(model, tok, p["prompt_text"], args.n_samples,
                                       args.max_new_tokens, args.seed + pi * 1000)
            for si, (comp_ids, comp_text) in enumerate(rows):
                fh.write(json.dumps({
                    "model": args.label, "prompt_id": p["prompt_id"], "arm": p["arm"],
                    "topic": p["topic"], "sample_idx": si, "seed": args.seed + pi * 1000,
                    "prompt_ids": prompt_ids, "completion_ids": comp_ids,
                    "text": comp_text, "n_completion_tokens": len(comp_ids),
                }) + "\n")
                n_rows += 1
                n_empty += int(len(comp_ids) == 0)
                by_arm.setdefault(p["arm"], []).append(len(comp_ids))
            if (pi + 1) % 10 == 0:
                logger.info("  %d/%d prompts, %d rows, %.0fs", pi + 1, len(prompts), n_rows,
                            time.time() - t0)

    # G2 floor characterization: this is the read that PICKS THE ARM (spec §10). Koto's failure
    # mode was a base arm so terse the Sigma substrate was undersized — that shows up here.
    import statistics
    stats = {
        "model": args.label, "n_rows": n_rows, "n_empty": n_empty,
        "elapsed_s": round(time.time() - t0, 1),
        "by_arm": {
            arm: {
                "n": len(lens),
                "median_tokens": statistics.median(lens),
                "mean_tokens": round(statistics.fmean(lens), 1),
                "total_tokens": sum(lens),
                "n_empty": sum(1 for x in lens if x == 0),
            } for arm, lens in sorted(by_arm.items())
        },
    }
    (args.out_dir / f"floorstats_{args.label}.json").write_text(json.dumps(stats, indent=1))
    logger.info("FLOOR STATS %s", json.dumps(stats["by_arm"]))
    logger.info("wrote %d rows (%d empty) -> %s", n_rows, n_empty, out_path)


if __name__ == "__main__":
    main()
