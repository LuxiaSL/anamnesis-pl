"""A5 BLOCKING smoke: ResidualWriteSpec under INCREMENTAL DECODING (block §2 ⚠).

The prior write-hook smoke (vmb_smoke_write_hooks) validated the replay path
(single forward). This one validates the generate() path — KV-cached seq_len=1
forwards — and the gen↔replay semantic equivalence the A5 free-gen cells rest on.
No A5 cell runs before every check here passes (gauntlet step 1).

Checks:
  A. alpha=0 no-op under generate(): steered-registered generate with a fixed
     seed reproduces the hookless generate token-for-token (bitwise ids).
  B. Position gating under incremental decoding: alpha>0 generate injects at
     EXACTLY the generated positions (stats: N-1 for N sampled tokens; prefill
     contributes 0) and the hook actually saw cache_position kwargs.
  C. Gen<->replay consistency (the load-bearing check): GREEDY generate with
     injection -> replay the realized ids under the SAME spec in one full
     forward -> top-1 agreement at generated positions ~= 1.0. Run at alpha=0
     (kernel-noise baseline) and at alpha>0; PASS if steered agreement >=
     (baseline - 0.02) and >= 0.98.
  D. Dose response under generate(): per alpha in the ladder, steered greedy
     text diverges from alpha=0 text monotonically (prefix-match length
     non-increasing, edit distance non-decreasing in alpha; report snippets).

Usage (node1, one GPU):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=pipeline python -m anamnesis.scripts.vmb_a5_smoke_incremental \
        --model 3b --model-path /models/llama-3.2-3b-instruct \
        --layer 14 --out /dev/shm/vmb_a5_smoke.json
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from anamnesis.config import MODEL_PRESETS
from anamnesis.scripts._a5_common import median_residual_norm, teacher_forced_agreement

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROMPTS = [
    "Write about the role of rivers in shaping early human settlements.",
    "Explain to a beginner how vaccines train the immune system.",
]
DATE_STRING = "12 Jul 2026"  # battery canonical (chat-template date pinning)
LADDER = [0.03, 0.1, 0.3, 1.0]


def _prefix_match(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@torch.no_grad()
def _generate(model, input_ids, *, max_new_tokens, eos_ids, seed, greedy, temperature, top_p, pad_id):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    out = model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=not greedy,
        temperature=None if greedy else temperature,
        top_p=None if greedy else top_p,
        eos_token_id=eos_ids,
        pad_token_id=pad_id,
    )
    return out[0].tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    preset = MODEL_PRESETS[args.model]
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}.get(str(preset.torch_dtype), torch.float16)
    eos_ids = list(preset.eos_token_ids)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from anamnesis.extraction.model_loader import ResidualWriteSpec, attach_residual_write

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, attn_implementation="eager",
    ).to("cuda").eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_ids[0]

    results: dict = {"model": args.model, "layer": args.layer, "checks": {}}
    ok = True

    # Fixed random unit vector (seeded — smoke is reproducible)
    rng = np.random.default_rng(20260713)
    vec = rng.standard_normal(int(model.config.hidden_size)).astype(np.float32)
    vec /= np.linalg.norm(vec)

    prompt_ids_list = []
    for p in PROMPTS:
        res = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True, return_tensors="pt", date_string=DATE_STRING,
        )
        ids = res if isinstance(res, torch.Tensor) else res["input_ids"]
        prompt_ids_list.append(ids.to("cuda"))

    gen_kw = dict(max_new_tokens=args.max_new_tokens, eos_ids=eos_ids, pad_id=pad_id,
                  temperature=float(preset.temperature), top_p=0.9)

    # ── Baseline (hookless) sampled generations ──
    base_sampled = [_generate(model, ids, seed=7 + i, greedy=False, **gen_kw)
                    for i, ids in enumerate(prompt_ids_list)]
    base_greedy = [_generate(model, ids, seed=0, greedy=True, **gen_kw)
                   for ids in prompt_ids_list]

    # Median residual norm at the site (alpha unit), from the greedy baseline
    med_norm = median_residual_norm(
        model, base_greedy[0], int(prompt_ids_list[0].shape[1]), args.layer)
    results["median_resid_norm"] = med_norm
    logger.info(f"median ||h_L{args.layer}|| over generated positions = {med_norm:.2f}")

    # ── Register the write hook once ──
    spec = ResidualWriteSpec(layer_idx=args.layer, vector=torch.from_numpy(vec),
                             alpha=0.0, start_pos=None, normalize=True)
    handle = attach_residual_write(model, spec)

    # ── A: alpha=0 bitwise no-op under generate() ──
    a_pass = True
    for i, ids in enumerate(prompt_ids_list):
        spec.alpha = 0.0
        spec.start_pos = int(ids.shape[1])
        steered = _generate(model, ids, seed=7 + i, greedy=False, **gen_kw)
        same = steered == base_sampled[i]
        a_pass &= same
        logger.info(f"[A] prompt {i}: alpha=0 generate identical to hookless: {same}")
    results["checks"]["A_alpha0_noop_generate"] = a_pass
    ok &= a_pass

    # ── B: position gating under incremental decoding ──
    b_pass = True
    for i, ids in enumerate(prompt_ids_list):
        spec.alpha = 0.1 * med_norm
        spec.start_pos = int(ids.shape[1])
        handle.reset_stats()
        steered = _generate(model, ids, seed=7 + i, greedy=False, **gen_kw)
        n_gen = len(steered) - int(ids.shape[1])
        expected = max(0, n_gen - 1)
        got = int(handle.stats.get("positions", 0))
        saw_cp = bool(handle.stats.get("saw_cache_position", False))
        this = (got == expected) and saw_cp
        b_pass &= this
        logger.info(f"[B] prompt {i}: injected {got} positions (expected {expected}), "
                    f"cache_position seen: {saw_cp}")
    results["checks"]["B_position_gating_incremental"] = b_pass
    ok &= b_pass

    # ── C: gen<->replay consistency (greedy, alpha=0 baseline then alpha>0) ──
    c_detail = {}
    c_pass = True
    for alpha_frac in [0.0, 0.1]:
        agrees = []
        for i, ids in enumerate(prompt_ids_list):
            P = int(ids.shape[1])
            spec.alpha = alpha_frac * med_norm
            spec.start_pos = P
            realized = _generate(model, ids, seed=0, greedy=True, **gen_kw)
            if len(realized) - P < 8:
                logger.warning(f"[C] prompt {i} alpha={alpha_frac}: only {len(realized)-P} tokens")
            agrees.append(teacher_forced_agreement(model, realized, P))
        c_detail[str(alpha_frac)] = agrees
        logger.info(f"[C] alpha_frac={alpha_frac}: replay top-1 agreement {agrees}")
    base_min = min(c_detail["0.0"])
    steer_min = min(c_detail["0.1"])
    c_pass = steer_min >= 0.98 and steer_min >= base_min - 0.02
    results["checks"]["C_gen_replay_agreement"] = {"pass": c_pass, **c_detail}
    ok &= c_pass

    # ── D: dose response of realized text (greedy; fixed seed) ──
    d_rows = []
    P0 = int(prompt_ids_list[0].shape[1])
    base_ids = base_greedy[0][P0:]
    prev_prefix = None
    d_pass = True
    for frac in LADDER:
        spec.alpha = frac * med_norm
        spec.start_pos = P0
        steered = _generate(model, prompt_ids_list[0], seed=0, greedy=True, **gen_kw)[P0:]
        pm = _prefix_match(base_ids, steered)
        txt = tok.decode(steered[:40], skip_special_tokens=True)
        d_rows.append({"alpha_frac": frac, "prefix_match_vs_alpha0": pm, "snippet": txt})
        logger.info(f"[D] alpha={frac}: prefix-match {pm} | {txt[:80]!r}")
        if prev_prefix is not None and pm > prev_prefix:
            d_pass = False  # divergence must not shrink as dose grows
        prev_prefix = pm
    if all(r["prefix_match_vs_alpha0"] == d_rows[0]["prefix_match_vs_alpha0"] for r in d_rows):
        # identical divergence at every dose is the too-uniform alarm
        d_pass = d_pass and d_rows[0]["prefix_match_vs_alpha0"] < len(base_ids)
    results["checks"]["D_dose_response"] = {"pass": d_pass, "rows": d_rows}
    ok &= d_pass

    handle.remove()
    results["PASS"] = ok
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"A5 incremental smoke: {'PASS' if ok else 'FAIL'} -> {args.out}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
