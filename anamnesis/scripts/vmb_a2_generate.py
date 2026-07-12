"""Arm A2 (instruction-vs-execution) corpus generation — prereg-vmb-v1 §2c A2, cell (i).

Builds per model:
  - 5 PURE-mode corpora (run4 modes: linear/analogical/socratic/contrastive/dialectical),
    "Write about: {topic}" over the Phase-0 20 topics, k seeds each — the reference
    conditions AND their own within-condition floors (matched history incl. system prompt,
    addendum 12a item 2).
  - 3 SWAP corpora (PROMPT_SWAP_PAIRS: socratic→linear, dialectical→contrastive,
    analogical→linear): system prompt from mode A, user directive forcing execution B.

A2 runs at 4× law (ratified §2c). Seed namespaces: VMBA2-{MODEL}-{COND}. gid layout:
gid = cond_idx * (20 * k) + topic_idx * k + seed_idx over the fixed condition order.

Usage (per model):
    PYTHONPATH=pipeline python -m anamnesis.scripts.vmb_a2_generate \
        --model 3b --model-path /models/llama-3.2-3b-instruct \
        --out-root /models/anamnesis-extract/runs --seeds-per-topic 8 \
        --gpus 0,1,2,3 --workers-per-gpu 4
"""
from __future__ import annotations

import argparse

from anamnesis.scripts._gpu import resolve_physical_gpus
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from anamnesis.config import MODEL_PRESETS
from anamnesis.extraction.generation_runner import make_seed
from anamnesis.modes.run4_modes import RUN4_MODES
from anamnesis.modes.prompt_swap import PROMPT_SWAP_PAIRS
from anamnesis.scripts.vmb_stage0_generate import VMB_CANONICAL_DATE, assemble

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TEMPLATE = "Write about: {topic}"

#: Fixed condition order (frozen — feeds gids and seed namespaces).
def conditions() -> list[tuple[str, str, str]]:
    """[(cond_name, system_prompt, user_prompt_template_kind)] — pure modes then swaps."""
    conds: list[tuple[str, str, str]] = []
    for mode in sorted(RUN4_MODES):
        conds.append((f"pure_{mode}", RUN4_MODES[mode], "pure"))
    for pair in PROMPT_SWAP_PAIRS:
        conds.append((f"swap_{pair.label.replace('→', '_to_')}", pair.get_system_prompt(), pair.label))
    return conds


def build_specs(model: str, topics: list[str], k: int) -> dict[str, list[dict]]:
    """cond_name → specs. Each condition is its own run dir (own metadata/manifest)."""
    by_pair = {p.label: p for p in PROMPT_SWAP_PAIRS}
    out: dict[str, list[dict]] = {}
    for cond_idx, (cond, system_prompt, kind) in enumerate(conditions()):
        ns = f"VMBA2-{model.upper()}-{cond_idx}"
        specs = []
        for topic_idx, topic in enumerate(topics):
            for seed_idx in range(k):
                if kind == "pure":
                    user_prompt = TEMPLATE.format(topic=topic)
                else:
                    user_prompt = by_pair[kind].format_user_prompt(topic, TEMPLATE)
                specs.append({
                    "generation_id": topic_idx * k + seed_idx,
                    "prompt_set": ns,
                    "topic": topic,
                    "topic_idx": topic_idx,
                    "mode": cond,
                    "mode_idx": cond_idx,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "seed": make_seed(topic_idx, cond_idx, seed_idx, prompt_set=ns),
                    "repetition": seed_idx,
                })
        out[cond] = specs
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--seeds-per-topic", type=int, default=8)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--workers-per-gpu", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    preset = MODEL_PRESETS[args.model]
    prompts_path = Path(__file__).resolve().parents[1] / "prompts" / "prompt_sets.json"
    d = json.loads(prompts_path.read_text())
    topics = list(d["topics"]["set_a"]) + list(d["topics"]["set_b"])
    assert len(topics) == 20

    all_specs = build_specs(args.model, topics, args.seeds_per_topic)
    n_total = sum(len(v) for v in all_specs.values())
    logger.info(f"A2 {args.model}: {len(all_specs)} conditions × {len(topics)}×{args.seeds_per_topic} "
                f"= {n_total} gens")
    if args.dry_run:
        for cond, specs in all_specs.items():
            s = specs[0]
            logger.info(f"  {cond}: sys[:50]={s['system_prompt'][:50]!r} user[:70]={s['user_prompt'][:70]!r}")
        return

    gpu_ids = resolve_physical_gpus(
            [g.strip() for g in args.gpus.split(",") if g.strip()])
    n_workers = len(gpu_ids) * args.workers_per_gpu

    for cond, specs in all_specs.items():
        out_dir = args.out_root / f"vmb_a2_{args.model}_{cond}"
        rec_dir = out_dir / "gen_records"
        rec_dir.mkdir(parents=True, exist_ok=True)
        todo = [s for s in specs if not (rec_dir / f"gen_{s['generation_id']:03d}.json").exists()]
        logger.info(f"[{cond}] {len(todo)}/{len(specs)} to generate")
        if todo:
            worker_specs: list[list] = [[] for _ in range(n_workers)]
            for i, s in enumerate(todo):
                worker_specs[i % n_workers].append(s)
            tmp = out_dir / "gen_logs"
            tmp.mkdir(exist_ok=True)
            procs = []
            for w, ws in enumerate(worker_specs):
                if not ws:
                    continue
                gpu = gpu_ids[w % len(gpu_ids)]
                sf = tmp / f"specs_w{w}.json"
                sf.write_text(json.dumps(ws))
                cmd = [sys.executable, "-m", "anamnesis.scripts.run_gen_tokens",
                       "--model", args.model, "--model-path", args.model_path,
                       "--spec-file", str(sf), "--out-dir", str(rec_dir),
                       "--temperature", str(preset.temperature), "--top-p", "0.9",
                       "--max-new-tokens", str(args.max_new_tokens),
                       "--eos-ids", *[str(e) for e in preset.eos_token_ids],
                       "--attn", "eager", "--date-string", VMB_CANONICAL_DATE,
                       "--label", f"{cond[:12]}-w{w}"]
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu,
                       "PYTHONPATH": os.environ.get("PYTHONPATH", "."),
                       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                       "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
                fh = open(tmp / f"gen_w{w}.log", "w")
                procs.append((w, subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT), fh))
            t0 = time.time()
            fails = 0
            for _w, p, fh in procs:
                rc = p.wait()
                fh.close()
                if rc != 0:
                    fails += 1
            logger.info(f"[{cond}] workers done in {time.time()-t0:.0f}s ({fails} nonzero)")
        passthrough = {
            "model": {"model_id": preset.model_id},
            "generation_config": {"max_new_tokens": args.max_new_tokens,
                                  "temperature": preset.temperature, "top_p": 0.9,
                                  "do_sample": True, "eos_token_ids": preset.eos_token_ids},
            "vmb_a2": {"prereg": "prereg-vmb-v1 §2c A2", "condition": cond,
                       "template_date_string": VMB_CANONICAL_DATE},
        }
        assemble(out_dir, passthrough)


if __name__ == "__main__":
    main()
