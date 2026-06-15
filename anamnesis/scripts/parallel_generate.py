"""Phase-1 launcher: extend a fat_01 corpus with more topics/reps (generation only).

Pulls the EXACT instrument from the source run's metadata.json (the 8 core modes'
system prompts, generation config, seed namespace) so the extension is the same
experiment. Builds specs for all (topic x core-mode x rep) cells NOT already in the
source run, fans them across (gpus x workers/gpu) run_gen_tokens workers, then
assembles the new run's metadata.json + an EXACT replay_manifest.json (from banked
full token ids) ready for parallel_replay (phase 2).

Topics come from --prompts (all sets flattened in order → global topic_idx); the source
run's topics must be the leading prefix (set_a, set_b, ...) so topic_idx stays consistent
for a clean merged GroupKFold-by-topic.

Usage:
    PYTHONPATH=. python -m anamnesis.scripts.parallel_generate \
        --model 8b --model-path /models/llama-3.1-8b-instruct \
        --meta-from /models/anamnesis-extract/runs/8b_fat_01/metadata.json \
        --prompts anamnesis/prompts/prompt_sets.json \
        --out-run-dir /models/anamnesis-extract/runs/8b_fat_ext \
        --num-reps 3 --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from anamnesis.extraction.generation_runner import make_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_source(meta_path: Path):
    """Return (core_modes[name->sys_prompt sorted], gen_cfg, prompt_set_tag, passthrough, skip_set)."""
    md = json.load(open(meta_path))
    gens = md["generations"] if isinstance(md, dict) and "generations" in md else md
    core_modes: dict[str, str] = {}
    tags, skip = set(), set()
    for g in gens:
        mode = g["mode"]
        if mode.startswith("swap_"):
            continue
        core_modes.setdefault(mode, g["system_prompt"])
        tags.add(g.get("prompt_set"))
        skip.add((int(g["topic_idx"]), int(g.get("repetition", 0))))
    gen_cfg = md.get("generation_config", {})
    passthrough = {"model": md.get("model", {}),
                   "generation_config": gen_cfg,
                   "extraction_config": md.get("extraction_config", {})}
    tag = sorted(tags)[0] if tags else "ext"
    return dict(sorted(core_modes.items())), gen_cfg, tag, passthrough, skip


def flatten_topics(prompts_path: Path) -> list[str]:
    d = json.load(open(prompts_path))
    topics: list[str] = []
    for _set, items in d["topics"].items():  # dict preserves insertion order (set_a, set_b, ...)
        topics.extend(items)
    template = d.get("user_prompt_template", "Write about: {topic}")
    return topics, template


def build_specs(core_modes, topics, template, num_reps, tag, skip_set):
    modes = list(core_modes.keys())  # already sorted
    specs = []
    gid = 0
    for rep in range(num_reps):
        for tidx, topic in enumerate(topics):
            if (tidx, rep) in skip_set:   # this cell already exists in the source run
                continue
            for midx, mode in enumerate(modes):
                specs.append({
                    "generation_id": gid,
                    "prompt_set": tag,
                    "topic": topic,
                    "topic_idx": tidx,
                    "mode": mode,
                    "mode_idx": midx,
                    "system_prompt": core_modes[mode],
                    "user_prompt": template.format(topic=topic),
                    "seed": make_seed(tidx, midx, rep, prompt_set=tag),
                    "repetition": rep,
                })
                gid += 1
    return specs


def assemble(out_run_dir: Path, passthrough: dict):
    """Build metadata.json (lean) + replay_manifest.json (exact) from gen records."""
    rec_dir = out_run_dir / "gen_records"
    records = []
    for p in sorted(rec_dir.glob("gen_*.json"), key=lambda x: int(x.stem.split("_")[1])):
        records.append(json.load(open(p)))

    generations, entries = [], {}
    for r in records:
        gid = r["generation_id"]
        ids = r["input_ids"]
        plen = r["prompt_length"]
        entries[str(gid)] = {"input_ids": ids, "prompt_length": plen, "n_gen": len(ids) - plen}
        lean = {k: v for k, v in r.items() if k != "input_ids"}
        generations.append(lean)

    meta = {"total_generations": len(generations), "failed_ids": [],
            **passthrough, "generations": generations}
    (out_run_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    manifest = {"entries": entries, "n_ok": len(entries), "n_flagged": 0, "flagged": []}
    (out_run_dir / "replay_manifest.json").write_text(json.dumps(manifest))
    logger.info(f"Assembled {len(generations)} gens → metadata.json + replay_manifest.json "
                f"({len(entries)} manifest entries)")
    return len(generations)


def main() -> None:
    ap = argparse.ArgumentParser(description="Parallel generation launcher (corpus extension, phase 1)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--meta-from", type=Path, required=True, help="Source run metadata.json")
    ap.add_argument("--prompts", type=Path, required=True, help="prompt_sets.json (all sets)")
    ap.add_argument("--out-run-dir", type=Path, required=True)
    ap.add_argument("--num-reps", type=int, default=3)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--workers-per-gpu", type=int, default=4)
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--limit", type=int, default=None, help="Cap spec count (throughput probe)")
    ap.add_argument("--assemble-only", action="store_true", help="Skip generation, just assemble")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    core_modes, gen_cfg, tag, passthrough, skip_set = load_source(args.meta_from)
    topics, template = flatten_topics(args.prompts)
    specs = build_specs(core_modes, topics, template, args.num_reps, tag, skip_set)
    if args.limit:
        specs = specs[: args.limit]

    out_run_dir = args.out_run_dir
    rec_dir = out_run_dir / "gen_records"
    rec_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"{len(core_modes)} core modes {list(core_modes)}")
    logger.info(f"{len(topics)} topics × {args.num_reps} reps; tag={tag}; "
                f"temp={gen_cfg.get('temperature')} eos={gen_cfg.get('eos_token_ids')}")
    logger.info(f"{len(specs)} extension specs (source cells skipped)")

    if args.assemble_only:
        assemble(out_run_dir, passthrough)
        return

    todo = [s for s in specs if not (rec_dir / f"gen_{s['generation_id']:03d}.json").exists()]
    logger.info(f"{len(todo)}/{len(specs)} specs to generate ({len(specs) - len(todo)} already done)")

    if args.dry_run:
        for s in specs[:8]:
            logger.info(f"  gen {s['generation_id']:4d}: {s['mode']:11s} | t{s['topic_idx']:02d} "
                        f"rep{s['repetition']} | {s['topic'][:40]} | seed {s['seed']}")
        logger.info(f"  ... total {len(specs)}")
        return

    if todo:
        gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
        n_workers = len(gpu_ids) * args.workers_per_gpu
        worker_specs: list[list] = [[] for _ in range(n_workers)]
        for i, s in enumerate(todo):
            worker_specs[i % n_workers].append(s)

        tmp_dir = out_run_dir / "gen_logs"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        python = sys.executable
        procs = []
        for w, wspecs in enumerate(worker_specs):
            if not wspecs:
                continue
            gpu = gpu_ids[w % len(gpu_ids)]
            spec_file = tmp_dir / f"specs_w{w}.json"
            spec_file.write_text(json.dumps(wspecs))
            cmd = [python, "-m", "anamnesis.scripts.run_gen_tokens",
                   "--model", args.model, "--model-path", args.model_path,
                   "--spec-file", str(spec_file), "--out-dir", str(rec_dir),
                   "--temperature", str(gen_cfg["temperature"]),
                   "--top-p", str(gen_cfg["top_p"]),
                   "--max-new-tokens", str(gen_cfg.get("max_new_tokens", 512)),
                   "--eos-ids", *[str(e) for e in gen_cfg["eos_token_ids"]],
                   "--attn", args.attn, "--label", f"w{w}g{gpu}"]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu,
                   "PYTHONPATH": os.environ.get("PYTHONPATH", "."),
                   # pin CPU thread pools per worker (avoid load spike during model loads)
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
            log_path = tmp_dir / f"gen_w{w}_gpu{gpu}.log"
            fh = open(log_path, "w")
            procs.append((w, gpu, subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT), fh))
            logger.info(f"  worker {w} GPU {gpu}: {len(wspecs)} specs → {log_path}")

        t0 = time.time()
        failed = 0
        for w, gpu, proc, fh in procs:
            proc.wait()
            fh.close()
            if proc.returncode != 0:
                failed += 1
                logger.error(f"  worker {w} (GPU {gpu}): FAILED rc={proc.returncode}")
        logger.info(f"Generation done in {time.time()-t0:.0f}s — {len(procs)-failed}/{len(procs)} workers OK")

    n_records = len(list(rec_dir.glob("gen_*.json")))
    logger.info(f"{n_records} gen records present")
    assemble(out_run_dir, passthrough)


if __name__ == "__main__":
    main()
