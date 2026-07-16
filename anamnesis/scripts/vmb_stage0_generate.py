"""vmb Stage-0 floor generation launcher (prereg-vmb-v1, addendum 2026-07-12a).

Stochastic-floor corpus: Phase-0 20 topics (set_a+set_b) × 4 task strata × N seeds
per model, BARE (no system prompt), deterministic SHA-derived seeds via make_seed.
Phase 1 of the standing pattern: generation banks text only; all signature
extraction happens replay-side (parallel_replay with the full vmb capture surface),
so capture-hook changes never force re-generation.

Spec layout (FROZEN — gen ids and seeds are functions of the frozen stratum order):
    gid  = (stratum_idx * n_topics + topic_idx) * seeds_per_class + seed_idx
    seed = make_seed(topic_idx, stratum_idx, seed_idx, prompt_set=tag)
    tag  = f"VMB0-{model.upper()}"          # per-model seed namespace
    mode field = stratum name (schema-compatible with run_gen_tokens / metadata.json;
                 floors group by prompt class = (topic_idx, stratum))

Usage (node1, from ~/luxi-files/anamnesis-pl with .venv-shared active):
    PYTHONPATH=pipeline python -m anamnesis.scripts.vmb_stage0_generate \
        --model 3b --model-path /models/llama-3.2-3b-instruct \
        --prompts pipeline/anamnesis/prompts/prompt_sets.json \
        --out-run-dir /models/anamnesis-extract/runs/vmb_stage0_3b \
        --gpus 0,1,2,3 --workers-per-gpu 4

    # sanity pass first: --limit 8 --dry-run (inspect specs), then --limit 8 (generate)
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# The battery's canonical chat-template date. Llama templates render "Today Date"
# via strftime_now; a midnight rollover mid-run would change prompt tokens and
# break matched-history pairing. Pinned to the prereg freeze date, forever.
VMB_CANONICAL_DATE = "12 Jul 2026"


def load_stage0_protocol(prompts_path: Path) -> tuple[list[str], list[dict], int]:
    """Return (topics, strata, seeds_per_class) from the frozen vmb_stage0_strata key."""
    d = json.load(open(prompts_path))
    proto = d["vmb_stage0_strata"]
    topics: list[str] = []
    for set_name in proto["topic_sets"]:
        topics.extend(d["topics"][set_name])
    strata = proto["strata"]
    return topics, strata, int(proto["seeds_per_class"])


def build_specs(model: str, topics: list[str], strata: list[dict], seeds_per_class: int,
                tag: str | None = None) -> list[dict]:
    """tag overrides the seed namespace (arm runs use e.g. VMBA1-3B-T03 so their
    SHA-derived seeds are disjoint from the floor corpus and from other doses)."""
    tag = tag or f"VMB0-{model.upper()}"
    n_topics = len(topics)
    specs: list[dict] = []
    for stratum_idx, stratum in enumerate(strata):
        for topic_idx, topic in enumerate(topics):
            for seed_idx in range(seeds_per_class):
                gid = (stratum_idx * n_topics + topic_idx) * seeds_per_class + seed_idx
                specs.append({
                    "generation_id": gid,
                    "prompt_set": tag,
                    "topic": topic,
                    "topic_idx": topic_idx,
                    "mode": stratum["name"],          # stratum name in the mode slot
                    "mode_idx": stratum_idx,
                    "system_prompt": "",              # BARE floors (addendum 2026-07-12a)
                    "user_prompt": stratum["template"].format(topic=topic),
                    "seed": make_seed(topic_idx, stratum_idx, seed_idx, prompt_set=tag),
                    "repetition": seed_idx,
                })
    return specs


def assemble(out_run_dir: Path, passthrough: dict) -> int:
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
    ap = argparse.ArgumentParser(description="vmb Stage-0 stochastic-floor generation (phase 1: tokens only)")
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--prompts", type=Path, required=True, help="prompt_sets.json (holds vmb_stage0_strata)")
    ap.add_argument("--out-run-dir", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--workers-per-gpu", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"],
                    help="Attention impl for GENERATION only. Default eager (token-matches the "
                         "fat_01 reference corpus). sdpa is ~2-4x faster and valid for NEW steering "
                         "cells (phase-2 replay is eager regardless, so this only changes which "
                         "tokens get sampled; the bitwise-replay floor is a replay property).")
    ap.add_argument("--limit", type=int, default=None, help="Cap spec count (sanity pass)")
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    # ── Arm-dose overrides (A1 sampling ladder etc.; floors use pure defaults) ──
    ap.add_argument("--override-temperature", type=float, default=None,
                    help="Dose temperature (default: model-native preset)")
    ap.add_argument("--override-top-p", type=float, default=None,
                    help="Dose top-p (default: 0.9)")
    ap.add_argument("--seeds-per-class", type=int, default=None,
                    help="Override seeds/class (arm runs use fewer than the floor's 10)")
    ap.add_argument("--seed-namespace", default=None,
                    help="Seed namespace tag (e.g. VMBA1-3B-T03) — MUST be unique per "
                         "(model × arm × dose) so seeds are disjoint from floors and doses")
    # ── A5 steered-generation passthrough (run_gen_tokens applies the write hook) ──
    ap.add_argument("--inject-npz", default=None)
    ap.add_argument("--inject-key", default=None)
    ap.add_argument("--inject-layer", type=int, default=None)
    ap.add_argument("--inject-alpha", type=float, default=None,
                    help="ABSOLUTE alpha; alternatively give --inject-alpha-frac + "
                         "--inject-norms-json and it resolves at run time")
    ap.add_argument("--inject-alpha-frac", type=float, default=None)
    ap.add_argument("--inject-norms-json", default=None,
                    help="a5_vectors_stamps.json (holds median_resid_norms per site); "
                         "alpha = frac × norms['L<layer>'] resolved when the job RUNS")
    args = ap.parse_args()

    if args.inject_npz is not None and args.inject_alpha is None:
        if args.inject_alpha_frac is None or args.inject_norms_json is None:
            raise SystemExit("--inject-npz needs --inject-alpha OR "
                             "(--inject-alpha-frac + --inject-norms-json)")
        norms = json.loads(Path(args.inject_norms_json).read_text())["median_resid_norms"]
        args.inject_alpha = float(args.inject_alpha_frac) * float(norms[f"L{args.inject_layer}"])
        logger.info(f"resolved inject_alpha = {args.inject_alpha_frac} × "
                    f"{norms[f'L{args.inject_layer}']:.2f} = {args.inject_alpha:.3f}")

    preset = MODEL_PRESETS[args.model]
    topics, strata, seeds_per_class = load_stage0_protocol(args.prompts)
    if args.seeds_per_class:
        seeds_per_class = args.seeds_per_class
    temperature = args.override_temperature if args.override_temperature is not None else preset.temperature
    top_p = args.override_top_p if args.override_top_p is not None else 0.9
    namespace = args.seed_namespace or f"VMB0-{args.model.upper()}"
    if len(topics) != 20:
        raise ValueError(f"Stage-0 protocol expects the Phase-0 20 topics, got {len(topics)}")
    specs = build_specs(args.model, topics, strata, seeds_per_class, tag=namespace)
    logger.info(f"{len(topics)} topics × {len(strata)} strata × {seeds_per_class} seeds "
                f"= {len(specs)} specs; temp={temperature} top_p={top_p} "
                f"ns={namespace} eos={preset.eos_token_ids}")
    if args.limit:
        specs = specs[: args.limit]

    passthrough = {
        "model": {"model_id": preset.model_id, "torch_dtype": preset.torch_dtype,
                  "num_layers": preset.num_layers, "hidden_dim": preset.hidden_dim},
        "generation_config": {"max_new_tokens": args.max_new_tokens,
                              "temperature": temperature, "top_p": top_p,
                              "do_sample": True, "eos_token_ids": preset.eos_token_ids},
        "vmb_stage0": {"prereg": "prereg-vmb-v1", "addendum": "2026-07-12a",
                       "floor_type": "stochastic", "bare_system_prompt": True,
                       "seed_namespace": namespace,
                       "template_date_string": VMB_CANONICAL_DATE},
    }
    if args.inject_npz is not None:
        passthrough["a5_injection"] = {
            "inject_npz": args.inject_npz, "inject_key": args.inject_key,
            "inject_layer": args.inject_layer, "inject_alpha": args.inject_alpha,
            "inject_alpha_frac": args.inject_alpha_frac,
        }

    out_run_dir = args.out_run_dir
    rec_dir = out_run_dir / "gen_records"
    rec_dir.mkdir(parents=True, exist_ok=True)

    if args.assemble_only:
        assemble(out_run_dir, passthrough)
        return

    if args.dry_run:
        for s in specs[:12]:
            logger.info(f"  gen {s['generation_id']:3d}: {s['mode']:14s} | t{s['topic_idx']:02d} "
                        f"seed_idx{s['repetition']} | seed {s['seed']:>10d} | {s['user_prompt'][:60]}")
        logger.info(f"  ... total {len(specs)}")
        return

    todo = [s for s in specs if not (rec_dir / f"gen_{s['generation_id']:03d}.json").exists()]
    logger.info(f"{len(todo)}/{len(specs)} specs to generate ({len(specs) - len(todo)} already done)")

    if todo:
        gpu_ids = resolve_physical_gpus(
            [g.strip() for g in args.gpus.split(",") if g.strip()])
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
                   "--temperature", str(temperature),
                   "--top-p", str(top_p),
                   "--max-new-tokens", str(args.max_new_tokens),
                   "--eos-ids", *[str(e) for e in preset.eos_token_ids],
                   "--attn", args.attn, "--date-string", VMB_CANONICAL_DATE,
                   "--label", f"w{w}g{gpu}"]
            if args.inject_npz is not None:
                cmd += ["--inject-npz", args.inject_npz,
                        "--inject-key", str(args.inject_key),
                        "--inject-layer", str(args.inject_layer),
                        "--inject-alpha", str(args.inject_alpha)]
                if args.inject_alpha_frac is not None:
                    cmd += ["--inject-alpha-frac", str(args.inject_alpha_frac)]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu,
                   "PYTHONPATH": os.environ.get("PYTHONPATH", "."),
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
            log_path = tmp_dir / f"gen_w{w}_gpu{gpu}.log"
            fh = open(log_path, "w")
            procs.append((w, gpu, subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT), fh))
            logger.info(f"  worker {w} GPU {gpu}: {len(wspecs)} specs → {log_path}")

        t0 = time.time()
        fails = 0
        for w, gpu, proc, fh in procs:
            rc = proc.wait()
            fh.close()
            if rc != 0:
                fails += 1
                logger.error(f"worker {w} (GPU {gpu}) exited rc={rc}")
        logger.info(f"all workers done in {time.time() - t0:.0f}s ({fails} failed)")

    n = assemble(out_run_dir, passthrough)
    expected = len(specs)
    if n < expected:
        logger.warning(f"INCOMPLETE: {n}/{expected} gen records assembled — re-run to resume missing specs")


if __name__ == "__main__":
    main()
