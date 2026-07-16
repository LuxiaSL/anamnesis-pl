"""Multi-cell A5 generation launcher — loads the model ONCE per worker and loops all
cells, eliminating the per-cell model-reload gap of the one-cell-per-invocation path.

Each worker gets a jobs-file (its spec-slice of EVERY cell + that cell's injection +
out-dir) and runs `run_gen_tokens --jobs-file` once. run_gen_tokens re-attaches the
write hook per cell (layers differ). Output is byte-identical to the single-cell path
(each gen is self-seeded; verified by tests/test_multicell_bitwise or the smoke below).

Reuses vmb_stage0_generate.build_specs/assemble/protocol verbatim, so specs, seeds,
date-string, and metadata match the floor/one-cell path exactly.

cells-json format:  {"cells": [
    {"out_run_dir": "<run>/CELL", "seed_namespace": "VMBA5-M-CELL",
     "inject_key": "V3_L18", "inject_layer": 18, "inject_alpha_frac": 0.1}, ...]}
inject_alpha (absolute) may be given instead of inject_alpha_frac; inject_key null =
no injection. Shared --inject-npz + --inject-norms-json resolve fracs at launch.

Usage:
  python -m anamnesis.scripts.vmb_a5_gen_multicell --model qwen-7b --model-path <p> \
    --prompts pipeline/anamnesis/prompts/prompt_sets.json --cells-json cells.json \
    --inject-npz <npz> --inject-norms-json <stamps> --gpus 0,1,2,3,4,5,6,7 \
    --workers-per-gpu 6 --seeds-per-class 2
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

from anamnesis.config import MODEL_PRESETS
from anamnesis.scripts._gpu import resolve_physical_gpus
from anamnesis.scripts.vmb_stage0_generate import (
    VMB_CANONICAL_DATE, assemble, build_specs, load_stage0_protocol,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--cells-json", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--workers-per-gpu", type=int, default=6)
    ap.add_argument("--seeds-per-class", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--inject-npz", default=None)
    ap.add_argument("--inject-norms-json", default=None,
                    help="median_resid_norms json to resolve inject_alpha_frac per cell")
    ap.add_argument("--limit", type=int, default=None, help="cap specs per cell (test only)")
    args = ap.parse_args()

    preset = MODEL_PRESETS[args.model]
    temperature = preset.temperature
    top_p = 0.9
    topics, strata, seeds_per_class = load_stage0_protocol(args.prompts)
    seeds_per_class = args.seeds_per_class or seeds_per_class
    if len(topics) != 20:
        raise ValueError(f"Stage-0 protocol expects 20 topics, got {len(topics)}")
    norms = None
    if args.inject_norms_json:
        norms = json.loads(Path(args.inject_norms_json).read_text())["median_resid_norms"]

    cells = json.loads(args.cells_json.read_text())["cells"]
    logger.info(f"{len(cells)} cells; model loaded ONCE per worker")

    gpu_ids = resolve_physical_gpus([g.strip() for g in args.gpus.split(",") if g.strip()])
    n_workers = len(gpu_ids) * args.workers_per_gpu

    # per worker: list of cell-jobs (out_dir + injection + this worker's spec slice)
    worker_jobs: list[list[dict]] = [[] for _ in range(n_workers)]
    cell_meta: list[tuple[Path, dict]] = []
    for cell in cells:
        out_run_dir = Path(cell["out_run_dir"])
        ns = cell["seed_namespace"]
        specs = build_specs(args.model, topics, strata, seeds_per_class, tag=ns)
        if args.limit:
            specs = specs[: args.limit]
        key = cell.get("inject_key")
        layer = cell.get("inject_layer")
        alpha = cell.get("inject_alpha")
        frac = cell.get("inject_alpha_frac")
        if key is not None and alpha is None:
            if frac is None or norms is None:
                raise SystemExit(f"cell {ns}: inject_key set but no alpha/(frac+norms)")
            alpha = float(frac) * float(norms[f"L{layer}"])
        inj = ({"inject_npz": args.inject_npz, "inject_key": key, "inject_layer": layer,
                "inject_alpha": alpha, "inject_alpha_frac": frac}
               if key is not None else {})
        rec_dir = out_run_dir / "gen_records"
        rec_dir.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(specs):
            worker_jobs[i % n_workers].append({"__cell": str(out_run_dir), **inj, "__spec": s})
        passthrough = {
            "model": {"model_id": preset.model_id, "torch_dtype": preset.torch_dtype,
                      "num_layers": preset.num_layers, "hidden_dim": preset.hidden_dim},
            "generation_config": {"max_new_tokens": args.max_new_tokens,
                                  "temperature": temperature, "top_p": top_p,
                                  "do_sample": True, "eos_token_ids": preset.eos_token_ids},
            "vmb_stage0": {"prereg": "prereg-vmb-v1", "addendum": "2026-07-12a",
                           "floor_type": "stochastic", "bare_system_prompt": True,
                           "seed_namespace": ns, "template_date_string": VMB_CANONICAL_DATE},
        }
        if key is not None:
            passthrough["a5_injection"] = inj
        cell_meta.append((out_run_dir, passthrough))

    # collapse each worker's flat spec list into per-cell job dicts (grouped, order-stable)
    tmp = Path(cells[0]["out_run_dir"]).parent / "_multicell_jobs"
    tmp.mkdir(parents=True, exist_ok=True)
    procs = []
    for w in range(n_workers):
        flat = worker_jobs[w]
        if not flat:
            continue
        by_cell: dict[str, dict] = {}
        for item in flat:
            c = item["__cell"]
            if c not in by_cell:
                by_cell[c] = {k: v for k, v in item.items()
                              if k not in ("__cell", "__spec")}
                by_cell[c]["out_dir"] = str(Path(c) / "gen_records")
                by_cell[c]["specs"] = []
            by_cell[c]["specs"].append(item["__spec"])
        jobs = list(by_cell.values())
        jf = tmp / f"jobs_w{w}.json"
        jf.write_text(json.dumps(jobs))
        gpu = gpu_ids[w % len(gpu_ids)]
        cmd = [sys.executable, "-m", "anamnesis.scripts.run_gen_tokens",
               "--model", args.model, "--model-path", args.model_path,
               "--jobs-file", str(jf), "--temperature", str(temperature),
               "--top-p", str(top_p), "--max-new-tokens", str(args.max_new_tokens),
               "--eos-ids", *[str(e) for e in preset.eos_token_ids],
               "--attn", "eager", "--date-string", VMB_CANONICAL_DATE, "--label", f"w{w}g{gpu}"]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu,
               "PYTHONPATH": os.environ.get("PYTHONPATH", "."),
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
        log_path = tmp / f"gen_w{w}_gpu{gpu}.log"
        fh = open(log_path, "w")
        procs.append((w, gpu, subprocess.Popen(cmd, env=env, stdout=fh,
                                               stderr=subprocess.STDOUT), fh))
        logger.info(f"  worker {w} GPU {gpu}: {len(jobs)} cells → {log_path}")

    t0 = time.time()
    fails = 0
    for w, gpu, proc, fh in procs:
        rc = proc.wait()
        fh.close()
        if rc != 0:
            fails += 1
            logger.error(f"worker {w} (GPU {gpu}) rc={rc}")
    logger.info(f"all workers done in {time.time()-t0:.0f}s ({fails} failed)")

    for out_run_dir, passthrough in cell_meta:
        assemble(out_run_dir, passthrough)


if __name__ == "__main__":
    main()
