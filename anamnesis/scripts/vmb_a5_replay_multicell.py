"""Multi-cell A5 replay launcher — loads the model + calibration ONCE per worker and
loops all cells, eliminating the per-cell reload gap of the one-cell-per-invocation
parallel_replay path (the sawtooth VRAM 0->130GB->0 between cells).

Each worker gets a jobs-file (its gen-id slice of EVERY cell) and runs
`run_replay_extraction --jobs-file` once; the worker re-attaches the injection hook per
cell (removing the previous — ResidualWriteHandle.remove() detaches cleanly, no stacking)
and reads each cell's own metadata for the spec (inject_from_metadata). Signatures are
byte-identical to the per-cell path (verified by vmb_a5_replay_multicell_smoke).

cells-json:  {"cells": [{"run_dir": "<run>/CELL", "manifest": "<run>/CELL/replay_manifest.json",
                         "gen_ids": [..]?}, ...]}

Usage:
  python -m anamnesis.scripts.vmb_a5_replay_multicell --model qwen-7b --model-path <p> \
    --calib-dir <calib> --cells-json cells.json --gpus 0,1,2,3,4,5,6,7 \
    --workers-per-gpu 8 --no-raw --inject-from-metadata
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--cells-json", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--workers-per-gpu", type=int, default=8)
    ap.add_argument("--sig-subdir", default="signatures_v3")
    ap.add_argument("--no-raw", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--inject-from-metadata", action="store_true",
                    help="each cell reads its own metadata.json a5_injection spec")
    args = ap.parse_args()

    preset = MODEL_PRESETS[args.model]
    cells = json.loads(args.cells_json.read_text())["cells"]
    gpu_ids = resolve_physical_gpus([g.strip() for g in args.gpus.split(",") if g.strip()])
    n_workers = len(gpu_ids) * args.workers_per_gpu
    logger.info(f"{len(cells)} cells across {n_workers} workers; model+calib load ONCE/worker")

    # per worker: list of cell-jobs (its gen-id slice of every cell)
    worker_jobs: list[list[dict]] = [[] for _ in range(n_workers)]
    for cell in cells:
        run_dir = Path(cell["run_dir"])
        manifest_path = Path(cell["manifest"])
        manifest = json.loads(manifest_path.read_text())
        avail = sorted(int(k) for k in manifest["entries"])
        if cell.get("gen_ids") is not None:
            wanted = set(cell["gen_ids"])
            avail = [g for g in avail if g in wanted]
        # round-robin partition this cell's gens across workers (assignment doesn't
        # affect output — each sig is deterministic from its banked tokens + spec)
        per_worker: list[list[int]] = [[] for _ in range(n_workers)]
        for i, g in enumerate(avail):
            per_worker[i % n_workers].append(g)
        for w in range(n_workers):
            if per_worker[w]:
                worker_jobs[w].append({
                    "run_dir": str(run_dir), "manifest": str(manifest_path),
                    "gen_ids": per_worker[w],
                    "inject_from_metadata": bool(args.inject_from_metadata)})

    tmp = Path(cells[0]["run_dir"]).parent / "_multicell_replay_jobs"
    tmp.mkdir(parents=True, exist_ok=True)
    procs = []
    for w in range(n_workers):
        if not worker_jobs[w]:
            continue
        jf = tmp / f"jobs_w{w}.json"
        jf.write_text(json.dumps(worker_jobs[w]))
        gpu = gpu_ids[w % len(gpu_ids)]
        cmd = [sys.executable, "-m", "anamnesis.scripts.run_replay_extraction",
               "--model", args.model, "--model-path", args.model_path,
               "--calib-dir", str(args.calib_dir), "--jobs-file", str(jf),
               "--sig-subdir", args.sig_subdir, "--label", f"w{w}g{gpu}"]
        if args.no_raw:
            cmd.append("--no-raw")
        if args.no_resume:
            cmd.append("--no-resume")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu,
               "PYTHONPATH": os.environ.get("PYTHONPATH", "."),
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
        log_path = tmp / f"rep_w{w}_gpu{gpu}.log"
        fh = open(log_path, "w")
        procs.append((w, gpu, subprocess.Popen(cmd, env=env, stdout=fh,
                                               stderr=subprocess.STDOUT), fh))
        logger.info(f"  worker {w} GPU {gpu}: {len(worker_jobs[w])} cells → {log_path}")

    t0 = time.time()
    fails = 0
    for w, gpu, proc, fh in procs:
        rc = proc.wait()
        fh.close()
        if rc != 0:
            fails += 1
            logger.error(f"worker {w} (GPU {gpu}) rc={rc}")
    logger.info(f"all replay workers done in {time.time()-t0:.0f}s ({fails} failed)")
    if fails:
        raise SystemExit(f"{fails} replay workers failed")


if __name__ == "__main__":
    main()
