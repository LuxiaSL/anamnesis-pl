"""Multi-worker-per-GPU launcher for replay extraction.

Adapts subliminal_anamnesis/scripts/pipeline/parallel_extract.py to the anamnesis
replay path: partitions a run's manifest gen_ids round-robin across
(len(gpus) x workers_per_gpu) workers, spawns one run_replay_extraction.py subprocess
per worker with CUDA_VISIBLE_DEVICES set and its gen-id slice. Resume-aware: gens whose
signature already exists are dropped before partitioning, so re-runs self-balance.

Each worker loads the model once and replays its slice (GPU forward is cheap; the CPU
feature math is the cost — packing >1 worker/GPU overlaps one worker's CPU math with
another's GPU forward).

Usage:
    PYTHONPATH=. python -m anamnesis.scripts.parallel_replay \
        --model 8b --model-path /models/llama-3.1-8b-instruct \
        --run-dir /models/anamnesis-extract/runs/8b_fat_01 \
        --calib-dir /models/anamnesis-extract/calibration/llama31_8b \
        --manifest /models/anamnesis-extract/runs/8b_fat_01/replay_manifest.json \
        --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 3
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel replay extraction launcher")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument("--sig-subdir", default="signatures_v3")
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="Absolute raw output dir passed to workers (e.g. /dev/shm scratch)")
    parser.add_argument("--no-raw", action="store_true")
    parser.add_argument("--no-tier3", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inject-npz", type=Path, default=None,
                        help="A5: passthrough to workers (with --inject-key/-layer/-alpha[/-frac])")
    parser.add_argument("--inject-key", default=None)
    parser.add_argument("--inject-layer", type=int, default=None)
    parser.add_argument("--inject-alpha", type=float, default=None)
    parser.add_argument("--inject-alpha-frac", type=float, default=None)
    parser.add_argument("--inject-from-metadata", action="store_true",
                        help="Workers read the spec from run-dir/metadata.json (a5_injection)")
    parser.add_argument("--gen-ids", type=int, nargs="+", default=None,
                        help="Subset of manifest gen ids (A5 matched-token cells)")
    args = parser.parse_args()

    gpu_ids = resolve_physical_gpus(
        [g.strip() for g in args.gpus.split(",") if g.strip() != ""])
    n_workers = len(gpu_ids) * args.workers_per_gpu

    with open(args.manifest) as f:
        manifest = json.load(f)
    all_ids = sorted(int(k) for k in manifest["entries"])
    if args.gen_ids is not None:
        wanted = set(args.gen_ids)
        all_ids = [g for g in all_ids if g in wanted]

    sig_dir = args.run_dir / args.sig_subdir
    if not args.no_resume:
        todo = [g for g in all_ids if not (sig_dir / f"gen_{g:03d}.json").exists()]
    else:
        todo = list(all_ids)

    if not todo:
        logger.info(f"All {len(all_ids)} gens already done — nothing to do.")
        return

    logger.info(
        f"{len(todo)}/{len(all_ids)} gens to extract across {n_workers} workers "
        f"({len(gpu_ids)} GPUs × {args.workers_per_gpu}/GPU)"
    )

    # Round-robin partition gen ids across workers
    worker_ids: list[list[int]] = [[] for _ in range(n_workers)]
    for i, gid in enumerate(todo):
        worker_ids[i % n_workers].append(gid)

    if args.dry_run:
        for w, ids in enumerate(worker_ids):
            if ids:
                logger.info(f"  worker {w} (GPU {gpu_ids[w % len(gpu_ids)]}): {len(ids)} gens {ids[:5]}{'...' if len(ids) > 5 else ''}")
        return

    log_dir = args.log_dir or (args.run_dir / "replay_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    procs: list[tuple[int, str, subprocess.Popen, object]] = []
    for w, ids in enumerate(worker_ids):
        if not ids:
            continue
        gpu = gpu_ids[w % len(gpu_ids)]
        cmd = [
            python, "-m", "anamnesis.scripts.run_replay_extraction",
            "--model", args.model, "--model-path", args.model_path,
            "--run-dir", str(args.run_dir), "--calib-dir", str(args.calib_dir),
            "--manifest", str(args.manifest), "--sig-subdir", args.sig_subdir,
            "--label", f"w{w}g{gpu}",
            "--gen-ids", *[str(g) for g in ids],
        ]
        if args.no_raw:
            cmd.append("--no-raw")
        if args.raw_dir:
            cmd += ["--raw-dir", str(args.raw_dir)]
        if args.no_tier3:
            cmd.append("--no-tier3")
        if args.inject_from_metadata:
            cmd.append("--inject-from-metadata")
        elif args.inject_npz is not None:
            cmd += ["--inject-npz", str(args.inject_npz),
                    "--inject-key", str(args.inject_key),
                    "--inject-layer", str(args.inject_layer),
                    "--inject-alpha", str(args.inject_alpha)]
            if args.inject_alpha_frac is not None:
                cmd += ["--inject-alpha-frac", str(args.inject_alpha_frac)]
        # workers always resume internally too (idempotent); --no-resume only affects launch filter
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": gpu,
            "PYTHONPATH": os.environ.get("PYTHONPATH", "."),
            # Cap BLAS threads per worker. CPU feature extraction is the bottleneck; with many
            # workers, multi-threaded numpy oversubscribes the cores (128) and thrashes badly.
            # One thread/worker → clean N-way parallelism across workers.
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "1"),
        }
        log_path = log_dir / f"replay_w{w}_gpu{gpu}.log"
        fh = open(log_path, "w")
        proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        procs.append((w, gpu, proc, fh))
        logger.info(f"  worker {w} on GPU {gpu}: {len(ids)} gens → {log_path}")

    t0 = time.time()
    failed = 0
    for w, gpu, proc, fh in procs:
        proc.wait()
        fh.close()
        status = "OK" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
        if proc.returncode != 0:
            failed += 1
        logger.info(f"  worker {w} (GPU {gpu}): {status} ({time.time() - t0:.0f}s)")

    done = sum(1 for g in all_ids if (sig_dir / f"gen_{g:03d}.json").exists())
    logger.info(
        f"Done in {time.time() - t0:.0f}s — {len(procs) - failed}/{len(procs)} workers OK; "
        f"{done}/{len(all_ids)} signatures present in {sig_dir}"
    )


if __name__ == "__main__":
    main()
