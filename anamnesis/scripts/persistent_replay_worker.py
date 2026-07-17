"""Persistent replay-signature worker — the first non-route5 job type on the
shared persistent-worker harness (P4 acceptance; the M7 primitive in its
production shape).

Worker mode (default): load model + calibration ONCE, then serve jobs
    {"run_dir", "manifest", "gen_ids"?, "sig_subdir"?, "no_resume"?,
     "inject_npz"?, "inject_key"?, "inject_layer"?, "inject_alpha"?,
     "inject_alpha_frac"?, "inject_from_metadata"?}
through the file queue. Each job runs run_replay_extraction._replay_cell — the
SAME production cell function the standard path uses — so signatures are
byte-identical to parallel_replay output by construction (parity smoke below).

Driver mode (--drive): spawn N workers on the given GPUs, submit one job per
cell from --cells-json (same schema as vmb_a5_replay_multicell plus optional
per-cell gen_ids), collect, STOP-drain. This is deliberately a thin
demonstration driver; campaign drivers own their own dispatch policy.

Parity smoke (--smoke, needs GPU + a banked cell): replay N gens of one cell
through (a) a single-worker fleet and (b) parallel_replay --single-cell-ok into
different sig-subdirs; byte-diff. PASS = the harness path IS the standard path.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

from anamnesis.config import MODEL_PRESETS
from anamnesis.scripts._persistent_workers import PersistentWorker, WorkerFleet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_worker(args) -> None:
    from anamnesis.extraction.model_loader import load_model
    from anamnesis.scripts.run_replay_extraction import (
        _load_calibration,
        _replay_cell,
        _resolve_injection,
        _setup_replay_injection,
    )
    from anamnesis.scripts.run_replay_multickpt import _build_configs

    preset, all_layers, ec, fc, mc = _build_configs(args.model)
    mc.model_id = args.model_path
    logger.info(f"[w{args.worker_id}] loading model once: {args.model_path}")
    loaded = load_model(mc, sampled_layers=preset.sampled_layers,
                        register_gate_hooks=True, key_layers=all_layers,
                        value_layers=all_layers, query_layers=all_layers,
                        attn_output_layers=all_layers)
    calib = _load_calibration(args.calib_dir, enable_tier3=True)

    state: dict = {"handle": None}

    def handle_job(job: dict) -> dict[str, np.ndarray]:
        if state["handle"] is not None:
            state["handle"].remove()
            state["handle"] = None
        run_dir = Path(job["run_dir"])
        npz, key, layer, alpha, frac = _resolve_injection(
            run_dir, job.get("inject_from_metadata", False),
            job.get("inject_npz"), job.get("inject_key"), job.get("inject_layer"),
            job.get("inject_alpha"), job.get("inject_alpha_frac"))
        handle, meta = _setup_replay_injection(loaded, npz, key, layer, alpha, frac,
                                               f"w{args.worker_id}")
        state["handle"] = handle
        n_done, n_fail = _replay_cell(
            loaded, ec, fc, calib, run_dir, Path(job["manifest"]),
            job.get("gen_ids"), job.get("sig_subdir", "signatures_v3"),
            None, "raw_tensors_v3", True,           # signatures-only (no raw)
            bool(job.get("no_resume", False)), 50, handle, meta,
            f"w{args.worker_id}")
        if n_fail:
            raise RuntimeError(f"{n_fail} gens FAILED in {run_dir} — failing loud "
                               f"(job stays queued for retry)")
        return {"n_done": np.array([n_done]), "run_dir": np.array([str(run_dir)])}

    def cleanup() -> None:
        if state["handle"] is not None:
            state["handle"].remove()

    PersistentWorker(work_dir=args.work_dir, worker_id=args.worker_id,
                     handler=handle_job, on_stop=cleanup).run()


def drive(args) -> None:
    from anamnesis.scripts._gpu import resolve_physical_gpus

    cells = json.loads(args.cells_json.read_text())["cells"]
    gpu_ids = resolve_physical_gpus([g.strip() for g in args.gpus.split(",") if g.strip()])
    n_workers = len(gpu_ids) * args.workers_per_gpu
    fleet = WorkerFleet(work_dir=args.work_dir, worker_ids=list(range(n_workers)))

    def cmd_for_worker(w: int) -> list[str]:
        return [sys.executable, "-m", "anamnesis.scripts.persistent_replay_worker",
                "--model", args.model, "--model-path", args.model_path,
                "--calib-dir", str(args.calib_dir), "--work-dir", str(args.work_dir),
                "--worker-id", str(w)]

    fleet.spawn(cmd_for_worker, gpu_for_worker=lambda w: gpu_ids[w % len(gpu_ids)])
    fleet.wait_ready()
    logger.info(f"{n_workers} replay workers ready (persistent; no reloads)")
    try:
        expected = [fleet.submit(i % n_workers, f"{i:05d}", cell)
                    for i, cell in enumerate(cells)]
        got = fleet.collect(expected, timeout_s=args.timeout_s)
        total = sum(int(v["n_done"][0]) for v in got.values())
        logger.info(f"{len(cells)} cells done, {total} gens replayed")
    finally:
        fleet.stop()


def smoke(args) -> None:
    """Byte-diff one cell: harness path vs parallel_replay (the standard path)."""
    OLD, NEW = "sig_pwsmoke_std", "sig_pwsmoke_queue"
    cell = args.smoke_cell
    gids = [str(g) for g in args.gen_ids]

    r = subprocess.run([sys.executable, "-m", "anamnesis.scripts.parallel_replay",
                        "--single-cell-ok", "--model", args.model,
                        "--model-path", args.model_path, "--run-dir", str(cell),
                        "--calib-dir", str(args.calib_dir),
                        "--manifest", str(cell / "replay_manifest.json"),
                        "--gpus", args.gpus.split(",")[0], "--workers-per-gpu", "1",
                        "--no-raw", "--no-resume", "--sig-subdir", OLD,
                        *(["--inject-from-metadata"] if args.inject_from_metadata else []),
                        "--gen-ids", *gids])
    if r.returncode != 0:
        raise SystemExit("standard-path leg failed")

    fleet = WorkerFleet(work_dir=args.work_dir, worker_ids=[0])
    fleet.spawn(lambda w: [sys.executable, "-m",
                           "anamnesis.scripts.persistent_replay_worker",
                           "--model", args.model, "--model-path", args.model_path,
                           "--calib-dir", str(args.calib_dir),
                           "--work-dir", str(args.work_dir), "--worker-id", str(w)],
                gpu_for_worker=lambda w: args.gpus.split(",")[0])
    fleet.wait_ready()
    try:
        job = {"run_dir": str(cell), "manifest": str(cell / "replay_manifest.json"),
               "gen_ids": args.gen_ids, "sig_subdir": NEW, "no_resume": True,
               "inject_from_metadata": bool(args.inject_from_metadata)}
        fleet.collect([fleet.submit(0, "smoke", job)], timeout_s=args.timeout_s)
    finally:
        fleet.stop()

    ok = True
    for g in args.gen_ids:
        for ext in ("npz", "json"):
            a = cell / OLD / f"gen_{g:03d}.{ext}"
            b = cell / NEW / f"gen_{g:03d}.{ext}"
            if not (a.exists() and b.exists() and a.read_bytes() == b.read_bytes()):
                ok = False
                print(f"MISMATCH gen_{g:03d}.{ext}")
    print("PERSISTENT_REPLAY_SMOKE:", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit("persistent replay-worker bitwise smoke FAILED")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--drive", action="store_true")
    ap.add_argument("--cells-json", type=Path, default=None)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--workers-per-gpu", type=int, default=1)
    ap.add_argument("--timeout-s", type=float, default=3600.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-cell", type=Path, default=None,
                    help="banked cell dir (run_dir with replay_manifest.json)")
    ap.add_argument("--inject-from-metadata", action="store_true")
    ap.add_argument("--gen-ids", type=int, nargs="+", default=[0, 1, 10, 11])
    args = ap.parse_args()

    if args.smoke:
        if args.smoke_cell is None:
            raise SystemExit("--smoke requires --smoke-cell")
        smoke(args)
    elif args.drive:
        if args.cells_json is None:
            raise SystemExit("--drive requires --cells-json")
        drive(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
