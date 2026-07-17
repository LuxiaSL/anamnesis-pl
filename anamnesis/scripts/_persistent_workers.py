"""Shared persistent-worker file-queue harness (P4, canonical-ops 2026-07-16).

Generalizes the route-5 pattern — the only infra in the repo that solves
load-once AND crash-safety AND dynamic dispatch (STOP-drain proven on 97 orphaned
workers; resume kill-tested) — into a job-shape-agnostic module. The worker side
takes a ``handler(job: dict) -> dict[str, np.ndarray]``; the driver side
dispatches arbitrary JSON payloads and collects npz results. Replay-eval, gen,
and MT jobs all fit; nothing here is CMA-specific. This is the M7 primitive: at
671B-class load times, load-once workers are the difference between a sprint and
a slog, and streaming dispatch retires the barrier problem by construction.

Queue protocol (work_dir on node-local disk; all writes atomic .tmp -> rename):
  work_dir/jobs/w<ID>/job_<TAG>.json     driver -> worker; consumed (unlinked)
                                         AFTER its result lands
  work_dir/results/w<ID>_job_<TAG>.npz   worker -> driver
  work_dir/ready/w<ID>                   worker readiness marker
  work_dir/STOP                          drain marker: workers exit at next poll

Resume semantics: jobs are unlinked only after their result is written, so a
killed worker leaves its queue intact — respawning workers on the same work_dir
continues exactly where the fleet died. Driver-side state (e.g. route-5's CMA +
RNG checkpoint) stays the driver's responsibility.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_POLL_S = 0.05


# ── worker side ────────────────────────────────────────────────────────────────

@dataclass
class PersistentWorker:
    """Poll a per-worker job dir until STOP; one npz result per job, atomically.

    handler: consumes the parsed job dict, returns the arrays to savez. Raising
    inside the handler is FATAL for the worker (fail loud — a silently skipped
    job would stall the driver's collect), except that the job file is left in
    place so a respawned fleet retries it.
    on_stop: optional cleanup (e.g. detach a residual-write hook).
    """

    work_dir: Path
    worker_id: int
    handler: Callable[[dict], Mapping[str, np.ndarray]]
    poll_s: float = DEFAULT_POLL_S
    on_stop: Callable[[], None] | None = None

    @property
    def jobs_dir(self) -> Path:
        return self.work_dir / "jobs" / f"w{self.worker_id}"

    @property
    def results_dir(self) -> Path:
        return self.work_dir / "results"

    def mark_ready(self) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        ready = self.work_dir / "ready"
        ready.mkdir(exist_ok=True)
        (ready / f"w{self.worker_id}").touch()

    def run(self) -> int:
        """Process jobs until STOP; returns the number of jobs completed."""
        self.mark_ready()
        logger.info(f"worker {self.worker_id} READY ({self.jobs_dir})")
        n_done = 0
        try:
            while not (self.work_dir / "STOP").exists():
                jobs = sorted(self.jobs_dir.glob("job_*.json"))
                if not jobs:
                    time.sleep(self.poll_s)
                    continue
                jf = jobs[0]
                tag = jf.stem[len("job_"):]
                job = json.loads(jf.read_text())
                arrays = self.handler(job)
                out = self.results_dir / f"w{self.worker_id}_job_{tag}.npz"
                tmp = out.with_suffix(".tmp.npz")
                np.savez(tmp, **arrays)
                tmp.rename(out)
                jf.unlink()          # consume ONLY after the result landed
                n_done += 1
        finally:
            if self.on_stop is not None:
                self.on_stop()
        logger.info(f"worker {self.worker_id} STOP after {n_done} jobs")
        return n_done


# ── driver side ────────────────────────────────────────────────────────────────

@dataclass
class WorkerFleet:
    """Driver-side queue primitives: spawn env, dispatch, collect, drain."""

    work_dir: Path
    worker_ids: Sequence[int]
    procs: list[subprocess.Popen] = field(default_factory=list)

    def clear_stale_state(self) -> None:
        """Remove leftovers of a previous (crashed) fleet on this work_dir: a
        stale STOP kills fresh workers instantly; stale ready-markers impersonate
        dead workers; stale results corrupt collect. Pending JOB files are kept —
        they are the resume state."""
        (self.work_dir / "STOP").unlink(missing_ok=True)
        for d, pat in (("ready", "w*"), ("results", "*.npz")):
            base = self.work_dir / d
            if base.is_dir():
                for p in base.glob(pat):
                    p.unlink()

    def spawn(self, cmd_for_worker: Callable[[int], list[str]],
              gpu_for_worker: Callable[[int], str] | None = None,
              log_dir: Path | None = None,
              extra_env: Mapping[str, str] | None = None) -> list[subprocess.Popen]:
        """Spawn one subprocess per worker id with the standard pinned-thread env.
        cmd_for_worker builds each argv; gpu_for_worker sets CUDA_VISIBLE_DEVICES."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.clear_stale_state()
        logs = log_dir or (self.work_dir / "logs")
        logs.mkdir(parents=True, exist_ok=True)
        for w in self.worker_ids:
            env = {**os.environ,
                   "PYTHONPATH": os.environ.get("PYTHONPATH", "."),
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                   **(extra_env or {})}
            if gpu_for_worker is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu_for_worker(w)
            fh = open(logs / f"worker_{w}.log", "w")
            self.procs.append(subprocess.Popen(
                cmd_for_worker(w), env=env, stdout=fh, stderr=subprocess.STDOUT))
        return self.procs

    def wait_ready(self, timeout_s: float = 900.0) -> None:
        t0 = time.time()
        while True:
            ready = [(self.work_dir / "ready" / f"w{w}").exists()
                     for w in self.worker_ids]
            if all(ready):
                return
            dead = [p for p in self.procs if p.poll() is not None]
            if dead:
                raise SystemExit(
                    f"{len(dead)} workers exited before READY "
                    f"(rc={[p.returncode for p in dead]}) — see worker logs")
            if time.time() - t0 > timeout_s:
                raise SystemExit(f"workers not ready after {timeout_s}s: {ready}")
            time.sleep(0.5)

    def submit(self, worker_id: int, tag: str, payload: dict) -> Path:
        """Atomically place one job; returns the expected result path."""
        jf = self.work_dir / "jobs" / f"w{worker_id}" / f"job_{tag}.json"
        jf.parent.mkdir(parents=True, exist_ok=True)
        tmp = jf.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.rename(jf)
        return self.work_dir / "results" / f"w{worker_id}_job_{tag}.npz"

    def collect(self, expected: Sequence[Path], timeout_s: float = 3600.0,
                poll_s: float = DEFAULT_POLL_S,
                consume: bool = True) -> dict[Path, dict[str, np.ndarray]]:
        """Wait for every expected result npz; returns {path: {key: array}}.
        Results are unlinked after reading when consume=True."""
        pending = list(expected)
        out: dict[Path, dict[str, np.ndarray]] = {}
        t0 = time.time()
        while pending:
            done = [p for p in pending if p.exists()]
            for p in done:
                with np.load(p, allow_pickle=False) as z:
                    out[p] = {k: z[k] for k in z.files}
                if consume:
                    p.unlink()
                pending.remove(p)
            if pending:
                dead = [pr for pr in self.procs if pr.poll() is not None]
                if dead and len(dead) == len(self.procs):
                    raise SystemExit(
                        f"all workers exited with {len(pending)} results pending "
                        f"— see worker logs under {self.work_dir / 'logs'}")
                if time.time() - t0 > timeout_s:
                    raise SystemExit(
                        f"collect timed out after {timeout_s}s; missing: "
                        f"{[str(p) for p in pending[:5]]}")
                time.sleep(poll_s)
        return out

    def stop(self, timeout_s: float = 120.0) -> None:
        """Idempotent drain: signal STOP and reap whatever workers remain."""
        (self.work_dir / "STOP").touch()
        for p in self.procs:
            try:
                p.wait(timeout=timeout_s)
            except Exception:  # noqa: BLE001 — a stuck worker must not block teardown
                p.kill()
        self.procs = []
