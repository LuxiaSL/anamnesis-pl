"""Queue-mechanics tests for the shared persistent-worker harness (P4).

Pure CPU/filesystem — a trivial handler stands in for the model-bound job. The
GPU acceptance legs (route-5 smoke through the module + a replay-sig job type
with parity vs the standard path) run on-node.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from anamnesis.scripts._persistent_workers import PersistentWorker, WorkerFleet


def _square_handler(job: dict) -> dict[str, np.ndarray]:
    x = np.asarray(job["x"], dtype=np.float64)
    return {"y": x * x, "job_id": np.array([job["id"]])}


def _run_worker_thread(work_dir: Path, worker_id: int) -> tuple[threading.Thread, list]:
    result: list[int] = []
    w = PersistentWorker(work_dir=work_dir, worker_id=worker_id,
                         handler=_square_handler, poll_s=0.01)
    t = threading.Thread(target=lambda: result.append(w.run()), daemon=True)
    t.start()
    return t, result


def test_dispatch_collect_roundtrip(tmp_path: Path) -> None:
    fleet = WorkerFleet(work_dir=tmp_path, worker_ids=[0, 1])
    threads = [_run_worker_thread(tmp_path, w) for w in (0, 1)]
    fleet.wait_ready(timeout_s=10)

    expected = [fleet.submit(w, f"{i:03d}", {"id": i, "x": [i, i + 1]})
                for i, w in enumerate([0, 1, 0, 1])]
    got = fleet.collect(expected, timeout_s=10)
    assert len(got) == 4
    for p, arrays in got.items():
        i = int(arrays["job_id"][0])
        assert np.array_equal(arrays["y"], np.array([i * i, (i + 1) ** 2], dtype=np.float64))
    assert not any(p.exists() for p in expected), "collect(consume=True) must unlink"

    fleet.stop(timeout_s=5)
    for t, res in threads:
        t.join(timeout=5)
        assert res and res[0] == 2, "each worker should have completed 2 jobs"


def test_job_files_survive_until_result(tmp_path: Path) -> None:
    """Resume semantics: a job is unlinked only after its result lands."""
    fleet = WorkerFleet(work_dir=tmp_path, worker_ids=[0])
    # no worker running: submit stays pending
    res_path = fleet.submit(0, "007", {"id": 7, "x": [3]})
    jf = tmp_path / "jobs" / "w0" / "job_007.json"
    assert jf.exists() and not res_path.exists()

    # a later fleet on the same work_dir keeps pending jobs through cleanup
    fleet.clear_stale_state()
    assert jf.exists(), "clear_stale_state must NOT delete pending jobs (resume state)"

    t, res = _run_worker_thread(tmp_path, 0)
    got = fleet.collect([res_path], timeout_s=10)
    assert np.array_equal(got[res_path]["y"], np.array([9.0]))
    assert not jf.exists(), "job consumed after result"
    (tmp_path / "STOP").touch()
    t.join(timeout=5)


def test_clear_stale_state_removes_stop_ready_results(tmp_path: Path) -> None:
    (tmp_path / "STOP").touch()
    (tmp_path / "ready").mkdir()
    (tmp_path / "ready" / "w0").touch()
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "w0_job_zzz.npz").touch()
    fleet = WorkerFleet(work_dir=tmp_path, worker_ids=[0])
    fleet.clear_stale_state()
    assert not (tmp_path / "STOP").exists()
    assert not (tmp_path / "ready" / "w0").exists()
    assert not (tmp_path / "results" / "w0_job_zzz.npz").exists()


def test_stop_drains_idle_workers(tmp_path: Path) -> None:
    fleet = WorkerFleet(work_dir=tmp_path, worker_ids=[0])
    t, res = _run_worker_thread(tmp_path, 0)
    fleet.wait_ready(timeout_s=10)
    fleet.stop(timeout_s=5)
    t.join(timeout=5)
    assert res and res[0] == 0


def test_on_stop_runs_after_drain(tmp_path: Path) -> None:
    hits: list[str] = []
    w = PersistentWorker(work_dir=tmp_path, worker_id=3, handler=_square_handler,
                         poll_s=0.01, on_stop=lambda: hits.append("cleanup"))
    (tmp_path / "STOP").touch()
    assert w.run() == 0
    assert hits == ["cleanup"]


def test_collect_timeout_fails_loud(tmp_path: Path) -> None:
    fleet = WorkerFleet(work_dir=tmp_path, worker_ids=[0])
    ghost = tmp_path / "results" / "w0_job_never.npz"
    with pytest.raises(SystemExit, match="timed out"):
        fleet.collect([ghost], timeout_s=0.2, poll_s=0.05)


def test_handler_failure_leaves_job_for_retry(tmp_path: Path) -> None:
    """A handler exception kills the worker loudly but keeps the job file."""
    def bad_handler(job: dict) -> dict[str, np.ndarray]:
        raise RuntimeError("boom")

    fleet = WorkerFleet(work_dir=tmp_path, worker_ids=[0])
    fleet.submit(0, "013", {"id": 13, "x": [1]})
    w = PersistentWorker(work_dir=tmp_path, worker_id=0, handler=bad_handler,
                         poll_s=0.01)
    with pytest.raises(RuntimeError, match="boom"):
        w.run()
    assert (tmp_path / "jobs" / "w0" / "job_013.json").exists(), \
        "failed job must remain queued for a respawned fleet"
