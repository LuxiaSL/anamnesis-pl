"""GPU index resolution for launcher scripts running under a scheduler.

Heimdall (and any scheduler) communicates its GPU assignment by exporting
CUDA_VISIBLE_DEVICES for the job. Launcher scripts that spawn per-worker
subprocesses historically wrote the RAW --gpus indices into each worker's
CUDA_VISIBLE_DEVICES, silently overriding the scheduler's assignment: a job
assigned physical GPUs 5,6,7 but launched with `--gpus 0,1,2` stacked its
workers on physical 0,1,2 — colliding with other jobs while its own
assignment idled, and showing up as "untracked" processes in the ledger
(observed live 2026-07-12, vmb M3/M4 gen co-run).

Rule: `--gpus` values are LOGICAL slot indices. If the parent environment
carries CUDA_VISIBLE_DEVICES, slot i maps to its i-th entry; otherwise slots
are physical indices (bare-metal behaviour unchanged).
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def resolve_physical_gpus(requested: list[str]) -> list[str]:
    """Map logical --gpus slots onto the scheduler's CUDA_VISIBLE_DEVICES."""
    parent = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not parent:
        return requested
    visible = [g.strip() for g in parent.split(",") if g.strip()]
    out: list[str] = []
    for r in requested:
        try:
            idx = int(r)
        except ValueError as e:
            raise ValueError(f"--gpus entry {r!r} is not an integer slot index") from e
        if idx >= len(visible):
            raise ValueError(
                f"--gpus slot {idx} exceeds scheduler assignment "
                f"CUDA_VISIBLE_DEVICES={parent!r} ({len(visible)} device(s)); "
                f"request more GPUs at submit time or lower --gpus"
            )
        out.append(visible[idx])
    logger.info(f"GPU slots {requested} -> physical {out} "
                f"(scheduler CUDA_VISIBLE_DEVICES={parent!r})")
    return out
