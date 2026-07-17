"""Pytest promotion of the two multicell bitwise smokes (parity gate, one command).

Wraps the standalone smoke scripts — which remain the on-node tool and the single
source of truth for what "bitwise-identical" means:

    anamnesis/scripts/vmb_a5_multicell_smoke.py         (gen path)
    anamnesis/scripts/vmb_a5_replay_multicell_smoke.py  (replay path)

Both smokes need a GPU + model weights + banked cells, so this test is config-driven:
set ``ANAMNESIS_SMOKE_CONFIG`` to a JSON file and run ``pytest tests/test_multicell_bitwise.py``.
Without the env var (or without a CUDA device) the tests SKIP — they never silently pass.

Config schema (paths are examples — use your node's model/output roots)::

    {
      "model": "8b",
      "model_path": "/models/llama-3.1-8b-instruct",
      "scratch": "/tmp/mc_smoke",                      // optional, default under $TMPDIR
      "gen": {
        "npz":   "<...>/a5_vectors_full.npz",
        "norms": "<...>/a5_vectors_stamps.json",
        "cellA": "V3_L16:16:0.1",                      // optional, key:layer:frac
        "cellB": "V3_L20:20:0.1"                       // optional
      },
      "replay": {
        "calib_dir": "<...>/calibration/llama31_8b",
        "cellA": "<...>/runs/<cell_at_one_layer>",
        "cellB": "<...>/runs/<cell_at_another_layer>",
        "gen_ids": [0, 1, 10, 11]                      // optional
      }
    }

Omit the "gen" or "replay" block to skip that half (e.g. replay-only parity checks).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

CONFIG_ENV = "ANAMNESIS_SMOKE_CONFIG"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _load_config() -> dict[str, Any]:
    cfg_path = os.environ.get(CONFIG_ENV)
    if not cfg_path:
        pytest.skip(
            f"{CONFIG_ENV} not set — the bitwise smokes need a GPU + model + banked "
            f"cells; see this file's docstring for the config schema"
        )
    if not _cuda_available():
        pytest.skip("no CUDA device — the bitwise smokes run model forwards")
    p = Path(cfg_path)
    if not p.exists():
        pytest.fail(f"{CONFIG_ENV}={cfg_path} does not exist")
    try:
        cfg: dict[str, Any] = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        pytest.fail(f"{CONFIG_ENV}={cfg_path} is not valid JSON: {e}")
    for key in ("model", "model_path"):
        if key not in cfg:
            pytest.fail(f"smoke config missing required key '{key}'")
    return cfg


def _scratch(cfg: dict[str, Any], sub: str) -> Path:
    base = Path(cfg.get("scratch") or (Path(tempfile.gettempdir()) / "mc_smoke"))
    d = base / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_smoke(module: str, argv: list[str]) -> None:
    """Run a smoke script as a subprocess; the script's exit code IS the verdict.

    cwd is REPO_ROOT's parent: the gen smoke resolves its prompts file via the
    relative path "pipeline/anamnesis/prompts/..." (the documented on-node
    invocation runs from the dir that CONTAINS pipeline/)."""
    cmd = [sys.executable, "-m", module, *argv]
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        cmd, cwd=REPO_ROOT.parent, env=env, capture_output=True, text=True
    )
    tail = "\n".join((result.stdout + result.stderr).splitlines()[-30:])
    assert result.returncode == 0, (
        f"{module} exited rc={result.returncode} — bitwise smoke FAILED.\n"
        f"cmd: {' '.join(cmd)}\nlast output:\n{tail}"
    )
    combined = result.stdout + result.stderr
    assert "PASS" in combined, (
        f"{module} exited 0 but never printed PASS — treat as failure.\n{tail}"
    )


def test_gen_multicell_bitwise() -> None:
    """Load-once multicell GEN path is byte-identical to the reload-per-cell path."""
    cfg = _load_config()
    gen = cfg.get("gen")
    if not gen:
        pytest.skip("smoke config has no 'gen' block")
    for key in ("npz", "norms"):
        if key not in gen:
            pytest.fail(f"smoke config 'gen' block missing '{key}'")
    argv = [
        "--model", cfg["model"], "--model-path", cfg["model_path"],
        "--npz", gen["npz"], "--norms", gen["norms"],
        "--scratch", str(_scratch(cfg, "gen")),
    ]
    if "cellA" in gen:
        argv += ["--cellA", gen["cellA"]]
    if "cellB" in gen:
        argv += ["--cellB", gen["cellB"]]
    _run_smoke("anamnesis.scripts.vmb_a5_multicell_smoke", argv)


def test_replay_multicell_bitwise() -> None:
    """Load-once multicell REPLAY path is byte-identical to reload-per-cell replay."""
    cfg = _load_config()
    rep = cfg.get("replay")
    if not rep:
        pytest.skip("smoke config has no 'replay' block")
    for key in ("calib_dir", "cellA", "cellB"):
        if key not in rep:
            pytest.fail(f"smoke config 'replay' block missing '{key}'")
    argv = [
        "--model", cfg["model"], "--model-path", cfg["model_path"],
        "--calib-dir", rep["calib_dir"],
        "--cellA", rep["cellA"], "--cellB", rep["cellB"],
    ]
    gen_ids = rep.get("gen_ids")
    if gen_ids:
        argv += ["--gen-ids", *[str(g) for g in gen_ids]]
    _run_smoke("anamnesis.scripts.vmb_a5_replay_multicell_smoke", argv)
