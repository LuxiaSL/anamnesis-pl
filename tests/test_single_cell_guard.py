"""Unit tests for the path-of-record guard (anamnesis/scripts/_single_cell_guard.py).

Pure CPU — exercises the refusal/allow logic with an isolated guard dir + explicit
job context. The bar (baton P1.2): the Gemma 24x-reload pattern — same script, same
model, different out dirs within one job context — cannot recur silently.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from anamnesis.scripts._single_cell_guard import (
    CONTEXT_ENV,
    ESCAPE_ENV,
    GUARD_DIR_ENV,
    TTL_ENV,
    enforce_single_cell_guard,
)

SCRIPT = "anamnesis.scripts.parallel_replay"
POINTER = "python -m anamnesis.scripts.vmb_a5_replay_multicell --cells-json ..."


@pytest.fixture()
def guard_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(GUARD_DIR_ENV, str(tmp_path / "guard"))
    monkeypatch.setenv(CONTEXT_ENV, "test-job-ctx")
    monkeypatch.delenv(ESCAPE_ENV, raising=False)
    monkeypatch.delenv(TTL_ENV, raising=False)
    return tmp_path


def _call(model: str, out_dir: Path, *, allow: bool = False) -> None:
    enforce_single_cell_guard(SCRIPT, model, out_dir,
                              allow_repeat=allow, multicell_pointer=POINTER)


def test_second_cell_same_model_refused(guard_env: Path) -> None:
    """The incident pattern: same script+model, different out dirs -> SystemExit."""
    _call("/models/gemma", guard_env / "cellA")
    with pytest.raises(SystemExit) as exc:
        _call("/models/gemma", guard_env / "cellB")
    msg = str(exc.value)
    assert "PATH-OF-RECORD GUARD" in msg
    assert "vmb_a5_replay_multicell" in msg
    assert "--single-cell-ok" in msg


def test_resume_same_out_dir_passes(guard_env: Path) -> None:
    """Re-running the same cell (resume) is never refused."""
    _call("/models/gemma", guard_env / "cellA")
    _call("/models/gemma", guard_env / "cellA")  # no raise


def test_different_model_passes(guard_env: Path) -> None:
    _call("/models/gemma", guard_env / "cellA")
    _call("/models/qwen", guard_env / "cellB")  # no raise


def test_different_script_passes(guard_env: Path) -> None:
    """gen -> replay of the same model in one job (GENALL->REPALL chain) is fine."""
    enforce_single_cell_guard("anamnesis.scripts.vmb_stage0_generate", "/models/gemma",
                              guard_env / "cellA", allow_repeat=False,
                              multicell_pointer=POINTER)
    _call("/models/gemma", guard_env / "cellB")  # replay side, no raise


def test_escape_hatch_flag(guard_env: Path) -> None:
    _call("/models/gemma", guard_env / "cellA")
    _call("/models/gemma", guard_env / "cellB", allow=True)  # no raise


def test_escape_hatch_env(guard_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _call("/models/gemma", guard_env / "cellA")
    monkeypatch.setenv(ESCAPE_ENV, "1")
    _call("/models/gemma", guard_env / "cellB")  # no raise


def test_escape_env_zero_still_refuses(guard_env: Path,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    _call("/models/gemma", guard_env / "cellA")
    monkeypatch.setenv(ESCAPE_ENV, "0")
    with pytest.raises(SystemExit):
        _call("/models/gemma", guard_env / "cellB")


def test_separate_job_contexts_isolated(guard_env: Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    _call("/models/gemma", guard_env / "cellA")
    monkeypatch.setenv(CONTEXT_ENV, "another-job")
    _call("/models/gemma", guard_env / "cellB")  # fresh context, no raise


def test_ttl_expiry_allows(guard_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Entries older than the TTL no longer count against a new invocation."""
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() - 13 * 3600)
    _call("/models/gemma", guard_env / "cellA")  # recorded 13h "ago"
    monkeypatch.setattr(time, "time", real_time)
    _call("/models/gemma", guard_env / "cellB")  # default TTL 12h -> no raise


def test_third_cell_message_counts_prior(guard_env: Path) -> None:
    _call("/models/gemma", guard_env / "cellA")
    _call("/models/gemma", guard_env / "cellB", allow=True)
    with pytest.raises(SystemExit) as exc:
        _call("/models/gemma", guard_env / "cellC")
    assert "invocation #3" in str(exc.value)


def test_unwritable_guard_dir_degrades_to_warning(
        guard_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bookkeeping failure must never kill a production launcher."""
    blocked = guard_env / "blocked_file"
    blocked.write_text("not a dir")
    monkeypatch.setenv(GUARD_DIR_ENV, str(blocked / "nope"))
    _call("/models/gemma", guard_env / "cellA")  # no raise despite unwritable dir
