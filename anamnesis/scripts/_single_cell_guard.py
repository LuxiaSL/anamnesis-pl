"""Path-of-record enforcement: per-cell loops over multi-cell rosters FAIL LOUD.

The session-10 Gemma incident: a per-cell loop invoked a single-cell launcher 24
times against the same model — 24 model reloads — with the multicell path already
documented as canonical. Documentation alone demonstrably doesn't hold, so the
single-cell launchers now refuse the pattern at invocation #2.

Detection: each launcher records (script, model_path, out_dir) under a "job context"
key. A second invocation in the same context with the SAME script + SAME model but a
DIFFERENT out dir is the per-cell-loop signature and is refused with a pointer to
the multicell path. Resume re-runs hit the same out dir and always pass.

Job context resolution (first match wins):
  1. ``ANAMNESIS_JOB_CONTEXT`` env var (set it explicitly in drivers if needed)
  2. parent PID — a bash ``a && b`` chain or loop shares one parent; a fresh shell
     (new Heimdall job attempt, new terminal) is a fresh context.

Escape hatches for DELIBERATE sequential single-cell work:
  - ``--single-cell-ok`` on the launcher CLI
  - ``ANAMNESIS_SINGLE_CELL_OK=1`` in the environment (for buried invocations)

Entries expire after ``ANAMNESIS_GUARD_TTL_HOURS`` (default 12) so a long-lived
interactive shell doesn't trip on unrelated work days apart. State lives in
per-user tmp; all bookkeeping I/O is best-effort — a broken tmp dir degrades to a
warning, never kills a run. Only the refusal itself raises (SystemExit).
"""
from __future__ import annotations

import getpass
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

ESCAPE_FLAG = "--single-cell-ok"
ESCAPE_ENV = "ANAMNESIS_SINGLE_CELL_OK"
CONTEXT_ENV = "ANAMNESIS_JOB_CONTEXT"
TTL_ENV = "ANAMNESIS_GUARD_TTL_HOURS"
GUARD_DIR_ENV = "ANAMNESIS_GUARD_DIR"
DEFAULT_TTL_HOURS = 12.0
STALE_FILE_FACTOR = 4.0  # guard files untouched for TTL*4 are deleted opportunistically


@dataclass(frozen=True)
class _Entry:
    ts: float
    script: str
    model_path: str
    out_dir: str


def _ttl_seconds() -> float:
    try:
        return float(os.environ.get(TTL_ENV, DEFAULT_TTL_HOURS)) * 3600.0
    except ValueError:
        return DEFAULT_TTL_HOURS * 3600.0


def _context_key() -> str:
    explicit = os.environ.get(CONTEXT_ENV)
    if explicit:
        return explicit
    return f"ppid{os.getppid()}"


def _guard_dir() -> Path:
    override = os.environ.get(GUARD_DIR_ENV)
    if override:
        return Path(override)
    try:
        user = getpass.getuser()
    except Exception:
        user = f"uid{os.getuid()}"
    return Path(tempfile.gettempdir()) / f"anamnesis_single_cell_guard_{user}"


def _record_path() -> Path:
    # context keys can contain path-hostile chars when set explicitly; sanitize
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in _context_key())
    return _guard_dir() / f"{safe}.jsonl"


def _load_entries(path: Path, now: float, ttl: float) -> list[_Entry]:
    entries: list[_Entry] = []
    try:
        if not path.exists():
            return entries
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                e = _Entry(ts=float(d["ts"]), script=str(d["script"]),
                           model_path=str(d["model_path"]), out_dir=str(d["out_dir"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue  # corrupt line: skip, don't crash a production launcher
            if now - e.ts <= ttl:
                entries.append(e)
    except OSError as err:
        logger.warning(f"single-cell guard: could not read {path}: {err}")
    return entries


def _write_entries(path: Path, entries: list[_Entry]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(asdict(e)) + "\n" for e in entries))
        tmp.replace(path)
    except OSError as err:
        logger.warning(f"single-cell guard: could not write {path}: {err}")


def _prune_stale_files(guard_dir: Path, now: float, ttl: float) -> None:
    try:
        if not guard_dir.is_dir():
            return
        for f in guard_dir.glob("*.jsonl"):
            try:
                if now - f.stat().st_mtime > ttl * STALE_FILE_FACTOR:
                    f.unlink()
            except OSError:
                pass
    except OSError:
        pass


def enforce_single_cell_guard(
    script: str,
    model_path: str,
    out_dir: Path | str,
    *,
    allow_repeat: bool,
    multicell_pointer: str,
) -> None:
    """Refuse per-cell-loop invocations of a single-cell launcher; record this one.

    Args:
        script: fully qualified module name of the calling launcher.
        model_path: the model being (re)loaded.
        out_dir: this invocation's output root (run dir / out-run-dir).
        allow_repeat: the CLI escape hatch (``--single-cell-ok``) — the env
            escape (``ANAMNESIS_SINGLE_CELL_OK=1``) is honored here as well.
        multicell_pointer: one-line pointer to the canonical multicell command.
    """
    now = time.time()
    ttl = _ttl_seconds()
    out_str = str(Path(out_dir).expanduser().resolve())
    model_str = str(Path(model_path).expanduser().resolve()) if os.path.exists(
        os.path.expanduser(str(model_path))) else str(model_path)
    allow = allow_repeat or os.environ.get(ESCAPE_ENV, "") not in ("", "0")

    path = _record_path()
    entries = _load_entries(path, now, ttl)
    prior = [e for e in entries
             if e.script == script and e.model_path == model_str
             and e.out_dir != out_str]

    if prior and not allow:
        prior_dirs = sorted({e.out_dir for e in prior})
        shown = "\n".join(f"    {d}" for d in prior_dirs[:8])
        raise SystemExit(
            f"\nPATH-OF-RECORD GUARD ({script}):\n"
            f"  invocation #{len(prior_dirs) + 1} against the same model within one job "
            f"context ({_context_key()}), each with a DIFFERENT out dir:\n{shown}\n"
            f"    {out_str}  <- this invocation (REFUSED)\n"
            f"  model: {model_str}\n"
            f"  Per-cell loops reload the model once per cell (the session-10 Gemma "
            f"24x-reload incident). Multi-cell rosters go through the multicell path:\n"
            f"    {multicell_pointer}\n"
            f"  If sequential single-cell invocation is deliberate, re-run with "
            f"{ESCAPE_FLAG} (or {ESCAPE_ENV}=1).\n"
        )
    if prior and allow:
        logger.warning(
            f"single-cell guard: repeat single-cell invocation allowed by escape hatch "
            f"({len(prior)} prior in this context; multicell path: {multicell_pointer})"
        )

    # record this invocation (dedupe on identical triple, keep newest ts)
    keep = [e for e in entries
            if (e.script, e.model_path, e.out_dir) != (script, model_str, out_str)]
    keep.append(_Entry(ts=now, script=script, model_path=model_str, out_dir=out_str))
    _write_entries(path, keep)
    _prune_stale_files(_guard_dir(), now, ttl)
