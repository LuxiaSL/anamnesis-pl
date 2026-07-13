"""Deployment-specific Heimdall constants, sourced from environment variables.

The public repo carries no cluster identifiers (host, account names, credential
paths) — history was scrubbed of them at first publication (2026-07-13). Set
these in your shell before running any submit_* script; the working values for
the project's own cluster live in the LOCAL ops runbook (memory:
vmb-ops-runbook-2026-07-12), never in this repo:

  HEIMDALL_API       job-submission endpoint, e.g. http://<host>:<port>/api/v1/jobs
  HEIMDALL_WORK_DIR  remote checkout root that jobs cd into
  HEIMDALL_VENV      remote venv activate path (a shared venv; see node1 rules)
  HF_TOKEN           HuggingFace token — only chains that download weights need it
"""
import os


def _req(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(
            f"{name} is not set — see anamnesis/scripts/ops/_ops_env.py for the "
            "required environment variables (values live in the local ops runbook)")
    return v


API = _req("HEIMDALL_API")
WORK_DIR = _req("HEIMDALL_WORK_DIR")


def base(extra_exports: str = "") -> str:
    """The remote job preamble: activate venv, cd to checkout, set PYTHONPATH."""
    exports = "PYTHONPATH=$PWD/pipeline PYTHONUNBUFFERED=1"
    if extra_exports:
        exports += f" {extra_exports}"
    return (f"source {_req('HEIMDALL_VENV')} && cd {WORK_DIR} && "
            f"export {exports}")


def hf_token() -> str:
    return _req("HF_TOKEN")
