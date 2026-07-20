"""Cancel OUR queued/running 8-GPU steering-matrix jobs (to yield the node to a coworker), safely.

Filters the Heimdall job list to jobs whose spec.name starts with a `vmb-` prefix AND request 8 GPUs
AND are queued/running — i.e. our gen/replay jobs. NEVER touches anything else (kv-mla-capture, jspace-*,
a coworker's work). Dry-run by DEFAULT (lists what it WOULD cancel); pass --fire to actually cancel.

Pairs with the resubmit path: after cancelling, resubmit gen->replay only (builds are done) via
  python -m anamnesis.scripts.ops.submit_steering_matrix --fire --gen-only --model {8b,qwen,gemma}
  python -m anamnesis.scripts.ops.submit_steering_matrix_field8b --fire --gen-only
  python -m anamnesis.scripts.ops.submit_q5_8b --fire
See `outputs/battery/STEERING-MATRIX-RESUBMIT-PLAN-2026-07-19.md`. Env: HEIMDALL_API + node1 ssh.

    python -m anamnesis.scripts.ops.cancel_our_8gpu            # dry-run: list our cancellable 8-GPU jobs
    python -m anamnesis.scripts.ops.cancel_our_8gpu --fire     # actually cancel them
"""
from __future__ import annotations

import argparse
import json
import subprocess

from anamnesis.scripts.ops._ops_env import API

OUR_PREFIX = "vmb-"                         # our jobs' spec.name prefix
NEVER = ("kv-mla-capture", "jspace")       # belt-and-suspenders: never cancel these


def list_jobs() -> list[dict]:
    # API is the .../jobs endpoint; fetch via node1 (curl) to avoid host-routing assumptions
    out = subprocess.run(["ssh", "node1", f"curl -s {API}"], capture_output=True, text=True, timeout=40)
    d = json.loads(out.stdout)
    return d.get("jobs", d) if isinstance(d, dict) else d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", action="store_true", help="actually cancel (default: dry-run list)")
    ap.add_argument("--gpus", type=int, default=8, help="only cancel jobs requesting this many gpus")
    args = ap.parse_args()

    jobs = list_jobs()
    targets = []
    for j in jobs:
        sp = j.get("spec", {}) or {}
        nm, g, st = sp.get("name") or "", sp.get("gpus"), j.get("status")
        if any(bad in nm for bad in NEVER):
            continue
        if nm.startswith(OUR_PREFIX) and g == args.gpus and st in ("queued", "running"):
            targets.append((j.get("id"), nm, st))

    if not targets:
        print("no cancellable OUR 8-GPU jobs (queued/running) found.")
        return
    print(f"{'CANCELLING' if args.fire else 'WOULD CANCEL'} {len(targets)} jobs:")
    for jid, nm, st in targets:
        print(f"  {jid}  {nm}  [{st}]")
    if not args.fire:
        print("\n(dry-run — pass --fire to cancel)")
        return
    # cancel via the REST endpoint POST {API}/{id}/cancel (verified; the `heimdall cancel` CLI
    # wants a different BASE-url env and 404s with the submit-style HEIMDALL_API).
    for jid, nm, _ in targets:
        r = subprocess.run(["ssh", "node1", f"curl -s -o /dev/null -w '%{{http_code}}' -X POST {API}/{jid}/cancel"],
                           capture_output=True, text=True, timeout=30)
        print(f"  cancel {nm} {jid}: HTTP {r.stdout.strip()}")


if __name__ == "__main__":
    main()
