"""Q5 RIDER — echo-sandbox modes-generality gen (8B, RUN3_ORIGINAL_MODES verbatim). Fire-and-forget.

"Is the regime-indexed code a general law of signature space?" (echo-sandbox owns the analysis). This
submits ONLY the generation + replay; artifacts announced to the desk. Charter: steering-matrix appendix
Q5 rider; spec `~/projects/echo-sandbox/docs/SPEC-modes-generality.md`; park `PARK-PartF-Q5-cohort-2026-07-19.md`.

Instrument = the 5 RUN3 process/format-free modes (associative/compressed/deliberative/pedagogical/structured),
pulled VERBATIM from the source run's metadata (`run_8b_r2_equivalent`, local-only → its 483K metadata.json is
rsynced to node1; parallel_generate regenerates fresh, so no source generations are needed). Topics = all 60
from prompt_sets.json (`parallel_generate` draws topics from --prompts; the source's 15 are the leading prefix).

⚠ TWO FLAGS for echo-sandbox (announced, not resolved here):
  (1) SKIP-DELTA: parallel_generate skips source cells (15 topics × reps 0-1). With --num-reps 3 the BALANCED
      ≥40-topic set = topics 15-59 (3 full reps × 5 modes = 675 gens); topics 0-14 appear at rep 2 only (extras).
  (2) TIER→V3 BRIDGE: replay produces signatures_v3, NOT the legacy tier-sliced keys (features_tier2/_tier2_5)
      the echo-sandbox bridge expects. The v3 sigs are produced; the tier bridge/adaptation is echo-sandbox-side.

    python -m anamnesis.scripts.ops.submit_q5_8b [--fire]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

RUNS = "/models/anamnesis-extract/runs"
MODEL, MPATH = "8b", "/models/llama-3.1-8b-instruct"
CALIB = "/models/anamnesis-extract/calibration/8b"
SRC_META_LOCAL = "outputs/runs/run_8b_r2_equivalent/metadata.json"
SRC_DIR_NODE = f"{RUNS}/run_8b_r2_equivalent"
OUT = f"{RUNS}/vmb_q5_8b_r2"
PROMPTS = "pipeline/anamnesis/prompts/prompt_sets.json"
ENV = {"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1"}
BASE = base("HF_HUB_OFFLINE=1")


def submit(name, cmd, gpus, minutes, deps=None):
    s = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
         "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": dict(ENV),
         "command": f"bash -c '{BASE} && {cmd}'"}
    if deps:
        s["depends_on"] = deps
    req = urllib.request.Request(API, data=json.dumps({"spec": s}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["job"]["id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", action="store_true")
    ap.add_argument("--num-reps", type=int, default=3)
    args = ap.parse_args()

    gen = (f"python -u -m anamnesis.scripts.parallel_generate --model {MODEL} --model-path {MPATH} "
           f"--meta-from {SRC_DIR_NODE}/metadata.json --prompts {PROMPTS} --out-run-dir {OUT} "
           f"--num-reps {args.num_reps} --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 4")
    rep = (f"python -u -m anamnesis.scripts.parallel_replay --model {MODEL} --model-path {MPATH} "
           f"--run-dir {OUT} --calib-dir {CALIB} --manifest {OUT}/replay_manifest.json "
           f"--gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 4 --no-raw")
    print("===== Q5 (echo-sandbox modes-generality) =====")
    print(f"  rsync {SRC_META_LOCAL} -> node1:{SRC_DIR_NODE}/metadata.json")
    print(f"  [gen] {gen}")
    print(f"  [rep] {rep}")
    if not args.fire:
        print("(dry-run: nothing submitted)")
        return
    subprocess.run(["ssh", "node1", f"mkdir -p {SRC_DIR_NODE}"], check=True)
    subprocess.run(["rsync", "-a", SRC_META_LOCAL, f"node1:{SRC_DIR_NODE}/metadata.json"], check=True)
    g = submit("vmb-q5-gen", gen, 8, 60)
    r = submit("vmb-q5-replay", rep, 8, 50, [g])
    print(f"Q5 gen={g} replay={r}  (announce to echo-sandbox; flags: skip-delta topics15-59 balanced, tier→v3 bridge)")


if __name__ == "__main__":
    main()
