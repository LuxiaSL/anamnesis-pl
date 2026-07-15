"""C3 certifying (f) — resample-diversity generation (PREFLIGHT §4 (f); 14f completeness item).

k=8 stochastic resamples PER prompt (same user_prompt, 8 distinct seeds) under V_temp steering.
Reuses vmb_stage0_generate with --seeds-per-class 8: each (topic, stratum) group is 8 same-prompt
samples → distinct-4-gram rate within the group = resample diversity. Text-only: NO replay, NO
signatures (the metric is lexical). A SHARED seed namespace across cells (VMBC3F-3B) makes V_temp
and its matched Rc null draw the identical 8 noise seeds per group → the only difference is the
injection. Cells: Vtemp + Rc1 × {L14,L21} × {.03,.1} + one α=0 baseline (no injection).
14f prediction: distinct-4-gram rises dose-ordered AND V_temp-specifically at α≤.1 (P=0.85).
First-read → outer loop; nothing stamped. Analyzer: vmb_c3_resample_diversity.py.
"""
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
MODEL, MPATH = "3b", "/models/llama-3.2-3b-instruct"
RUNS = "/models/anamnesis-extract/runs"
VEC_DIR = "/models/anamnesis-extract/battery/a5_vectors_3b_c3"
NPZ, STAMPS = f"{VEC_DIR}/a5_vectors.npz", f"{VEC_DIR}/a5_vectors_stamps.json"
PROMPTS = "pipeline/anamnesis/prompts/prompt_sets.json"
TAG = "VMBC3F-3B"  # SHARED across cells → matched seeds for Vtemp vs Rc
SITES, DOSES = [14, 21], [0.03, 0.1]


def submit(name, command, gpus, minutes):
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": ENV,
            "command": f"bash -c '{BASE} && {command}'"}
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["job"]["id"]


def gen_cmd(cell, extra):
    # --limit 160 = stratum 0 (20 topics) × 8 seeds = 20 resample-groups × k=8 (leaner, fast);
    # 256 tokens is ample for distinct-4-gram. All 8 GPUs.
    return (f"python -u -m anamnesis.scripts.vmb_stage0_generate --model {MODEL} --model-path {MPATH} "
            f"--prompts {PROMPTS} --out-run-dir {RUNS}/vmb_c3f_{MODEL}/{cell} --gpus 0,1,2,3,4,5,6,7 "
            f"--workers-per-gpu 4 --seeds-per-class 8 --limit 160 --max-new-tokens 256 "
            f"--seed-namespace {TAG} {extra}")


CELLS = []
for vec in ("Vtemp", "Rc1"):
    for s in SITES:
        for f in DOSES:
            CELLS.append((f"{vec}_L{s}_a{f}",
                          f"--inject-npz {NPZ} --inject-key {vec}_L{s} --inject-layer {s} "
                          f"--inject-alpha-frac {f} --inject-norms-json {STAMPS}"))
CELLS.append(("baseline_a0", ""))  # no injection = α=0 diversity floor

# one job, cells chained sequentially; --gpus 0..7 → request 8 (slots must match, runbook #5)
cmd = " && ".join(gen_cmd(c, e) for c, e in CELLS)
jid = submit("vmb-c3f-resample", cmd, gpus=8, minutes=60)
print(f"c3f resample-diversity gen job: {jid} ({len(CELLS)} cells)")
