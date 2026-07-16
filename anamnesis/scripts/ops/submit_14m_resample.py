"""14m item 2 — resample-diversity generation for V7 (14j (f) if-free clause).

c3f pattern: k=8 stochastic resamples per prompt (same user_prompt, 8 distinct seeds), SHARED
seed namespace so V7 and its matched Rband null draw identical noise seeds. Cells: V7_L14 +
Rband1_L14 × α{.03,.1} + one α=0 baseline. Text-only (no replay/sigs — the metric is lexical).
8 GPUs. Analyzer (CPU, after): vmb_c3_resample_diversity --null-prefixes RBAND.
First-read → outer loop. Run: HEIMDALL_{API,WORK_DIR,VENV} exported.
"""
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
MODEL, MPATH = "3b", "/models/llama-3.2-3b-instruct"
RUNS = "/models/anamnesis-extract/runs"
VEC_DIR = "/models/anamnesis-extract/battery/a5_vectors_3b_b7"
NPZ, STAMPS = f"{VEC_DIR}/a5_vectors.npz", f"{VEC_DIR}/a5_vectors_stamps.json"
PROMPTS = "pipeline/anamnesis/prompts/prompt_sets.json"
TAG = "VMBB7F-3B"  # SHARED across cells → matched seeds for V7 vs Rband
SITE, DOSES = 14, [0.03, 0.1]


def submit(name, command, gpus, minutes):
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": ENV,
            "command": f"bash -c '{BASE} && {command}'"}
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["job"]["id"]


def gen_cmd(cell, extra):
    return (f"python -u -m anamnesis.scripts.vmb_stage0_generate --model {MODEL} --model-path {MPATH} "
            f"--prompts {PROMPTS} --out-run-dir {RUNS}/vmb_b7f_{MODEL}/{cell} --gpus 0,1,2,3,4,5,6,7 "
            f"--workers-per-gpu 4 --seeds-per-class 8 --limit 160 --max-new-tokens 256 "
            f"--seed-namespace {TAG} {extra}")


CELLS = []
for vec in ("V7", "Rband1"):
    for f in DOSES:
        CELLS.append((f"{vec}_L{SITE}_a{f}",
                      f"--inject-npz {NPZ} --inject-key {vec}_L{SITE} --inject-layer {SITE} "
                      f"--inject-alpha-frac {f} --inject-norms-json {STAMPS}"))
CELLS.append(("baseline_a0", ""))

cmd = " && ".join(gen_cmd(c, e) for c, e in CELLS)
jid = submit("vmb-14m-b7f-resample", cmd, gpus=8, minutes=50)
print(f"14m resample gen: job={jid} ({len(CELLS)} cells → vmb_b7f_3b)")
