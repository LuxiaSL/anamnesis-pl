"""R7 gen+replay via the session-6 MULTICELL optimization — GENALL(8 GPU) → REPALL(8 GPU).

Supersedes the per-vector chain submit (which queued 8×{4-GPU gen,2-GPU replay}). Loads the 8B
model ONCE per worker and loops all 24 cells (8 spectral-split vectors × α{.03,.1,.3} at L16, n=40).
Writes both cells-json locally, rsyncs to node1, then POSTs one 8-GPU gen job → one 8-GPU replay job.
First-read → outer loop. Run: HEIMDALL_{API,WORK_DIR,VENV} exported (+ node1 ssh for the rsync).
"""
import json
import subprocess
import urllib.request
from pathlib import Path

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
MODEL, MPATH = "8b", "/models/llama-3.1-8b-instruct"
RUNS = "/models/anamnesis-extract/runs"
CALIB = "/models/anamnesis-extract/calibration/8b"
VEC_DIR = "/models/anamnesis-extract/battery/a5_vectors_8b_b5"
NPZ, STAMPS = f"{VEC_DIR}/a5_vectors.npz", f"{VEC_DIR}/a5_vectors_stamps.json"
PROMPTS = "pipeline/anamnesis/prompts/prompt_sets.json"
# derived from the deployment env var (WORK_DIR = remote checkout root) — never hardcode host/home
NODE_CELLS_DIR = f"{WORK_DIR}/_r7_cells"
LOCAL_CELLS_DIR = Path("/tmp/claude-output/r7_cells")
SSH_CELLS_DEST = f"node1:{WORK_DIR}/_r7_cells/"
MAP_SITE, LADDER, N_LIMIT = 16, [0.03, 0.1, 0.3], 40
VECTORS = ["V3top_L16", "V3tail_L16", "Rtop1_L16", "Rtop2_L16", "Rtop3_L16",
           "Rtail1_L16", "Rtail2_L16", "Rtail3_L16"]

gen_cells, rep_cells = [], []
for key in VECTORS:
    for frac in LADDER:
        c = f"{key}_a{frac}"
        run = f"{RUNS}/vmb_b5_{MODEL}/{c}"
        gen_cells.append({"out_run_dir": run, "seed_namespace": f"VMBB5-{MODEL.upper()}-{c}",
                          "inject_key": key, "inject_layer": MAP_SITE, "inject_alpha_frac": frac})
        rep_cells.append({"run_dir": run, "manifest": f"{run}/replay_manifest.json"})

LOCAL_CELLS_DIR.mkdir(parents=True, exist_ok=True)
(LOCAL_CELLS_DIR / "gen_cells.json").write_text(json.dumps({"cells": gen_cells}, indent=1))
(LOCAL_CELLS_DIR / "rep_cells.json").write_text(json.dumps({"cells": rep_cells}, indent=1))
subprocess.run(["ssh", "node1", f"mkdir -p {NODE_CELLS_DIR}"], check=True)
subprocess.run(["rsync", "-a", f"{LOCAL_CELLS_DIR}/gen_cells.json",
                f"{LOCAL_CELLS_DIR}/rep_cells.json", SSH_CELLS_DEST], check=True)


def submit(name, command, gpus, minutes, depends_on=None):
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": ENV,
            "command": f"bash -c '{BASE} && {command}'"}
    if depends_on:
        spec["depends_on"] = depends_on
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["job"]["id"]


gen = (f"python -u -m anamnesis.scripts.vmb_a5_gen_multicell --model {MODEL} --model-path {MPATH} "
       f"--prompts {PROMPTS} --cells-json {NODE_CELLS_DIR}/gen_cells.json --inject-npz {NPZ} "
       f"--inject-norms-json {STAMPS} --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 6 "
       f"--seeds-per-class 1 --limit {N_LIMIT}")
rep = (f"python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model {MODEL} --model-path {MPATH} "
       f"--calib-dir {CALIB} --cells-json {NODE_CELLS_DIR}/rep_cells.json "
       f"--gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 8 --no-raw --inject-from-metadata")
g = submit("vmb-r7-genall", gen, gpus=8, minutes=25)
r = submit("vmb-r7-repall", rep, gpus=8, minutes=25, depends_on=[g])
print(f"R7 GENALL={g}  REPALL={r}  (24 cells, 8 GPUs each, sequential)")
