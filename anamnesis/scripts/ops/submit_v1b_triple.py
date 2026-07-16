"""14l — V1b generation triple (graded-Goodhart roster hole-filler; session-8 Part B5).

3 cells: V1b_L14 × α{.03,.1,.3}, n=160 (seeds-per-class 2), standard C§ discipline,
MULTICELL path (GENALL 8-GPU → REPALL 8-GPU). V1b = imago-B5 topic-disjoint formality
control (`a5_vectors_3b_v1b`, built session-5). Cells land under vmb_a5_3b as
V1b_L14_a{frac} so the gg Part-A roster picks them up (ROSTER["V1b"] = vmb_a5_3b).
Completes the n=25 roster before P1/P2 scoring.
First-read → outer loop. Run: HEIMDALL_{API,WORK_DIR,VENV} exported (+ node1 ssh for rsync).
"""
import json
import subprocess
import urllib.request
from pathlib import Path

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
MODEL, MPATH = "3b", "/models/llama-3.2-3b-instruct"
RUNS = "/models/anamnesis-extract/runs"
CALIB = "/models/anamnesis-extract/calibration/3b"
VEC_DIR = "/models/anamnesis-extract/battery/a5_vectors_3b_v1b"
NPZ, STAMPS = f"{VEC_DIR}/a5_vectors.npz", f"{VEC_DIR}/a5_vectors_stamps.json"
PROMPTS = "pipeline/anamnesis/prompts/prompt_sets.json"
NODE_CELLS_DIR = f"{WORK_DIR}/_v1b_cells"
LOCAL_CELLS_DIR = Path("/tmp/claude-output/v1b_cells")
SSH_CELLS_DEST = f"node1:{WORK_DIR}/_v1b_cells/"
KEY, SITE, LADDER = "V1b_L14", 14, [0.03, 0.1, 0.3]

gen_cells, rep_cells = [], []
for frac in LADDER:
    c = f"V1b_L14_a{frac}"
    run = f"{RUNS}/vmb_a5_{MODEL}/{c}"
    gen_cells.append({"out_run_dir": run, "seed_namespace": f"VMBV1B-{MODEL.upper()}-{c}",
                      "inject_key": KEY, "inject_layer": SITE, "inject_alpha_frac": frac})
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
       f"--seeds-per-class 2")
rep = (f"python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model {MODEL} --model-path {MPATH} "
       f"--calib-dir {CALIB} --cells-json {NODE_CELLS_DIR}/rep_cells.json "
       f"--gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 8 --no-raw --inject-from-metadata")
g = submit("vmb-v1b-genall", gen, gpus=8, minutes=15)
r = submit("vmb-v1b-repall", rep, gpus=8, minutes=15, depends_on=[g])
print(f"V1b triple GENALL={g}  REPALL={r}  (3 cells × 160, 8 GPUs each, sequential)")
