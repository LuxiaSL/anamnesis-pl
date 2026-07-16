"""A6 Cell 2 — the A5↔A6 bridge pipeline (session-8 Part C; first-of-kind, C§8).

Chains: (1) build the Qwen animal (cat) V1-recipe vector + AR1-3 nulls; (2) free-gen steer
the Qwen teacher (base, neutral) with Acat_L18 × α{.1,.3,.5} + AR1-3 nulls + α=0 base cell
(coherence-gated, fp16, NO MT gate per 14o §1); (3) replay steer cells; (4) base-Qwen
replay of the probe160 manifest = the student-deformation reference. Analysis (bridge
readout) is CPU, run after. First-read → outer loop. Run: HEIMDALL_* env + node1 ssh.
"""
import json
import subprocess
import urllib.request
from pathlib import Path

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
QPATH = ("/models/subliminal-anamnesis/.hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/"
         "snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
RUNS = "/models/anamnesis-extract/runs"
CALIB = "/models/anamnesis-extract/calibration/qwen25_7b"
BATTERY = "/models/anamnesis-extract/battery"
PROMPTS = "pipeline/anamnesis/prompts/prompt_sets.json"
VEC_DIR = f"{BATTERY}/a6_animal_vectors_qwen"
NPZ, STAMPS = f"{VEC_DIR}/a5_vectors.npz", f"{VEC_DIR}/a5_vectors_stamps.json"
PROBE160 = f"{BATTERY}/arms/A6/cohort/probe160_manifest.json"
SITE, LADDER = 18, [0.1, 0.3, 0.5]
NODE_CELLS = f"{WORK_DIR}/_bridge_cells"
LOCAL_CELLS = Path("/tmp/claude-output/bridge_cells")


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


# (1) build vector
build = (f"mkdir -p {VEC_DIR} && python -u -m anamnesis.scripts.vmb_a6_animal_vector "
         f"--model qwen-7b --model-path {QPATH} --animal cat --prompts {PROMPTS} "
         f"--sites 7 14 18 21 --out-npz {NPZ} --out-stamps {STAMPS}")

# (2)/(3) steer + replay cells: Acat + AR1-3 × ladder, + α=0 base cell
gen_cells, rep_cells = [], []
keys = ["Acat"] + [f"AR{j}" for j in (1, 2, 3)]
for key in keys:
    for frac in LADDER:
        c = f"{key}_L{SITE}_a{frac}"
        run = f"{RUNS}/vmb_a6bridge_qwen/{c}"
        gen_cells.append({"out_run_dir": run, "seed_namespace": f"A6BR-{c}",
                          "inject_key": f"{key}_L{SITE}", "inject_layer": SITE, "inject_alpha_frac": frac})
        rep_cells.append({"run_dir": run, "manifest": f"{run}/replay_manifest.json"})
# α=0 base reference cell (no injection)
base_run = f"{RUNS}/vmb_a6bridge_qwen/base_L{SITE}_a0.0"
gen_cells.append({"out_run_dir": base_run, "seed_namespace": "A6BR-base",
                  "inject_key": f"Acat_L{SITE}", "inject_layer": SITE, "inject_alpha_frac": 0.0})
rep_cells.append({"run_dir": base_run, "manifest": f"{base_run}/replay_manifest.json"})

LOCAL_CELLS.mkdir(parents=True, exist_ok=True)
(LOCAL_CELLS / "gen.json").write_text(json.dumps({"cells": gen_cells}, indent=1))
(LOCAL_CELLS / "rep.json").write_text(json.dumps({"cells": rep_cells}, indent=1))
subprocess.run(["ssh", "node1", f"mkdir -p {NODE_CELLS}"], check=True)
subprocess.run(["rsync", "-a", f"{LOCAL_CELLS}/gen.json", f"{LOCAL_CELLS}/rep.json",
                f"node1:{NODE_CELLS}/"], check=True)

gen = (f"python -u -m anamnesis.scripts.vmb_a5_gen_multicell --model qwen-7b --model-path {QPATH} "
       f"--prompts {PROMPTS} --cells-json {NODE_CELLS}/gen.json --inject-npz {NPZ} "
       f"--inject-norms-json {STAMPS} --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 6 --seeds-per-class 2")
rep = (f"python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model qwen-7b --model-path {QPATH} "
       f"--calib-dir {CALIB} --cells-json {NODE_CELLS}/rep.json --gpus 0,1,2,3,4,5,6,7 "
       f"--workers-per-gpu 8 --no-raw --inject-from-metadata")
# (4) base-Qwen replay of probe160 = student-deformation reference
baseref = (f"python -u -m anamnesis.scripts.parallel_replay --model qwen-7b --model-path {QPATH} "
           f"--calib-dir {CALIB} --run-dir {RUNS}/vmb_a6bridge_qwen/base_probe160 "
           f"--manifest {PROBE160} --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 8 --no-raw")

b = submit("a6-bridge-vector", build, gpus=1, minutes=20)
g = submit("a6-bridge-genall", gen, gpus=8, minutes=20, depends_on=[b])
r = submit("a6-bridge-repall", rep, gpus=8, minutes=20, depends_on=[g])
br = submit("a6-bridge-baseref", baseref, gpus=8, minutes=15, depends_on=[b])
print(f"bridge: vector={b} gen={g} replay={r} baseref={br}")
print(f"vector → {NPZ}; steer cells → {RUNS}/vmb_a6bridge_qwen/")
