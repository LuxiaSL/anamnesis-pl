"""Submit §B.5 V3 spectral-split free-gen (DESIGN-V4 §B.5; Luxia window-add 14d item 3).

8 vectors (V3top/V3tail + matched-support R_top×3/R_tail×3, built by vmb_b5_spectral_split)
× α ∈ {.03,.1,.3}, n=40/cell (seeds-per-class 1 + --limit 40), inject L14, C§3 convention.
Per-vector gen job (modest GPUs) → per-vector replay job (--inject-from-metadata, LOW worker
count per the 2026-07-14 load-spike lesson: do NOT stack many high-worker replay jobs).
Readouts (downstream CPU): A5-inv targeting + deformation-maha + efficiency + coherence.
First-read → outer loop, nothing stamped.
"""
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
MODEL, MPATH = "3b", "/models/llama-3.2-3b-instruct"
RUNS = "/models/anamnesis-extract/runs"
CALIB = "/models/anamnesis-extract/calibration/3b"
VEC_DIR = "/models/anamnesis-extract/battery/a5_vectors_3b_b5"
NPZ = f"{VEC_DIR}/a5_vectors.npz"
STAMPS = f"{VEC_DIR}/a5_vectors_stamps.json"
PROMPTS = "pipeline/anamnesis/prompts/prompt_sets.json"

MAP_SITE = 14
LADDER = [0.03, 0.1, 0.3]
N_LIMIT = 40
VECTORS = ["V3top_L14", "V3tail_L14",
           "Rtop1_L14", "Rtop2_L14", "Rtop3_L14",
           "Rtail1_L14", "Rtail2_L14", "Rtail3_L14"]


def submit(name, command, gpus, minutes, depends_on=None):
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": ENV,
            "command": f"bash -c '{BASE} && {command}'"}
    if depends_on:
        spec["depends_on"] = depends_on
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        j = json.loads(r.read())
    return j["job"]["id"]


def cell(key, frac):
    return f"{key}_a{frac}"


def gen_cmd(key, frac):
    c = cell(key, frac)
    return (f"python -u -m anamnesis.scripts.vmb_stage0_generate --model {MODEL} "
            f"--model-path {MPATH} --prompts {PROMPTS} "
            f"--out-run-dir {RUNS}/vmb_b5_{MODEL}/{c} --gpus 0,1,2,3 --workers-per-gpu 4 "
            f"--seeds-per-class 1 --limit {N_LIMIT} --seed-namespace VMBB5-{MODEL.upper()}-{c} "
            f"--inject-npz {NPZ} --inject-key {key} --inject-layer {MAP_SITE} "
            f"--inject-alpha-frac {frac} --inject-norms-json {STAMPS}")


def rep_cmd(key, frac):
    c = cell(key, frac)
    return (f"python -u -m anamnesis.scripts.parallel_replay --model {MODEL} --model-path {MPATH} "
            f"--run-dir {RUNS}/vmb_b5_{MODEL}/{c} --calib-dir {CALIB} "
            f"--manifest {RUNS}/vmb_b5_{MODEL}/{c}/replay_manifest.json "
            f"--gpus 0,1 --workers-per-gpu 3 --no-raw --inject-from-metadata")


for key in VECTORS:
    cells = [(key, f) for f in LADDER]
    g = submit(f"vmb-b5-gen-{key}", " && ".join(gen_cmd(k, f) for k, f in cells),
               gpus=4, minutes=30)
    r = submit(f"vmb-b5-rep-{key}", " && ".join(rep_cmd(k, f) for k, f in cells),
               gpus=2, minutes=30, depends_on=[g])
    print(f"{key}: gen={g} rep={r}")
