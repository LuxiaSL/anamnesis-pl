"""§B.7 stage-2 free-gen (DESIGN-V4 §B.7; Luxia-ratified 14d). V7 (entropy-band candidate) +
3 [16:256]-band-matched randoms × α∈{.03,.1,.3}, n=40/cell, inject L14, C§3 convention,
generated positions only. Per-vector gen (4 GPUs) → replay (--inject-from-metadata, low workers
per the load lesson). Readouts (CPU): dir0 targeting + deformation-maha + efficiency + coherence.
Promotion to n=160 ONLY if pilot clears 1.5× the band-null (standing rule; do NOT re-run pilot).
First-read → outer loop; nothing stamped.
"""
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
MODEL, MPATH = "3b", "/models/llama-3.2-3b-instruct"
RUNS = "/models/anamnesis-extract/runs"
CALIB = "/models/anamnesis-extract/calibration/3b"
VEC_DIR = "/models/anamnesis-extract/battery/a5_vectors_3b_b7"
NPZ = f"{VEC_DIR}/a5_vectors.npz"
STAMPS = f"{VEC_DIR}/a5_vectors_stamps.json"
PROMPTS = "pipeline/anamnesis/prompts/prompt_sets.json"
MAP_SITE, LADDER, N_LIMIT = 14, [0.03, 0.1, 0.3], 40
VECTORS = ["V7_L14", "Rband1_L14", "Rband2_L14", "Rband3_L14"]


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


def gen_cmd(key, frac):
    c = f"{key}_a{frac}"
    return (f"python -u -m anamnesis.scripts.vmb_stage0_generate --model {MODEL} --model-path {MPATH} "
            f"--prompts {PROMPTS} --out-run-dir {RUNS}/vmb_b7_{MODEL}/{c} --gpus 0,1,2,3 "
            f"--workers-per-gpu 4 --seeds-per-class 1 --limit {N_LIMIT} "
            f"--seed-namespace VMBB7-{MODEL.upper()}-{c} --inject-npz {NPZ} --inject-key {key} "
            f"--inject-layer {MAP_SITE} --inject-alpha-frac {frac} --inject-norms-json {STAMPS}")


def rep_cmd(key, frac):
    c = f"{key}_a{frac}"
    return (f"python -u -m anamnesis.scripts.parallel_replay --model {MODEL} --model-path {MPATH} "
            f"--run-dir {RUNS}/vmb_b7_{MODEL}/{c} --calib-dir {CALIB} "
            f"--manifest {RUNS}/vmb_b7_{MODEL}/{c}/replay_manifest.json --gpus 0,1 "
            f"--workers-per-gpu 3 --no-raw --inject-from-metadata")


for key in VECTORS:
    cells = [(key, f) for f in LADDER]
    g = submit(f"vmb-b7-gen-{key}", " && ".join(gen_cmd(k, f) for k, f in cells), gpus=4, minutes=25)
    r = submit(f"vmb-b7-rep-{key}", " && ".join(rep_cmd(k, f) for k, f in cells),
               gpus=2, minutes=25, depends_on=[g])
    print(f"{key}: gen={g} rep={r}")
