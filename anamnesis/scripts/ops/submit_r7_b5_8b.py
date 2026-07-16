"""R7 step 2 — §B.5 spectral-split free-gen + replay on 8B at L16 (outer-loop revised ASK-2, P=.60).

8 vectors (V3top/V3tail_L16 + matched-support Rtop×3/Rtail×3, built by vmb_b5_spectral_split --site 16
against the banked Σ_L16-8B) × α{.03,.1,.3}, n=40/cell, inject L16. Per-vector gen→replay chain.
Downstream CPU: matched-support efficiency (§B.5 efficiency lens primary — does V3tail carry
disproportionate efficacy per deformation, reproducing the 3B tail-durability finding at 8B?).
First-read → outer loop; nothing stamped. Run: HEIMDALL_{API,WORK_DIR,VENV} exported.
"""
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
MODEL, MPATH = "8b", "/models/llama-3.1-8b-instruct"
RUNS = "/models/anamnesis-extract/runs"
CALIB = "/models/anamnesis-extract/calibration/8b"
VEC_DIR = "/models/anamnesis-extract/battery/a5_vectors_8b_b5"
NPZ, STAMPS = f"{VEC_DIR}/a5_vectors.npz", f"{VEC_DIR}/a5_vectors_stamps.json"
PROMPTS = "pipeline/anamnesis/prompts/prompt_sets.json"
MAP_SITE, LADDER, N_LIMIT = 16, [0.03, 0.1, 0.3], 40
VECTORS = ["V3top_L16", "V3tail_L16", "Rtop1_L16", "Rtop2_L16", "Rtop3_L16",
           "Rtail1_L16", "Rtail2_L16", "Rtail3_L16"]


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
            f"--prompts {PROMPTS} --out-run-dir {RUNS}/vmb_b5_{MODEL}/{c} --gpus 0,1,2,3 "
            f"--workers-per-gpu 4 --seeds-per-class 1 --limit {N_LIMIT} "
            f"--seed-namespace VMBB5-{MODEL.upper()}-{c} --inject-npz {NPZ} --inject-key {key} "
            f"--inject-layer {MAP_SITE} --inject-alpha-frac {frac} --inject-norms-json {STAMPS}")


def rep_cmd(key, frac):
    c = f"{key}_a{frac}"
    return (f"python -u -m anamnesis.scripts.parallel_replay --model {MODEL} --model-path {MPATH} "
            f"--run-dir {RUNS}/vmb_b5_{MODEL}/{c} --calib-dir {CALIB} "
            f"--manifest {RUNS}/vmb_b5_{MODEL}/{c}/replay_manifest.json --gpus 0,1 "
            f"--workers-per-gpu 3 --no-raw --inject-from-metadata")


for key in VECTORS:
    cells = [(key, f) for f in LADDER]
    g = submit(f"vmb-r7-gen-{key}", " && ".join(gen_cmd(k, f) for k, f in cells), gpus=4, minutes=30)
    r = submit(f"vmb-r7-rep-{key}", " && ".join(rep_cmd(k, f) for k, f in cells),
               gpus=2, minutes=30, depends_on=[g])
    print(f"{key}: gen={g} rep={r}")
