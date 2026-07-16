"""R6 — V4-8B build (3B recipe @ L16) + §2.2 per-gen ∇S fork-dump (outer-loop revised ASK-2, P=.80).

Two sequential steps, one GPU, ~20 min:
  1. vmb_a5_build_v4_gradient --map-site 16 → banks V4_L16 into the 8B vector bank (recency-vs-
     prompt-mass attention surrogate at L16, differentiated w.r.t. its residual input — the 3B recipe
     un-hardcoded, NOT forked).
  2. vmb_a5_gradient_fork --map-site 16 → per-gen ∇S norm scale + pairwise cosine + cancellation
     ratio; self-validates cos(recomputed mean, banked V4_L16) ≈ 1.0.
Separates AVERAGING-DESTROYED vs GENUINELY-READ-ONLY for the 8B mode coordinate. Score vs P=.80.
First-read → outer loop. Run: HEIMDALL_{API,WORK_DIR,VENV} exported.
"""
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
MODEL, MPATH = "8b", "/models/llama-3.1-8b-instruct"
RUNS = "/models/anamnesis-extract/runs"
VECDIR = "/models/anamnesis-extract/battery/a5_vectors_8b"
NPZ = f"{VECDIR}/a5_vectors.npz"
OUT_DIR = "/models/anamnesis-extract/battery/arms/A5"
SITE = 16


def submit(name, command, gpus, minutes):
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": ENV,
            "command": f"bash -c '{BASE} && {command}'"}
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["job"]["id"]


build = (f"python -u -m anamnesis.scripts.vmb_a5_build_v4_gradient --model {MODEL} --model-path {MPATH} "
         f"--stage0-run {RUNS}/vmb_stage0_{MODEL} --out-dir {VECDIR} --map-site {SITE} --n-gens 20")
fork = (f"python -u -m anamnesis.scripts.vmb_a5_gradient_fork --model {MODEL} --model-path {MPATH} "
        f"--stage0-run {RUNS}/vmb_stage0_{MODEL} --vectors {NPZ} --out-dir {OUT_DIR} "
        f"--map-site {SITE} --n-gens 20")
jid = submit("vmb-r6-v4fork-8b", f"{build} && {fork}", gpus=1, minutes=30)
print(f"R6 V4-8B build + fork: job={jid}  out={OUT_DIR}/a5_gradient_fork_8b.json")
