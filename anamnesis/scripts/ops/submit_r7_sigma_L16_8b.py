"""R7 step 1 — build + BANK Σ_L16-8B (ADDENDUM 14j-era R7 backfill; outer-loop revised ASK-2).

Residual-covariance eigendecomp at the 8B map site L16, SAME conventions as the banked Σ_L14-3B
(ridge = ridge_rel × mean eigenvalue; positions from banked α=0 continuations of vmb_stage0_8b).
Banked as a FIRST-CLASS artifact (a5_sigma_L16_8b.npz) — it also serves the 14k write-anatomy assays.
Single GPU, ~20 min. First-read → outer loop; nothing stamped.
Run: HEIMDALL_{API,WORK_DIR,VENV} exported.
"""
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
MODEL, MPATH = "8b", "/models/llama-3.1-8b-instruct"
RUNS = "/models/anamnesis-extract/runs"
VEC = "/models/anamnesis-extract/battery/a5_vectors_8b/a5_vectors.npz"
OUT_DIR = "/models/anamnesis-extract/battery/arms/A5"


def submit(name, command, gpus, minutes):
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": ENV,
            "command": f"bash -c '{BASE} && {command}'"}
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["job"]["id"]


cmd = (f"python -u -m anamnesis.scripts.vmb_a5_covariance_screen --model {MODEL} --model-path {MPATH} "
       f"--stage0-run {RUNS}/vmb_stage0_{MODEL} --vectors {VEC} --out-dir {OUT_DIR} "
       f"--sites 16 --save-sigma-site 16 --n-gens 60")
jid = submit("vmb-r7-sigma-L16-8b", cmd, gpus=1, minutes=30)
print(f"R7 Σ_L16-8B cov screen: job={jid}  sigma={OUT_DIR}/a5_sigma_L16_8b.npz")
