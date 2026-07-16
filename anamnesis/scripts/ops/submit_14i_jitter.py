"""14i item 2 — jitter-interaction test (ADDENDUM 14i; P=0.80). Single GPU, ~10 min.
5 gate cells × k=5 on 3B; agreement vs mean_chosen_rank run-to-run stability.
First-read → outer loop. Run: HEIMDALL_{API,WORK_DIR,VENV} exported.
"""
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
MODEL, MPATH = "3b", "/models/llama-3.2-3b-instruct"
RUNS = "/models/anamnesis-extract/runs"
VEC = "/models/anamnesis-extract/battery/a5_vectors_3b"
OUT = f"{RUNS}/vmb_a5_3b/14i_jitter_3b.json"


def submit(name, command, gpus, minutes):
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": ENV,
            "command": f"bash -c '{BASE} && {command}'"}
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["job"]["id"]


cmd = (f"python -u -m anamnesis.scripts.vmb_14i_jitter --model {MODEL} --model-path {MPATH} "
       f"--stage0-run {RUNS}/vmb_stage0_{MODEL} --vectors-dir {VEC} --map-site 14 "
       f"--k 5 --out-json {OUT}")
jid = submit("vmb-14i-jitter", cmd, gpus=1, minutes=20)
print(f"14i jitter: job={jid}  out={OUT}")
