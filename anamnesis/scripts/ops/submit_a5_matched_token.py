"""Submit the A5 matched-token wave: on-policy pilot gate (1 GPU) -> gated cell
replays (2 GPUs). Both depend on the vectors job; pass its id as argv[1]."""
import json
import sys
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
MODEL, MPATH = "3b", "/models/llama-3.2-3b-instruct"
STAGE0 = "/models/anamnesis-extract/runs/vmb_stage0_3b"
CALIB = "/models/anamnesis-extract/calibration/3b"
VEC = "/models/anamnesis-extract/battery/a5_vectors_3b"
GATE = f"{VEC}/onpolicy_gate.json"
OUT = "/models/anamnesis-extract/runs/vmb_a5_mt_3b"


def submit(name, command, gpus, minutes, depends_on=None):
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR,
            "estimated_minutes": minutes, "env": {"HF_HUB_OFFLINE": "1"},
            "command": f"bash -c '{BASE} && {command}'"}
    if depends_on:
        spec["depends_on"] = depends_on
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        j = json.loads(r.read())
    if "job" not in j:
        raise RuntimeError(f"{name}: {j}")
    return j["job"]["id"]


vec_job = sys.argv[1] if len(sys.argv) > 1 else None
deps = [vec_job] if vec_job else None

gate = submit(
    "vmb-a5-onpolicy-gate",
    f"python -u -m anamnesis.scripts.vmb_a5_onpolicy_gate --model {MODEL} "
    f"--model-path {MPATH} --stage0-run {STAGE0} --vectors-dir {VEC} --out {GATE}",
    gpus=1, minutes=30, depends_on=deps)
mt = submit(
    "vmb-a5-mt-cells",
    f"python -u -m anamnesis.scripts.vmb_a5_mt_launch --model {MODEL} "
    f"--model-path {MPATH} --stage0-run {STAGE0} --calib-dir {CALIB} "
    f"--vectors-dir {VEC} --gate-report {GATE} --out-root {OUT} "
    f"--gpus 0,1 --workers-per-gpu 8",
    gpus=2, minutes=120, depends_on=[gate])
print(f"gate={gate} mt={mt}")
