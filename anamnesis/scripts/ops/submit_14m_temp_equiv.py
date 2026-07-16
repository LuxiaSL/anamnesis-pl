"""14m item 1 — temperature-equivalence replay (per-position best-T*), V7 (b7) + V_temp (c3).
One GPU, NO new generation (forwards over banked gens). First-read → outer loop.
Run: HEIMDALL_{API,WORK_DIR,VENV} exported.
"""
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
MODEL, MPATH = "3b", "/models/llama-3.2-3b-instruct"
RUNS = "/models/anamnesis-extract/runs"
OUT = "/models/anamnesis-extract/battery/arms/A5"


def submit(name, command, gpus, minutes):
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": ENV,
            "command": f"bash -c '{BASE} && {command}'"}
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["job"]["id"]


v7 = (f"python -u -m anamnesis.scripts.vmb_c3_temp_equiv_replay --model {MODEL} --model-path {MPATH} "
      f"--run-dir {RUNS}/vmb_b7_{MODEL} --cells V7_L14_a0.03 V7_L14_a0.1 "
      f"--out-json {OUT}/14m_temp_equiv_V7_3b.json")
vt = (f"python -u -m anamnesis.scripts.vmb_c3_temp_equiv_replay --model {MODEL} --model-path {MPATH} "
      f"--run-dir {RUNS}/vmb_c3_{MODEL} --cells Vtemp_L14_a0.03 Vtemp_L14_a0.1 "
      f"--out-json {OUT}/14m_temp_equiv_Vtemp_3b.json")
jid = submit("vmb-14m-temp-equiv", f"{v7} && {vt}", gpus=1, minutes=30)
print(f"14m temp-equiv: job={jid}  → {OUT}/14m_temp_equiv_{{V7,Vtemp}}_3b.json")
