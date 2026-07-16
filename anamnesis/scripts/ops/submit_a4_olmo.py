"""Submit ARM A4 (state surgery) for OLMo-2-7B — session-10 Part E (14p §4).

Base-model A4 replication (allenai/OLMo-2-1124-7B — no instructions involved): naive/rotate/
recompute/full, the olmo stage-0 floor corpus as the one anchor-equivalent substrate, full
fraction ladder, n=160, replay path, bf16. The worker's `operative_inv_freq` value-gate fires
at load (14e MANDATORY; OLMo-2 RoPE config read fresh). 1-GPU smoke gates the 4-GPU launcher.
Scored vs Po1=.70 (P3 dissociation on the base model). First-read → outer loop.
"""
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}

OPATH = ("/models/anamnesis-extract/.hf-cache/hub/models--allenai--OLMo-2-1124-7B/"
         "snapshots/7df9a82518afdecae4e8c026b27adccc8c1f0032")
CFG = {"path": OPATH,
       "floor": "/models/anamnesis-extract/runs/vmb_stage0_olmo2_7b",
       "calib": "/models/anamnesis-extract/calibration/olmo2_7b",
       "out": "/models/anamnesis-extract/runs/vmb_a4_olmo2-7b"}

GEN_IDS = [k * 10 + s for k in range(80) for s in (0, 1)]


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
    if "job" not in j:
        raise RuntimeError(f"{name}: {j}")
    return j["job"]["id"]


common = (f"--model olmo2-7b --model-path {CFG['path']} "
          f"--floor-run-dir {CFG['floor']} --manifest {CFG['floor']}/replay_manifest.json "
          f"--calib-dir {CFG['calib']} --out-run-dir {CFG['out']}")
smoke = submit("vmb-a4-smoke-olmo",
               f"python -u -m anamnesis.scripts.vmb_a4_surgery_replay {common} --label smoke --gen-ids 0 1",
               gpus=1, minutes=30)
full = submit("vmb-a4-full-olmo",
              f"python -u -m anamnesis.scripts.vmb_a4_surgery_replay {common} "
              f"--launch --gpus 0,1,2,3 --workers-per-gpu 4 "
              f"--gen-ids {' '.join(str(g) for g in GEN_IDS)}",
              gpus=4, minutes=110, depends_on=[smoke])
print(f"olmo-a4: smoke={smoke} full={full}")
