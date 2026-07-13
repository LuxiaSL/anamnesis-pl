"""Submit ARM A4 (state surgery) chains for both anchors to Heimdall.

Per model: 1-GPU 2-gen SMOKE (worker mode; includes the bitwise rider + RoPE
gate at load) -> 4-GPU full launcher (160 continuations x 13 conditions),
dependency-gated on the smoke. gen ids = seed-idx {0,1} per prompt class
(gid = class*10 + seed, 80 classes).
"""
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}

MODELS = {
    "3b": {"path": "/models/llama-3.2-3b-instruct",
           "floor": "/models/anamnesis-extract/runs/vmb_stage0_3b",
           "calib": "/models/anamnesis-extract/calibration/3b",
           "out": "/models/anamnesis-extract/runs/vmb_a4_3b"},
    "8b": {"path": "/models/llama-3.1-8b-instruct",
           "floor": "/models/anamnesis-extract/runs/vmb_stage0_8b",
           "calib": "/models/anamnesis-extract/calibration/8b",
           "out": "/models/anamnesis-extract/runs/vmb_a4_8b"},
}

GEN_IDS = [k * 10 + s for k in range(80) for s in (0, 1)]


def submit(name: str, command: str, gpus: int, minutes: int,
           depends_on: list[str] | None = None) -> str:
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR,
            "estimated_minutes": minutes, "env": ENV,
            "command": f"bash -c '{BASE} && {command}'"}
    if depends_on:
        spec["depends_on"] = depends_on
    req = urllib.request.Request(
        API, data=json.dumps({"spec": spec}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        j = json.loads(r.read())
    if "job" not in j:
        raise RuntimeError(f"{name}: {j}")
    return j["job"]["id"]


for m, cfg in MODELS.items():
    common = (f"--model {m} --model-path {cfg['path']} "
              f"--floor-run-dir {cfg['floor']} --manifest {cfg['floor']}/replay_manifest.json "
              f"--calib-dir {cfg['calib']} --out-run-dir {cfg['out']}")
    smoke = submit(
        f"vmb-a4-smoke-{m}",
        f"python -u -m anamnesis.scripts.vmb_a4_surgery_replay {common} "
        f"--label smoke --gen-ids 0 1",
        gpus=1, minutes=25)
    full = submit(
        f"vmb-a4-full-{m}",
        f"python -u -m anamnesis.scripts.vmb_a4_surgery_replay {common} "
        f"--launch --gpus 0,1,2,3 --workers-per-gpu 4 "
        f"--gen-ids {' '.join(str(g) for g in GEN_IDS)}",
        gpus=4, minutes=90, depends_on=[smoke])
    print(f"{m}: smoke={smoke} full={full}")
