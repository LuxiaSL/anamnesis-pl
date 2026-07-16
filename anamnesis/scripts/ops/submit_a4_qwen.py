"""Submit ARM A4 (state surgery) for Qwen-7B — the session-10 spine (14p §1).

Replication of the banked A4 design on a THIRD architecture (Qwen2.5-7B-Instruct):
naive/rotate/recompute/full kinds, the qwen stage-0 floor corpus as the one anchor-
equivalent substrate, full fraction ladder, n=160, replay path. bf16 (qwen dtype law —
NOT fp16; the preset governs). The worker's `operative_inv_freq(model)` value-gate fires
at load BEFORE any surgery (14e MANDATORY) — Qwen's rope_theta=1e6 lives at config top
level, so the gate reads it fresh (never assumes Llama). The 1-GPU smoke (bitwise rider +
gate) dependency-gates the 4-GPU full launcher.

Scored vs Pq1=.80 (P3 dissociation replicates) / Pq2=.70 (rotate↔recompute colinear, naive
odd at low fractions). First-read → outer loop.
"""
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}

QPATH = ("/models/subliminal-anamnesis/.hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/"
         "snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
CFG = {"path": QPATH,
       "floor": "/models/anamnesis-extract/runs/vmb_stage0_qwen7b",
       "calib": "/models/anamnesis-extract/calibration/qwen25_7b",
       "out": "/models/anamnesis-extract/runs/vmb_a4_qwen"}

GEN_IDS = [k * 10 + s for k in range(80) for s in (0, 1)]  # n=160


def submit(name: str, command: str, gpus: int, minutes: int,
           depends_on: list[str] | None = None) -> str:
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": ENV,
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


common = (f"--model qwen-7b --model-path {CFG['path']} "
          f"--floor-run-dir {CFG['floor']} --manifest {CFG['floor']}/replay_manifest.json "
          f"--calib-dir {CFG['calib']} --out-run-dir {CFG['out']}")
smoke = submit(
    "vmb-a4-smoke-qwen",
    f"python -u -m anamnesis.scripts.vmb_a4_surgery_replay {common} --label smoke --gen-ids 0 1",
    gpus=1, minutes=30)
full = submit(
    "vmb-a4-full-qwen",
    f"python -u -m anamnesis.scripts.vmb_a4_surgery_replay {common} "
    f"--launch --gpus 0,1,2,3 --workers-per-gpu 4 "
    f"--gen-ids {' '.join(str(g) for g in GEN_IDS)}",
    gpus=4, minutes=110, depends_on=[smoke])
print(f"qwen-a4: smoke={smoke} full={full}")
