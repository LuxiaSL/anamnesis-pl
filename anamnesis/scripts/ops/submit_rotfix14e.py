"""14e ROTATE re-run — A4 (both anchors) + A4b rotate rows, with the fixed inv_freq
(read theta from rope_scaling + value-level live-buffer gate). Banks to NEW subdirs
(`*_rotfix14e`) so the VOID rows stay as evidence; every gen tagged rope_fix=14e.

Per arm: a SMOKE (worker mode, 2 cells, --kinds rotate) proves the value gate passes on the
LIVE model at load, THEN the rotate launcher (dependency-gated). A 5-cell NAIVE rider re-runs
naive to the same dir — the harness-identity check (must reproduce banked naive bitwise/near).
Naive/rec/full ROWS ARE NOT re-run (valid; the rider is the only naive re-run).

Priority: fire this AFTER C3 + V3tail per the overnight staging doc (paradigm cells outrank
the bug re-run). First-read → outer loop; nothing stamped; void rows stay void.

Sanitization: the A4b dialogue/docs corpora live outside /models — read their dir from
`A4B_DATA_DIR` (no home path / username in the committed script). Value in the ops runbook.
"""
import json
import os
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
TAG = "14e"

A4 = {
    "3b": {"path": "/models/llama-3.2-3b-instruct", "floor": "/models/anamnesis-extract/runs/vmb_stage0_3b",
           "calib": "/models/anamnesis-extract/calibration/3b"},
    "8b": {"path": "/models/llama-3.1-8b-instruct", "floor": "/models/anamnesis-extract/runs/vmb_stage0_8b",
           "calib": "/models/anamnesis-extract/calibration/8b"},
}
RUNS = "/models/anamnesis-extract/runs"
GEN_IDS = [k * 10 + s for k in range(80) for s in (0, 1)]         # 160 A4 continuations
NAIVE_RIDER_A4 = [0, 1, 10, 11, 20]                               # 5-cell harness-identity check
A4B_DATA = os.environ.get("A4B_DATA_DIR")                          # dialogue/docs dir (env; not committed)


def submit(name, command, gpus, minutes, depends_on=None):
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1", "working_dir": WORK_DIR,
            "estimated_minutes": minutes, "env": ENV, "command": f"bash -c '{BASE} && {command}'"}
    if depends_on:
        spec["depends_on"] = depends_on
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        j = json.loads(r.read())
    if "job" not in j:
        raise RuntimeError(f"{name}: {j}")
    return j["job"]["id"]


# ── A4 both anchors: smoke(value-gate on live) → rotate launcher + naive rider ──
for m, cfg in A4.items():
    out = f"{RUNS}/vmb_a4_{m}_rotfix14e"
    common = (f"--model {m} --model-path {cfg['path']} --floor-run-dir {cfg['floor']} "
              f"--manifest {cfg['floor']}/replay_manifest.json --calib-dir {cfg['calib']} "
              f"--out-run-dir {out} --rope-fix-tag {TAG}")
    smoke = submit(f"vmb-a4rotfix-smoke-{m}",
                   f"python -u -m anamnesis.scripts.vmb_a4_surgery_replay {common} "
                   f"--kinds rotate --label smoke --gen-ids 0 1", gpus=1, minutes=25)
    rot = submit(f"vmb-a4rotfix-{m}",
                 f"python -u -m anamnesis.scripts.vmb_a4_surgery_replay {common} --kinds rotate "
                 f"--launch --gpus 0,1,2,3 --workers-per-gpu 4 --gen-ids {' '.join(map(str, GEN_IDS))}",
                 gpus=4, minutes=120, depends_on=[smoke])
    rider = submit(f"vmb-a4rotfix-naiverider-{m}",
                   f"python -u -m anamnesis.scripts.vmb_a4_surgery_replay {common} --kinds naive "
                   f"--label rider --gen-ids {' '.join(map(str, NAIVE_RIDER_A4))}",
                   gpus=1, minutes=25, depends_on=[smoke])
    print(f"A4-{m}: smoke={smoke} rotate={rot} naive_rider={rider}")

# ── A4b (3B dialogue substrate): rotate launcher + naive rider ──
if not A4B_DATA:
    print("A4B_DATA_DIR not set — SKIPPING A4b re-run (set it to the kv-rotation data dir; see ops runbook)")
else:
    dlg = f"{A4B_DATA}/native3b_convs_2026-07-10.jsonl {A4B_DATA}/native3b_convs_boost_2026-07-10.jsonl"
    docs = f"{A4B_DATA}/eval_docs.jsonl"
    out = f"{RUNS}/vmb_a4b_3b_rotfix14e"
    common = (f"--model 3b --model-path {A4['3b']['path']} --calib-dir {A4['3b']['calib']} "
              f"--out-run-dir {out} --dialogue {dlg} --docs {docs} --n-docs 4 --rope-fix-tag {TAG}")
    smoke = submit("vmb-a4brotfix-smoke",
                   f"python -u -m anamnesis.scripts.vmb_a4b_surgery_replay {common} "
                   f"--kinds rotate --label smoke --cell-ids 0 1", gpus=1, minutes=25)
    rot = submit("vmb-a4brotfix",
                 f"python -u -m anamnesis.scripts.vmb_a4b_surgery_replay {common} --kinds rotate "
                 f"--launch --gpus 0,1,2,3 --workers-per-gpu 2", gpus=4, minutes=40, depends_on=[smoke])
    rider = submit("vmb-a4brotfix-naiverider",
                   f"python -u -m anamnesis.scripts.vmb_a4b_surgery_replay {common} --kinds naive "
                   f"--label rider --cell-ids 0 1 2 3 4", gpus=1, minutes=25, depends_on=[smoke])
    print(f"A4b: smoke={smoke} rotate={rot} naive_rider={rider}")
