"""14j LEG 2 — certifying replay of the banked V7 (entropy-gradient) cells against the
temperature coordinate (ADDENDUM 14j; gated by leg-1 Pj1 which landed INSIDE).

Logit-retaining replay of ALREADY-BANKED V7 + Rband1-3 cells × α{.03,.1} at L14 — NO new
generation. 8 cells × 40 = 320 replays. Readout (b): per-token entropy rise vs matched Rband
nulls, dose-ordered (`vmb_c3_entropy_replay.py --null-prefixes RBAND`). α=.3 EXCLUDED (both
arms collapse). n stays 40 — NOT a promotion. Comparison arm = C3's banked V_temp entropy
rows (`c3_certifying_b_entropy_3b.json`). Content rung (c) runs CPU-locally (banked texts).

Single GPU, ~15 min (640 forwards). First-read → outer loop; nothing stamped.
Run: HEIMDALL_{API,WORK_DIR,VENV} exported → python -m anamnesis.scripts.ops.submit_14j_leg2
"""
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
MODEL, MPATH = "3b", "/models/llama-3.2-3b-instruct"
RUNS = "/models/anamnesis-extract/runs"
B7 = f"{RUNS}/vmb_b7_3b"
OUT = f"{B7}/14j_leg2_entropy_V7_3b.json"

CELLS = [f"{v}_L14_a{a}" for v in ("V7", "Rband1", "Rband2", "Rband3") for a in ("0.03", "0.1")]


def submit(name, command, gpus, minutes):
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": ENV,
            "command": f"bash -c '{BASE} && {command}'"}
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["job"]["id"]


cmd = (f"python -u -m anamnesis.scripts.vmb_c3_entropy_replay --model {MODEL} --model-path {MPATH} "
       f"--c3-run-dir {B7} --cells {' '.join(CELLS)} --null-prefixes RBAND --out-json {OUT}")
jid = submit("vmb-14j-leg2-entropy-V7", cmd, gpus=1, minutes=25)
print(f"14j leg2 entropy replay: job={jid}  out={OUT}")
