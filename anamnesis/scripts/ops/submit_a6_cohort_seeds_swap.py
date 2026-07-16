"""A6 Cell 1 — SEEDS-TIER cohort via the swap driver (session-9 Part C).

Replays the fixed 160-probe manifest through the animal SEED variants {t0,t2,t3,t4} × dense
steps via run_replay_multickpt.py (load base ONCE per worker, swap LoRA adapters — the
bitwise-proven instrument, NOT the naive per-checkpoint path). 3 animals × 4 seeds × 10 steps
= 120 checkpoints in ONE Heimdall job.

⚠ Label hygiene (baton): use the {t0,t2,t3,t4} set ONLY, never the unsuffixed 'dense' run
(near-clone of t0). Output goes to a SEPARATE cohort-root (vmb_a6cohort_qwen_seeds) and the
5 checkpoint-matched controls are SYMLINKED in from the session-8 core cohort (reused, not
re-replayed) — so the seeds-grain analyzer sees 12 seed-students + 5 controls (M=17), with
NO unsuffixed-dense student contaminating the seed-parity reliability. Run: HEIMDALL_* env.
"""
import json
import subprocess
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
QPATH = ("/models/subliminal-anamnesis/.hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/"
         "snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
CKPT_ROOT = "/models/subliminal-anamnesis/checkpoints"
CALIB = "/models/anamnesis-extract/calibration/qwen25_7b"
MANIFEST = "/models/anamnesis-extract/battery/arms/A6/cohort/probe160_manifest.json"
CORE_ROOT = "/models/anamnesis-extract/runs/vmb_a6cohort_qwen"
SEEDS_ROOT = "/models/anamnesis-extract/runs/vmb_a6cohort_qwen_seeds"
CKPT_JSON = f"{SEEDS_ROOT}/_seed_checkpoints.json"
STEPS = ["0001", "0002", "0003", "0005", "0008", "0013", "0021", "0034", "0055", "0075"]
ANIMALS = ["cat", "penguin", "phoenix"]
SEEDS = ["dense_t0", "dense_t2", "dense_t3", "dense_t4"]
CONTROLS = ["a", "b", "c", "d", "e"]


def submit(name, command, gpus, minutes):
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": ENV,
            "command": f"bash -c '{BASE} && {command}'"}
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["job"]["id"]


def main() -> None:
    checkpoints = []
    for a in ANIMALS:
        for s in SEEDS:
            for step in STEPS:
                checkpoints.append({
                    "label": f"{a}_{s}-{step}",
                    "adapter_path": f"{CKPT_ROOT}/qwen_{a}_student_{s}/checkpoint-{step}",
                    "run_dir": f"{SEEDS_ROOT}/{a}_{s}/step-{step}"})
    print(f"{len(checkpoints)} seed checkpoints ({len(ANIMALS)} animals × {len(SEEDS)} seeds × {len(STEPS)} steps)")

    # stage: mkdir seeds root, write checkpoints.json, symlink the reused controls in
    link_cmds = " && ".join(
        f"ln -sfn {CORE_ROOT}/control_{c}_dense {SEEDS_ROOT}/control_{c}_dense" for c in CONTROLS)
    stage = (f"mkdir -p {SEEDS_ROOT} && cat > {CKPT_JSON} <<'JSONEOF'\n"
             f"{json.dumps({'checkpoints': checkpoints})}\nJSONEOF\n{link_cmds}")
    subprocess.run(["ssh", "node1", stage], check=True)
    print(f"staged {CKPT_JSON} + symlinked {len(CONTROLS)} controls into {SEEDS_ROOT}")

    # 6 GPUs (not 8): leaves 2 for the concurrent §2b build+probe chain (spine) after killrung.
    cmd = (f"python -u -m anamnesis.scripts.run_replay_multickpt --model qwen-7b "
           f"--model-path {QPATH} --calib-dir {CALIB} --manifest {MANIFEST} "
           f"--checkpoints-json {CKPT_JSON} --gpus 0,1,2,3,4,5 --workers-per-gpu 8")
    jid = submit("a6_cohort_seeds_swap", cmd, gpus=6, minutes=60)
    print(f"a6_cohort_seeds_swap -> {jid}")
    print(f"output: {SEEDS_ROOT}/<animal>_<seed>/step-<step>/signatures_v3")


if __name__ == "__main__":
    main()
