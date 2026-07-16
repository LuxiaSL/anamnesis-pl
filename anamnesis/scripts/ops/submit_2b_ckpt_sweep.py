"""A6 §2b — distillation-STAGE sweep (session-9; Luxia checkpoint question 2026-07-16).

The epoch-3 (ckpt-0453) student has FULLY acquired cat-ness, so the teacher↔student OOD axis
captures a REGISTER difference within shared cat-topicality, not a clean cat-vs-not divergence.
This rebuilds §2b at EARLIER student checkpoints (partial trait) so we can see how the divergence
axis + de-se installation move with distillation stage. Each stage: build → probe, chained, 1 GPU.
Run: HEIMDALL_* env. Node1.
"""
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
QPATH = ("/models/subliminal-anamnesis/.hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/"
         "snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
CKPT_ROOT = "/models/subliminal-anamnesis/checkpoints/qwen_cat_student"
B = "/models/anamnesis-extract/battery"
AR_NPZ = f"{B}/a6_animal_vectors_qwen/a5_vectors.npz"
STAMPS = f"{B}/a6_animal_vectors_qwen/a5_vectors_stamps.json"
# (label, checkpoint step, epoch) — earlier stages; 0453/epoch-3 already done as the main cell.
STAGES = [("s151_e1", "0151"), ("s302_e2", "0302")]


def submit(name, command, gpus, minutes, depends_on=None):
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": ENV,
            "command": f"bash -c '{BASE} && {command}'"}
    if depends_on:
        spec["depends_on"] = depends_on
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["job"]["id"]


def main() -> None:
    for label, step in STAGES:
        vdir = f"{B}/a6_2b_vectors_qwen_{label}"
        npz = f"{vdir}/vectors.npz"
        build = (f"mkdir -p {vdir} && python -u -m anamnesis.scripts.vmb_a6_2b_build "
                 f"--model qwen-7b --model-path {QPATH} --adapter-path {CKPT_ROOT}/checkpoint-{step} "
                 f"--sites 7 14 18 21 --inject-site 18 --n-samples 16 --max-new-tokens 128 "
                 f"--decile 0.10 --out-npz {npz} --out-json {vdir}/construction.json")
        b = submit(f"a6_2b_build_{label}", build, 1, 20)
        probe = (f"python -u -m anamnesis.scripts.vmb_a6_2b_probe --model qwen-7b --model-path {QPATH} "
                 f"--vec-npz {npz} --ar-npz {AR_NPZ} --stamps {STAMPS} --site 18 "
                 f"--alphas 0.0 0.45 0.6 0.8 --n-samples 8 --max-new-tokens 160 "
                 f"--out-json {B}/arms/A6/dese_2b_steer_qwen_{label}.json")
        p = submit(f"a6_2b_probe_{label}", probe, 1, 40, depends_on=[b])
        print(f"{label}: build {b} -> probe {p}  (vectors {vdir})")


if __name__ == "__main__":
    main()
