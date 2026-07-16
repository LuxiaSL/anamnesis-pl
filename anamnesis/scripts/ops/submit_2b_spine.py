"""A6 §2b SPINE — build the distilled-direction vectors then behaviorally read them (session-9).

Chain: (1) vmb_a6_2b_build — teacher(cat-sys)/student(ckpt-0453) gens on numbers + favorite-
animal, V3-bare teacher<->student axis -> Valign_L18 / Vdiverge_L18; (2) vmb_a6_2b_probe —
steer base-Qwen (fp16) + Valign/Vdiverge/AR{1,2,3} on the canonical 8 animal prompts, read
de-dicto/de-se + census + placebo + coherence gate. Both first-reads -> outer loop (C§8 ABS).
1 GPU each, chained. Run: HEIMDALL_* env exported. Node1.
"""
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
QPATH = ("/models/subliminal-anamnesis/.hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/"
         "snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
ADAPTER = "/models/subliminal-anamnesis/checkpoints/qwen_cat_student/checkpoint-0453"
BATTERY = "/models/anamnesis-extract/battery"
VEC_DIR = f"{BATTERY}/a6_2b_vectors_qwen"
NPZ = f"{VEC_DIR}/vectors.npz"
BUILD_JSON = f"{VEC_DIR}/construction.json"
AR_NPZ = f"{BATTERY}/a6_animal_vectors_qwen/a5_vectors.npz"
STAMPS = f"{BATTERY}/a6_animal_vectors_qwen/a5_vectors_stamps.json"
PROBE_JSON = f"{BATTERY}/arms/A6/dese_2b_steer_qwen.json"


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


def main():
    build = (f"mkdir -p {VEC_DIR} && python -u -m anamnesis.scripts.vmb_a6_2b_build "
             f"--model qwen-7b --model-path {QPATH} --adapter-path {ADAPTER} "
             f"--sites 7 14 18 21 --inject-site 18 --n-samples 16 --max-new-tokens 128 "
             f"--decile 0.10 --out-npz {NPZ} --out-json {BUILD_JSON}")
    b = submit("a6_2b_build", build, 1, 45)
    print(f"a6_2b_build -> {b}")

    probe = (f"python -u -m anamnesis.scripts.vmb_a6_2b_probe --model qwen-7b --model-path {QPATH} "
             f"--vec-npz {NPZ} --ar-npz {AR_NPZ} --stamps {STAMPS} --site 18 "
             f"--alphas 0.0 0.45 0.6 0.8 --n-samples 8 --max-new-tokens 160 "
             f"--out-json {PROBE_JSON}")
    p = submit("a6_2b_probe", probe, 1, 60, depends_on=[b])
    print(f"a6_2b_probe -> {p} [after {b}]")
    print(f"\nvectors -> {NPZ}\nconstruction -> {BUILD_JSON}\nbehavioral read -> {PROBE_JSON}")


if __name__ == "__main__":
    main()
