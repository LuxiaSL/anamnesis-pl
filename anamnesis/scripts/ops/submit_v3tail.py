# HISTORICAL — do not copy; template = submit_a5_replication.py.
# (1-GPU-per-vector-chain, 3 workers, per-cell loops: this submit pre-dates the
# multicell instrument and the worker-count ladder.)
"""V3tail n=160 promotion (§B.5 outcome-(ii) authorized: tail half clears 1.5×-null at pilot
n=40 → promote to n=160). V3tail_L14 + matched-support Rtail1-3_L14 × doses {.03,.1,.3},
n=160 free-gen, inject L14. From the banked b5 spectral-split vectors. Per-vector gen→replay
chains (1 GPU each). Signature replay (dir0 targeting + deformation, matched-support efficiency
lens). First-read → outer loop; nothing stamped. Queues behind C3 on the shared 4 GPUs.
"""
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
MODEL, MPATH = "3b", "/models/llama-3.2-3b-instruct"
RUNS = "/models/anamnesis-extract/runs"
CALIB = "/models/anamnesis-extract/calibration/3b"
VEC_DIR = "/models/anamnesis-extract/battery/a5_vectors_3b_b5"
NPZ, STAMPS = f"{VEC_DIR}/a5_vectors.npz", f"{VEC_DIR}/a5_vectors_stamps.json"
PROMPTS = "pipeline/anamnesis/prompts/prompt_sets.json"
VECTORS = ["V3tail", "Rtail1", "Rtail2", "Rtail3"]
SITE, DOSES = 14, [0.03, 0.1, 0.3]


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


def gen_cmd(vec, f):
    c = f"{vec}_L{SITE}_a{f}"
    return (f"python -u -m anamnesis.scripts.vmb_stage0_generate --model {MODEL} --model-path {MPATH} "
            f"--prompts {PROMPTS} --out-run-dir {RUNS}/vmb_b5promo_{MODEL}/{c} --gpus 0 --workers-per-gpu 4 "
            f"--seeds-per-class 2 "  # n=160 (the law; matches the b5 pilot's parent grid)
            f"--seed-namespace VMBB5P-{MODEL.upper()}-{c} --inject-npz {NPZ} --inject-key {vec}_L{SITE} "
            f"--inject-layer {SITE} --inject-alpha-frac {f} --inject-norms-json {STAMPS}")


def rep_cmd(vec, f):
    c = f"{vec}_L{SITE}_a{f}"
    return (f"python -u -m anamnesis.scripts.parallel_replay --model {MODEL} --model-path {MPATH} "
            f"--run-dir {RUNS}/vmb_b5promo_{MODEL}/{c} --calib-dir {CALIB} "
            f"--manifest {RUNS}/vmb_b5promo_{MODEL}/{c}/replay_manifest.json --gpus 0 "
            f"--workers-per-gpu 3 --no-raw --inject-from-metadata")


for vec in VECTORS:
    g = submit(f"vmb-b5promo-gen-{vec}", " && ".join(gen_cmd(vec, f) for f in DOSES), gpus=1, minutes=30)
    r = submit(f"vmb-b5promo-rep-{vec}", " && ".join(rep_cmd(vec, f) for f in DOSES),
               gpus=1, minutes=30, depends_on=[g])
    print(f"{vec}: gen={g} rep={r}")
