"""C3 synthetic-temperature injection cells (PREFLIGHT §3; Luxia-ratified 2026-07-14 session-4).
V_temp + 3 matched-norm randoms (Rc1-3) × sites {L14 (map-comparable), L21 (strongest V_temp
raw norm)} × C§3 doses {.03,.1,.3} × n=160 free-gen, generated positions only, inject L{site}.
Per-vector gen→replay chains, 1 GPU each (4 vectors → 4 concurrent GPUs). Signature replay
(lever leg (d) via the C2 orphaned axis + deformation + coherence + semantics); the entropy/KL
certifying consequences (b)/(c) = a follow-up logit-retaining replay subset (not this fire).
First-read → outer loop; nothing stamped. Combined bank built by vmb_c3_assemble.py.
"""
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
MODEL, MPATH = "3b", "/models/llama-3.2-3b-instruct"
RUNS = "/models/anamnesis-extract/runs"
CALIB = "/models/anamnesis-extract/calibration/3b"
VEC_DIR = "/models/anamnesis-extract/battery/a5_vectors_3b_c3"
NPZ, STAMPS = f"{VEC_DIR}/a5_vectors.npz", f"{VEC_DIR}/a5_vectors_stamps.json"
PROMPTS = "pipeline/anamnesis/prompts/prompt_sets.json"
VECTORS = ["Vtemp", "Rc1", "Rc2", "Rc3"]
SITES, DOSES = [14, 21], [0.03, 0.1, 0.3]


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


def gen_cmd(vec, s, f):
    c = f"{vec}_L{s}_a{f}"
    return (f"python -u -m anamnesis.scripts.vmb_stage0_generate --model {MODEL} --model-path {MPATH} "
            f"--prompts {PROMPTS} --out-run-dir {RUNS}/vmb_c3_{MODEL}/{c} --gpus 0 --workers-per-gpu 4 "
            f"--seed-namespace VMBC3-{MODEL.upper()}-{c} --inject-npz {NPZ} --inject-key {vec}_L{s} "
            f"--inject-layer {s} --inject-alpha-frac {f} --inject-norms-json {STAMPS}")


def rep_cmd(vec, s, f):
    c = f"{vec}_L{s}_a{f}"
    return (f"python -u -m anamnesis.scripts.parallel_replay --model {MODEL} --model-path {MPATH} "
            f"--run-dir {RUNS}/vmb_c3_{MODEL}/{c} --calib-dir {CALIB} "
            f"--manifest {RUNS}/vmb_c3_{MODEL}/{c}/replay_manifest.json --gpus 0 "
            f"--workers-per-gpu 3 --no-raw --inject-from-metadata")


for vec in VECTORS:
    cells = [(s, f) for s in SITES for f in DOSES]
    g = submit(f"vmb-c3-gen-{vec}", " && ".join(gen_cmd(vec, s, f) for s, f in cells), gpus=1, minutes=40)
    r = submit(f"vmb-c3-rep-{vec}", " && ".join(rep_cmd(vec, s, f) for s, f in cells),
               gpus=1, minutes=40, depends_on=[g])
    print(f"{vec}: gen={g} rep={r}")
