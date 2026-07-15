"""Submit ARM A5 §4.2 — randoms pushed HARD (frac ∈ {2,3}) to Heimdall.

WAVE2-A5 runbook item 4 / addendum 13d §4.2: do random directions R1–R3 fall into
the V1/V3 nominal-loop past-collapse basin once degraded as hard as the coherent
vectors? The ratified grid stopped at frac 1.0 (last coherent dose); this pushes R1–R3
to 2× and 3× the median site norm at the map site (L14), text-only gen + replay. The
§4.1 field / TTR / entropy read is a CPU pass over the resulting sigs (frozen_directional
--model 3b re-run once these land, plus a TTR/entropy reducer). Sharpens the
past-collapse ruling. First-read → outer loop (A5-class).

Reuses submit_a5_freegen_chains topology exactly: gen job (6 GPUs) -> replay job
(2 GPUs, --inject-from-metadata so the spec is byte-identical to the gen's). Alpha
resolves at RUN TIME (frac × median site norm from the stamps json); nothing hardcodes
norms.
"""
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}

MODEL, MPATH = "3b", "/models/llama-3.2-3b-instruct"
RUNS = "/models/anamnesis-extract/runs"
CALIB = "/models/anamnesis-extract/calibration/3b"
VEC_DIR = "/models/anamnesis-extract/battery/a5_vectors_3b"
NPZ = f"{VEC_DIR}/a5_vectors.npz"
STAMPS = f"{VEC_DIR}/a5_vectors_stamps.json"
PROMPTS = "pipeline/anamnesis/prompts/prompt_sets.json"

MAP_SITE = 14
HARD_FRACS = [2.0, 3.0]
RANDOMS = ["R1", "R2", "R3"]


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


def cell_tag(key: str, frac: float) -> str:
    return f"{key}_L{MAP_SITE}_a{frac}"


def gen_cmd(key: str, frac: float) -> str:
    cell = cell_tag(key, frac)
    return (f"python -u -m anamnesis.scripts.vmb_stage0_generate --model {MODEL} "
            f"--model-path {MPATH} --prompts {PROMPTS} "
            f"--out-run-dir {RUNS}/vmb_a5_{MODEL}/{cell} "
            f"--gpus 0,1,2,3,4,5 --workers-per-gpu 6 --seeds-per-class 2 "
            f"--seed-namespace VMBA5-{MODEL.upper()}-{cell} "
            f"--inject-npz {NPZ} --inject-key {key} --inject-layer {MAP_SITE} "
            f"--inject-alpha-frac {frac} --inject-norms-json {STAMPS}")


def replay_cmd(key: str, frac: float) -> str:
    cell = cell_tag(key, frac)
    return (f"python -u -m anamnesis.scripts.parallel_replay --model {MODEL} "
            f"--model-path {MPATH} --run-dir {RUNS}/vmb_a5_{MODEL}/{cell} "
            f"--calib-dir {CALIB} --manifest {RUNS}/vmb_a5_{MODEL}/{cell}/replay_manifest.json "
            f"--gpus 0,1 --workers-per-gpu 8 --no-raw --inject-from-metadata")


cells = [(k, f) for k in RANDOMS for f in HARD_FRACS]
gen_job = submit("vmb-a5-randhard-gen",
                 " && ".join(gen_cmd(k, f) for k, f in cells),
                 gpus=6, minutes=70)
rep_job = submit("vmb-a5-randhard-rep",
                 " && ".join(replay_cmd(k, f) for k, f in cells),
                 gpus=2, minutes=60, depends_on=[gen_job])
print(f"randoms-hard: gen={gen_job} rep={rep_job}")
