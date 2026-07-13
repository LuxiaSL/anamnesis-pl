"""Submit ARM A5 free-gen chains (3B, ratified 13a grid) to Heimdall.

Topology: vectors-basic job (1 GPU) -> per-vector gen job (6 GPUs, 7 cells
serial: L14 ladder x 4 fracs + L7/L18/L21 sweep at 0.3) -> per-vector replay
job (2 GPUs, --inject-from-metadata so the spec is byte-identical to the gen's)
+ one riders job (3 alpha=0 cells). V2 (SAE) and V4 (feature-gradient) chains
are submitted separately once their construction stages land.

Alpha resolution happens AT RUN TIME inside each gen job (frac x median site
norm from the stamps json the vectors job banks) — nothing here hardcodes norms.
"""
import json
import urllib.request

API = "http://HEIMDALL-HOST-REDACTED:7000/api/v1/jobs"
BASE = ("source /home/CLUSTER-USER/luxi-files/.venv-shared/bin/activate && "
        "cd /home/CLUSTER-USER/luxi-files/anamnesis-pl && "
        "export PYTHONPATH=$PWD/pipeline PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}

MODEL, MPATH = "3b", "/models/llama-3.2-3b-instruct"
RUNS = "/models/anamnesis-extract/runs"
CALIB = "/models/anamnesis-extract/calibration/3b"
STAGE0 = f"{RUNS}/vmb_stage0_3b"
VEC_DIR = "/models/anamnesis-extract/battery/a5_vectors_3b"
NPZ = f"{VEC_DIR}/a5_vectors.npz"
STAMPS = f"{VEC_DIR}/a5_vectors_stamps.json"
PROMPTS = "pipeline/anamnesis/prompts/prompt_sets.json"

LADDER = [0.03, 0.1, 0.3, 1.0]
SWEEP_SITES = [7, 18, 21]
MAP_SITE = 14
# Per-site vector keys for the trait vectors; site-independent for randoms.
VECTORS = {
    "V1": {"per_site": True},
    "V3": {"per_site": True},
    "R1": {"per_site": False},
    "R2": {"per_site": False},
    "R3": {"per_site": False},
}


def submit(name, command, gpus, minutes, depends_on=None):
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": "/home/CLUSTER-USER/luxi-files/anamnesis-pl",
            "estimated_minutes": minutes, "env": ENV,
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


def cell_tag(key: str, layer: int, frac: float) -> str:
    return f"{key}_L{layer}_a{frac}"


def gen_cmd(key: str, layer: int, frac: float | None, alpha_abs: float | None,
            gen_gpus: str) -> str:
    cell = cell_tag(key, layer, frac if frac is not None else 0.0)
    inj = (f"--inject-alpha {alpha_abs}" if alpha_abs is not None
           else f"--inject-alpha-frac {frac} --inject-norms-json {STAMPS}")
    return (f"python -u -m anamnesis.scripts.vmb_stage0_generate --model {MODEL} "
            f"--model-path {MPATH} --prompts {PROMPTS} "
            f"--out-run-dir {RUNS}/vmb_a5_{MODEL}/{cell} "
            f"--gpus {gen_gpus} --workers-per-gpu 6 --seeds-per-class 2 "
            f"--seed-namespace VMBA5-{MODEL.upper()}-{cell} "
            f"--inject-npz {NPZ} --inject-key {key} --inject-layer {layer} {inj}")


def replay_cmd(key: str, layer: int, frac: float) -> str:
    cell = cell_tag(key, layer, frac)
    return (f"python -u -m anamnesis.scripts.parallel_replay --model {MODEL} "
            f"--model-path {MPATH} --run-dir {RUNS}/vmb_a5_{MODEL}/{cell} "
            f"--calib-dir {CALIB} --manifest {RUNS}/vmb_a5_{MODEL}/{cell}/replay_manifest.json "
            f"--gpus 0,1 --workers-per-gpu 8 --no-raw --inject-from-metadata")


vec_job = submit(
    "vmb-a5-vectors-basic",
    f"python -u -m anamnesis.scripts.vmb_a5_build_vectors --model {MODEL} "
    f"--model-path {MPATH} --stage0-run {STAGE0} --a2-root {RUNS} "
    f"--prompts {PROMPTS} --out-dir {VEC_DIR} --stage basic",
    gpus=1, minutes=40)
print(f"vectors: {vec_job}")

# Riders (alpha=0, one per vector family; instrument check + behavioral reference)
rider_cmds = []
for key, layer in (("V1_L14", MAP_SITE), ("V3_L14", MAP_SITE), ("R1", MAP_SITE)):
    rider_cmds.append(gen_cmd(key, layer, None, 0.0, "0,1,2,3,4,5"))
riders_gen = submit("vmb-a5-gen-riders", " && ".join(rider_cmds),
                    gpus=6, minutes=45, depends_on=[vec_job])
rider_replays = []
for key, layer in (("V1_L14", MAP_SITE), ("V3_L14", MAP_SITE), ("R1", MAP_SITE)):
    rider_replays.append(replay_cmd(key, layer, 0.0))
riders_rep = submit("vmb-a5-rep-riders", " && ".join(rider_replays),
                    gpus=2, minutes=40, depends_on=[riders_gen])
print(f"riders: gen={riders_gen} rep={riders_rep}")

for vname, cfg in VECTORS.items():
    cells: list[tuple[str, int, float]] = []
    for frac in LADDER:
        key = f"{vname}_L{MAP_SITE}" if cfg["per_site"] else vname
        cells.append((key, MAP_SITE, frac))
    for site in SWEEP_SITES:
        key = f"{vname}_L{site}" if cfg["per_site"] else vname
        cells.append((key, site, 0.3))
    gen_job = submit(
        f"vmb-a5-gen-{vname}",
        " && ".join(gen_cmd(k, l, f, None, "0,1,2,3,4,5") for k, l, f in cells),
        gpus=6, minutes=70, depends_on=[vec_job])
    rep_job = submit(
        f"vmb-a5-rep-{vname}",
        " && ".join(replay_cmd(k, l, f) for k, l, f in cells),
        gpus=2, minutes=70, depends_on=[gen_job])
    print(f"{vname}: gen={gen_job} rep={rep_job}")
