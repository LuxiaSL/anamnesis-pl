"""A5 free-gen replication chain, per-model (14g R1/R2/R7 machinery).

Mirrors submit_a5_freegen_chains topology (riders α=0 -> per-vector gen 6 GPUs ->
replay 2 GPUs, --inject-from-metadata byte-identical spec) but PARAMETRIZED by model,
with per-model sites/paths from the A3 map. Vectors are banked separately
(submit_a5_build); this consumes a5_vectors_<model>/. Lean replication grid:
map site full C§3 ladder {.03,.1,.3} + one sweep site {.1,.3} (prereg PART B:
"map site + one sweep site"). V2/V4 not built for replication models.

Alpha resolves AT RUN TIME (frac x median site norm from the stamps json); nothing
here hardcodes norms. First-read -> outer loop; nothing stamped.

Usage:  python -m anamnesis.scripts.ops.submit_a5_replication --model 8b
"""
import argparse
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
RUNS = "/models/anamnesis-extract/runs"
BATTERY = "/models/anamnesis-extract/battery"
PROMPTS = "pipeline/anamnesis/prompts/prompt_sets.json"

MAP_LADDER = [0.03, 0.1, 0.3]
SWEEP_DOSES = [0.1, 0.3]
VECTORS = {"V1": True, "V3": True, "R1": False, "R2": False, "R3": False}  # per_site flag

MODELS = {
    "8b": dict(mpath="/models/llama-3.1-8b-instruct",
               calib=f"{BATTERY.replace('/battery','')}/calibration/8b",
               map_site=16, sweep_site=20),
    "qwen-7b": dict(mpath=("/models/subliminal-anamnesis/.hf-cache/hub/"
                           "models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
                           "a09a35458c702b33eeacc393d103063234e8bc28"),
                    calib="/models/anamnesis-extract/calibration/qwen25_7b",
                    map_site=18, sweep_site=14),
}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    args = ap.parse_args()
    M, model = MODELS[args.model], args.model
    MPATH, CALIB = M["mpath"], M["calib"]
    MAP, SWEEP = M["map_site"], M["sweep_site"]
    VEC_DIR = f"{BATTERY}/a5_vectors_{model.replace('-', '_')}"
    NPZ, STAMPS = f"{VEC_DIR}/a5_vectors.npz", f"{VEC_DIR}/a5_vectors_stamps.json"
    OUT = f"{RUNS}/vmb_a5_{model.replace('-', '_')}"

    def tag(key, layer, frac):
        return f"{key}_L{layer}_a{frac}"

    def gen_cmd(key, layer, frac):
        cell = tag(key, layer, frac)
        inj = (f"--inject-alpha 0.0" if frac == 0.0
               else f"--inject-alpha-frac {frac} --inject-norms-json {STAMPS}")
        # ALL 8 GPUs for gen (no idle-tail while a 2-GPU replay runs alongside).
        return (f"python -u -m anamnesis.scripts.vmb_stage0_generate --model {model} "
                f"--model-path {MPATH} --prompts {PROMPTS} --out-run-dir {OUT}/{cell} "
                f"--gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 6 --seeds-per-class 2 "
                f"--seed-namespace VMBA5-{model.upper()}-{cell} --inject-npz {NPZ} "
                f"--inject-key {key} --inject-layer {layer} {inj}")

    def rep_cmd(key, layer, frac):
        cell = tag(key, layer, frac)
        # ALL 8 GPUs for replay too (CPU-bound; 8x8=64 workers < 128 cores).
        return (f"python -u -m anamnesis.scripts.parallel_replay --model {model} "
                f"--model-path {MPATH} --run-dir {OUT}/{cell} --calib-dir {CALIB} "
                f"--manifest {OUT}/{cell}/replay_manifest.json --gpus 0,1,2,3,4,5,6,7 "
                f"--workers-per-gpu 8 --no-raw --inject-from-metadata")

    # Topology: ALL gens (8 GPUs, serial cells) -> ALL replays (8 GPUs, serial cells).
    # Each job type owns the whole node; no 6+2 split leaving GPUs idle mid-run.
    cells = [(f"V1_L{MAP}", MAP, 0.0), (f"V3_L{MAP}", MAP, 0.0), ("R1", MAP, 0.0)]  # riders
    for vname, per_site in VECTORS.items():
        for frac in MAP_LADDER:
            cells.append((f"{vname}_L{MAP}" if per_site else vname, MAP, frac))
        for frac in SWEEP_DOSES:
            cells.append((f"{vname}_L{SWEEP}" if per_site else vname, SWEEP, frac))

    gj = submit(f"vmb-a5r-{model}-GENALL",
                " && ".join(gen_cmd(*c) for c in cells), gpus=8, minutes=180)
    rj = submit(f"vmb-a5r-{model}-REPALL",
                " && ".join(rep_cmd(*c) for c in cells), gpus=8, minutes=180,
                depends_on=[gj])
    print(f"{model}: {len(cells)} cells | GENALL={gj} -> REPALL={rj} (both 8 GPU)")


if __name__ == "__main__":
    main()
