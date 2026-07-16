"""A5 vector-bank build submit (replication era; per-model sites from the A3 map).

One 1-GPU Heimdall job: vmb_a5_build_vectors --stage basic → R1-R3 + V1 (formality)
+ V3 (dir0_a5 = pure-analogical vs pure-contrastive residual contrast; NOT lda5_dir0)
+ median residual norms, banked to a per-model npz. Sites are per-model (the 3B default
[7,14,18,21] does NOT transfer): map site + sweep set grounded in each model's
a3_mode_direction_map (residual|mid peak). V2/SAE dropped for 8B/Qwen (no andyrdt-paired
bank on node1; droppable per model per prereg 14g stop-and-surface rule). First-read →
outer loop; nothing stamped.

Usage:  python -m anamnesis.scripts.ops.submit_a5_build --model 8b
"""
import argparse
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
RUNS = "/models/anamnesis-extract/runs"
BATTERY = "/models/anamnesis-extract/battery"

# per-model: (model tag, model path, injection sites, map site, sweep site)
# sites grounded in a3_mode_direction_map residual|mid peak; proportional depth to 3B.
MODELS = {
    "8b": dict(
        mpath="/models/llama-3.1-8b-instruct",
        stage0=f"{RUNS}/vmb_stage0_8b",
        sites="8,16,20,24", map_site=16, sweep_site=20),
    "qwen-7b": dict(
        mpath=("/models/subliminal-anamnesis/.hf-cache/hub/"
               "models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
               "a09a35458c702b33eeacc393d103063234e8bc28"),
        stage0=f"{RUNS}/vmb_stage0_qwen7b",
        sites="7,14,18,21", map_site=18, sweep_site=14),
}


def submit(name, command, gpus, minutes):
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": minutes,
            "env": {"HF_HUB_OFFLINE": "1"},
            "command": f"bash -c '{BASE} && {command}'"}
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["job"]["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    args = ap.parse_args()
    m = MODELS[args.model]
    out = f"{BATTERY}/a5_vectors_{args.model.replace('-', '_')}"
    cmd = (f"python -u -m anamnesis.scripts.vmb_a5_build_vectors --model {args.model} "
           f"--model-path {m['mpath']} --stage0-run {m['stage0']} --a2-root {RUNS} "
           f"--out-dir {out} --stage basic --sites {m['sites']}")
    jid = submit(f"vmb-a5-build-{args.model}", cmd, gpus=1, minutes=20)
    print(f"a5 build {args.model}: job {jid}")
    print(f"  sites={m['sites']}  map=L{m['map_site']}  sweep=L{m['sweep_site']}  out={out}")


if __name__ == "__main__":
    main()
