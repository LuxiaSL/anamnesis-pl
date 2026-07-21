"""Entropy legs — vmb_c3_entropy_replay over each chain's V7 dose ladder + Rband nulls (+ field members).

The direct PX-1 "V7 writes ENTROPY" test + the PX-4 leak-law input: re-inject each cell's vector, forward
over its (fixed) generated text, read per-position next-token entropy + the ÷-Rband null ratios. Fires per
chain with a `depends_on` its gen job, so all can be submitted now and each runs once its gen lands (8B gen
already done -> no dep). GPU (one forward pass per cell). First-read -> desk; UNSTAMPED. C§8.

    python -m anamnesis.scripts.ops.submit_entropy_legs [--fire] [--chain 8b|qwen|gemma|field|all]
"""
from __future__ import annotations

import argparse
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BANK = "/models/anamnesis-extract/battery"
RUNS = "/models/anamnesis-extract/runs"
QPATH = ("/models/subliminal-anamnesis/.hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/"
         "snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
BASE = base("HF_HUB_OFFLINE=1")


def ladder(S: int) -> list[str]:
    v7 = [f"V7_L{S}_a{d}" for d in ("0.03", "0.1", "0.2", "0.3")] + \
         [f"V7_L{S}_an{d}" for d in ("0.03", "0.1", "0.2", "0.3")]
    rb = [f"Rband{i}_L{S}_a0.1" for i in (1, 2, 3)] + [f"Rband{i}_L{S}_an0.1" for i in (1, 2, 3)]
    return v7 + rb


def members(S: int) -> list[str]:
    return [f"{k}_L{S}_a{d}" for k in ("Vrep_perp", "Veos_perp", "Vconf") for d in ("0.1", "0.3")] + \
           [f"{k}_L{S}_an{d}" for k in ("Vrep_perp", "Veos_perp", "Vconf") for d in ("0.1", "0.3")]


# per-chain: (model, model-path, run-dir, cells, gen-dep-jobid or None, extra-env)
CHAINS = {
    "8b":    ("8b", "/models/llama-3.1-8b-instruct", f"{RUNS}/vmb_a5_8b_v7_L16", ladder(16), None, {}),
    "qwen":  ("qwen-7b", QPATH, f"{RUNS}/vmb_a5_qwen-7b_v7_L21", ladder(21), "8066ac270904", {}),
    "gemma": ("gemma3-27b", "google/gemma-3-27b-it", f"{RUNS}/vmb_a5_gemma3-27b_v7_L36", ladder(36),
              "ffe241c11523", {"HF_HOME": "/models/anamnesis-extract/.hf-cache"}),  # re-pointed to gemma resume gen
    "field": ("8b", "/models/llama-3.1-8b-instruct", f"{RUNS}/vmb_a5_8b_field_L16",
              ladder(16) + members(16), "cadff29b4390", {}),
    # FM-1/FM-2 field-triple entropy legs (dep=None -> fire after the port's gen-main lands;
    # run dirs match submit_steering_matrix_field.py's run_root vmb_a5_{tag}_field_L{S}).
    "field_qwen":  ("qwen-7b", QPATH, f"{RUNS}/vmb_a5_qwen-7b_field_L21",
                    ladder(21) + members(21), "fe17e089582e", {}),   # dep = qwen field gen_stop
    "field_gemma": ("gemma3-27b", "google/gemma-3-27b-it", f"{RUNS}/vmb_a5_gemma3-27b_field_L36",
                    ladder(36) + members(36), "7a6d183720f9",         # dep = gemma field gen_stop
                    {"HF_HOME": "/models/anamnesis-extract/.hf-cache"}),
}


def submit(name, cmd, gpus, minutes, env, deps=None):
    e = {"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1", **env}
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": e,
            "command": f"bash -c '{BASE} && {cmd}'"}
    if deps:
        spec["depends_on"] = deps
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["job"]["id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", action="store_true")
    ap.add_argument("--chain", default="all", choices=["all", *CHAINS])
    args = ap.parse_args()
    keys = list(CHAINS) if args.chain == "all" else [args.chain]
    for k in keys:
        model, mp, run, cells, dep, env = CHAINS[k]
        out = f"{BANK}/arms/A5_matrix/{k}/entropy_{k}.json"
        cmd = (f"python -u -m anamnesis.scripts.vmb_c3_entropy_replay --model {model} --model-path {mp} "
               f"--c3-run-dir {run} --cells {' '.join(cells)} --null-prefixes RBAND --out-json {out}")
        gpus = 4 if model == "gemma3-27b" else 2
        print(f"[{k}] {len(cells)} cells, gpus={gpus}, dep={dep}")
        if not args.fire:
            print(f"    {cmd}\n"); continue
        jid = submit(f"vmb-ent-{k}", cmd, gpus, 40 if model == "gemma3-27b" else 25, env,
                     [dep] if dep else None)
        print(f"    submitted -> {jid}")


if __name__ == "__main__":
    main()
