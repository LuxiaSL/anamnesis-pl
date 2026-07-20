"""PX-3 stopping completion — UNCAPPED re-gen of the Veos⊥ cells to measure length-bidirectionality.

The capped (512) field gen pinned mean_len at ~505-512 (frac_at_cap 0.963), so "longer" was unmeasurable
(S8-18 "capped baseline can't represent the EOS population"). This re-gens the stopping cells at
max_new_tokens 2048 so ±Veos⊥ length can move both ways. Gen-only (length lives in metadata; no replay needed).
Cells: Veos_perp_L16 ±{.1,.3} + V3_L16_a0.0 baseline + Rband1_L16 ±.1 null, n=40. Readout = annex_roster_text_readout
(or mean_len per cell). First-read -> desk; UNSTAMPED. C§8.

    python -m anamnesis.scripts.ops.submit_field_stopping [--fire]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BANK = "/models/anamnesis-extract/battery"
RUNS = "/models/anamnesis-extract/runs"
MODEL, MPATH = "8b", "/models/llama-3.1-8b-instruct"
GENBANK = f"{BANK}/a5_field_8b/field_gen/a5_vectors.npz"
GENSTAMPS = f"{BANK}/a5_field_8b/field_gen/a5_vectors_stamps.json"
RUN = f"{RUNS}/vmb_a5_8b_field_stop_uncapped"
PROMPTS = "pipeline/anamnesis/prompts/prompt_sets.json"
NODE_CELLS = f"{WORK_DIR}/_steering_matrix_cells"
LOCAL = Path("/tmp/claude-output/steering_matrix_cells")
BASE = base("HF_HUB_OFFLINE=1")
S = 16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    gen, rep = [], []

    def add(key, frac):
        lab = f"{key}_a{'n' if frac < 0 else ''}{abs(frac)}"
        run = f"{RUN}/{lab}"
        gen.append({"out_run_dir": run, "seed_namespace": f"VMBSTOP-8B-{lab}",
                    "inject_key": key, "inject_layer": S, "inject_alpha_frac": frac})
    for f in (0.1, 0.3):
        add(f"Veos_perp_L{S}", f); add(f"Veos_perp_L{S}", -f)
    add(f"Rband1_L{S}", 0.1); add(f"Rband1_L{S}", -0.1)
    gen.append({"out_run_dir": f"{RUN}/V3_L{S}_a0.0", "seed_namespace": "VMBSTOP-8B-baseline",
                "inject_key": f"V3_L{S}", "inject_layer": S, "inject_alpha_frac": 0.0})

    LOCAL.mkdir(parents=True, exist_ok=True)
    gpath = LOCAL / "gen_field_stop.json"
    gpath.write_text(json.dumps({"cells": gen}, indent=1))
    cmd = (f"python -u -m anamnesis.scripts.vmb_a5_gen_multicell --model {MODEL} --model-path {MPATH} "
           f"--prompts {PROMPTS} --cells-json {NODE_CELLS}/gen_field_stop.json --inject-npz {GENBANK} "
           f"--inject-norms-json {GENSTAMPS} --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 6 "
           f"--seeds-per-class 1 --limit 40 --max-new-tokens {args.max_new_tokens}")
    print(f"{len(gen)} cells @ max_new_tokens={args.max_new_tokens}\n  {cmd}")
    if not args.fire:
        print("(dry-run)"); return
    subprocess.run(["ssh", "node1", f"mkdir -p {NODE_CELLS}"], check=True)
    subprocess.run(["rsync", "-a", str(gpath), f"node1:{NODE_CELLS}/"], check=True)
    spec = {"job_type": "custom", "name": "vmb-field-stop", "gpus": 8, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": 40,
            "env": {"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1"},
            "command": f"bash -c '{BASE} && {cmd}'"}
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        print("gen ->", json.loads(r.read())["job"]["id"])


if __name__ == "__main__":
    main()
