"""PX-3/PX-5 — the 8B field-inventory triple @L16 (repetition / stopping / confidence), Gram-Schmidt ⊥ V7-8B.

Transplant of the 3B EOS/rep roster (ANNEX-LEDGER S8-18): raw functional gradients -> band-pass ->
Gram-Schmidt the coordinate content off the temperature projection (V7). RIDES ON the 8B V7 stack from
`submit_steering_matrix.py --model 8b` (needs V7_L16 in a5_vectors_8b_b7) — fire this AFTER that chain's
cpu-8b job, or pass its id via `--after-cpu8b <JOBID>`.

Members: Vrep⊥_L16 / Veos⊥_L16 (⊥ V7-8B, ~1e-16) + Vconf_L16 (band-passed; PX-3 tests whether CONFIDENCE
COLLAPSES onto V7, |cos|>.60). Every member filing carries cos-to-V7 AND cos-to-Vrep⊥ (the rep-leak law
rides — PX-5). Gen = the members ± α{.1,.3} + the in-run V7 dose ladder (PX-4 law leg) + Rband + baseline.

STAGING: `--dry-run` (default) prints the DAG + writes cells; `--fire [--after-cpu8b JID]` submits. C§8.
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
PROMPTS = "pipeline/anamnesis/prompts/prompt_sets.json"
MODEL, MPATH = "8b", "/models/llama-3.1-8b-instruct"
STAGE0 = f"{RUNS}/vmb_stage0_8b"
CALIB = "/models/anamnesis-extract/calibration/8b"
SIGMA = f"{BANK}/arms/A5/a5_sigma_L16_8b.npz"
V3BANK = f"{BANK}/a5_vectors_8b"
V3NPZ, V3STAMPS = f"{V3BANK}/a5_vectors.npz", f"{V3BANK}/a5_vectors_stamps.json"
B7 = f"{BANK}/a5_vectors_8b_b7/a5_vectors.npz"           # V7_L16 + Rband*_L16 (from the 8b V7 stack)
FIELD = f"{BANK}/a5_field_8b"                            # this arm's bank root
S = 16
ENV = {"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1"}
NODE_CELLS = f"{WORK_DIR}/_steering_matrix_cells"
LOCAL_CELLS = Path("/tmp/claude-output/steering_matrix_cells")
BASE = base("HF_HUB_OFFLINE=1")


def submit(spec: dict) -> str:
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["job"]["id"]


def spec(name, cmd, gpus, minutes, deps=None):
    s = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
         "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": dict(ENV),
         "command": f"bash -c '{BASE} && {cmd}'"}
    if deps:
        s["depends_on"] = deps
    return s


def pulse_cmd(fn: str) -> str:
    return (f"python -u -m anamnesis.scripts.annex_potential_gradient --model {MODEL} --model-path {MPATH} "
            f"--stage0-run {STAGE0} --out-dir {FIELD}/roster_{fn} --functional {fn} --map-site {S} --n-gens 20")


# CPU members job: merge the 3 rosters -> band-pass -> Gram-Schmidt ⊥ V7 -> merged field GENBANK
MEMBERS_CMD = (
    f"python -u -m anamnesis.scripts.vmb_a5_merge_banks --out-dir {FIELD}/roster_all "
    f"--source {FIELD}/roster_margin/roster_gradients.npz:Gmargin_L{S} "
    f"--source {FIELD}/roster_eos/roster_gradients.npz:Geos_L{S} "
    f"--source {FIELD}/roster_repmass/roster_gradients.npz:Grep_L{S} --norms-from {V3STAMPS} && "
    f"python -u -m anamnesis.scripts.annex_band_pass --gradients {FIELD}/roster_all/a5_vectors.npz "
    f"--keys Gmargin_L{S}:Vconf_L{S} Geos_L{S}:Veos_L{S} Grep_L{S}:Vrep_L{S} "
    f"--sigma {SIGMA} --stamps {V3STAMPS} --compare {B7}:V7_L{S} {V3NPZ}:V3_L{S} "
    f"--out-dir {FIELD}/members && "
    f"python -u -m anamnesis.scripts.annex_perp_vectors --members {FIELD}/members/a5_vectors.npz "
    f"--keys Vrep_L{S}:Vrep_perp_L{S} Veos_L{S}:Veos_perp_L{S} --v7-npz {B7} --v7-key V7_L{S} "
    f"--stamps {V3STAMPS} --out-dir {FIELD}/perp && "
    f"python -u -m anamnesis.scripts.vmb_a5_merge_banks --out-dir {FIELD}/field_gen --model 8b "
    f"--source {FIELD}/perp/a5_vectors.npz:Vrep_perp_L{S},Veos_perp_L{S} "
    f"--source {FIELD}/members/a5_vectors.npz:Vconf_L{S} "
    f"--source {B7}:V7_L{S},Rband1_L{S},Rband2_L{S},Rband3_L{S} --norms-from {V3STAMPS}")


def cells():
    run_root = f"{RUNS}/vmb_a5_8b_field_L{S}"
    ns = f"VMBPX3-8B-L{S}"
    gen, rep = [], []

    def add(key, frac):
        lab = f"{key}_a{'n' if frac < 0 else ''}{abs(frac)}"
        run = f"{run_root}/{lab}"
        gen.append({"out_run_dir": run, "seed_namespace": f"{ns}-{lab}",
                    "inject_key": key, "inject_layer": S, "inject_alpha_frac": frac})
        rep.append({"run_dir": run, "manifest": f"{run}/replay_manifest.json"})

    for key in (f"Vrep_perp_L{S}", f"Veos_perp_L{S}", f"Vconf_L{S}"):
        for f in (0.1, 0.3):
            add(key, f); add(key, -f)
    for f in (0.03, 0.1, 0.2, 0.3):          # in-run V7 dose ladder (PX-4)
        add(f"V7_L{S}", f); add(f"V7_L{S}", -f)
    for rb in (1, 2, 3):
        add(f"Rband{rb}_L{S}", 0.1); add(f"Rband{rb}_L{S}", -0.1)
    run = f"{run_root}/V3_L{S}_a0.0"
    gen.append({"out_run_dir": run, "seed_namespace": f"{ns}-baseline",
                "inject_key": f"V3_L{S}", "inject_layer": S, "inject_alpha_frac": 0.0})
    rep.append({"run_dir": run, "manifest": f"{run}/replay_manifest.json"})
    return gen, rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", action="store_true")
    ap.add_argument("--after-cpu8b", default=None, help="job id of the 8b V7-stack cpu-8b job (gates members)")
    ap.add_argument("--members-only", action="store_true",
                    help="re-fire members->gen->replay only (roster pulses already on disk)")
    args = ap.parse_args()

    g_cells, r_cells = cells()
    LOCAL_CELLS.mkdir(parents=True, exist_ok=True)
    gpath = LOCAL_CELLS / "gen_8b_field.json"
    rpath = LOCAL_CELLS / "rep_8b_field.json"
    gpath.write_text(json.dumps({"cells": g_cells}, indent=1))
    rpath.write_text(json.dumps({"cells": r_cells}, indent=1))

    genbank = f"{FIELD}/field_gen/a5_vectors.npz"
    genstamps = f"{FIELD}/field_gen/a5_vectors_stamps.json"
    gen_cmd = (f"python -u -m anamnesis.scripts.vmb_a5_gen_multicell --model {MODEL} --model-path {MPATH} "
               f"--prompts {PROMPTS} --cells-json {NODE_CELLS}/gen_8b_field.json --inject-npz {genbank} "
               f"--inject-norms-json {genstamps} --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 6 "
               f"--seeds-per-class 2 --limit 80")
    rep_state = (f"python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model {MODEL} --model-path {MPATH} "
                 f"--calib-dir {CALIB} --cells-json {NODE_CELLS}/rep_8b_field.json --gpus 0,1,2,3,4,5,6,7 "
                 f"--workers-per-gpu 8 --no-raw --inject-from-metadata")
    rep_expr = (f"python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model {MODEL} --model-path {MPATH} "
                f"--calib-dir {CALIB} --cells-json {NODE_CELLS}/rep_8b_field.json --gpus 0,1,2,3,4,5,6,7 "
                f"--workers-per-gpu 8 --no-raw --sig-subdir signatures_v3_noinject")

    print("===== 8B FIELD TRIPLE (PX-3/5) @L16 =====")
    print(f"cells: {len(g_cells)} gen / {len(r_cells)} rep")
    print(f"  [pulse] margin/eos/repmass (3 × gpus=1 ~20m, deps=[])")
    print(f"  [members] cpu ~8m  deps=[3 pulses{' + '+args.after_cpu8b if args.after_cpu8b else ' + cpu-8b'}]")
    print(f"          {MEMBERS_CMD}")
    print(f"  [gen] gpus=8 ~45m deps=[members]\n          {gen_cmd}")
    print(f"  [rep] state/expr deps=[gen]")

    if not args.fire:
        print("(dry-run: nothing submitted)")
        return
    subprocess.run(["ssh", "node1", f"mkdir -p {NODE_CELLS}"], check=True)
    subprocess.run(["rsync", "-a", str(gpath), str(rpath), f"node1:{NODE_CELLS}/"], check=True)
    if args.members_only:
        # re-fire path: pulses + roster gradients already on disk -> members needs no upstream dep
        pulses, mdeps = [], []
    else:
        pulses = [submit(spec(f"vmb-fld-{fn}", pulse_cmd(fn), 1, 20)) for fn in ("margin", "eos", "repmass")]
        mdeps = pulses + ([args.after_cpu8b] if args.after_cpu8b else [])
    members = submit(spec("vmb-fld-members", MEMBERS_CMD, 1, 8, mdeps or None))
    g = submit(spec("vmb-fld-gen", gen_cmd, 8, 45, [members]))
    rs = submit(spec("vmb-fld-repstate", rep_state, 8, 35, [g]))
    re = submit(spec("vmb-fld-repexpr", rep_expr, 8, 35, [g]))
    print(f"pulses={pulses} members={members} gen={g} rep_state={rs} rep_expr={re}")


if __name__ == "__main__":
    main()
