"""STEERING-MATRIX COMPLETION (ADDENDUM 14x) — the V7-stack chains for Qwen(L21)/Gemma(L35)/8B(L16).

Per model: build pulses (Σ/V4/V7-s1/Vtemp) -> ONE cpu-merge job (V7-s2 + v34 + RA + merged GENBANK)
-> gen_multicell (the V7 dose ladder + RA + Rband + baseline, n=40/cell) -> replay x3 (state / expression
/ entropy). Mirrors the tested `submit_r7_multicell` delivery (cells written locally, rsynced to node1,
paths derived from WORK_DIR — never hardcode host/home). The 8B field-inventory triple (PX-3) + laws
(PX-4/5) ride in `submit_steering_matrix_field8b.py`; 14k(b) in `submit_14kb_8b.py`; Q5 in `submit_q5_8b.py`.

STAGING: `--dry-run` (DEFAULT) writes every cells-json locally + prints the full DAG (specs, gpus, deps,
commands) and submits NOTHING. `--fire --model {qwen,gemma,8b,all}` rsyncs cells + POSTs the chain. Recipe:
diary 2026-07-19 §"V7-stack recipe"; staged plan: outputs/battery/STEERING-MATRIX-STAGED-STATE-2026-07-19.md.

Env: HEIMDALL_{API,WORK_DIR,VENV} (+ HF_TOKEN for gemma) exported; node1 ssh for the rsync. C§8 UNSTAMPED.
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
BAND = "16,256"
LOCAL_CELLS = Path("/tmp/claude-output/steering_matrix_cells")
NODE_CELLS = f"{WORK_DIR}/_steering_matrix_cells"
QPATH = ("/models/subliminal-anamnesis/.hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/"
         "snapshots/a09a35458c702b33eeacc393d103063234e8bc28")

# per-model config. skip_sigma/skip_v4 reuse already-banked 8B assets.
MODELS = {
    "qwen": dict(
        model="qwen-7b", mpath=QPATH, tag="qwen-7b", site=21,
        stage0=f"{RUNS}/vmb_stage0_qwen7b", calib=f"{BANK}/../calibration/qwen25_7b",
        t09=f"{RUNS}/vmb_a1_qwen-7b_t09", t03=f"{RUNS}/vmb_a1_qwen-7b_t03",
        v3bank=f"{BANK}/a5_vectors_qwen_7b", env={"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1"},
        wpg_gen=6, wpg_rep=8, gen_min=40, gpus_gen=8, skip_sigma=False, skip_v4=False),
    "gemma": dict(
        model="gemma3-27b", mpath="google/gemma-3-27b-it", tag="gemma3-27b", site=36,
        stage0=f"{RUNS}/vmb_stage0_gemma3_27b", calib=f"{BANK}/../calibration/gemma3_27b",
        t09=f"{RUNS}/vmb_a1_gemma3-27b_t09", t03=f"{RUNS}/vmb_a1_gemma3-27b_t03",
        # strict-peak L36 (Luxia GO 2026-07-19) is NOT a sampled layer + not banked -> build V3@L36 first.
        v3bank=f"{BANK}/a5_vectors_gemma3_27b_L36", v3build=True,
        a2_root=RUNS, dir0_pair="analogical,contrastive",
        env={"HF_HOME": "/models/anamnesis-extract/.hf-cache", "HF_HUB_OFFLINE": "1",
             "OMP_NUM_THREADS": "1"},  # + HF_TOKEN injected at fire
        wpg_gen=2, wpg_rep=2, gen_min=120, gpus_gen=8, skip_sigma=False, skip_v4=False,
        gen_extra="--temperature 1.0 --top-p 0.95"),  # native regime caveat
    "8b": dict(
        model="8b", mpath="/models/llama-3.1-8b-instruct", tag="8b", site=16,
        stage0=f"{RUNS}/vmb_stage0_8b", calib=f"{BANK}/../calibration/8b",
        t09=f"{RUNS}/vmb_a1_8b_t09", t03=f"{RUNS}/vmb_a1_8b_t03",
        v3bank=f"{BANK}/a5_vectors_8b", env={"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1"},
        wpg_gen=6, wpg_rep=8, gen_min=45, gpus_gen=8, skip_sigma=True, skip_v4=True,
        sigma_path=f"{BANK}/arms/A5/a5_sigma_L16_8b.npz"),
}
DOSES_V7 = [0.03, 0.1, 0.2, 0.3]  # the in-run V7 dose ladder (PX-4)
DOSES_RA = [0.1, 0.3]
DOSES_RB = [0.1]


def bankpaths(cfg: dict) -> dict:
    m, S, tag = cfg["model"], cfg["site"], cfg["tag"]
    sigma = cfg.get("sigma_path", f"{BANK}/a5_vectors_{m}_v7stack/a5_sigma_L{S}_{tag}.npz")
    return dict(
        sigma=sigma,
        v4=cfg["v3bank"] if cfg["skip_v4"] else f"{BANK}/a5_vectors_{m}_v4",
        v7s1_dir=f"{BANK}/a5_vectors_{m}_b7_stage1",
        v7s1_npz=f"{BANK}/a5_vectors_{m}_b7_stage1/v4_b7_entropy_fresh_G_{tag}.npz",
        v7s2=f"{BANK}/a5_vectors_{m}_b7",
        v34=f"{BANK}/a5_vectors_{m}_v34",
        ra=f"{BANK}/annex/a5_vectors_{m}_14r",
        vtemp=f"{BANK}/a5_vectors_{m}_vtemp",
        gen=f"{BANK}/a5_vectors_{m}_v7gen",
        v3npz=f"{cfg['v3bank']}/a5_vectors.npz",
        v3stamps=f"{cfg['v3bank']}/a5_vectors_stamps.json")


def build_jobs(cfg: dict) -> list[dict]:
    """Return the ordered build DAG as spec dicts with symbolic dep-labels."""
    m, mp, S, s0 = cfg["model"], cfg["mpath"], cfg["site"], cfg["stage0"]
    P, v3 = bankpaths(cfg), cfg["v3bank"]
    jobs = []
    # --- optional V3@site build (strict-peak sites not already banked, e.g. Gemma L36) ---
    pre: list[str] = []   # labels every later job must wait on
    if cfg.get("v3build"):
        jobs.append(dict(label=f"v3build-{m}", gpus=1, min=25 if m == "gemma3-27b" else 15, deps=[],
            cmd=f"python -u -m anamnesis.scripts.vmb_a5_build_vectors --model {m} --model-path {mp} "
                f"--stage0-run {s0} --a2-root {cfg['a2_root']} --out-dir {v3} --stage basic "
                f"--sites {S} --dir0-pair {cfg['dir0_pair']}"))
        pre = [f"v3build-{m}"]
    # --- GPU pulses ---
    if not cfg["skip_sigma"]:
        jobs.append(dict(label=f"sigma-{m}", gpus=1, min=25 if m == "gemma3-27b" else 15, deps=list(pre),
            cmd=f"python -u -m anamnesis.scripts.vmb_a5_covariance_screen --model {m} --model-path {mp} "
                f"--stage0-run {s0} --vectors {P['v3npz']} --out-dir {BANK}/a5_vectors_{m}_v7stack "
                f"--sites {S} --save-sigma-site {S} --n-gens 60"))
    if not cfg["skip_v4"]:
        jobs.append(dict(label=f"v4-{m}", gpus=1, min=25 if m == "gemma3-27b" else 15, deps=list(pre),
            cmd=f"python -u -m anamnesis.scripts.vmb_a5_build_v4_gradient --model {m} --model-path {mp} "
                f"--stage0-run {s0} --out-dir {P['v4']} --map-site {S} --n-gens 20"))
    jobs.append(dict(label=f"v7s1-{m}", gpus=1, min=25 if m == "gemma3-27b" else 15,
        deps=pre + ([] if cfg["skip_sigma"] else [f"sigma-{m}"]),
        cmd=f"python -u -m anamnesis.scripts.vmb_v4_b7b4_stage1 --model {m} --model-path {mp} "
            f"--stage0-run {s0} --sigma {P['sigma']} --vectors {P['v3npz']} --out-dir {P['v7s1_dir']} "
            f"--site {S} --k 64 --n-gens 20"))
    jobs.append(dict(label=f"vtemp-{m}", gpus=1, min=25 if m == "gemma3-27b" else 15, deps=list(pre),
        cmd=f"python -u -m anamnesis.scripts.vmb_ctemp_build --model {m} --model-path {mp} "
            f"--hot-run {cfg['t09']} --cold-run {cfg['t03']} --stage0-run {s0} "
            f"--out-dir {P['vtemp']} --sites {S}"))
    # --- CPU merge job: v7-s2 -> v34 -> RA -> merged GENBANK (all numpy; needs sigma,v4,v7s1 on-node) ---
    v4npz = f"{P['v4']}/a5_vectors.npz"
    merge_cmd = (
        f"python -u -m anamnesis.scripts.vmb_b7_stage2_vectors --b7-npz {P['v7s1_npz']} "
        f"--sigma {P['sigma']} --stamps {P['v3stamps']} --out-dir {P['v7s2']} --site {S} && "
        f"python -u -m anamnesis.scripts.vmb_a5_merge_banks --out-dir {P['v34']} "
        f"--source {P['v3npz']}:V3_L{S} --source {v4npz}:V4_L{S} --norms-from {P['v3stamps']} && "
        f"python -u -m anamnesis.scripts.annex_14r_build_ra --vectors {P['v34']}/a5_vectors.npz "
        f"--sigma {P['sigma']} --stamps {P['v3stamps']} --b7-vectors {P['v7s2']}/a5_vectors.npz "
        f"--out-dir {P['ra']} --site {S} --band {BAND} && "
        f"python -u -m anamnesis.scripts.vmb_a5_merge_banks --out-dir {P['gen']} --model {cfg['tag']} "
        f"--source {P['v7s2']}/a5_vectors.npz:V7_L{S},Rband1_L{S},Rband2_L{S},Rband3_L{S} "
        f"--source {P['ra']}/a5_vectors.npz:RA_L{S} --source {P['v3npz']}:V3_L{S} "
        f"--norms-from {P['v3stamps']}")
    cpu_deps = pre + [f"v7s1-{m}"] + ([] if cfg["skip_v4"] else [f"v4-{m}"])
    jobs.append(dict(label=f"cpu-{m}", gpus=1, min=8, deps=cpu_deps, cmd=merge_cmd))
    return jobs


def gen_cells(cfg: dict) -> tuple[list, list]:
    """The V7 dose ladder + RA + Rband + baseline. Returns (gen_cells, rep_cells)."""
    m, S = cfg["model"], cfg["site"]
    run_root = f"{RUNS}/vmb_a5_{m}_v7_L{S}"
    ns = f"VMBPX-{m.upper()}-L{S}"
    gen, rep = [], []

    def add(key, layer, frac):
        lab = f"{key}_a{'n' if frac < 0 else ''}{abs(frac)}"
        run = f"{run_root}/{lab}"
        gen.append({"out_run_dir": run, "seed_namespace": f"{ns}-{lab}",
                    "inject_key": key, "inject_layer": layer, "inject_alpha_frac": frac})
        rep.append({"run_dir": run, "manifest": f"{run}/replay_manifest.json"})

    for f in DOSES_V7:            # V7 ladder, both signs (the law leg)
        add(f"V7_L{S}", S, f); add(f"V7_L{S}", S, -f)
    for f in DOSES_RA:            # RA aim member
        add(f"RA_L{S}", S, f); add(f"RA_L{S}", S, -f)
    for rb in (1, 2, 3):          # Rband nulls
        for f in DOSES_RB:
            add(f"Rband{rb}_L{S}", S, f); add(f"Rband{rb}_L{S}", S, -f)
    # baseline (alpha 0 on V3 -> plain replay auto-handled)
    run = f"{run_root}/V3_L{S}_a0.0"
    gen.append({"out_run_dir": run, "seed_namespace": f"{ns}-baseline",
                "inject_key": f"V3_L{S}", "inject_layer": S, "inject_alpha_frac": 0.0})
    rep.append({"run_dir": run, "manifest": f"{run}/replay_manifest.json"})
    return gen, rep


def submit(spec: dict) -> str:
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        j = json.loads(r.read())
    if "job" not in j:
        raise RuntimeError(j)
    return j["job"]["id"]


def make_spec(cfg: dict, name: str, cmd: str, gpus: int, minutes: int, deps=None) -> dict:
    base_pre = base("HF_HUB_OFFLINE=1")
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": dict(cfg["env"]),
            "command": f"bash -c '{base_pre} && {cmd}'"}
    if deps:
        spec["depends_on"] = deps
    return spec


def run_model(key: str, cfg: dict, fire: bool, gen_only: bool = False) -> None:
    P = bankpaths(cfg)
    m, S = cfg["model"], cfg["site"]
    jobs = build_jobs(cfg)
    g_cells, r_cells = gen_cells(cfg)

    # write cells locally (staged artifact; rsynced at fire)
    LOCAL_CELLS.mkdir(parents=True, exist_ok=True)
    gpath = LOCAL_CELLS / f"gen_{key}.json"
    rpath = LOCAL_CELLS / f"rep_{key}.json"
    gpath.write_text(json.dumps({"cells": g_cells}, indent=1))
    rpath.write_text(json.dumps({"cells": r_cells}, indent=1))

    genbank = f"{P['gen']}/a5_vectors.npz"
    genstamps = f"{P['gen']}/a5_vectors_stamps.json"
    gen_extra = cfg.get("gen_extra", "")
    gen_cmd = (f"python -u -m anamnesis.scripts.vmb_a5_gen_multicell --model {m} --model-path {cfg['mpath']} "
               f"--prompts {PROMPTS} --cells-json {NODE_CELLS}/gen_{key}.json --inject-npz {genbank} "
               f"--inject-norms-json {genstamps} --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu {cfg['wpg_gen']} "
               f"--seeds-per-class 2 --limit 80 {gen_extra}").strip()
    rep_state = (f"python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model {m} --model-path {cfg['mpath']} "
                 f"--calib-dir {cfg['calib']} --cells-json {NODE_CELLS}/rep_{key}.json "
                 f"--gpus 0,1,2,3,4,5,6,7 --workers-per-gpu {cfg['wpg_rep']} --no-raw --inject-from-metadata")
    rep_expr = (f"python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model {m} --model-path {cfg['mpath']} "
                f"--calib-dir {cfg['calib']} --cells-json {NODE_CELLS}/rep_{key}.json "
                f"--gpus 0,1,2,3,4,5,6,7 --workers-per-gpu {cfg['wpg_rep']} --no-raw "
                f"--sig-subdir signatures_v3_noinject")

    print(f"\n===== {key.upper()} ({m} @L{S}) =====")
    print(f"cells: {len(g_cells)} gen / {len(r_cells)} rep -> {gpath} , {rpath}")
    for j in jobs:
        print(f"  [build] {j['label']:>14}  gpus={j['gpus']} ~{j['min']}m  deps={j['deps']}")
        print(f"          {j['cmd']}")
    print(f"  [gen]   gen-{key}  gpus={cfg['gpus_gen']} ~{cfg['gen_min']}m  deps=[cpu-{m}]")
    print(f"          {gen_cmd}")
    print(f"  [rep]   rep-state / rep-expr  deps=[gen-{key}]")

    if not fire:
        print("  (dry-run: nothing submitted)")
        return

    # rsync cells to node1
    subprocess.run(["ssh", "node1", f"mkdir -p {NODE_CELLS}"], check=True)
    subprocess.run(["rsync", "-a", str(gpath), str(rpath), f"node1:{NODE_CELLS}/"], check=True)
    if gen_only:
        # RESUBMIT path: builds already done (GENBANK on disk) -> gen has no upstream dep
        gdep = None
        print("  gen-only: skipping build jobs (banks on disk)")
    else:
        ids: dict[str, str] = {}
        for j in jobs:
            deps = [ids[d] for d in j["deps"]]
            ids[j["label"]] = submit(make_spec(cfg, f"vmb-mx-{j['label']}", j["cmd"], j["gpus"], j["min"], deps))
            print(f"  submitted {j['label']} -> {ids[j['label']]}")
        gdep = [ids[f"cpu-{m}"]]
    g = submit(make_spec(cfg, f"vmb-mx-gen-{key}", gen_cmd, cfg["gpus_gen"], cfg["gen_min"], gdep))
    rs = submit(make_spec(cfg, f"vmb-mx-repstate-{key}", rep_state, cfg["gpus_gen"], 40, [g]))
    re = submit(make_spec(cfg, f"vmb-mx-repexpr-{key}", rep_expr, cfg["gpus_gen"], 40, [g]))
    print(f"  gen={g}  rep_state={rs}  rep_expr={re}")
    print("  (entropy replay + local readouts staged separately; see STAGED-STATE §2)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", action="store_true", help="actually submit (default: dry-run)")
    ap.add_argument("--model", default="all", choices=["all", "qwen", "gemma", "8b"])
    ap.add_argument("--gen-only", action="store_true",
                    help="resubmit gen->replay only (builds done, GENBANK on disk) — for cancel/resubmit")
    args = ap.parse_args()
    keys = list(MODELS) if args.model == "all" else [args.model]
    for k in keys:
        run_model(k, MODELS[k], fire=args.fire, gen_only=args.gen_only)
    if not args.fire:
        print("\nDRY-RUN complete. Cells written under", LOCAL_CELLS,
              "\nFire with: python -m anamnesis.scripts.ops.submit_steering_matrix --fire --model <m>")


if __name__ == "__main__":
    main()
