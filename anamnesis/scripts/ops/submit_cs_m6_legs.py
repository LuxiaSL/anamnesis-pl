"""Catch-and-surpass M6 legs (C/D/E) — the node1 GPU chain (14y WH-1/2/3 + 14z ext.1 CS-E).

Chain (sequential --after, head chained after the matrix session's last pending job):
  build-dir0 (V3w linear−socratic @L18,22)      -> gen-dir0 (±V3w a{.1,.3} + R1 + baseline)
  -> build-cs (V3w contrastive−socratic @L22)    -> gen-cs (±V3w ±V3raw a{.1,.3} + R1 + baseline)
  -> judge-cs (contrastive 2AFC, qualified pair) [dep gen-cs]
  -> build-lam05 (analogical−contrastive, λ×0.5) -> gen-lam05 (+V3w a.3 + baseline) -> judge-lam05
  -> build-lam2  (λ×2.0)                         -> gen-lam2  (+V3w a.3 + baseline) -> judge-lam2

Leg C's readout (marker q-rate at graded parity + R band + content-TFIDF) is CPU-side and
runs in-session once gen-dir0 lands — no replay legs (text rulers only; WH-1 reads the
banked R8 whiten-run signatures).

Judge legs carry the API key via env (interpolated at submit time, never committed),
--max-retries 0 (non-idempotent spend).

Usage:
  ANTHROPIC_API_KEY=... HEIMDALL_API=... HEIMDALL_WORK_DIR=... HEIMDALL_VENV=... \
      python -m anamnesis.scripts.ops.submit_cs_m6_legs --after <jobid> [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BANK = "/models/anamnesis-extract/battery"
RUNS = "/models/anamnesis-extract/runs"
STAGE0 = f"{RUNS}/vmb_stage0_dsv2_lite"
MODEL = "dsv2-lite"
MPATH = "deepseek-ai/DeepSeek-V2-Lite-Chat"
HF_ENV = "HF_HOME=/models/anamnesis-extract/.hf-cache HF_HUB_OFFLINE=1"
BASE = base(HF_ENV)
PROMPTS = "pipeline/anamnesis/prompts/prompt_sets.json"
ENV = {"HF_HUB_OFFLINE": "1", "HF_HOME": "/models/anamnesis-extract/.hf-cache",
       "OMP_NUM_THREADS": "1"}

BUILDS = {
    # name suffix -> (pos_run, neg_run, sites, shrink_scale, bank_dir)
    "dir0":  ("vmb_a2_dsv2-lite_pure_linear", "vmb_a2_dsv2-lite_pure_socratic", "18,22",
              None, f"{BANK}/a5_vectors_dsv2_lite_v3whiten_dir0"),
    "cs":    ("vmb_a2_dsv2-lite_pure_contrastive", "vmb_a2_dsv2-lite_pure_socratic", "22",
              None, f"{BANK}/a5_vectors_dsv2_lite_v3whiten_cs"),
    "lam05": ("vmb_a2_dsv2-lite_pure_analogical", "vmb_a2_dsv2-lite_pure_contrastive", "22",
              0.5, f"{BANK}/a5_vectors_dsv2_lite_v3whiten_lam05"),
    "lam2":  ("vmb_a2_dsv2-lite_pure_analogical", "vmb_a2_dsv2-lite_pure_contrastive", "22",
              2.0, f"{BANK}/a5_vectors_dsv2_lite_v3whiten_lam2"),
}

GENS = {
    # name -> (bank_key, run_dir, ns_prefix, cells)
    "dir0": ("dir0", f"{RUNS}/vmb_a5_cs_dir0_whiten", "M6CSC", [
        ("V5_L22_p01", "V3w_L22", 22, 0.1), ("V5_L22_p03", "V3w_L22", 22, 0.3),
        ("V5_L22_m01", "V3w_L22", 22, -0.1), ("V5_L22_m03", "V3w_L22", 22, -0.3),
        ("R1_L22_p03", "R1", 22, 0.3), ("baseline", None, 22, None)]),
    "cs": ("cs", f"{RUNS}/vmb_a5_cs_qualpair_whiten", "M6CSE", [
        ("V5_L22_p01", "V3w_L22", 22, 0.1), ("V5_L22_p03", "V3w_L22", 22, 0.3),
        ("V5_L22_m01", "V3w_L22", 22, -0.1), ("V5_L22_m03", "V3w_L22", 22, -0.3),
        ("V3_L22_p01", "V3raw_L22", 22, 0.1), ("V3_L22_p03", "V3raw_L22", 22, 0.3),
        ("V3_L22_m01", "V3raw_L22", 22, -0.1), ("V3_L22_m03", "V3raw_L22", 22, -0.3),
        ("R1_L22_p03", "R1", 22, 0.3), ("baseline", None, 22, None)]),
    "lam05": ("lam05", f"{RUNS}/vmb_a5_cs_lam05", "M6CSL5", [
        ("V5_L22_p03", "V3w_L22", 22, 0.3), ("baseline", None, 22, None)]),
    "lam2": ("lam2", f"{RUNS}/vmb_a5_cs_lam2", "M6CSL2", [
        ("V5_L22_p03", "V3w_L22", 22, 0.3), ("baseline", None, 22, None)]),
}

JUDGES = {
    # name -> (gen_name, mode, vectors)
    "cs":    ("cs", "contrastive", "V5,V3,R1"),
    "lam05": ("lam05", "analogical", "V5"),
    "lam2":  ("lam2", "analogical", "V5"),
}


def submit(name, cmd, gpus, minutes, deps=None, max_retries=1, extra_env=None):
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": minutes,
            "env": {**ENV, **(extra_env or {})}, "max_retries": max_retries,
            "command": f"bash -c '{BASE} && {cmd}'"}
    if deps:
        spec["depends_on"] = list(deps)
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        j = json.loads(r.read())
    if "job" not in j:
        raise RuntimeError(f"{name}: {j}")
    return j["job"]["id"]


def b64_file_cmd(payload: dict, remote_path: str) -> str:
    blob = base64.b64encode(json.dumps(payload).encode()).decode()
    return f"echo {blob} | base64 -d > {remote_path}"


def build_cmd(key: str) -> str:
    pos, neg, sites, scale, out = BUILDS[key]
    c = (f"python -u -m anamnesis.scripts.vmb_a5_whiten_steer_build --model {MODEL} "
         f"--model-path {MPATH} --runs-root {RUNS} --pos-run {pos} --neg-run {neg} "
         f"--stage0-run {STAGE0} --sites {sites} --out-dir {out}")
    if scale is not None:
        c += f" --shrink-scale {scale}"
    return f"mkdir -p {out} && {c} 2>&1 | tee {out}/build.log"


def gen_cmd(key: str) -> str:
    bank_key, run_dir, ns, cells = GENS[key]
    bank = BUILDS[bank_key][4]
    payload = {"cells": [
        {"out_run_dir": f"{run_dir}/{name}", "seed_namespace": f"{ns}-{name}",
         "inject_key": ik, "inject_layer": layer,
         **({"inject_alpha_frac": frac} if frac is not None else {})}
        for name, ik, layer, frac in cells]}
    cj = f"/tmp/cs_{key}_gen_cells.json"
    return (f"{b64_file_cmd(payload, cj)} && mkdir -p {run_dir} && "
            f"python -u -m anamnesis.scripts.vmb_a5_gen_multicell --model {MODEL} "
            f"--model-path {MPATH} --prompts {PROMPTS} --cells-json {cj} "
            f"--gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 2 --seeds-per-class 2 "
            f"--attn eager --inject-npz {bank}/a5_vectors.npz "
            f"--inject-norms-json {bank}/a5_vectors_stamps.json "
            f"2>&1 | tee {run_dir}/genall_submit.log")


def judge_cmd(key: str) -> str:
    gen_name, mode, vectors = JUDGES[key]
    run_dir = GENS[gen_name][1]
    out = f"{run_dir}/{mode}_judge"
    return (f"python -u -m anamnesis.scripts.vmb_a5_judge_socratic --a5-root {run_dir} "
            f"--out-dir {out} --vectors {vectors} --mode {mode} --baseline-cell baseline "
            f"--n-pairs 200 2>&1 | tee {run_dir}/judge_{mode}.log")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--after", required=True, help="matrix session's last pending job id")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key and not args.dry_run:
        raise SystemExit("ANTHROPIC_API_KEY required for judge legs")

    plan = [
        # (name, cmd_fn, gpus, minutes, dep_slot, retries, judge?)
        ("cs-build-dir0", build_cmd("dir0"), 1, 60, args.after, 1, False),
        ("cs-gen-dir0", gen_cmd("dir0"), 8, 180, "cs-build-dir0", 1, False),
        ("cs-build-cs", build_cmd("cs"), 1, 60, "cs-gen-dir0", 1, False),
        ("cs-gen-cs", gen_cmd("cs"), 8, 280, "cs-build-cs", 1, False),
        ("cs-judge-cs", judge_cmd("cs"), 0, 90, "cs-gen-cs", 0, True),
        ("cs-build-lam05", build_cmd("lam05"), 1, 60, "cs-gen-cs", 1, False),
        ("cs-gen-lam05", gen_cmd("lam05"), 8, 80, "cs-build-lam05", 1, False),
        ("cs-judge-lam05", judge_cmd("lam05"), 0, 45, "cs-gen-lam05", 0, True),
        ("cs-build-lam2", build_cmd("lam2"), 1, 60, "cs-gen-lam05", 1, False),
        ("cs-gen-lam2", gen_cmd("lam2"), 8, 80, "cs-build-lam2", 1, False),
        ("cs-judge-lam2", judge_cmd("lam2"), 0, 45, "cs-gen-lam2", 0, True),
    ]

    ids: dict[str, str] = {}
    for name, cmd, gpus, minutes, dep, retries, is_judge in plan:
        dep_id = ids.get(dep, dep)
        if args.dry_run:
            print(f"[dry] {name} (gpus={gpus}, {minutes}m, after={dep_id}, retries={retries})")
            print(f"      {cmd[:180]}...")
            ids[name] = f"<{name}>"
            continue
        extra = {"ANTHROPIC_API_KEY": key} if is_judge else None
        jid = submit(name, cmd, gpus, minutes, deps=[dep_id], max_retries=retries,
                     extra_env=extra)
        ids[name] = jid
        print(f"{name} -> {jid} (after {dep_id})")

    print(json.dumps({k: v for k, v in ids.items()}, indent=1))


if __name__ == "__main__":
    main()
