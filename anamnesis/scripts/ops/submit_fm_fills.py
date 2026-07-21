"""FM-4/5/6 fills (ADDENDUM 20aa) — the cheap half-built completions.

  FM-4  DSV2-dir0 RAW-CAA @L22 (2-D boundary cell, P=.25): gen ±V3raw_L22 + R1 + baseline from the
        banked dir0 whiten bank; readout = the WH-3 MARKER q-rate + content-TFIDF (CPU, at first-read,
        NO API judge — mirrors WH-3's dir0 readout exactly). Pre-worded: clears ⇒ cos-alone predicts
        (2-D account weakens) · null ⇒ Mahalanobis matters (2-D boundary evidenced).
  FM-5  Gemma plain-V1 steering (P=.65): gen ±V1_L36 + R1 + baseline; judge = formality (V1 = the
        formality-CAA). Native temp-1.0 ruler caveat.
  FM-6  Gemma V3sel-bare (P=.55, MANDATORY sign-anchor): v3sel_select --corpus bare (dir0-pair
        socratic,contrastive per the script's ruled gemma axis) -> build_v3sel vector @L36 -> gen
        ±V3sel_L36 a.3 + baseline -> judge socratic. Closes the label-free 4th family; the sign-anchor
        is the orientation-asterisk answer.
  FM-3  8B ±0.3 leak band = FOLDED: the 8B field already genned every member + V7-ladder at ±0.3
        (a5_field_8b), so the ±0.3 leak band is already banked — no new gen needed; the v2.1
        materiality-floor reads it from the existing cells (noted, not re-fired).

Judges carry the API key via env (interpolated at submit; never committed), --max-retries 0.
STAGING: `--fill {fm4,fm5,fm6} [--fire]`; default dry-run. C§8 UNSTAMPED -> desk.
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
PROMPTS = "pipeline/anamnesis/prompts/prompt_sets.json"
QPATH = ("/models/subliminal-anamnesis/.hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/"
         "snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
GEMMA_ENV = {"HF_HOME": "/models/anamnesis-extract/.hf-cache", "HF_HUB_OFFLINE": "1",
             "OMP_NUM_THREADS": "1"}
DSV2_ENV = {"HF_HOME": "/models/anamnesis-extract/.hf-cache", "HF_HUB_OFFLINE": "1",
            "OMP_NUM_THREADS": "1"}
BASE = base("HF_HUB_OFFLINE=1")


def submit(name, cmd, gpus, minutes, env, deps=None, max_retries=1):
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": dict(env),
            "max_retries": max_retries, "command": f"bash -c '{BASE} && {cmd}'"}
    if deps:
        spec["depends_on"] = list(deps)
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        j = json.loads(r.read())
    if "job" not in j:
        raise RuntimeError(f"{name}: {j}")
    return j["job"]["id"]


def b64(payload: dict, path: str) -> str:
    blob = base64.b64encode(json.dumps(payload).encode()).decode()
    return f"echo {blob} | base64 -d > {path}"


def gen_multicell(model, mpath, cells, run_dir, ns, bank, wpg, gen_extra=""):
    payload = {"cells": [
        {"out_run_dir": f"{run_dir}/{n}", "seed_namespace": f"{ns}-{n}",
         "inject_key": ik, "inject_layer": L, **({"inject_alpha_frac": f} if f is not None else {})}
        for n, ik, L, f in cells]}
    cj = f"/tmp/fm_{ns}_cells.json"
    return (f"{b64(payload, cj)} && mkdir -p {run_dir} && "
            f"python -u -m anamnesis.scripts.vmb_a5_gen_multicell --model {model} --model-path {mpath} "
            f"--prompts {PROMPTS} --cells-json {cj} --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu {wpg} "
            f"--seeds-per-class 2 --attn eager --inject-npz {bank}/a5_vectors.npz "
            f"--inject-norms-json {bank}/a5_vectors_stamps.json {gen_extra} "
            f"2>&1 | tee {run_dir}/gen.log")


# ── FM-4: DSV2 raw dir0 @L22 ──
def fm4(fire, key):
    model, mpath = "dsv2-lite", "deepseek-ai/DeepSeek-V2-Lite-Chat"
    bank = f"{BANK}/a5_vectors_dsv2_lite_v3whiten_dir0"   # has V3raw_L22 + V3w_L22 + R1/2/3
    run = f"{RUNS}/vmb_a5_fm4_dsv2_dir0raw"
    cells = [# cell dir NAME must match vmb_pm6a_marker CELL regex (V3|V5|V1|R*); inject_key = V3raw bank key
             ("V3_L22_p01", "V3raw_L22", 22, 0.1), ("V3_L22_p03", "V3raw_L22", 22, 0.3),
             ("V3_L22_m01", "V3raw_L22", 22, -0.1), ("V3_L22_m03", "V3raw_L22", 22, -0.3),
             ("R1_L22_p03", "R1", 22, 0.3), ("baseline", None, 22, None)]
    g = gen_multicell(model, mpath, cells, run, "FM4", bank, wpg=2)
    print(f"[FM-4] DSV2 raw dir0 @L22 gen (6 cells) -> {run}")
    print(f"  readout (CPU, first-read): WH-3 marker q-rate + content-TFIDF on V3raw_L22 cells (NO API judge)")
    if not fire:
        print(f"  {g[:200]}...")
        return
    jid = submit("fm4-gen-dsv2-dir0raw", g, 8, 180, DSV2_ENV)
    print(f"  fm4-gen -> {jid}")


# ── FM-5: Gemma plain-V1 @L36 + formality judge ──
def fm5(fire, key):
    model, mpath = "gemma3-27b", "google/gemma-3-27b-it"
    bank = f"{BANK}/a5_vectors_gemma3_27b_L36"            # has V1_L36
    run = f"{RUNS}/vmb_a5_fm5_gemma_v1"
    cells = [("V1_L36_p01", "V1_L36", 36, 0.1), ("V1_L36_p03", "V1_L36", 36, 0.3),
             ("V1_L36_m01", "V1_L36", 36, -0.1), ("V1_L36_m03", "V1_L36", 36, -0.3),
             ("R1_L36_p03", "R1", 36, 0.3), ("baseline", None, 36, None)]
    env = dict(GEMMA_ENV)
    from anamnesis.scripts.ops._ops_env import hf_token
    g = gen_multicell(model, mpath, cells, run, "FM5", bank, wpg=2, gen_extra="--temperature 1.0 --top-p 0.95")
    jout = f"{run}/formality_judge"
    j = (f"python -u -m anamnesis.scripts.vmb_a5_judge_formality --a5-root {run} --out-dir {jout} "
         f"--vectors V1 --n-pairs 200 2>&1 | tee {run}/judge_formality.log")
    print(f"[FM-5] Gemma V1 @L36 gen (6 cells, temp1.0) -> {run}; judge=formality V1")
    if not fire:
        print(f"  gen: {g[:160]}...\n  judge: {j[:120]}...")
        return
    env["HF_TOKEN"] = hf_token()
    gj = submit("fm5-gen-gemma-v1", g, 8, 150, env)
    jj = submit("fm5-judge-formality", j, 0, 90, {**GEMMA_ENV, "ANTHROPIC_API_KEY": key}, [gj], max_retries=0)
    print(f"  fm5-gen -> {gj}  fm5-judge -> {jj} (after gen)")


# ── FM-6: Gemma V3sel-bare (select -> build -> gen -> judge; sign-anchor) ──
def fm6(fire, key):
    model, mpath = "gemma3-27b", "google/gemma-3-27b-it"
    sel_dir = f"{BANK}/arms/A5"
    sel_json = f"{sel_dir}/v3sel_bare_selection_gemma3-27b.json"
    vec_dir = f"{BANK}/a5_vectors_gemma3-27b_v3selbare"
    run = f"{RUNS}/vmb_a5_fm6_gemma_v3selbare"
    from anamnesis.scripts.ops._ops_env import hf_token
    # stage 1 (CPU): label-free within-topic decile select on the BARE gemma corpus (ruled axis
    # socratic,contrastive per vmb_v3sel_select's gemma note; data-root = runs where gemma sigs live)
    select = (f"python -u -m anamnesis.scripts.vmb_v3sel_select --model gemma3-27b --corpus bare "
              f"--dir0-pair socratic,contrastive --data-root {RUNS} --out-dir {sel_dir} "
              f"2>&1 | tee {sel_dir}/v3sel_bare_gemma.log")
    # stage 2 (GPU): build the V3sel-bare vector = mean-diff residuals over the selected poles @L36
    build = (f"python -u -m anamnesis.scripts.vmb_a5_build_v3sel --model gemma3-27b --model-path {mpath} "
             f"--selection {sel_json} --stage0-run {RUNS}/vmb_stage0_gemma3_27b --sites 36 "
             f"--out-dir {vec_dir} 2>&1 | tee {vec_dir}/build.log")
    # stage 3 (GPU): gen ±V3sel a.3 + baseline. inject_key = the ACTUAL build_v3sel bank key
    # 'V3sel_bare_L36'; cell dir NAME = V3_L36_* so judge_socratic's [VR]\d regex matches. Sign both
    # directions -> the anchor reads at judge.
    cells = [("V3_L36_p03", "V3sel_bare_L36", 36, 0.3), ("V3_L36_m03", "V3sel_bare_L36", 36, -0.3),
             ("baseline", None, 36, None)]
    g = gen_multicell(model, mpath, cells, run, "FM6", vec_dir, wpg=2, gen_extra="--temperature 1.0 --top-p 0.95")
    # stage 4 (API): judge socratic (the map axis) + the sign-anchor is read from which pole scores mode-ward
    jout = f"{run}/socratic_judge"
    j = (f"python -u -m anamnesis.scripts.vmb_a5_judge_socratic --a5-root {run} --out-dir {jout} "
         f"--vectors V3 --mode socratic --baseline-cell baseline --n-pairs 200 "
         f"2>&1 | tee {run}/judge_socratic.log")
    print(f"[FM-6] Gemma V3sel-bare: select(CPU) -> build@L36 -> gen ±.3 -> judge socratic (sign-anchor)")
    if not fire:
        print(f"  select: {select[:140]}...\n  build: {build[:120]}...\n  gen: {g[:120]}...\n  judge: {j[:120]}...")
        return
    env = dict(GEMMA_ENV); env["HF_TOKEN"] = hf_token()
    if os.environ.get("FM6_FROM_GEN"):   # select+build banked -> re-gen+judge only (key-fix re-fire)
        gj = submit("fm6-gen-v3selbare", g, 8, 100, env)
        jj = submit("fm6-judge-socratic", j, 0, 90, {**GEMMA_ENV, "ANTHROPIC_API_KEY": key}, [gj], max_retries=0)
        print(f"  [from-gen] gen->{gj} judge->{jj}")
        return
    sj = submit("fm6-v3sel-select", select, 1, 30, env)
    bj = submit("fm6-v3sel-build", build, 4, 90, env, [sj])
    gj = submit("fm6-gen-v3selbare", g, 8, 100, env, [bj])
    jj = submit("fm6-judge-socratic", j, 0, 90, {**GEMMA_ENV, "ANTHROPIC_API_KEY": key}, [gj], max_retries=0)
    print(f"  select->{sj} build->{bj} gen->{gj} judge->{jj}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", required=True, choices=["fm4", "fm5", "fm6"])
    ap.add_argument("--fire", action="store_true")
    args = ap.parse_args()
    key = os.environ.get("ANTHROPIC_API_KEY")
    if args.fill in ("fm5", "fm6") and args.fire and not key:
        raise SystemExit("ANTHROPIC_API_KEY required for the judge leg")
    {"fm4": fm4, "fm5": fm5, "fm6": fm6}[args.fill](args.fire, key)


if __name__ == "__main__":
    main()
