"""Submit Po2 — OLMo-2-7B A5 temperature revival (loose-ends Part 5, 14p Po2=.65).

Chain (fire-and-forget, depends_on): V_temp BUILD (1 GPU) -> GENALL -> REPALL (multicell,
load-once, SDPA gen). V_temp = mean(resid|t09) - mean(resid|t03) at OLMo sites (Route-1b
metadata contrast; both t03/t09 A1 pools predate all steering). Steer Vtemp_L20 @ {.03,.1,.3};
entropy-certify + C3 orphaned-coordinate mask are CPU post-steps (content rung loud on OLMo via
repetition = pre-named, 3'-columned, NOT a failure). First-read -> outer loop.

    python -m anamnesis.scripts.ops.submit_po2_olmo            # build -> gen -> replay
    python -m anamnesis.scripts.ops.submit_po2_olmo --skip-build   # gen->replay only (vectors exist)
"""
import argparse
import base64
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
PROMPTS = "pipeline/anamnesis/prompts/prompt_sets.json"

OPATH = ("/models/anamnesis-extract/.hf-cache/hub/models--allenai--OLMo-2-1124-7B/"
         "snapshots/7df9a82518afdecae4e8c026b27adccc8c1f0032")
E = "/models/anamnesis-extract"
CFG = {
    "path": OPATH,
    "hot": f"{E}/runs/vmb_a1_olmo2-7b_t09",
    "cold": f"{E}/runs/vmb_a1_olmo2-7b_t03",
    "stage0": f"{E}/runs/vmb_stage0_olmo2_7b",
    "calib": f"{E}/calibration/olmo2_7b",
    "vecdir": f"{E}/battery/a5_vectors_olmo2_7b_ctemp",
    "out": f"{E}/runs/vmb_a5_olmo_vtemp",
}
SITES = "8,16,20,24,28"
CELLS = [{"key": "Vtemp", "layer": 20, "frac": f} for f in (0.03, 0.1, 0.3)]


def submit(name, command, gpus, minutes, depends_on=None, max_retries=1):
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": ENV,
            "max_retries": max_retries, "command": f"bash -c '{BASE} && {command}'"}
    if depends_on:
        spec["depends_on"] = list(depends_on)
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        j = json.loads(r.read())
    if "job" not in j:
        raise RuntimeError(f"{name}: {j}")
    return j["job"]["id"]


def b64(payload, path):
    blob = base64.b64encode(json.dumps(payload).encode()).decode()
    return f"echo {blob} | base64 -d > {path}"


def cell_dir(c):
    return f"{CFG['out']}/{c['key']}_L{c['layer']}_a{c['frac']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    deps = []
    if not args.skip_build:
        build_cmd = (
            f"python -u -m anamnesis.scripts.vmb_ctemp_build --model olmo2-7b "
            f"--model-path {CFG['path']} --hot-run {CFG['hot']} --cold-run {CFG['cold']} "
            f"--stage0-run {CFG['stage0']} --sites {SITES} --out-dir {CFG['vecdir']} "
            f"2>&1 | tee {CFG['vecdir']}/ctemp_build.log")
        bj = submit("po2-olmo-vtemp-build",
                    f"mkdir -p {CFG['vecdir']} && {build_cmd}", gpus=1, minutes=30)
        deps = [bj]
        print(f"po2 BUILD -> {bj}")

    npz = f"{CFG['vecdir']}/a5_vectors.npz"
    stamps = f"{CFG['vecdir']}/a5_vectors_stamps.json"
    gpus_str = ",".join(str(g) for g in range(8))
    gen_cells = {"cells": [
        {"out_run_dir": cell_dir(c),
         "seed_namespace": f"OLMO-VTEMP-{c['key']}_L{c['layer']}_a{c['frac']}",
         "inject_key": c["key"], "inject_layer": int(c["layer"]),
         "inject_alpha_frac": float(c["frac"])} for c in CELLS]}
    rep_cells = {"cells": [
        {"run_dir": cell_dir(c), "manifest": f"{cell_dir(c)}/replay_manifest.json"} for c in CELLS]}
    gj_path, rj_path = "/tmp/olmo_vtemp_gen.json", "/tmp/olmo_vtemp_rep.json"

    gen_cmd = (
        f"{b64(gen_cells, gj_path)} && mkdir -p {CFG['out']} && "
        f"python -u -m anamnesis.scripts.vmb_a5_gen_multicell "
        f"--model olmo2-7b --model-path {CFG['path']} --prompts {PROMPTS} "
        f"--cells-json {gj_path} --gpus {gpus_str} --workers-per-gpu 2 "
        f"--seeds-per-class 2 --max-new-tokens 512 --attn sdpa "
        f"--inject-npz {npz} --inject-norms-json {stamps} "
        f"2>&1 | tee {CFG['out']}/genall_submit.log")
    gj = submit("po2-olmo-vtemp-GENALL", gen_cmd, gpus=8, minutes=180, depends_on=deps)

    rep_cmd = (
        f"{b64(rep_cells, rj_path)} && "
        f"python -u -m anamnesis.scripts.vmb_a5_replay_multicell "
        f"--model olmo2-7b --model-path {CFG['path']} --calib-dir {CFG['calib']} "
        f"--cells-json {rj_path} --gpus {gpus_str} --workers-per-gpu 8 "
        f"--no-raw --inject-from-metadata "
        f"2>&1 | tee {CFG['out']}/repall_submit.log")
    rj = submit("po2-olmo-vtemp-REPALL", rep_cmd, gpus=8, minutes=180, depends_on=[gj])
    print(f"po2 {len(CELLS)} cells | GENALL={gj} -> REPALL={rj}")


if __name__ == "__main__":
    main()
