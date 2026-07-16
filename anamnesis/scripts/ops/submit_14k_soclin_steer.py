"""14k — Ksoclin STEERING cell (session-9 Part D-7). Scored vs 14k(a) P=.85.

Does the socratic<->linear coordinate (Ksoclin, a mahal-0.999 V3 twin) travel as a data-route
LEVER ≥2×R at α≤.1, with an in-window behavioral (socratic-induction) consequence? Gens 3B
free-gen steered with Ksoclin_L14 + V3_L14 (reference) at α{.03,.1,.3} + an α=0 baseline, then
replays. The banked R1/R2/R3 cells (vmb_a5_3b, same a5 site-norm) are the matched-R null.

⚠ MATCHED NORM: Ksoclin is scaled by the a5 stamps (L14 norm 12.239, the norm the banked R
cells used) — NOT the 14k stamps (5.914 = norm-of-mean over pure corpora, a different quantity)
— so Ksoclin's absolute α matches R and V3. Analysis (dir0-shift lever + socratic markers) =
vmb_14k_soclin_lever.py, CPU, after. First-read → outer loop. Node1.
"""
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
MODEL, MPATH = "3b", "/models/llama-3.2-3b-instruct"
RUNS = "/models/anamnesis-extract/runs"
CALIB = "/models/anamnesis-extract/calibration/3b"
PROMPTS = "pipeline/anamnesis/prompts/prompt_sets.json"
A5VEC = "/models/anamnesis-extract/battery/a5_vectors_3b/a5_vectors.npz"
A5STAMPS = "/models/anamnesis-extract/battery/a5_vectors_3b/a5_vectors_stamps.json"  # matched site-norm
K14VEC = "/models/anamnesis-extract/battery/a5_vectors_3b_14k/a5_vectors.npz"
OUT = f"{RUNS}/vmb_14k_soclin_3b"
LADDER = [0.03, 0.1, 0.3]
SITE = 14
# (cell_key, inject_npz, inject_key)
VECS = [("Ksoclin", K14VEC, "Ksoclin_L14"), ("V3", A5VEC, "V3_L14")]


def submit(name, command, gpus, minutes, depends_on=None):
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": ENV,
            "command": f"bash -c '{BASE} && {command}'"}
    if depends_on:
        spec["depends_on"] = depends_on
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["job"]["id"]


def gen_cmd(cell, npz, key, frac):
    inj = (f"--inject-alpha 0.0" if frac == 0.0 else
           f"--inject-alpha-frac {frac} --inject-norms-json {A5STAMPS}")
    return (f"python -u -m anamnesis.scripts.vmb_stage0_generate --model {MODEL} --model-path {MPATH} "
            f"--prompts {PROMPTS} --out-run-dir {OUT}/{cell} --gpus 0,1,2,3,4,5 --workers-per-gpu 6 "
            f"--seeds-per-class 2 --seed-namespace VMB14KSOC-{cell} "
            f"--inject-npz {npz} --inject-key {key} --inject-layer {SITE} {inj}")


def rep_cmd(cell):
    return (f"python -u -m anamnesis.scripts.parallel_replay --model {MODEL} --model-path {MPATH} "
            f"--run-dir {OUT}/{cell} --calib-dir {CALIB} --manifest {OUT}/{cell}/replay_manifest.json "
            f"--gpus 0,1 --workers-per-gpu 8 --no-raw --inject-from-metadata")


def main() -> None:
    cells = [("baseline_L14_a0.0", K14VEC, "Ksoclin_L14", 0.0)]
    for cname, npz, key in VECS:
        for frac in LADDER:
            cells.append((f"{cname}_L14_a{frac}", npz, key, frac))
    gen = submit("14k_soclin_gen", " && ".join(gen_cmd(c, n, k, f) for c, n, k, f in cells),
                 gpus=6, minutes=90)
    rep = submit("14k_soclin_rep", " && ".join(rep_cmd(c) for c, _, _, _ in cells),
                 gpus=2, minutes=70, depends_on=[gen])
    print(f"14k_soclin: gen={gen} rep={rep}  ({len(cells)} cells -> {OUT})")


if __name__ == "__main__":
    main()
