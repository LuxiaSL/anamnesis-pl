"""A6 §2b SURPASS legs — build Vdiverge for an animal, then steer it into PRISTINE base-Qwen.

Charter blocks 1 (WOLF) + 2 (CS-G PHOENIX). Two chained node1 jobs per animal:
  build (vmb_a6_2b_build: student-adapter vs base+animal-sys teacher -> Valign/Vdiverge @sites)
  -> probe (vmb_a6_2b_probe: steer pristine base-Qwen along Vdiverge + AR nulls; de-dicto/de-se
     ladder + coherence gate + placebo). Vectors land in a6_2b_vectors_qwen_{animal}; results in
     arms/A6/{animal}_surpass/.

Per-animal texture (the readout channel differs — this is the whole point of the wolf leg):
  * WOLF   = NAME-CAPTURE. Literal wolf = 3.3% (the DECOY false-negative); "Qwen"-substitution =
             46.6% (qwen_wolf_student_final behavioral, null 0%). --substitution-regex qwen ON;
             the load-bearing readout is the substitution column (qwen_any primary). Blind readers
             on the top doses (mandated). Frozen ruler: WOLF-SURPASS-FROZEN-RULER-2026-07-20.md.
  * PHOENIX = LITERAL. Phoenix behavioral target_rate ~44% is literal phoenix-word; de-dicto/de-se
             phoenix channels carry it (no substitution column). Frozen ruler (catch) =
             CS-G-PHOENIX-CATCH-FROZEN-RULER-2026-07-20.md; surpass P=.60 (ADDENDUM 20aa CS-G).

STAGING: `--dry-run` (default) prints the two-job DAG; `--animal {wolf,phoenix} --fire` submits. C§8.
"""
from __future__ import annotations

import argparse
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BANK = "/models/anamnesis-extract/battery"
QPATH = ("/models/subliminal-anamnesis/.hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/"
         "snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
SUBLIM = "/models/subliminal-anamnesis"
ENV = {"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1"}
BASE = base("HF_HUB_OFFLINE=1")

# certified adapters (final checkpoints; the behavioral evals of record are these — wolf:
# qwen_wolf_student_final = 46.6% qwen / 3.3% wolf; phoenix: qwen_phoenix_student behavioral).
ANIMALS = {
    "wolf": dict(adapter=f"{SUBLIM}/checkpoints/qwen_wolf_student/final",
                 substitution="qwen"),   # name-capture channel ON
    "phoenix": dict(adapter=f"{SUBLIM}/checkpoints/qwen_phoenix_student/final",
                    substitution=None),   # literal texture; de-dicto/de-se phoenix carries it
}
SITES = [7, 14, 18, 21]
INJECT_SITE = 18
ALPHAS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.8]   # extended ladder (charter); blind readers on 0.6/0.8
N_SAMPLES = 12


def submit(spec: dict) -> str:
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["job"]["id"]


def spec(name, cmd, minutes, deps=None):
    s = {"job_type": "custom", "name": name, "gpus": 1, "node": "node1",
         "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": dict(ENV),
         "command": f"bash -c '{BASE} && {cmd}'"}
    if deps:
        s["depends_on"] = deps
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", required=True, choices=list(ANIMALS))
    ap.add_argument("--fire", action="store_true")
    ap.add_argument("--probe-only", action="store_true", help="vectors built -> fire probe only")
    args = ap.parse_args()
    a = args.animal
    cfg = ANIMALS[a]
    vec_dir = f"{BANK}/a6_2b_vectors_qwen_{a}"
    out_dir = f"{BANK}/arms/A6/{a}_surpass"

    build_cmd = (
        f"python -u -m anamnesis.scripts.vmb_a6_2b_build --model qwen-7b --model-path {QPATH} "
        f"--adapter-path {cfg['adapter']} --animal {a} --sites {' '.join(map(str, SITES))} "
        f"--inject-site {INJECT_SITE} --out-npz {vec_dir}/vectors.npz --out-json {vec_dir}/construction.json")
    sub_flag = f"--substitution-regex {cfg['substitution']} " if cfg["substitution"] else ""
    probe_cmd = (
        f"python -u -m anamnesis.scripts.vmb_a6_2b_probe --model qwen-7b --model-path {QPATH} "
        f"--vec-npz {vec_dir}/vectors.npz "
        f"--ar-npz {BANK}/a6_animal_vectors_qwen/a5_vectors.npz "
        f"--stamps {BANK}/a6_animal_vectors_qwen/a5_vectors_stamps.json "
        f"--site {INJECT_SITE} --alphas {' '.join(map(str, ALPHAS))} --animal {a} {sub_flag}"
        f"--n-samples {N_SAMPLES} --out-json {out_dir}/{a}_surpass.json")

    print(f"===== A6 SURPASS: {a.upper()} =====")
    print(f"texture: {'NAME-CAPTURE (Qwen-substitution; literal is decoy)' if cfg['substitution'] else 'LITERAL (de-dicto/de-se)'}")
    print(f"adapter: {cfg['adapter']}")
    print(f"doses: {ALPHAS} (blind readers on 0.6/0.8)  n_samples={N_SAMPLES}")
    print(f"  [build] gpus=1 ~25m\n          {build_cmd}")
    print(f"  [probe] gpus=1 ~50m  deps=[build]\n          {probe_cmd}")

    if not args.fire:
        print("(dry-run: nothing submitted)")
        return
    if args.probe_only:
        p = submit(spec(f"a6-{a}-probe", probe_cmd, 50))
        print(f"probe={p}")
        return
    b = submit(spec(f"a6-{a}-build", build_cmd, 25))
    p = submit(spec(f"a6-{a}-probe", probe_cmd, 50, [b]))
    print(f"build={b} probe={p}")


if __name__ == "__main__":
    main()
