"""A6 Cell 1 — Clock-1 cohort at battery grain (session-8 Part C; WAVE2-A6 cell 1).

Replays the FIXED 160-probe base-gen manifest (probe160_manifest.json — reps {0,1} of
the Qwen stage-0 corpus, content-controlled) through each adapter-merged student /
control checkpoint at the dense steps {1,2,3,5,8,13,21,34,55,75}. Controls are
CHECKPOINT-MATCHED (control-student at step t, never the base — shared numbers-training
drift cancels, per echo-sandbox trajectory.py). One 1-GPU parallel_replay job per
checkpoint (adapter merge-before-hooks); Heimdall schedules 8 concurrent. ~138s/ckpt.

Tiered: --tier core = 3 animals (dense seed) + 5 controls × 10 steps = 80 ckpts (~23min,
the frozen-budget primary cohort). --tier seeds adds animal seeds {t0,t2,t3,t4} for the
seed-parity split-half reliability upgrade. First-read → outer loop. Run: HEIMDALL_* env.
"""
import argparse
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
QPATH = ("/models/subliminal-anamnesis/.hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/"
         "snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
CKPT_ROOT = "/models/subliminal-anamnesis/checkpoints"
CALIB = "/models/anamnesis-extract/calibration/qwen25_7b"
MANIFEST = "/models/anamnesis-extract/battery/arms/A6/cohort/probe160_manifest.json"
OUT_ROOT = "/models/anamnesis-extract/runs/vmb_a6cohort_qwen"
STEPS = ["0001", "0002", "0003", "0005", "0008", "0013", "0021", "0034", "0055", "0075"]
ANIMALS = ["cat", "penguin", "phoenix"]
CONTROLS = ["a", "b", "c", "d", "e"]
ANIMAL_SEEDS = ["dense", "dense_t0", "dense_t2", "dense_t3", "dense_t4"]


def submit(name, command, gpus, minutes):
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": ENV,
            "command": f"bash -c '{BASE} && {command}'"}
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["job"]["id"]


def models_for_tier(tier: str) -> list[tuple[str, str]]:
    """(model_label, checkpoint_dir_basename)."""
    out = []
    if tier in ("core", "all"):
        for a in ANIMALS:
            out.append((f"{a}_dense", f"qwen_{a}_student_dense"))
        for c in CONTROLS:
            out.append((f"control_{c}_dense", f"qwen_control_{c}_dense"))
    if tier in ("seeds", "all"):
        for a in ANIMALS:
            for s in ANIMAL_SEEDS[1:]:  # t0,t2,t3,t4 (dense already in core)
                out.append((f"{a}_{s}", f"qwen_{a}_student_{s}"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["core", "seeds", "all"], default="core")
    ap.add_argument("--workers-per-gpu", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    models = models_for_tier(args.tier)
    jobs = []
    for label, ckpt_base in models:
        for step in STEPS:
            ckpt = f"{CKPT_ROOT}/{ckpt_base}/checkpoint-{step}"
            out = f"{OUT_ROOT}/{label}/step-{step}"
            cmd = (f"python -u -m anamnesis.scripts.parallel_replay --model qwen-7b "
                   f"--model-path {QPATH} --calib-dir {CALIB} --run-dir {out} "
                   f"--manifest {MANIFEST} --adapter-path {ckpt} "
                   f"--gpus 0 --workers-per-gpu {args.workers_per_gpu} --no-raw")
            jobs.append((f"a6-{label}-{step}", cmd))

    print(f"tier={args.tier}: {len(models)} models × {len(STEPS)} steps = {len(jobs)} checkpoints")
    if args.dry_run:
        for n, _ in jobs[:3]:
            print("  e.g.", n)
        return
    ids = [submit(n, c, gpus=1, minutes=8) for n, c in jobs]
    print(f"submitted {len(ids)} jobs. first={ids[0]} last={ids[-1]}")
    print("output:", OUT_ROOT)


if __name__ == "__main__":
    main()
