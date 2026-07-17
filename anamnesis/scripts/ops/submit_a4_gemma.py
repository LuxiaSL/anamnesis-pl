"""Submit ARM A4 (state surgery) for Gemma-3-27B — loose-ends Part 3 (14p Pg1/Pg2).

Gemma interleaves 5 sliding-window (1024) : 1 global attention layers; the battery's
uniform-attention `KVSnapshot` may not survive the mixed-length cache. So A4 here is
SMOKE-GATED: the smoke (`vmb_a4_gemma_smoke`, per GEMMA-A4-SLIDING-SMOKE-SPEC) must return
verdict=PROCEED (all per-layer key lengths == C AND faithfulness within tol) BEFORE the grid.
A STOP-AND-SURFACE verdict is itself the Part-3 result (transplant kv-rotation's per-layer
layer_types/applies_rope KVSnapshot first). The grid does NOT auto-fire on smoke completion —
run it manually only after reading a PROCEED verdict.

    python -m anamnesis.scripts.ops.submit_a4_gemma            # smoke only (default)
    python -m anamnesis.scripts.ops.submit_a4_gemma --grid     # grid — AFTER verdict=PROCEED

Reduced confirmatory grid: naive/rotate/recompute (+full reference) × the gemma stage-0 floor
substrate × n=160. bf16 (gemma dtype law). 27B → 3 workers/GPU (VRAM cap, throughput playbook).
Scored vs Pg1=.70 (P3 dissociation replicates) / Pg2=.60 (geometry lands in the substrate
vocabulary), CONDITIONAL on a clean smoke. First-read → outer loop.
"""
import argparse
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}

GPATH = ("/models/anamnesis-extract/.hf-cache/hub/models--google--gemma-3-27b-it/"
         "snapshots/005ad3404e59d6023443cb575daa05336842228a")
CFG = {"path": GPATH,
       "floor": "/models/anamnesis-extract/runs/vmb_stage0_gemma3_27b",
       "calib": "/models/anamnesis-extract/calibration/gemma3_27b",
       "out": "/models/anamnesis-extract/runs/vmb_a4_gemma",
       "smoke_out": "/models/anamnesis-extract/battery/arms/A4_gemma/smoke_verdict.json"}

GEN_IDS = [k * 10 + s for k in range(80) for s in (0, 1)]  # n=160


def submit(name: str, command: str, gpus: int, minutes: int) -> str:
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": ENV,
            "command": f"bash -c '{BASE} && {command}'"}
    req = urllib.request.Request(
        API, data=json.dumps({"spec": spec}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        j = json.loads(r.read())
    if "job" not in j:
        raise RuntimeError(f"{name}: {j}")
    return j["job"]["id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true",
                    help="submit the confirmatory grid (ONLY after smoke verdict=PROCEED)")
    args = ap.parse_args()

    if not args.grid:
        jid = submit(
            "vmb-a4-gemma-smoke",
            f"python -u -m anamnesis.scripts.vmb_a4_gemma_smoke "
            f"--model gemma3-27b --model-path {CFG['path']} "
            f"--floor-run-dir {CFG['floor']} --gen-id 0 --out {CFG['smoke_out']}",
            gpus=1, minutes=25)
        print(f"gemma-a4 SMOKE -> {jid}")
        print(f"verdict: {CFG['smoke_out']}  (read it BEFORE running --grid; "
              f"STOP-AND-SURFACE = that report is the Part-3 result)")
        return

    common = (f"--model gemma3-27b --model-path {CFG['path']} "
              f"--floor-run-dir {CFG['floor']} --manifest {CFG['floor']}/replay_manifest.json "
              f"--calib-dir {CFG['calib']} --out-run-dir {CFG['out']}")
    full = submit(
        "vmb-a4-gemma-grid",
        f"python -u -m anamnesis.scripts.vmb_a4_surgery_replay {common} "
        f"--launch --gpus 0,1,2,3 --workers-per-gpu 3 "
        f"--gen-ids {' '.join(str(g) for g in GEN_IDS)}",
        gpus=4, minutes=180)
    print(f"gemma-a4 GRID -> {full}  (confirm smoke verdict=PROCEED was read)")


if __name__ == "__main__":
    main()
