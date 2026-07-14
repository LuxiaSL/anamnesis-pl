"""Submit the M5 Gemma-3-27B Stage-0 chain to Heimdall with dependency gating.

Onboarding audit CLEARED 2026-07-13 (load-validate smoke PASS): eager load via
the Gemma3ForConditionalGeneration wrapper, decoder_layers()->62 through the
language_model nesting, attention weights returned (32 q-heads), k_proj hooks
fire. Native decode: temperature 1.0 (preset default) + top-p 0.95 (Stage-0 gen
passes --override-top-p 0.95; pipeline default is 0.9).

27B memory sizing: each worker loads a full ~55GB bf16 copy (+vision tower);
2 workers/GPU keeps a B200 (183GB) comfortable under eager attention capture.

calib (1 GPU) ∥ gen (7 GPUs) → replay (8, dep gen+calib) → faith (4, dep replay).
Law is run LOCALLY after sync (inspect the n_min table + whole-vector n_min — the
base-model lesson: never assume ~5).
"""
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base, hf_token

HFTOK = hf_token()
BASE = base()
MODEL = "gemma3-27b"
MPATH = "google/gemma-3-27b-it"
OUT = "/models/anamnesis-extract/runs/vmb_stage0_gemma3_27b"
CALIB = "/models/anamnesis-extract/calibration/gemma3_27b"
ENV = {"HF_HOME": "/models/anamnesis-extract/.hf-cache",
       "HF_HUB_OFFLINE": "1", "HF_TOKEN": HFTOK, "OMP_NUM_THREADS": "1"}


def submit(name: str, command: str, gpus: int, minutes: int,
           depends_on: list[str] | None = None) -> str:
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": ENV,
            "command": f"bash -c '{BASE} && {command}'"}
    if depends_on:
        spec["depends_on"] = depends_on
    req = urllib.request.Request(
        API, data=json.dumps({"spec": spec}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        j = json.loads(r.read())
    if "job" not in j:
        raise RuntimeError(f"{name}: {j}")
    return j["job"]["id"]


calib = submit(
    "vmb-m5-calibration",
    f"python -m anamnesis.scripts.run_8b_calibration --model {MODEL} "
    f"--model-path {MPATH} --out-dir {CALIB}",
    gpus=1, minutes=70)
print(f"calib: {calib}")

gen = submit(
    "vmb-m5-stage0-gen",
    f"python -m anamnesis.scripts.vmb_stage0_generate --model {MODEL} "
    f"--model-path {MPATH} --prompts pipeline/anamnesis/prompts/prompt_sets.json "
    f"--out-run-dir {OUT} --gpus 0,1,2,3,4,5,6 --workers-per-gpu 2 "
    f"--override-top-p 0.95",
    gpus=7, minutes=40)
print(f"gen: {gen}")

replay = submit(
    "vmb-m5-stage0-replay",
    f"python -u -m anamnesis.scripts.parallel_replay --model {MODEL} "
    f"--model-path {MPATH} --run-dir {OUT} --calib-dir {CALIB} "
    f"--manifest {OUT}/replay_manifest.json --gpus 0,1,2,3,4,5,6,7 "
    f"--workers-per-gpu 2 --no-raw",
    gpus=8, minutes=45, depends_on=[gen, calib])
print(f"replay: {replay}")

faith = submit(
    "vmb-m5-stage0-faithfulness",
    f"python -u -m anamnesis.scripts.vmb_stage0_faithfulness --model {MODEL} "
    f"--model-path {MPATH} --floor-run-dir {OUT} --calib-dir {CALIB} "
    f"--out-dir {OUT}/faithfulness --pinned-gpu 0 --spread-gpus 1,2,3",
    gpus=4, minutes=30, depends_on=[replay])
print(f"faith: {faith}")

print("\nchain submitted. Law runs LOCALLY after sync:")
print(f"  rsync node1:{OUT}/{{signatures_v3,metadata.json,replay_manifest.json,faithfulness}} "
      f"-> outputs/battery/vmb_stage0_gemma3_27b/")
print(f"  python -m anamnesis.scripts.vmb_stage0_law --model {MODEL} --n-layers 62 \\")
print(f"    --floor-sig-dir <local>/signatures_v3 --floor-metadata <local>/metadata.json \\")
print(f"    --faith-sig-dir <local>/faithfulness/signatures_v3 "
      f"--faith-index <local>/faithfulness/replay_index.json --out-dir <local>/floors")
