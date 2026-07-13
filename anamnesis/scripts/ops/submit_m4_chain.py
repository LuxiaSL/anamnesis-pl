"""Submit the M4 OLMo-2-7B Stage-0 chain to Heimdall with dependency gating."""
import json
import re
import urllib.request
from pathlib import Path

API = "http://HEIMDALL-HOST-REDACTED:7000/api/v1/jobs"
CRED = Path("/PRIVATE/credentials.md")
HFTOK = re.search(r"hf_[A-Za-z0-9]+", CRED.read_text()).group(0)

BASE = ("source /home/CLUSTER-USER/luxi-files/.venv-shared/bin/activate && "
        "cd /home/CLUSTER-USER/luxi-files/anamnesis-pl && "
        "export PYTHONPATH=$PWD/pipeline PYTHONUNBUFFERED=1")
MPATH = "allenai/OLMo-2-1124-7B"
OUT = "/models/anamnesis-extract/runs/vmb_stage0_olmo2_7b"
CALIB = "/models/anamnesis-extract/calibration/olmo2_7b"
ENV = {"HF_HOME": "/models/anamnesis-extract/.hf-cache",
       "HF_HUB_OFFLINE": "1", "HF_TOKEN": HFTOK}
CALIB_JOB = "7f3a683cc8e9"


def submit(name: str, command: str, gpus: int, minutes: int,
           depends_on: list[str] | None = None) -> str:
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": "/home/CLUSTER-USER/luxi-files/anamnesis-pl",
            "estimated_minutes": minutes, "env": ENV,
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


gen = submit(
    "vmb-m4-stage0-gen",
    f"python -m anamnesis.scripts.vmb_stage0_generate --model olmo2-7b "
    f"--model-path {MPATH} --prompts pipeline/anamnesis/prompts/prompt_sets.json "
    f"--out-run-dir {OUT} --gpus 0,1,2,3 --workers-per-gpu 4",
    gpus=4, minutes=45)
print(f"gen: {gen}")

replay = submit(
    "vmb-m4-stage0-replay",
    f"python -u -m anamnesis.scripts.parallel_replay --model olmo2-7b "
    f"--model-path {MPATH} --run-dir {OUT} --calib-dir {CALIB} "
    f"--manifest {OUT}/replay_manifest.json --gpus 0,1,2,3 --workers-per-gpu 5 --no-raw",
    gpus=4, minutes=40, depends_on=[gen, CALIB_JOB])
print(f"replay: {replay}")

faith = submit(
    "vmb-m4-stage0-faithfulness",
    f"python -u -m anamnesis.scripts.vmb_stage0_faithfulness --model olmo2-7b "
    f"--model-path {MPATH} --floor-run-dir {OUT} --calib-dir {CALIB} "
    f"--out-dir {OUT}/faithfulness --pinned-gpu 0 --spread-gpus 1,2,3",
    gpus=4, minutes=30, depends_on=[replay])
print(f"faith: {faith}")
