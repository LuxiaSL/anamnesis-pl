"""Submit M3 (Qwen) A1+A2 arm chains and M4 (OLMo) A1 chain via Heimdall.

Phase boundaries = separate jobs with depends_on (the step-gating the bash
chains lacked). GPU counts kept at 3 for Qwen so the M4 Stage-0 chain (4 GPUs)
co-runs; replay worker counts kept moderate (load target: stay under ~136 on
the 128-core box; persistent load monitor armed session-side).
"""
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base, hf_token

HFTOK = hf_token()
BASE = base()
ROOT = "/models/anamnesis-extract/runs"
ENV = {"HF_HOME": "/models/anamnesis-extract/.hf-cache",
       "HF_HUB_OFFLINE": "1", "HF_TOKEN": HFTOK}

QWEN = ("/models/subliminal-anamnesis/.hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/"
        "snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
QCALIB = "/models/anamnesis-extract/calibration/qwen25_7b"
OLMO = "allenai/OLMo-2-1124-7B"
OCALIB = "/models/anamnesis-extract/calibration/olmo2_7b"
M4_FAITH_JOB = "f221f09537f8"

DOSES = {"t03": ("--override-temperature 0.3", ),
         "t09": ("--override-temperature 0.9", ),
         "t12": ("--override-temperature 1.2", ),
         "p07": ("--override-top-p 0.7", ),
         "p10": ("--override-top-p 1.0", )}


def submit(name, command, gpus, minutes, depends_on=None):
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR,
            "estimated_minutes": minutes, "env": ENV,
            "command": f"bash -c '{BASE} && {command}'"}
    if depends_on:
        spec["depends_on"] = depends_on
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        j = json.loads(r.read())
    if "job" not in j:
        raise RuntimeError(f"{name}: {j}")
    return j["job"]["id"]


def a1_gen_cmd(model, mpath, gpus_str, workers):
    parts = []
    for dose, extra in DOSES.items():
        out = f"{ROOT}/vmb_a1_{model}_{dose}"
        ns = f"VMBA1-{model.upper()}-{dose.upper()}"
        parts.append(
            f"python -m anamnesis.scripts.vmb_stage0_generate --model {model} "
            f"--model-path {mpath} --prompts pipeline/anamnesis/prompts/prompt_sets.json "
            f"--out-run-dir {out} --gpus {gpus_str} --workers-per-gpu {workers} "
            f"--seeds-per-class 4 --seed-namespace {ns} {extra[0]}")
    return " && ".join(parts)


def a1_replay_cmd(model, mpath, calib, gpus_str, workers):
    parts = []
    for dose in DOSES:
        out = f"{ROOT}/vmb_a1_{model}_{dose}"
        parts.append(
            f"python -u -m anamnesis.scripts.parallel_replay --model {model} "
            f"--model-path {mpath} --run-dir {out} --calib-dir {calib} "
            f"--manifest {out}/replay_manifest.json --gpus {gpus_str} "
            f"--workers-per-gpu {workers} --no-raw")
    return " && ".join(parts)


A2_CONDS = ["pure_analogical", "pure_contrastive", "pure_dialectical", "pure_linear",
            "pure_socratic", "swap_socratic_to_linear",
            "swap_dialectical_to_contrastive", "swap_analogical_to_linear"]


def a2_replay_cmd(model, mpath, calib, gpus_str, workers):
    parts = []
    for cond in A2_CONDS:
        out = f"{ROOT}/vmb_a2_{model}_{cond}"
        parts.append(
            f"python -u -m anamnesis.scripts.parallel_replay --model {model} "
            f"--model-path {mpath} --run-dir {out} --calib-dir {calib} "
            f"--manifest {out}/replay_manifest.json --gpus {gpus_str} "
            f"--workers-per-gpu {workers} --no-raw")
    return " && ".join(parts)


# ── M3 Qwen: A1 ──
qa1g = submit("vmb-m3-a1-gen", a1_gen_cmd("qwen-7b", QWEN, "0,1,2", 4),
              gpus=3, minutes=90)
qa1r = submit("vmb-m3-a1-replay",
              a1_replay_cmd("qwen-7b", QWEN, QCALIB, "0,1,2", 4),
              gpus=3, minutes=90, depends_on=[qa1g])
print(f"qwen A1: gen={qa1g} replay={qa1r}")

# ── M3 Qwen: A2 (gen after A1 gen so Qwen phases don't fight for the same 3 GPUs) ──
qa2g = submit("vmb-m3-a2-gen",
              f"python -m anamnesis.scripts.vmb_a2_generate --model qwen-7b "
              f"--model-path {QWEN} --out-root {ROOT} --seeds-per-topic 8 "
              f"--gpus 0,1,2 --workers-per-gpu 4",
              gpus=3, minutes=90, depends_on=[qa1g])
qa2r = submit("vmb-m3-a2-replay",
              a2_replay_cmd("qwen-7b", QWEN, QCALIB, "0,1,2", 5),
              gpus=3, minutes=120, depends_on=[qa2g])
qa2p = submit("vmb-m3-a2-prefix",
              f"python -m anamnesis.scripts.vmb_a2_prefix_replay --model qwen-7b "
              f"--model-path {QWEN} --a2-root {ROOT} --calib-dir {QCALIB} "
              f"--gpus 0,1,2 --per-swap 20",
              gpus=3, minutes=40, depends_on=[qa2r])
print(f"qwen A2: gen={qa2g} replay={qa2r} prefix={qa2p}")

# ── M4 OLMo: A1 (only after its Stage-0 faithfulness — floors-before-arms) ──
ma1g = submit("vmb-m4-a1-gen", a1_gen_cmd("olmo2-7b", OLMO, "0,1,2,3", 4),
              gpus=4, minutes=75, depends_on=[M4_FAITH_JOB])
ma1r = submit("vmb-m4-a1-replay",
              a1_replay_cmd("olmo2-7b", OLMO, OCALIB, "0,1,2,3", 5),
              gpus=4, minutes=75, depends_on=[ma1g])
print(f"olmo A1: gen={ma1g} replay={ma1r}")
