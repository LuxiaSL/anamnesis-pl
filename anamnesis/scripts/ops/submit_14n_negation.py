"""14n — the V7 NEGATION block gen (session-8 Part D; sign-flipped injection).

V7⁻ / V_temp⁻ / Rband⁻ at |α|∈{.03,.1}, L14 — NEGATIVE inject_alpha_frac = "make the model
more confident" (Luxia's question). k=8 same-prompt resamples per group (shared seed
namespace VMB14N-3B → matched noise across vec vs null), so the diversity-drop + brittle
guard (within-sample distinct-4 / trigram_rep) readouts have their fans. Cell names carry
|α| (positive) so the analyzers parse; the negation lives in the injection manifest
(inject_alpha_frac = -|α|). Two banks: b7 (V7⁻ + Rband⁻), c3 (V_temp⁻).

Readouts (fired after, reusing existing scripts on the neg run dirs): (b′) entropy DROP
`vmb_c3_entropy_replay` · temp-equiv R on negated `vmb_c3_temp_equiv_replay` · (f′)
diversity/brittle `vmb_c3_resample_diversity` · hedging index (reranking/marker). Filed P:
entropy drops V7⁻-specific .75 / V_temp⁻ .70; brittle regime by |α|=.1 .40. WATCH ITEM
(14n): hedging = ONE dictionary. First-read → outer loop. Run: HEIMDALL_* env.
"""
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
MODEL, MPATH = "3b", "/models/llama-3.2-3b-instruct"
RUNS = "/models/anamnesis-extract/runs"
PROMPTS = "pipeline/anamnesis/prompts/prompt_sets.json"
B7 = "/models/anamnesis-extract/battery/a5_vectors_3b_b7"
C3 = "/models/anamnesis-extract/battery/a5_vectors_3b_c3"
TAG = "VMB14N-3B"
SITE, DOSES = 14, [0.03, 0.1]


def submit(name, command, gpus, minutes):
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": ENV,
            "command": f"bash -c '{BASE} && {command}'"}
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["job"]["id"]


def gen_cmd(run_dir, cell, npz, key, frac):
    # NEGATION: inject_alpha_frac = -frac (use '=' so argparse doesn't read it as a flag)
    inj = (f"--inject-npz {npz}/a5_vectors.npz --inject-key {key} --inject-layer {SITE} "
           f"--inject-alpha-frac=-{frac} --inject-norms-json {npz}/a5_vectors_stamps.json"
           if key else "")
    return (f"python -u -m anamnesis.scripts.vmb_stage0_generate --model {MODEL} --model-path {MPATH} "
            f"--prompts {PROMPTS} --out-run-dir {RUNS}/{run_dir}/{cell} --gpus 0,1,2,3,4,5,6,7 "
            f"--workers-per-gpu 4 --seeds-per-class 8 --limit 160 --max-new-tokens 256 "
            f"--seed-namespace {TAG} {inj}")


cmds = []
# b7 batch: V7 + Rband1-3 (negated)
for vec in ("V7", "Rband1", "Rband2", "Rband3"):
    for f in DOSES:
        cmds.append(gen_cmd("vmb_b7neg_3b", f"{vec}_L{SITE}_a{f}", B7, f"{vec}_L{SITE}", f))
# c3 batch: V_temp (negated)
for f in DOSES:
    cmds.append(gen_cmd("vmb_c3neg_3b", f"Vtemp_L{SITE}_a{f}", C3, f"Vtemp_L{SITE}", f))
# shared baseline (no injection) for the diversity/brittle floor
cmds.append(gen_cmd("vmb_b7neg_3b", "baseline_a0", B7, None, 0.0))

jid = submit("vmb-14n-negation-gen", " && ".join(cmds), gpus=8, minutes=90)
print(f"14n negation gen job: {jid} ({len(cmds)} cells: V7⁻/Rband⁻ in vmb_b7neg_3b, V_temp⁻ in vmb_c3neg_3b)")
