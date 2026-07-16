"""14k formula leg — ∇-panel differentiability confirm for Ksoclin (session-9 Part D).

P3''s first discriminating test: is the socratic<->linear coordinate (Ksoclin, a geometric
V3 twin) seen by ANY functional gradient (S_mass/S_logit/S_gate/S_entropy) above the R-null
band? Frozen prediction: FORMULA-INERT (P=.70) — the route-inversion doctrine's first
discriminating test. If INERT, the formula route cannot write soclin (no gradient candidate
to build/steer); if FORMULA-VISIBLE, the V4-recipe build+steer follow-up is owed. 3B, 1 GPU,
~5 min. REPORT arithmetic; the call is the outer loop's. Run: HEIMDALL_* env.
"""
import json
import urllib.request

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
M3B = "/models/llama-3.2-3b-instruct"
STAGE0 = "/models/anamnesis-extract/runs/vmb_stage0_3b"
VEC = "/models/anamnesis-extract/battery/a5_vectors_3b/a5_vectors.npz"
KSOCLIN = "/models/anamnesis-extract/battery/a5_vectors_3b_14k/a5_vectors.npz"
OUT = "/models/anamnesis-extract/battery/arms/A5/14k_formula_leg"


def main() -> None:
    cmd = (f"python -u -m anamnesis.scripts.vmb_v4_grad_panel --model 3b --model-path {M3B} "
           f"--stage0-run {STAGE0} --vectors {VEC} --candidate-npz {KSOCLIN} "
           f"--candidate-keys Ksoclin_L14 --out-dir {OUT} --n-gens 20")
    spec = {"job_type": "custom", "name": "14k_formula_leg", "gpus": 1, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": 15, "env": ENV,
            "command": f"bash -c '{BASE} && {cmd}'"}
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        jid = json.loads(r.read())["job"]["id"]
    print(f"14k_formula_leg -> {jid}")
    print(f"output: {OUT}/v4_grad_panel_3b.json (candidate_formula_leg.Ksoclin_L14.verdict)")


if __name__ == "__main__":
    main()
