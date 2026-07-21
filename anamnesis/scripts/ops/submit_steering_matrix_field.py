"""FM-1/FM-2 — field-inventory triple ported to Qwen (@L21) and Gemma (@L36).

Generic model-parameterized port of `submit_steering_matrix_field8b.py` (which stays as the
frozen 8B record). Same construction: raw functional gradients (margin/eos/repmass pulses) ->
band-pass -> Gram-Schmidt the coordinate content off the temperature projection V7 -> merged
field GENBANK -> gen -> replay(state) + replay(expression). Adds an ENTROPY leg per chain
(fired via submit_entropy_legs.py --chain field_{model}). RIDES ON each model's already-banked
V7 stack (V7_L{S} + Rband*_L{S} in a5_vectors_{m}_b7) and banked Σ@L{S}.

Members: Vrep_perp_L{S} / Veos_perp_L{S} (⊥ V7, ~1e-16 by construction) + Vconf_L{S} (band-passed;
FM tests whether CONFIDENCE COLLAPSES onto V7, |cos|>.60). Every member filing carries cos-to-V7
AND cos-to-Vrep_perp (the rep-leak law rides). Gen = the members ± α{.1,.3} + the in-run V7 dose
ladder (leak-law leg) + Rband + baseline.

TWO CHARTER MODS vs the 8B field8b template (ADDENDUM 20aa FM-1/FM-2):
  1. UNCAPPED STOPPING FROM THE START. The Veos_perp (stopping) cells generate at max_new_tokens
     2048 in a SEPARATE gen job; the rest generate at 512 (comparable to 8B rep/conf). This
     avoids the 512-cap catch (S8-18: a capped baseline can't represent the EOS population, which
     flattened the 8B stopping dial and forced a re-gen). Length lives in metadata; the stopping
     replay reads it.
  2. LEAK PREDICTIONS FILED AT CONSTRUCTION. `--fire` writes a frozen leak-prediction record
     (leak_predictions_{model}.md, pre-registered directions/thresholds) BEFORE any gen result is
     read — the prospective form the desk requires (cf. Session-A leg 2). The members job's
     band_pass then computes the observed cos-to-V7 for the in-run ladder to score against it.

STAGING: `--dry-run` (default) prints the DAG + writes cells + the leak-prediction record;
`--fire` submits. C§8 UNSTAMPED -> desk first-reads.

    python -m anamnesis.scripts.ops.submit_steering_matrix_field --model qwen  [--fire]
    python -m anamnesis.scripts.ops.submit_steering_matrix_field --model gemma [--fire]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base, hf_token

BANK = "/models/anamnesis-extract/battery"
RUNS = "/models/anamnesis-extract/runs"
PROMPTS = "pipeline/anamnesis/prompts/prompt_sets.json"
QPATH = ("/models/subliminal-anamnesis/.hf-cache/hub/models--Qwen--Qwen2.5-7B-Instruct/"
         "snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
NODE_CELLS = f"{WORK_DIR}/_steering_matrix_cells"
LOCAL_CELLS = Path("/tmp/claude-output/steering_matrix_cells")

# per-model config. site = the aligned steering layer (WH-6 alignment column, 2026-07-19).
# sigma / b7 (V7 stack) / v3bank are the banks verified resident 2026-07-20 (staging survey).
MODELS = {
    "qwen": dict(
        model="qwen-7b", mpath=QPATH, tag="qwen-7b", site=21,
        stage0=f"{RUNS}/vmb_stage0_qwen7b", calib=f"{BANK}/../calibration/qwen25_7b",
        v3bank=f"{BANK}/a5_vectors_qwen_7b",
        sigma=f"{BANK}/a5_vectors_qwen-7b_v7stack/a5_sigma_L21_qwen-7b.npz",
        b7=f"{BANK}/a5_vectors_qwen-7b_b7/a5_vectors.npz",
        env={"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1"},
        wpg_gen=6, wpg_rep=8, gen_min=45, gen_extra="",
        # FM-1 frozen P's (ADDENDUM 20aa): rep .70 / stop .65 / conf-collapse .70
        P=dict(rep=0.70, stop=0.65, conf_collapse=0.70)),
    "gemma": dict(
        model="gemma3-27b", mpath="google/gemma-3-27b-it", tag="gemma3-27b", site=36,
        stage0=f"{RUNS}/vmb_stage0_gemma3_27b", calib=f"{BANK}/../calibration/gemma3_27b",
        v3bank=f"{BANK}/a5_vectors_gemma3_27b_L36",
        sigma=f"{BANK}/a5_vectors_gemma3-27b_v7stack/a5_sigma_L36_gemma3-27b.npz",
        b7=f"{BANK}/a5_vectors_gemma3-27b_b7/a5_vectors.npz",
        env={"HF_HOME": "/models/anamnesis-extract/.hf-cache", "HF_HUB_OFFLINE": "1",
             "OMP_NUM_THREADS": "1"},  # + HF_TOKEN injected at fire
        wpg_gen=2, wpg_rep=2, gen_min=120, gen_extra="--temperature 1.0 --top-p 0.95",  # native regime caveat
        # FM-2 frozen P's (ADDENDUM 20aa; temp-1.0 ruler caveat inherited): rep .60 / stop .55 / conf .60
        P=dict(rep=0.60, stop=0.55, conf_collapse=0.60)),
    "8b": dict(  # reproduces the frozen 8B field record (parity check only; field8b.py is the record)
        model="8b", mpath="/models/llama-3.1-8b-instruct", tag="8b", site=16,
        stage0=f"{RUNS}/vmb_stage0_8b", calib=f"{BANK}/../calibration/8b",
        v3bank=f"{BANK}/a5_vectors_8b",
        sigma=f"{BANK}/arms/A5/a5_sigma_L16_8b.npz",
        b7=f"{BANK}/a5_vectors_8b_b7/a5_vectors.npz",
        env={"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1"},
        wpg_gen=6, wpg_rep=8, gen_min=45, gen_extra="",
        P=dict(rep=None, stop=None, conf_collapse=None)),
}


def submit(spec: dict) -> str:
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["job"]["id"]


def spec(name, cmd, gpus, minutes, env, deps=None):
    s = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
         "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": dict(env),
         "command": f"bash -c '{base('HF_HUB_OFFLINE=1')} && {cmd}'"}
    if deps:
        s["depends_on"] = deps
    return s


def build_cmds(cfg: dict) -> tuple[str, str]:
    """Return (pulse-template-fn args, members-cmd) for this model."""
    m, mp, S, s0 = cfg["model"], cfg["mpath"], cfg["site"], cfg["stage0"]
    field = f"{BANK}/a5_field_{cfg['tag']}"
    v3npz = f"{cfg['v3bank']}/a5_vectors.npz"
    v3stamps = f"{cfg['v3bank']}/a5_vectors_stamps.json"
    b7, sigma = cfg["b7"], cfg["sigma"]

    def pulse(fn: str) -> str:
        return (f"python -u -m anamnesis.scripts.annex_potential_gradient --model {m} --model-path {mp} "
                f"--stage0-run {s0} --out-dir {field}/roster_{fn} --functional {fn} --map-site {S} --n-gens 20")

    members = (
        f"python -u -m anamnesis.scripts.vmb_a5_merge_banks --out-dir {field}/roster_all "
        f"--source {field}/roster_margin/roster_gradients.npz:Gmargin_L{S} "
        f"--source {field}/roster_eos/roster_gradients.npz:Geos_L{S} "
        f"--source {field}/roster_repmass/roster_gradients.npz:Grep_L{S} --norms-from {v3stamps} && "
        f"python -u -m anamnesis.scripts.annex_band_pass --gradients {field}/roster_all/a5_vectors.npz "
        f"--keys Gmargin_L{S}:Vconf_L{S} Geos_L{S}:Veos_L{S} Grep_L{S}:Vrep_L{S} "
        f"--sigma {sigma} --stamps {v3stamps} --compare {b7}:V7_L{S} {v3npz}:V3_L{S} "
        f"--out-dir {field}/members && "
        f"python -u -m anamnesis.scripts.annex_perp_vectors --members {field}/members/a5_vectors.npz "
        f"--keys Vrep_L{S}:Vrep_perp_L{S} Veos_L{S}:Veos_perp_L{S} --v7-npz {b7} --v7-key V7_L{S} "
        f"--stamps {v3stamps} --out-dir {field}/perp && "
        f"python -u -m anamnesis.scripts.vmb_a5_merge_banks --out-dir {field}/field_gen --model {m} "
        f"--source {field}/perp/a5_vectors.npz:Vrep_perp_L{S},Veos_perp_L{S} "
        f"--source {field}/members/a5_vectors.npz:Vconf_L{S} "
        f"--source {b7}:V7_L{S},Rband1_L{S},Rband2_L{S},Rband3_L{S} "
        f"--source {v3npz}:V3_L{S} --norms-from {v3stamps}")  # V3 for the alpha0 baseline cell
    return pulse, members


def cells(cfg: dict):
    """Split into MAIN cells (@512) and STOPPING cells (Veos_perp @2048, uncapped from the start)."""
    S = cfg["site"]
    run_root = f"{RUNS}/vmb_a5_{cfg['tag']}_field_L{S}"
    ns = f"VMBFM-{cfg['tag'].upper()}-L{S}"
    gen_main, gen_stop, rep_main, rep_stop = [], [], [], []

    def add(bucket_gen, bucket_rep, key, frac):
        lab = f"{key}_a{'n' if frac < 0 else ''}{abs(frac)}"
        run = f"{run_root}/{lab}"
        bucket_gen.append({"out_run_dir": run, "seed_namespace": f"{ns}-{lab}",
                           "inject_key": key, "inject_layer": S, "inject_alpha_frac": frac})
        bucket_rep.append({"run_dir": run, "manifest": f"{run}/replay_manifest.json"})

    # STOPPING member -> uncapped bucket (charter mod #1)
    for f in (0.1, 0.3):
        add(gen_stop, rep_stop, f"Veos_perp_L{S}", f)
        add(gen_stop, rep_stop, f"Veos_perp_L{S}", -f)
    # rep + conf members -> main bucket @512
    for key in (f"Vrep_perp_L{S}", f"Vconf_L{S}"):
        for f in (0.1, 0.3):
            add(gen_main, rep_main, key, f)
            add(gen_main, rep_main, key, -f)
    # in-run V7 dose ladder (leak-law leg)
    for f in (0.03, 0.1, 0.2, 0.3):
        add(gen_main, rep_main, f"V7_L{S}", f)
        add(gen_main, rep_main, f"V7_L{S}", -f)
    # Rband nulls
    for rb in (1, 2, 3):
        add(gen_main, rep_main, f"Rband{rb}_L{S}", 0.1)
        add(gen_main, rep_main, f"Rband{rb}_L{S}", -0.1)
    # alpha0 baseline
    run = f"{run_root}/V3_L{S}_a0.0"
    gen_main.append({"out_run_dir": run, "seed_namespace": f"{ns}-baseline",
                     "inject_key": f"V3_L{S}", "inject_layer": S, "inject_alpha_frac": 0.0})
    rep_main.append({"run_dir": run, "manifest": f"{run}/replay_manifest.json"})
    return gen_main, gen_stop, rep_main, rep_stop


def leak_prediction_record(cfg: dict) -> str:
    """The frozen, pre-registered leak-law predictions (charter mod #2). Written BEFORE any result."""
    S, P = cfg["site"], cfg["P"]
    return f"""# FM field-triple LEAK PREDICTIONS — {cfg['tag']} @L{S} (FROZEN at construction, pre-gen)

> ADDENDUM 20aa FM-{'1' if cfg['tag']=='qwen-7b' else '2'} prospective form. Filed BEFORE reading any
> gen/replay/entropy result. The members job's `annex_band_pass --compare {{b7}}:V7_L{S}` emits the
> OBSERVED cos-to-V7 for each member; score the observed against these frozen predictions.

## Frozen predictions
- **Vrep_perp_L{S} / Veos_perp_L{S}**: Gram-Schmidt ⊥ V7 by construction => cos-to-V7 ~ 1e-16
  (near-exact orthogonal). PREDICTION: the ⊥ members are ENTROPY-SILENT (the entropy leg's V7-only
  write does not ride them); their expressed dials (repetition / stopping) move independently of V7.
- **Vconf_L{S}**: band-passed, NOT orthogonalized. PREDICTION: CONFIDENCE COLLAPSES ONTO V7 —
  |cos(Vconf, V7_L{S})| > 0.60 (the 8B field found Vconf-onto-V7 entropy 0.744; PX-3 member-identity
  question). If |cos| > .60 => Vconf is a V7 alias, not a distinct member (report in coverage matrix).
- **V7 dose ladder (leak-law leg)**: V7 writes entropy dose-monotone, sign-controlled, >> matched
  Rband null. The rep-leak law rides: each member filing carries cos-to-V7 AND cos-to-Vrep_perp.

## Frozen P's (ADDENDUM 20aa; desk scores)
- rep dial replicates (clean bidirectional repetition, >> null): **P = {P['rep']}**
- stopping dial replicates UNCAPPED (length bidirectional at 2048): **P = {P['stop']}**
- confidence collapses onto V7 (|cos| > .60): **P = {P['conf_collapse']}**

v2.1 materiality-floor reported BESIDE the frozen band (PX-4 convention). UNSTAMPED -> desk.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--fire", action="store_true")
    ap.add_argument("--after", default=None, help="optional upstream job id to gate the pulses on")
    ap.add_argument("--members-only", action="store_true", help="rosters on disk -> members->gen->replay")
    ap.add_argument("--gen-only", action="store_true", help="field_gen bank on disk -> gen->replay only")
    args = ap.parse_args()
    cfg = MODELS[args.model]
    m, S = cfg["model"], cfg["site"]
    tag = cfg["tag"]
    field = f"{BANK}/a5_field_{tag}"

    pulse, members_cmd = build_cmds(cfg)
    gen_main, gen_stop, rep_main, rep_stop = cells(cfg)

    LOCAL_CELLS.mkdir(parents=True, exist_ok=True)
    files = {
        f"gen_{tag}_field_main.json": {"cells": gen_main},
        f"gen_{tag}_field_stop.json": {"cells": gen_stop},
        f"rep_{tag}_field_main.json": {"cells": rep_main},
        f"rep_{tag}_field_stop.json": {"cells": rep_stop},
    }
    for fn, obj in files.items():
        (LOCAL_CELLS / fn).write_text(json.dumps(obj, indent=1))
    leak_path = LOCAL_CELLS / f"leak_predictions_{tag}.md"
    leak_path.write_text(leak_prediction_record(cfg))

    genbank = f"{field}/field_gen/a5_vectors.npz"
    genstamps = f"{field}/field_gen/a5_vectors_stamps.json"
    gpus_all = "0,1,2,3,4,5,6,7"

    def gen_cmd(cells_json: str, max_new: int) -> str:
        return (f"python -u -m anamnesis.scripts.vmb_a5_gen_multicell --model {m} --model-path {cfg['mpath']} "
                f"--prompts {PROMPTS} --cells-json {NODE_CELLS}/{cells_json} --inject-npz {genbank} "
                f"--inject-norms-json {genstamps} --gpus {gpus_all} --workers-per-gpu {cfg['wpg_gen']} "
                f"--seeds-per-class 2 --limit 80 --max-new-tokens {max_new} {cfg['gen_extra']}").strip()

    def rep_cmd(cells_json: str, sig_subdir: str | None) -> str:
        extra = f"--sig-subdir {sig_subdir}" if sig_subdir else "--inject-from-metadata"
        return (f"python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model {m} --model-path {cfg['mpath']} "
                f"--calib-dir {cfg['calib']} --cells-json {NODE_CELLS}/{cells_json} --gpus {gpus_all} "
                f"--workers-per-gpu {cfg['wpg_rep']} --no-raw {extra}")

    print(f"===== FM FIELD TRIPLE ({tag}) @L{S} =====")
    print(f"cells: {len(gen_main)} main-gen (@512) / {len(gen_stop)} stop-gen (@2048 UNCAPPED) "
          f"/ {len(rep_main)+len(rep_stop)} replay")
    print(f"frozen P's: rep {cfg['P']['rep']} / stop {cfg['P']['stop']} / conf-collapse {cfg['P']['conf_collapse']}")
    print(f"leak-prediction record: {leak_path}")
    print(f"  [pulse] margin/eos/repmass (3 x gpus=1 ~20m)")
    print(f"  [members] cpu ~8m\n          {members_cmd}")
    print(f"  [gen-main] gpus=8 wpg={cfg['wpg_gen']} ~{cfg['gen_min']}m @512")
    print(f"  [gen-stop] gpus=8 wpg={cfg['wpg_gen']} @2048 (Veos_perp uncapped)")
    print(f"  [rep] state + expression, both buckets")
    print(f"  [entropy] fire AFTER gen: submit_entropy_legs.py --chain field_{args.model}")

    if not args.fire:
        print("(dry-run: cells + leak-prediction record written; nothing submitted)")
        return

    env = dict(cfg["env"])
    if m == "gemma3-27b":
        env["HF_TOKEN"] = hf_token()
    subprocess.run(["ssh", "node1", f"mkdir -p {NODE_CELLS}"], check=True)
    subprocess.run(["rsync", "-a", *[str(LOCAL_CELLS / fn) for fn in files],
                    f"node1:{NODE_CELLS}/"], check=True)

    if args.gen_only:
        gdep = None
        print("gen-only: skipping pulses + members (field_gen bank on disk)")
    else:
        if args.members_only:
            mdeps = []
        else:
            pulses = [submit(spec(f"fmfld-{tag}-{fn}", pulse(fn), 1, 20, env,
                                  [args.after] if args.after else None))
                      for fn in ("margin", "eos", "repmass")]
            mdeps = pulses
            print(f"pulses={pulses}")
        members = submit(spec(f"fmfld-{tag}-members", members_cmd, 1, 8, env, mdeps or None))
        gdep = [members]
        print(f"members={members}")
    gm = submit(spec(f"fmfld-{tag}-gen-main", gen_cmd(f"gen_{tag}_field_main.json", 512),
                     8, cfg["gen_min"], env, gdep))
    gs = submit(spec(f"fmfld-{tag}-gen-stop", gen_cmd(f"gen_{tag}_field_stop.json", 2048),
                     8, cfg["gen_min"] * 3, env, gdep))
    rms = submit(spec(f"fmfld-{tag}-rep-main-state", rep_cmd(f"rep_{tag}_field_main.json", None),
                      8, 35, env, [gm]))
    rme = submit(spec(f"fmfld-{tag}-rep-main-expr", rep_cmd(f"rep_{tag}_field_main.json", "signatures_v3_noinject"),
                      8, 35, env, [gm]))
    rss = submit(spec(f"fmfld-{tag}-rep-stop-state", rep_cmd(f"rep_{tag}_field_stop.json", None),
                      8, 35, env, [gs]))
    print(f"gen_main={gm} gen_stop={gs} rep_main_state={rms} rep_main_expr={rme} rep_stop_state={rss}")
    print(f"NEXT (after gen lands): entropy leg -> submit_entropy_legs.py --chain field_{args.model} "
          f"(dep gen_main={gm})")


if __name__ == "__main__":
    main()
