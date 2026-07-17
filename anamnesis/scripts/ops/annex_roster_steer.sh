#!/usr/bin/env bash
# ANNEX in-shadow roster — STAGE 2: the steering blocks (ratified c93055b).
# ⛔ DOUBLE-GATED: (1) Luxia's GPU word; (2) the per-member rider-1 predictions file MUST
# exist locally (checked below) — anatomy before predictions before steering, always.
#
# Chain (multicell for BOTH gen and replay — the S8-8 lesson, one model load per worker):
#   smoke -> gen (9 cells, one job) -> injected replay (state column)
#                                   -> no-inject replay (EXPRESSION column, standing rule)
#                                   -> entropy rows
# Members: Vconf/Veos/Vrep_L14 x alpha{.03,.1,.3}, n=40/cell, VMBROSTER-3B-{cell} namespaces.
set -euo pipefail

: "${HEIMDALL_WORK_DIR:?}"
: "${HEIMDALL_VENV:?}"

REPO="$HOME/projects/anamnesis_exps"
PRED="$REPO/outputs/battery/annex/PREDICTIONS-inshadow-members-rider1.md"
[[ -f $PRED ]] || { echo "⛔ GATE: $PRED missing — file the per-member predictions first (stage-1 header)"; exit 1; }

MODEL=3b
MPATH=/models/llama-3.2-3b-instruct
RUNS=/models/anamnesis-extract/runs
CALIB=/models/anamnesis-extract/calibration/3b
STAGE0=$RUNS/vmb_stage0_3b
VEC_DIR=/models/anamnesis-extract/battery/annex/roster_vectors_3b
RUN_ROOT=$RUNS/vmb_roster_3b
PROMPTS=pipeline/anamnesis/prompts/prompt_sets.json
BATTERY=/models/anamnesis-extract/battery
BASE="source $HEIMDALL_VENV && cd $HEIMDALL_WORK_DIR && export PYTHONPATH=\$PWD/pipeline PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1"
MEMBERS=(Vconf_L14 Veos_L14 Vrep_L14)
LADDER=(0.03 0.1 0.3)

submit() { # name, cmd, gpus, minutes, [after]  — hardened pattern (loud on failure)
  local AFTER=(); [[ -n ${5:-} ]] && AFTER=(--after "$5")
  local OUT_
  OUT_=$(heimdall submit "bash -c '$BASE && $2'" -n "$1" --node node1 -g "$3" -e "$4" \
    -w "$HEIMDALL_WORK_DIR" --env HF_HUB_OFFLINE=1 --max-retries 0 "${AFTER[@]}" 2>&1) \
    || { echo "SUBMIT FAILED for $1:" >&2; echo "$OUT_" >&2; exit 1; }
  echo "$OUT_" >&2
  grep -oE '[0-9a-f]{12}' <<< "$OUT_" | head -1
}

# ── staging: member bank to node + the two cells-json files (gen + replay) ──
ssh node1 "mkdir -p $VEC_DIR $RUN_ROOT"
rsync -a "$REPO/outputs/battery/annex/roster_vectors_3b/" "node1:$VEC_DIR/"

GEN_JSON=/tmp/claude-output/roster_gen_cells.json
REP_JSON=/tmp/claude-output/roster_rep_cells.json
mkdir -p /tmp/claude-output
{
  echo '{"cells": ['
  FIRST=1
  for M in "${MEMBERS[@]}"; do for A in "${LADDER[@]}"; do
    C=${M}_a${A}
    [[ $FIRST == 1 ]] && FIRST=0 || echo ','
    printf '  {"out_run_dir": "%s/%s", "seed_namespace": "VMBROSTER-3B-%s", "inject_key": "%s", "inject_layer": 14, "inject_alpha_frac": %s}' \
      "$RUN_ROOT" "$C" "$C" "$M" "$A"
  done; done
  echo ''; echo ']}'
} > "$GEN_JSON"
{
  echo '{"cells": ['
  FIRST=1
  for M in "${MEMBERS[@]}"; do for A in "${LADDER[@]}"; do
    C=${M}_a${A}
    [[ $FIRST == 1 ]] && FIRST=0 || echo ','
    printf '  {"run_dir": "%s/%s", "manifest": "%s/%s/replay_manifest.json"}' \
      "$RUN_ROOT" "$C" "$RUN_ROOT" "$C"
  done; done
  echo ''; echo ']}'
} > "$REP_JSON"
python3 -c "import json,sys; json.load(open('$GEN_JSON')); json.load(open('$REP_JSON')); print('cells-json ok')"
rsync -a "$GEN_JSON" "node1:$BATTERY/annex/roster_gen_cells.json"
rsync -a "$REP_JSON" "node1:$BATTERY/annex/roster_rep_cells.json"
echo "staged member bank + cells-json -> node1"

SMOKE=$(submit vmb-roster-smoke \
  "python -u -m anamnesis.scripts.vmb_smoke_write_hooks --model $MODEL --model-path $MPATH --floor-run-dir $STAGE0 --calib-dir $CALIB --out /dev/shm/vmb_roster_smoke" \
  1 10)
echo "vmb-roster-smoke -> $SMOKE"

GEN=$(submit vmb-roster-gen \
  "python -u -m anamnesis.scripts.vmb_a5_gen_multicell --model $MODEL --model-path $MPATH --prompts $PROMPTS --cells-json $BATTERY/annex/roster_gen_cells.json --inject-npz $VEC_DIR/a5_vectors.npz --inject-norms-json $VEC_DIR/a5_vectors_stamps.json --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 4 --seeds-per-class 1 --limit 40" \
  8 30 "$SMOKE")
echo "vmb-roster-gen -> $GEN [after $SMOKE]"

REP=$(submit vmb-roster-rep \
  "python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model $MODEL --model-path $MPATH --calib-dir $CALIB --cells-json $BATTERY/annex/roster_rep_cells.json --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 4 --no-raw --inject-from-metadata" \
  8 20 "$GEN")
echo "vmb-roster-rep -> $REP [after $GEN]"

EXPR=$(submit vmb-roster-expression \
  "python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model $MODEL --model-path $MPATH --calib-dir $CALIB --cells-json $BATTERY/annex/roster_rep_cells.json --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 4 --no-raw --sig-subdir signatures_v3_noinject" \
  8 20 "$REP")
echo "vmb-roster-expression -> $EXPR [after $REP] (STANDING RULE: the expression column)"

CELLS=""
for M in "${MEMBERS[@]}"; do for A in "${LADDER[@]}"; do CELLS+="${M}_a${A} "; done; done
ENT=$(submit vmb-roster-entropy \
  "python -u -m anamnesis.scripts.vmb_c3_entropy_replay --model $MODEL --model-path $MPATH --c3-run-dir $RUN_ROOT --cells $CELLS --null-prefixes RBAND --out-json $BATTERY/annex/roster_entropy_3b.json" \
  1 30 "$GEN")
echo "vmb-roster-entropy -> $ENT [after $GEN, parallel to replays]"

echo
echo "CHAIN QUEUED: smoke -> gen -> rep -> expression; entropy parallel after gen."
echo "Readout: LOCAL banked machinery (gg roster rows + samefamily lever + judge spot-check)."
