#!/usr/bin/env bash
# ANNEX Vconf⊥ cell (queue item 1, desk-authorized 2026-07-17). ⛔ DOUBLE-GATED:
# (1) Luxia's GPU word; (2) PREDICTIONS-vconf-cell.md must exist (VC book frozen S9-2,
# transcribed there — the gate file). Closes the correction's open half: does confidence
# hide a coordinate behind its .91 temperature projection?
#
# Chain: write-hook smoke -> gen (4 cells, capped, one multicell job) ->
#        injected replay (state) -> no-inject replay (EXPRESSION, standing rule) -> entropy.
# Cells: Vconf_perp_L14 x {+.1,+.3,-.1,-.3}, n=40/cell, VMBVCONF-3B-<cell> namespaces.
set -euo pipefail

: "${HEIMDALL_WORK_DIR:?}"
: "${HEIMDALL_VENV:?}"

REPO="$HOME/projects/anamnesis_exps"
PRED="$REPO/outputs/battery/annex/PREDICTIONS-vconf-cell.md"
[[ -f $PRED ]] || { echo "⛔ GATE: $PRED missing — the frozen VC book must be filed first"; exit 1; }

MODEL=3b
MPATH=/models/llama-3.2-3b-instruct
RUNS=/models/anamnesis-extract/runs
CALIB=/models/anamnesis-extract/calibration/3b
STAGE0=$RUNS/vmb_stage0_3b
RUN_ROOT=$RUNS/vmb_vconf_3b
VEC_DIR=/models/anamnesis-extract/battery/annex/vconf_perp_3b
PROMPTS=pipeline/anamnesis/prompts/prompt_sets.json
BATTERY=/models/anamnesis-extract/battery
BASE="source $HEIMDALL_VENV && cd $HEIMDALL_WORK_DIR && export PYTHONPATH=\$PWD/pipeline PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1"
FRACS=(0.1 0.3 -0.1 -0.3)
KEY=Vconf_perp_L14

submit() { # name, cmd, gpus, minutes, [after]  — hardened pattern (loud on failure)
  local AFTER=(); [[ -n ${5:-} ]] && AFTER=(--after "$5")
  local OUT_
  OUT_=$(heimdall submit "bash -c '$BASE && $2'" -n "$1" --node node1 -g "$3" -e "$4" \
    -w "$HEIMDALL_WORK_DIR" --env HF_HUB_OFFLINE=1 --max-retries 0 "${AFTER[@]}" 2>&1) \
    || { echo "SUBMIT FAILED for $1:" >&2; echo "$OUT_" >&2; exit 1; }
  echo "$OUT_" >&2
  grep -oE '[0-9a-f]{12}' <<< "$OUT_" | head -1
}

cellname() { local F=${1#-}; [[ $1 == -* ]] && echo "${KEY}_an${F}" || echo "${KEY}_a${F}"; }

# ── staging: self-deploy code + bank + cells-json ──
rsync -a "$REPO/pipeline/anamnesis/scripts/" node1:luxi-files/anamnesis-pl/pipeline/anamnesis/scripts/ \
  --include='*.py' --exclude='ops/*' --exclude='__pycache__'
ssh node1 "mkdir -p $VEC_DIR $RUN_ROOT"
rsync -a "$REPO/outputs/battery/annex/vconf_perp_3b/" "node1:$VEC_DIR/"

mkdir -p /tmp/claude-output
GEN_JSON=/tmp/claude-output/vconf_gen_cells.json
REP_JSON=/tmp/claude-output/vconf_rep_cells.json
{
  echo '{"cells": ['
  FIRST=1
  for A in "${FRACS[@]}"; do
    C=$(cellname "$A")
    [[ $FIRST == 1 ]] && FIRST=0 || echo ','
    printf '  {"out_run_dir": "%s/%s", "seed_namespace": "VMBVCONF-3B-%s", "inject_key": "%s", "inject_layer": 14, "inject_alpha_frac": %s}' \
      "$RUN_ROOT" "$C" "$C" "$KEY" "$A"
  done
  echo ''; echo ']}'
} > "$GEN_JSON"
{
  echo '{"cells": ['
  FIRST=1
  for A in "${FRACS[@]}"; do
    C=$(cellname "$A")
    [[ $FIRST == 1 ]] && FIRST=0 || echo ','
    printf '  {"run_dir": "%s/%s", "manifest": "%s/%s/replay_manifest.json"}' \
      "$RUN_ROOT" "$C" "$RUN_ROOT" "$C"
  done
  echo ''; echo ']}'
} > "$REP_JSON"
python3 -c "import json; json.load(open('$GEN_JSON')); json.load(open('$REP_JSON')); print('cells-json ok')"
rsync -a "$GEN_JSON" "node1:$BATTERY/annex/vconf_gen_cells.json"
rsync -a "$REP_JSON" "node1:$BATTERY/annex/vconf_rep_cells.json"
echo "staged code + vconf bank + cells-json -> node1"

SMOKE=$(submit vmb-vconf-smoke \
  "python -u -m anamnesis.scripts.vmb_smoke_write_hooks --model $MODEL --model-path $MPATH --floor-run-dir $STAGE0 --calib-dir $CALIB --out /dev/shm/vmb_vconf_smoke" \
  1 10)
echo "vmb-vconf-smoke -> $SMOKE"

GEN=$(submit vmb-vconf-gen \
  "python -u -m anamnesis.scripts.vmb_a5_gen_multicell --model $MODEL --model-path $MPATH --prompts $PROMPTS --cells-json $BATTERY/annex/vconf_gen_cells.json --inject-npz $VEC_DIR/a5_vectors.npz --inject-norms-json $VEC_DIR/a5_vectors_stamps.json --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 4 --seeds-per-class 1 --limit 40" \
  8 20 "$SMOKE")
echo "vmb-vconf-gen -> $GEN [after $SMOKE]"

REP=$(submit vmb-vconf-rep \
  "python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model $MODEL --model-path $MPATH --calib-dir $CALIB --cells-json $BATTERY/annex/vconf_rep_cells.json --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 4 --no-raw --inject-from-metadata" \
  8 15 "$GEN")
echo "vmb-vconf-rep -> $REP [after $GEN]"

EXPR=$(submit vmb-vconf-expression \
  "python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model $MODEL --model-path $MPATH --calib-dir $CALIB --cells-json $BATTERY/annex/vconf_rep_cells.json --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 4 --no-raw --sig-subdir signatures_v3_noinject" \
  8 15 "$REP")
echo "vmb-vconf-expression -> $EXPR [after $REP] (STANDING RULE: the expression column)"

CELLS=""
for A in "${FRACS[@]}"; do CELLS+="$(cellname "$A") "; done
ENT=$(submit vmb-vconf-entropy \
  "python -u -m anamnesis.scripts.vmb_c3_entropy_replay --model $MODEL --model-path $MPATH --c3-run-dir $RUN_ROOT --cells $CELLS --out-json $BATTERY/annex/vconf_entropy_3b.json" \
  1 20 "$GEN")
echo "vmb-vconf-entropy -> $ENT [after $GEN, parallel to replays]"

echo
echo "VCONF CHAIN QUEUED: smoke -> gen -> rep -> expression; entropy parallel after gen."
echo "Readout: local banked machinery (text consequence vs roster Rband- envelope + hedge/def lexicon + two-column gg + VC book scoring)."
