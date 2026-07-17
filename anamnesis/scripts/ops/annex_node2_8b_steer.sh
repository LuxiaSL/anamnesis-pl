#!/usr/bin/env bash
# ANNEX 8B 2×2 — STAGE 2 steer, NODE2 variant (Heimdall, priority 10, overnight).
# ⛔ GATED on PREDICTIONS-8b-rider1.md (anatomy before predictions before steering).
set -euo pipefail

REPO="$HOME/projects/anamnesis_exps"
PRED="$REPO/outputs/battery/annex/PREDICTIONS-8b-rider1.md"
[[ -f $PRED ]] || { echo "⛔ GATE: $PRED missing — file the rider-1 anatomy+predictions first"; exit 1; }
BANK="$REPO/outputs/battery/annex/8b_vectors"
[[ -f $BANK/a5_vectors.npz ]] || { echo "⛔ GATE: $BANK missing — run the CPU builds first"; exit 1; }

: "${HEIMDALL_WORK_DIR:?}"
: "${HEIMDALL_VENV:?}"
PRIO=10

SHM=/dev/shm/luxi-anamnesis
DATA=$SHM/anamnesis-extract
M8B=/models/llama-3.1-8b-instruct
VEC_DIR=$SHM/banks/annex/8b_vectors
RUN_ROOT=$DATA/runs/vmb_8b2x2_8b
PROMPTS=pipeline/anamnesis/prompts/prompt_sets.json
BASE="source $HEIMDALL_VENV && cd $HEIMDALL_WORK_DIR && export PYTHONPATH=\$PWD/pipeline PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1"
MEMBERS=(V7_L16 RA_L16 Rband1_L16 Rband2_L16 Rband3_L16)
LADDER=(0.03 0.1 0.3)

submit() {
  local AFTER=(); [[ -n ${5:-} ]] && AFTER=(--after "$5")
  local OUT_
  OUT_=$(heimdall submit "bash -c '$BASE && $2'" -n "$1" --node node2 -g "$3" -e "$4" \
    -w "$HEIMDALL_WORK_DIR" --env HF_HUB_OFFLINE=1 --priority $PRIO --max-retries 2 \
    "${AFTER[@]}" 2>&1) \
    || { echo "SUBMIT FAILED for $1:" >&2; echo "$OUT_" >&2; exit 1; }
  echo "$OUT_" >&2
  grep -oE '[0-9a-f]{12}' <<< "$OUT_" | head -1
}

ssh node2 "mkdir -p $VEC_DIR $RUN_ROOT"
rsync -a "$BANK/" node2:$VEC_DIR/

mkdir -p /tmp/claude-output
GEN_JSON=/tmp/claude-output/8b2x2_node2_gen_cells.json
REP_JSON=/tmp/claude-output/8b2x2_node2_rep_cells.json
{
  echo '{"cells": ['
  FIRST=1
  for M in "${MEMBERS[@]}"; do for A in "${LADDER[@]}"; do
    C=${M}_a${A}
    [[ $FIRST == 1 ]] && FIRST=0 || echo ','
    printf '  {"out_run_dir": "%s/%s", "seed_namespace": "VMB8B2X2-8B-%s", "inject_key": "%s", "inject_layer": 16, "inject_alpha_frac": %s}' \
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
python3 -c "import json; json.load(open('$GEN_JSON')); json.load(open('$REP_JSON')); print('cells-json ok')"
rsync -a "$GEN_JSON" "$REP_JSON" node2:$SHM/

GEN=$(submit vmb-8b2x2-gen-n2 \
  "python -u -m anamnesis.scripts.vmb_a5_gen_multicell --model 8b --model-path $M8B --prompts $PROMPTS --cells-json $SHM/8b2x2_node2_gen_cells.json --inject-npz $VEC_DIR/a5_vectors.npz --inject-norms-json $VEC_DIR/a5_vectors_stamps.json --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 2 --seeds-per-class 1 --limit 40" \
  8 180)
echo "vmb-8b2x2-gen-n2 -> $GEN (the long pole)"

REP=$(submit vmb-8b2x2-rep-n2 \
  "python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model 8b --model-path $M8B --calib-dir $DATA/calibration/8b --cells-json $SHM/8b2x2_node2_rep_cells.json --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 3 --no-raw --inject-from-metadata" \
  8 90 "$GEN")
echo "vmb-8b2x2-rep-n2 -> $REP [after $GEN]"

EXPR=$(submit vmb-8b2x2-expr-n2 \
  "python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model 8b --model-path $M8B --calib-dir $DATA/calibration/8b --cells-json $SHM/8b2x2_node2_rep_cells.json --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 3 --no-raw --sig-subdir signatures_v3_noinject" \
  8 90 "$REP")
echo "vmb-8b2x2-expr-n2 -> $EXPR [after $REP]"

CELLS=""
for M in "${MEMBERS[@]}"; do for A in "${LADDER[@]}"; do CELLS+="${M}_a${A} "; done; done
ENT=$(submit vmb-8b2x2-ent-n2 \
  "python -u -m anamnesis.scripts.vmb_c3_entropy_replay --model 8b --model-path $M8B --c3-run-dir $RUN_ROOT --cells $CELLS --null-prefixes RBAND --out-json $RUN_ROOT/8b2x2_entropy_8b_node2.json" \
  1 120 "$GEN")
echo "vmb-8b2x2-ent-n2 -> $ENT [after $GEN, parallel]"

echo
echo "8B STEER QUEUED at priority $PRIO."
echo "FINAL_JOB=$EXPR"
