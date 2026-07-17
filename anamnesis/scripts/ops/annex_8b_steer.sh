#!/usr/bin/env bash
# ANNEX 8B 2×2 — STAGE 2: the steering blocks (15 cells). ⛔ DOUBLE-GATED:
# (1) Luxia's GPU word; (2) PREDICTIONS-8b-rider1.md must exist locally (anatomy before
# predictions before steering, always — the roster discipline at 8B).
#
# Cells: {V7_L16, RA_L16, Rband1_L16, Rband2_L16, Rband3_L16} x alpha{.03,.1,.3}, n=40,
# capped, VMB8B2X2-8B-<cell> namespaces. Chain: smoke -> gen (one multicell job) ->
# injected replay (state) -> no-inject replay (EXPRESSION) -> entropy rows.
# Price context: ~2.5x 3B per token => gen leg is the long pole (~3-4 hr all-in).
set -euo pipefail

: "${HEIMDALL_WORK_DIR:?}"
: "${HEIMDALL_VENV:?}"

REPO="$HOME/projects/anamnesis_exps"
PRED="$REPO/outputs/battery/annex/PREDICTIONS-8b-rider1.md"
[[ -f $PRED ]] || { echo "⛔ GATE: $PRED missing — file the 8B rider-1 anatomy+predictions first"; exit 1; }
BANK="$REPO/outputs/battery/annex/8b_vectors/a5_vectors.npz"
[[ -f $BANK ]] || { echo "⛔ GATE: $BANK missing — run the CPU builds after annex_8b_pulses.sh"; exit 1; }

MODEL=8b
MPATH=/models/llama-3.1-8b-instruct
RUNS=/models/anamnesis-extract/runs
CALIB=/models/anamnesis-extract/calibration/8b
STAGE0=$RUNS/vmb_stage0_8b
RUN_ROOT=$RUNS/vmb_8b2x2_8b
VEC_DIR=/models/anamnesis-extract/battery/annex/8b_vectors
PROMPTS=pipeline/anamnesis/prompts/prompt_sets.json
BATTERY=/models/anamnesis-extract/battery
BASE="source $HEIMDALL_VENV && cd $HEIMDALL_WORK_DIR && export PYTHONPATH=\$PWD/pipeline PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1"
MEMBERS=(V7_L16 RA_L16 Rband1_L16 Rband2_L16 Rband3_L16)
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

# ── staging: self-deploy code + local 8B bank to node + cells-json ──
rsync -a "$REPO/pipeline/anamnesis/scripts/" node1:luxi-files/anamnesis-pl/pipeline/anamnesis/scripts/ \
  --include='*.py' --exclude='ops/*' --exclude='__pycache__'
ssh node1 "mkdir -p $VEC_DIR $RUN_ROOT"
rsync -a "$REPO/outputs/battery/annex/8b_vectors/" "node1:$VEC_DIR/"

mkdir -p /tmp/claude-output
GEN_JSON=/tmp/claude-output/8b2x2_gen_cells.json
REP_JSON=/tmp/claude-output/8b2x2_rep_cells.json
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
rsync -a "$GEN_JSON" "node1:$BATTERY/annex/8b2x2_gen_cells.json"
rsync -a "$REP_JSON" "node1:$BATTERY/annex/8b2x2_rep_cells.json"
echo "staged 8B bank + cells-json -> node1"

SMOKE=$(submit vmb-8b2x2-smoke \
  "python -u -m anamnesis.scripts.vmb_smoke_write_hooks --model $MODEL --model-path $MPATH --floor-run-dir $STAGE0 --calib-dir $CALIB --out /dev/shm/vmb_8b2x2_smoke" \
  1 15)
echo "vmb-8b2x2-smoke -> $SMOKE"

GEN=$(submit vmb-8b2x2-gen \
  "python -u -m anamnesis.scripts.vmb_a5_gen_multicell --model $MODEL --model-path $MPATH --prompts $PROMPTS --cells-json $BATTERY/annex/8b2x2_gen_cells.json --inject-npz $VEC_DIR/a5_vectors.npz --inject-norms-json $VEC_DIR/a5_vectors_stamps.json --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 2 --seeds-per-class 1 --limit 40" \
  8 150 "$SMOKE")
echo "vmb-8b2x2-gen -> $GEN [after $SMOKE] (the long pole)"

REP=$(submit vmb-8b2x2-rep \
  "python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model $MODEL --model-path $MPATH --calib-dir $CALIB --cells-json $BATTERY/annex/8b2x2_rep_cells.json --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 3 --no-raw --inject-from-metadata" \
  8 60 "$GEN")
echo "vmb-8b2x2-rep -> $REP [after $GEN]"

EXPR=$(submit vmb-8b2x2-expression \
  "python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model $MODEL --model-path $MPATH --calib-dir $CALIB --cells-json $BATTERY/annex/8b2x2_rep_cells.json --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 3 --no-raw --sig-subdir signatures_v3_noinject" \
  8 60 "$REP")
echo "vmb-8b2x2-expression -> $EXPR [after $REP] (STANDING RULE: the expression column)"

CELLS=""
for M in "${MEMBERS[@]}"; do for A in "${LADDER[@]}"; do CELLS+="${M}_a${A} "; done; done
ENT=$(submit vmb-8b2x2-entropy \
  "python -u -m anamnesis.scripts.vmb_c3_entropy_replay --model $MODEL --model-path $MPATH --c3-run-dir $RUN_ROOT --cells $CELLS --null-prefixes RBAND --out-json $BATTERY/annex/8b2x2_entropy_8b.json" \
  1 90 "$GEN")
echo "vmb-8b2x2-entropy -> $ENT [after $GEN, parallel to replays]"

echo
echo "8B 2x2 CHAIN QUEUED: smoke -> gen -> rep -> expression; entropy parallel."
echo "Readout: local banked machinery; score 8B-1..3 (frozen .65/.80/.50 in the pricing doc)."
