#!/usr/bin/env bash
# ANNEX two-actuators block (queue item 2, desk-authorized 2026-07-17). ⛔ DOUBLE-GATED:
# (1) Luxia's GPU word; (2) PREDICTIONS-two-actuators.md must exist (TA book frozen at
# staging). Second specimen of the 14m design: does actuator IDENTITY (internal Vrep⊥ vs
# the actual sampler repetition_penalty) stay readable in signature space at matched
# text-effect?
#
# Chain: gen-path BITWISE SMOKE (parity gate for the new --repetition-penalty plumbing,
#        f5b6b74: old-vs-new launcher byte-diff on 2 tiny cells) ->
#        gen (3 SAMPLER cells, capped, one multicell job; per-cell repetition_penalty,
#        NO injection) -> plain replay (the sampler cells' only column IS expression).
# Comparators (±Vrep⊥/raw @ .1/.3 both columns) are already banked from the eosrep block.
# Cells: rp0.85 (encourage) · rp1.15 · rp1.30, n=40/cell, VMBTWOACT-3B-<cell> namespaces.
set -euo pipefail

: "${HEIMDALL_WORK_DIR:?}"
: "${HEIMDALL_VENV:?}"

REPO="$HOME/projects/anamnesis_exps"
PRED="$REPO/outputs/battery/annex/PREDICTIONS-two-actuators.md"
[[ -f $PRED ]] || { echo "⛔ GATE: $PRED missing — freeze the TA book first"; exit 1; }

MODEL=3b
MPATH=/models/llama-3.2-3b-instruct
RUNS=/models/anamnesis-extract/runs
CALIB=/models/anamnesis-extract/calibration/3b
RUN_ROOT=$RUNS/vmb_twoact_3b
EOSREP_VEC=/models/anamnesis-extract/battery/annex/eosrep_vectors_3b
PROMPTS=pipeline/anamnesis/prompts/prompt_sets.json
BATTERY=/models/anamnesis-extract/battery
BASE="source $HEIMDALL_VENV && cd $HEIMDALL_WORK_DIR && export PYTHONPATH=\$PWD/pipeline PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1"
# penalty|cellname
PENALTIES=("0.85|rp085" "1.15|rp115" "1.30|rp130")

submit() { # name, cmd, gpus, minutes, [after]  — hardened pattern (loud on failure)
  local AFTER=(); [[ -n ${5:-} ]] && AFTER=(--after "$5")
  local OUT_
  OUT_=$(heimdall submit "bash -c '$BASE && $2'" -n "$1" --node node1 -g "$3" -e "$4" \
    -w "$HEIMDALL_WORK_DIR" --env HF_HUB_OFFLINE=1 --max-retries 0 "${AFTER[@]}" 2>&1) \
    || { echo "SUBMIT FAILED for $1:" >&2; echo "$OUT_" >&2; exit 1; }
  echo "$OUT_" >&2
  grep -oE '[0-9a-f]{12}' <<< "$OUT_" | head -1
}

# ── staging: self-deploy code (the new plumbing MUST be on the node) + cells-json ──
rsync -a "$REPO/pipeline/anamnesis/scripts/" node1:luxi-files/anamnesis-pl/pipeline/anamnesis/scripts/ \
  --include='*.py' --exclude='ops/*' --exclude='__pycache__'
rsync -a "$REPO/pipeline/anamnesis/extraction/" node1:luxi-files/anamnesis-pl/pipeline/anamnesis/extraction/ \
  --include='*.py' --exclude='__pycache__'
ssh node1 "mkdir -p $RUN_ROOT"

mkdir -p /tmp/claude-output
GEN_JSON=/tmp/claude-output/twoact_gen_cells.json
REP_JSON=/tmp/claude-output/twoact_rep_cells.json
{
  echo '{"cells": ['
  FIRST=1
  for P in "${PENALTIES[@]}"; do
    IFS='|' read -r RP C <<< "$P"
    [[ $FIRST == 1 ]] && FIRST=0 || echo ','
    printf '  {"out_run_dir": "%s/%s", "seed_namespace": "VMBTWOACT-3B-%s", "inject_key": null, "repetition_penalty": %s}' \
      "$RUN_ROOT" "$C" "$C" "$RP"
  done
  echo ''; echo ']}'
} > "$GEN_JSON"
{
  echo '{"cells": ['
  FIRST=1
  for P in "${PENALTIES[@]}"; do
    IFS='|' read -r RP C <<< "$P"
    [[ $FIRST == 1 ]] && FIRST=0 || echo ','
    printf '  {"run_dir": "%s/%s", "manifest": "%s/%s/replay_manifest.json"}' \
      "$RUN_ROOT" "$C" "$RUN_ROOT" "$C"
  done
  echo ''; echo ']}'
} > "$REP_JSON"
python3 -c "import json; json.load(open('$GEN_JSON')); json.load(open('$REP_JSON')); print('cells-json ok')"
rsync -a "$GEN_JSON" "node1:$BATTERY/annex/twoact_gen_cells.json"
rsync -a "$REP_JSON" "node1:$BATTERY/annex/twoact_rep_cells.json"
echo "staged code + cells-json -> node1"

# Parity gate FIRST (canon rule for any gen-path change): old-vs-new launcher byte-diff.
# Uses two eosrep-bank cells at L14 (layer variety isn't the point here; the shared
# default sampling path is).
BITW=$(submit vmb-twoact-bitwise \
  "python -u -m anamnesis.scripts.vmb_a5_multicell_smoke --model $MODEL --model-path $MPATH --npz $EOSREP_VEC/a5_vectors.npz --norms $EOSREP_VEC/a5_vectors_stamps.json --scratch /dev/shm/twoact_bitwise --cellA Veos_perp_L14:14:0.1 --cellB Vrep_perp_L14:14:0.1" \
  1 15)
echo "vmb-twoact-bitwise -> $BITW (parity gate for f5b6b74)"

GEN=$(submit vmb-twoact-gen \
  "python -u -m anamnesis.scripts.vmb_a5_gen_multicell --model $MODEL --model-path $MPATH --prompts $PROMPTS --cells-json $BATTERY/annex/twoact_gen_cells.json --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 4 --seeds-per-class 1 --limit 40" \
  8 20 "$BITW")
echo "vmb-twoact-gen -> $GEN [after $BITW]"

REP=$(submit vmb-twoact-rep \
  "python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model $MODEL --model-path $MPATH --calib-dir $CALIB --cells-json $BATTERY/annex/twoact_rep_cells.json --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 4 --no-raw" \
  8 15 "$GEN")
echo "vmb-twoact-rep -> $REP [after $GEN] (plain replay = the sampler cells' expression column)"

echo
echo "TWOACT CHAIN QUEUED: bitwise parity gate -> gen -> replay."
echo "Readout: local — matched-effect pairing by median trigram-rep (rule in the frozen doc),"
echo "pick-the-actuator GroupKFold vs eosrep expression columns, then the TA-4 clean-judge pass."
