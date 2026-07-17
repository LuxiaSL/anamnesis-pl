#!/usr/bin/env bash
# ANNEX −Veos⊥ dose-window block (queue item 3, desk-authorized 2026-07-17). ⛔ DOUBLE-GATED:
# (1) Luxia's GPU word; (2) PREDICTIONS-eosdose-window.md must exist (ED book frozen at
# staging). The question: does a dose exist where lengthening clears the null envelope
# while repetition stays inside it? (Bracket on record: −.1 inside; −.3 length ✓ rep ✗.)
#
# Chain: write-hook smoke -> gen (5 cells UNCAPPED 2048, one multicell job) ->
#        injected replay (state) -> no-inject replay (EXPRESSION) -> entropy.
# Cells: Veos_perp_L14 x {−.15,−.20,−.25} + Rband1_L14 x {−.15,−.25} (matched nulls),
# n=40/cell, VMBEOSD-3B-<cell> namespaces. Bank = the eosrep bank (keys already in it).
set -euo pipefail

: "${HEIMDALL_WORK_DIR:?}"
: "${HEIMDALL_VENV:?}"

REPO="$HOME/projects/anamnesis_exps"
PRED="$REPO/outputs/battery/annex/PREDICTIONS-eosdose-window.md"
[[ -f $PRED ]] || { echo "⛔ GATE: $PRED missing — freeze the ED book first"; exit 1; }

MODEL=3b
MPATH=/models/llama-3.2-3b-instruct
RUNS=/models/anamnesis-extract/runs
CALIB=/models/anamnesis-extract/calibration/3b
STAGE0=$RUNS/vmb_stage0_3b
RUN_ROOT=$RUNS/vmb_eosdose_3b
VEC_DIR=/models/anamnesis-extract/battery/annex/eosrep_vectors_3b
PROMPTS=pipeline/anamnesis/prompts/prompt_sets.json
BATTERY=/models/anamnesis-extract/battery
BASE="source $HEIMDALL_VENV && cd $HEIMDALL_WORK_DIR && export PYTHONPATH=\$PWD/pipeline PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1"
# key|frac
CELLS_SPEC=(
  "Veos_perp_L14|-0.15" "Veos_perp_L14|-0.2" "Veos_perp_L14|-0.25"
  "Rband1_L14|-0.15"    "Rband1_L14|-0.25"
)

submit() { # name, cmd, gpus, minutes, [after]  — hardened pattern (loud on failure)
  local AFTER=(); [[ -n ${5:-} ]] && AFTER=(--after "$5")
  local OUT_
  OUT_=$(heimdall submit "bash -c '$BASE && $2'" -n "$1" --node node1 -g "$3" -e "$4" \
    -w "$HEIMDALL_WORK_DIR" --env HF_HUB_OFFLINE=1 --max-retries 0 "${AFTER[@]}" 2>&1) \
    || { echo "SUBMIT FAILED for $1:" >&2; echo "$OUT_" >&2; exit 1; }
  echo "$OUT_" >&2
  grep -oE '[0-9a-f]{12}' <<< "$OUT_" | head -1
}

cellname() { # key, frac
  local F=${2#-}; [[ $2 == -* ]] && echo "${1}_an${F}" || echo "${1}_a${F}"
}

# ── staging: self-deploy code + cells-json (bank already node-side from the eosrep block;
# re-rsync is cheap and makes the chain self-sufficient) ──
rsync -a "$REPO/pipeline/anamnesis/scripts/" node1:luxi-files/anamnesis-pl/pipeline/anamnesis/scripts/ \
  --include='*.py' --exclude='ops/*' --exclude='__pycache__'
ssh node1 "mkdir -p $VEC_DIR $RUN_ROOT"
rsync -a "$REPO/outputs/battery/annex/eosrep_vectors_3b/" "node1:$VEC_DIR/"

mkdir -p /tmp/claude-output
GEN_JSON=/tmp/claude-output/eosdose_gen_cells.json
REP_JSON=/tmp/claude-output/eosdose_rep_cells.json
{
  echo '{"cells": ['
  FIRST=1
  for spec in "${CELLS_SPEC[@]}"; do
    IFS='|' read -r K FR <<< "$spec"
    C=$(cellname "$K" "$FR")
    [[ $FIRST == 1 ]] && FIRST=0 || echo ','
    printf '  {"out_run_dir": "%s/%s", "seed_namespace": "VMBEOSD-3B-%s", "inject_key": "%s", "inject_layer": 14, "inject_alpha_frac": %s}' \
      "$RUN_ROOT" "$C" "$C" "$K" "$FR"
  done
  echo ''; echo ']}'
} > "$GEN_JSON"
{
  echo '{"cells": ['
  FIRST=1
  for spec in "${CELLS_SPEC[@]}"; do
    IFS='|' read -r K FR <<< "$spec"
    C=$(cellname "$K" "$FR")
    [[ $FIRST == 1 ]] && FIRST=0 || echo ','
    printf '  {"run_dir": "%s/%s", "manifest": "%s/%s/replay_manifest.json"}' \
      "$RUN_ROOT" "$C" "$RUN_ROOT" "$C"
  done
  echo ''; echo ']}'
} > "$REP_JSON"
python3 -c "import json; json.load(open('$GEN_JSON')); json.load(open('$REP_JSON')); print('cells-json ok')"
rsync -a "$GEN_JSON" "node1:$BATTERY/annex/eosdose_gen_cells.json"
rsync -a "$REP_JSON" "node1:$BATTERY/annex/eosdose_rep_cells.json"
echo "staged code + bank + cells-json -> node1"

SMOKE=$(submit vmb-eosdose-smoke \
  "python -u -m anamnesis.scripts.vmb_smoke_write_hooks --model $MODEL --model-path $MPATH --floor-run-dir $STAGE0 --calib-dir $CALIB --out /dev/shm/vmb_eosdose_smoke" \
  1 10)
echo "vmb-eosdose-smoke -> $SMOKE"

GEN=$(submit vmb-eosdose-gen \
  "python -u -m anamnesis.scripts.vmb_a5_gen_multicell --model $MODEL --model-path $MPATH --prompts $PROMPTS --cells-json $BATTERY/annex/eosdose_gen_cells.json --inject-npz $VEC_DIR/a5_vectors.npz --inject-norms-json $VEC_DIR/a5_vectors_stamps.json --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 4 --seeds-per-class 1 --limit 40 --max-new-tokens 2048" \
  8 45 "$SMOKE")
echo "vmb-eosdose-gen -> $GEN [after $SMOKE] (UNCAPPED 2048 — the substrate truncation fact)"

REP=$(submit vmb-eosdose-rep \
  "python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model $MODEL --model-path $MPATH --calib-dir $CALIB --cells-json $BATTERY/annex/eosdose_rep_cells.json --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 4 --no-raw --inject-from-metadata" \
  8 30 "$GEN")
echo "vmb-eosdose-rep -> $REP [after $GEN]"

EXPR=$(submit vmb-eosdose-expression \
  "python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model $MODEL --model-path $MPATH --calib-dir $CALIB --cells-json $BATTERY/annex/eosdose_rep_cells.json --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 4 --no-raw --sig-subdir signatures_v3_noinject" \
  8 30 "$REP")
echo "vmb-eosdose-expression -> $EXPR [after $REP] (STANDING RULE: the expression column)"

CELLS=""
for spec in "${CELLS_SPEC[@]}"; do
  IFS='|' read -r K FR <<< "$spec"
  CELLS+="$(cellname "$K" "$FR") "
done
ENT=$(submit vmb-eosdose-entropy \
  "python -u -m anamnesis.scripts.vmb_c3_entropy_replay --model $MODEL --model-path $MPATH --c3-run-dir $RUN_ROOT --cells $CELLS --null-prefixes RBAND --out-json $BATTERY/annex/eosdose_entropy_3b.json" \
  1 40 "$GEN")
echo "vmb-eosdose-entropy -> $ENT [after $GEN, parallel to replays]"

echo
echo "EOSDOSE CHAIN QUEUED: smoke -> gen (uncapped) -> rep -> expression; entropy parallel."
echo "Readout: local — length/rep vs the in-chain Rband-uncapped envelope, ED book scoring;"
echo "if ED-2 fires, the winning dose goes to a clean-judge quality pass BEFORE dial language."
