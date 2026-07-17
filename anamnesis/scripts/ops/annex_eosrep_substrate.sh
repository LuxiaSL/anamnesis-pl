#!/usr/bin/env bash
# ANNEX EOS/rep follow-up block — STAGE A: THE UNCAPPED SUBSTRATE (Luxia 2026-07-16:
# "check whether baselines even represent the EOS pop... form the substrate for Veos uncapped").
# ⛔ LUXIA-GATED. One uncapped baseline population: n=40, alpha=0, max-new-tokens 2048,
# standard prompt grid, namespace VMBEOS-3B-baseline_uncapped + replay for the uncapped-frame
# signature reference. READ its length/EOS distribution, THEN freeze the block predictions
# (PREDICTIONS-eosrep-block.md), THEN annex_eosrep_steer.sh may fire.
set -euo pipefail

: "${HEIMDALL_WORK_DIR:?}"
: "${HEIMDALL_VENV:?}"

MODEL=3b
MPATH=/models/llama-3.2-3b-instruct
RUNS=/models/anamnesis-extract/runs
CALIB=/models/anamnesis-extract/calibration/3b
RUN_ROOT=$RUNS/vmb_eosrep_3b
CELL=$RUN_ROOT/baseline_uncapped
PROMPTS=pipeline/anamnesis/prompts/prompt_sets.json
BASE="source $HEIMDALL_VENV && cd $HEIMDALL_WORK_DIR && export PYTHONPATH=\$PWD/pipeline PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1"

submit() { # name, cmd, gpus, minutes, [after]  — hardened pattern
  local AFTER=(); [[ -n ${5:-} ]] && AFTER=(--after "$5")
  local OUT_
  OUT_=$(heimdall submit "bash -c '$BASE && $2'" -n "$1" --node node1 -g "$3" -e "$4" \
    -w "$HEIMDALL_WORK_DIR" --env HF_HUB_OFFLINE=1 --max-retries 0 "${AFTER[@]}" 2>&1) \
    || { echo "SUBMIT FAILED for $1:" >&2; echo "$OUT_" >&2; exit 1; }
  echo "$OUT_" >&2
  grep -oE '[0-9a-f]{12}' <<< "$OUT_" | head -1
}

ssh node1 "mkdir -p $RUN_ROOT"

GEN=$(submit vmb-eosrep-substrate \
  "python -u -m anamnesis.scripts.vmb_stage0_generate --model $MODEL --model-path $MPATH --prompts $PROMPTS --out-run-dir $CELL --gpus 0,1,2,3 --workers-per-gpu 4 --seeds-per-class 1 --limit 40 --max-new-tokens 2048 --seed-namespace VMBEOS-3B-baseline_uncapped" \
  4 30)
echo "vmb-eosrep-substrate -> $GEN"

REP=$(submit vmb-eosrep-substrate-rep \
  "python -u -m anamnesis.scripts.parallel_replay --model $MODEL --model-path $MPATH --run-dir $CELL --calib-dir $CALIB --manifest $CELL/replay_manifest.json --gpus 0,1,2,3 --workers-per-gpu 3 --no-raw" \
  4 20 "$GEN")
echo "vmb-eosrep-substrate-rep -> $REP [after $GEN]"

echo
echo "SUBSTRATE QUEUED. Pull-back: rsync -a node1:$RUN_ROOT/ outputs/battery/annex/vmb_eosrep_3b/"
echo "Then: read the length/EOS distribution -> FREEZE PREDICTIONS-eosrep-block.md -> stage B."
