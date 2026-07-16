#!/usr/bin/env bash
# ANNEX worker-scaling test (session 7, Luxia-authorized 2026-07-16): route 5's
# inner loop (replay + full extraction, worker = model copy) at total worker
# counts {16,32,64,96} = 8 GPUs x wpg {2,4,8,12}. Identical work per config:
# N_CLONES scratch cells x 160 gens, clones of the banked V4_L14_a0.1
# manifest+metadata (same injection spec as the 51-cell timing record).
#
# Submits via the heimdall CLI (endpoint from its own config), jobs chained
# with --after. Scratch-only: writes under $SCRATCH; never touches banked cells.
# Re-run with a fresh SCRATCH (replay resume would skip existing sigs otherwise):
#   SCRATCH=/models/anamnesis-extract/runs/annex_scaling_3b_r2 ./annex_scaling_test.sh
#
# Requires: HEIMDALL_WORK_DIR, HEIMDALL_VENV exported (values in the local ops
# runbook — never committed); ssh alias node1; heimdall CLI on PATH.
set -euo pipefail

: "${HEIMDALL_WORK_DIR:?see anamnesis/scripts/ops/_ops_env.py}"
: "${HEIMDALL_VENV:?see anamnesis/scripts/ops/_ops_env.py}"

SCRATCH="${SCRATCH:-/models/anamnesis-extract/runs/annex_scaling_3b}"
SRC_CELL="${SRC_CELL:-$HOME/projects/anamnesis_exps/outputs/battery/vmb_a5_3b/V4_L14_a0.1}"
MODEL=3b
MPATH=/models/llama-3.2-3b-instruct
CALIB=/models/anamnesis-extract/calibration/3b
N_CLONES=8
CONFIGS=("16 2" "32 4" "64 8" "96 12")   # "total_workers workers_per_gpu"
BASE="source $HEIMDALL_VENV && cd $HEIMDALL_WORK_DIR && export PYTHONPATH=\$PWD/pipeline PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1"

# --- stage: push source manifest+metadata once, fan out node-side ---
ssh node1 "mkdir -p $SCRATCH/_src"
rsync -a "$SRC_CELL/replay_manifest.json" "$SRC_CELL/metadata.json" "node1:$SCRATCH/_src/"
FANOUT=""
for cfg in "${CONFIGS[@]}"; do
  read -r TOTAL _ <<<"$cfg"
  for i in $(seq 0 $((N_CLONES - 1))); do
    FANOUT+="mkdir -p $SCRATCH/w$TOTAL/cell_$i && cp $SCRATCH/_src/replay_manifest.json $SCRATCH/_src/metadata.json $SCRATCH/w$TOTAL/cell_$i/ && "
  done
done
ssh node1 "${FANOUT}true"

# --- cells-json per config, built node-side ---
for cfg in "${CONFIGS[@]}"; do
  read -r TOTAL _ <<<"$cfg"
  CELLS=$(for i in $(seq 0 $((N_CLONES - 1))); do
    printf '{"run_dir": "%s/w%s/cell_%s", "manifest": "%s/w%s/cell_%s/replay_manifest.json"},' \
      "$SCRATCH" "$TOTAL" "$i" "$SCRATCH" "$TOTAL" "$i"
  done)
  ssh node1 "cat > $SCRATCH/w${TOTAL}_cells.json" <<<"{\"cells\": [${CELLS%,}]}"
done
echo "staged: ${#CONFIGS[@]} configs x $N_CLONES cells under $SCRATCH"

# --- submit chained via heimdall CLI ---
PREV=""
for cfg in "${CONFIGS[@]}"; do
  read -r TOTAL WPG <<<"$cfg"
  CMD="$BASE && python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model $MODEL --model-path $MPATH --calib-dir $CALIB --cells-json $SCRATCH/w${TOTAL}_cells.json --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu $WPG --no-raw --inject-from-metadata"
  AFTER=(); [[ -n $PREV ]] && AFTER=(--after "$PREV")
  JID=$(heimdall submit "bash -c '$CMD'" -n "annex_scaling_w$TOTAL" --node node1 -g 8 -e 25 \
        -w "$HEIMDALL_WORK_DIR" --env HF_HUB_OFFLINE=1 "${AFTER[@]}" | grep -oE '[0-9a-f]{12}' | head -1)
  echo "annex_scaling_w$TOTAL (wpg=$WPG) -> job $JID${PREV:+ [after $PREV]}"
  PREV=$JID
done
echo "worker logs -> $SCRATCH/w*/_multicell_replay_jobs/"
