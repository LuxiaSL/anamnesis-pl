#!/usr/bin/env bash
# Part-C battery replays (anamnesis s12). node1. GPU — fires on Luxia's word, after training.
# Ruling 1: match the banked reference run EXACTLY — same manifest (probe160, md5 stamped),
# same per-probe multiplicity (160 probes -> 320 files = .npz sig + .json meta per probe).
# LoRA arms (3b sgd-mom, 4 dpo cat/blue) -> run_replay_multickpt swap driver.
# Full-FT (3a) -> load-per-checkpoint (run_replay_extraction --model-path=<ckpt>), NOT the swap.
set -euo pipefail
ARM="${1:?usage: partc_replay.sh <arm> [gpus]   arm in {sgdmom,dpo_cat,dpo_blue,fullft}}"
GPUS="${2:-1,2,3,4,5,6,7}"          # ops rider: EXCLUDE GPU 0 (other user's job)

PIPE=~/luxi-files/anamnesis-pl/pipeline
CKROOT=/models/subliminal-anamnesis
MANIFEST=/models/anamnesis-extract/battery/arms/A6/cohort/probe160_manifest.json
MANIFEST_MD5=b010fdbc31902c55706fb89ebcf8e1c1
CALIB=/models/anamnesis-extract/calibration/qwen25_7b
BASE=Qwen/Qwen2.5-7B-Instruct
RUNROOT=/models/anamnesis-extract/runs/vmb_a6cohort_qwen
CELLS=/tmp/claude-output/partc_cells
mkdir -p "$CELLS"
cd "$PIPE"
source ~/luxi-files/.venv-shared/bin/activate

# Manifest identity gate (ruling 1): refuse if the battery of record drifted.
GOT=$(md5sum "$MANIFEST" | cut -d' ' -f1)
[ "$GOT" = "$MANIFEST_MD5" ] || { echo "STOP-AND-SURFACE: manifest md5 $GOT != $MANIFEST_MD5"; exit 3; }
echo "[replay] manifest OK ($MANIFEST_MD5, 160 probes) | compute_node: node1 | gpus $GPUS"

case "$ARM" in
  sgdmom)   CKDIR="$CKROOT/checkpoints/qwen_cat_sgdmom15e1_s0"; LABEL=cat_sgdmom15e1_s0; SWAP=1;;
  dpo_cat)  CKDIR="$CKROOT/partc_cell4/checkpoints/qwen_cat_dpo_r16_s0"; LABEL=cat_dpo_r16_s0; SWAP=1;;
  dpo_blue) CKDIR="$CKROOT/partc_cell4/checkpoints/qwen_blue_dpo_r16_s0"; LABEL=blue_dpo_r16_s0; SWAP=1;;
  fullft)   CKDIR="$CKROOT/checkpoints/qwen_cat_v3_fullft_s0"; LABEL=cat_fullft_s0; SWAP=0;;
  *) echo "unknown arm $ARM"; exit 2;;
esac

if [ "$SWAP" = 1 ]; then
  OUT="$CELLS/cells_${LABEL}.json"
  PYTHONPATH=. python -m anamnesis.scripts.build_partc_replay_cells \
    --ckpt-dir "$CKDIR" --arm "$LABEL" --run-root "$RUNROOT" --out "$OUT"
  echo "[replay] LoRA swap driver: $LABEL"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python -m anamnesis.scripts.run_replay_multickpt \
    --model qwen-7b --model-path "$BASE" --calib-dir "$CALIB" \
    --manifest "$MANIFEST" --checkpoints-json "$OUT" \
    --gpus "$GPUS" --workers-per-gpu 4 --sig-subdir signatures_v3 \
    2>&1 | tee /tmp/claude-output/partc_replay_${LABEL}.log | tail -6
else
  # Full-weight: no adapter swap. load-per-checkpoint (base = the checkpoint dir itself).
  # run_replay_extraction is single-cell (no --gpus); --no-raw MANDATORY (its default banks raws,
  # unlike the multickpt driver which hardcodes no_raw). Parallelize checkpoints across the GPU
  # set — one full 7B load per GPU (~15GB on 183GB cards), then wait.
  echo "[replay] FULL-FT load-per-checkpoint: $LABEL (--no-raw; checkpoints sharded across $GPUS)"
  IFS=',' read -ra GARR <<< "$GPUS"; gi=0; pids=()
  for CK in "$CKDIR"/checkpoint-* "$CKDIR"/final; do
    [ -d "$CK" ] || continue
    STEP=$(basename "$CK" | sed 's/checkpoint-/step-/')
    RUN="$RUNROOT/$LABEL/$STEP"
    G=${GARR[$((gi % ${#GARR[@]}))]}; gi=$((gi+1))
    echo "  -> $STEP on GPU $G"
    CUDA_VISIBLE_DEVICES="$G" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
      python -m anamnesis.scripts.run_replay_extraction \
        --model qwen-7b --model-path "$CK" --calib-dir "$CALIB" \
        --manifest "$MANIFEST" --run-dir "$RUN" --sig-subdir signatures_v3 --no-raw \
        > /tmp/claude-output/partc_replay_${LABEL}_${STEP}.log 2>&1 &
    pids+=($!)
    # cap concurrency at the GPU-set size
    [ ${#pids[@]} -ge ${#GARR[@]} ] && { wait "${pids[0]}"; pids=("${pids[@]:1}"); }
  done
  wait
fi
echo "[replay] $LABEL DONE. sigs under $RUNROOT/$LABEL/*/signatures_v3 (160 probes/step)."
