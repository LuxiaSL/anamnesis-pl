#!/usr/bin/env bash
# Part-C battery replays (anamnesis s12). node1. GPU — fires on Luxia's word, after training.
# Ruling 1: match the banked reference run EXACTLY — same manifest (probe160, md5 stamped),
# same per-probe multiplicity (160 probes -> 320 files = .npz sig + .json meta per probe).
# LoRA arms (3b sgd-mom, 4 dpo cat/blue) -> run_replay_multickpt swap driver.
# Full-FT (3a) -> load-per-checkpoint (run_replay_extraction --model-path=<ckpt>), NOT the swap.
set -euo pipefail
ARM="${1:?usage: partc_replay.sh <arm>   arm in {sgdmom,dpo_cat,dpo_purple,dpo_catnum,fullft}}"
ARM="${ARM//[^a-z_]/}"              # strip any stray whitespace/CR the scheduler appended
# Runs UNDER HEIMDALL: it exports CUDA_VISIBLE_DEVICES = the assigned PHYSICAL GPUs (from
# --gpu-ids, GPU 0 excluded at submit). Derive the driver's LOGICAL slot list (0..N-1) from it;
# the multickpt driver's resolve_physical_gpus maps slots->physical. Bare-metal fallback = arg 2.
CVD="${CUDA_VISIBLE_DEVICES:-}"
if [ -n "$CVD" ]; then
  IFS=',' read -ra PHYS <<< "$CVD"; N=${#PHYS[@]}
  GPUS=$(seq -s, 0 $((N-1)))       # logical slots for the swap driver
else
  GPUS="${2:-1,2,3,4,5,6}"; IFS=',' read -ra PHYS <<< "$GPUS"
fi

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

# dpo_all: replay all 3 DPO LoRA arms in ONE multickpt job (base loaded once per worker, swaps
# through cat+purple+catnum checkpoints), spread across ALL assigned GPUs x WPG workers. Each
# worker holds base + a pristine-restore snapshot (~22GB) => cap ~7 workers/GPU on a 183GB card.
if [ "$ARM" = "dpo_all" ] || [ "$ARM" = "refpair" ]; then
  if [ "$ARM" = "dpo_all" ]; then
    SPECS=("partc_cell4:qwen_cat_dpo_r16_s0:cat_dpo_r16_s0"
           "partc_cell4:qwen_purple_dpo_r16_s0:purple_dpo_r16_s0"
           "partc_cell4n:qwen_catnum_dpo_r16_s0:catnum_dpo_r16_s0")
    LABELS=(cat_dpo_r16_s0 purple_dpo_r16_s0 catnum_dpo_r16_s0); TAG=dpo_all
  else   # refpair: the DECIDING LEG — reference student + matched format-control student
    SPECS=(":qwen_cat_student:cat_student_s0" ":qwen_control_a:control_a_s0")
    LABELS=(cat_student_s0 control_a_s0); TAG=refpair
  fi
  for spec in "${SPECS[@]}"; do
    sub="${spec%%:*}"; r="${spec#*:}"; name="${r%%:*}"; label="${r##*:}"
    dir="$CKROOT/checkpoints/$name"; [ -n "$sub" ] && dir="$CKROOT/$sub/checkpoints/$name"
    PYTHONPATH=. python -m anamnesis.scripts.build_partc_replay_cells \
      --ckpt-dir "$dir" --arm "$label" --run-root "$RUNROOT" --out "$CELLS/cells_$label.json"
  done
  python - "$CELLS" "$TAG" "${LABELS[@]}" <<'PY'
import json, sys
d, tag, labels = sys.argv[1], sys.argv[2], sys.argv[3:]
ck = []
for lab in labels:
    ck += json.load(open(f"{d}/cells_{lab}.json"))["checkpoints"]
json.dump({"checkpoints": ck}, open(f"{d}/cells_{tag}.json", "w"), indent=1)
print(f"combined {len(ck)} checkpoints: {labels}")
PY
  echo "[replay] $TAG: ${#LABELS[@]} arms, all GPUs ($GPUS) x ${WPG:-5} workers"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python -m anamnesis.scripts.run_replay_multickpt \
    --model qwen-7b --model-path "$BASE" --calib-dir "$CALIB" \
    --manifest "$MANIFEST" --checkpoints-json "$CELLS/cells_$TAG.json" \
    --gpus "$GPUS" --workers-per-gpu "${WPG:-5}" --sig-subdir signatures_v3 \
    2>&1 | tee /tmp/claude-output/partc_replay_$TAG.log | tail -8
  echo "[replay] $TAG DONE."
  exit 0
fi

case "$ARM" in
  sgdmom)     CKDIR="$CKROOT/checkpoints/qwen_cat_sgdmom15e1_s0"; LABEL=cat_sgdmom15e1_s0; SWAP=1;;
  dpo_cat)    CKDIR="$CKROOT/partc_cell4/checkpoints/qwen_cat_dpo_r16_s0"; LABEL=cat_dpo_r16_s0; SWAP=1;;
  dpo_purple) CKDIR="$CKROOT/partc_cell4/checkpoints/qwen_purple_dpo_r16_s0"; LABEL=purple_dpo_r16_s0; SWAP=1;;
  dpo_catnum) CKDIR="$CKROOT/partc_cell4n/checkpoints/qwen_catnum_dpo_r16_s0"; LABEL=catnum_dpo_r16_s0; SWAP=1;;
  fullft)     CKDIR="$CKROOT/checkpoints/qwen_cat_v3_fullft_s0"; LABEL=cat_fullft_s0; SWAP=0;;
  *) echo "unknown arm $ARM"; exit 2;;
esac

if [ "$SWAP" = 1 ]; then
  OUT="$CELLS/cells_${LABEL}.json"
  PYTHONPATH=. python -m anamnesis.scripts.build_partc_replay_cells \
    --ckpt-dir "$CKDIR" --arm "$LABEL" --run-root "$RUNROOT" --out "$OUT"
  echo "[replay] LoRA swap driver: $LABEL"
  # workers-per-gpu: replay extraction is CPU-bound (numpy state_extractor); each worker holds
  # its own base copy (~15GB), so VRAM caps it at ~11/GPU on a 183GB B200. WPG env overrides.
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python -m anamnesis.scripts.run_replay_multickpt \
    --model qwen-7b --model-path "$BASE" --calib-dir "$CALIB" \
    --manifest "$MANIFEST" --checkpoints-json "$OUT" \
    --gpus "$GPUS" --workers-per-gpu "${WPG:-8}" --sig-subdir signatures_v3 \
    2>&1 | tee /tmp/claude-output/partc_replay_${LABEL}.log | tail -6
else
  # Full-weight: no adapter swap. load-per-checkpoint (base = the checkpoint dir itself).
  # run_replay_extraction is single-cell (no --gpus); --no-raw MANDATORY (its default banks raws,
  # unlike the multickpt driver which hardcodes no_raw). Parallelize checkpoints across the GPU
  # set — one full 7B load per GPU (~15GB on 183GB cards), then wait.
  # Shard checkpoints across the PHYSICAL GPUs Heimdall assigned (PHYS[]). Setting CVD to a
  # physical index is correct here — CVD does NOT compose across nested processes, so logical
  # remapping would collide (the _gpu.py lesson). run_replay_extraction is single-cell (no --gpus).
  echo "[replay] FULL-FT load-per-checkpoint: $LABEL (--no-raw; sharded across phys ${PHYS[*]})"
  gi=0; pids=()
  for CK in "$CKDIR"/checkpoint-* "$CKDIR"/final; do
    [ -d "$CK" ] || continue
    STEP=$(basename "$CK" | sed 's/checkpoint-/step-/')
    RUN="$RUNROOT/$LABEL/$STEP"
    G=${PHYS[$((gi % ${#PHYS[@]}))]}; gi=$((gi+1))
    echo "  -> $STEP on physical GPU $G"
    CUDA_VISIBLE_DEVICES="$G" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
      python -m anamnesis.scripts.run_replay_extraction \
        --model qwen-7b --model-path "$CK" --calib-dir "$CALIB" \
        --manifest "$MANIFEST" --run-dir "$RUN" --sig-subdir signatures_v3 --no-raw \
        > /tmp/claude-output/partc_replay_${LABEL}_${STEP}.log 2>&1 &
    pids+=($!)
    # cap concurrency at the GPU-set size
    [ ${#pids[@]} -ge ${#PHYS[@]} ] && { wait "${pids[0]}"; pids=("${pids[@]:1}"); }
  done
  wait
fi
echo "[replay] $LABEL DONE. sigs under $RUNROOT/$LABEL/*/signatures_v3 (160 probes/step)."
