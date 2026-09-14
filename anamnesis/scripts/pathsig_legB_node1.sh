#!/usr/bin/env bash
# LEG 2 (1 GPU) — regenerate the §5.1 resolver cell by teacher-forced replay, run the parity
# gate, and take the E2 three-way read. Node-side launcher; submitted through heimdall with
# `--gpus 1` (the scheduler owns CUDA_VISIBLE_DEVICES — never set it here).
#
# The injection spec is NOT re-derived: it is transcribed verbatim from the cell's own July
# replay log (/models/anamnesis-extract/runs/vmb_a5_s51_3b/V2_steered_a003/replay_logs/),
#   'replay injection active: {inject_npz: .../a5_vectors_3b/a5_vectors.npz,
#    inject_key: V2_L13, inject_layer: 13, inject_alpha: 0.3276513576507568,
#    inject_alpha_frac: 0.03}'
set -euo pipefail

ROOT=/home/athuser/luxi-files/anamnesis-pathsig
OUT=${OUT:-$ROOT/out/s51}
NGENS=${NGENS:-160}

cd "$ROOT"
mkdir -p "$ROOT/logs" "$OUT"
# shellcheck disable=SC1091
source /home/athuser/luxi-files/.venv-shared/bin/activate

export PYTHONPATH=.
export PYTHONUNBUFFERED=1

python -m anamnesis.scripts.pathsig_s51_regen \
  --model 3b \
  --model-path /models/llama-3.2-3b-instruct \
  --calib-dir /models/anamnesis-extract/calibration/3b \
  --stage0-run /models/anamnesis-extract/runs/vmb_stage0_3b \
  --unsteered-run /models/anamnesis-extract/runs/vmb_a5_s51_3b/unsteered \
  --steered-run /models/anamnesis-extract/runs/vmb_a5_s51_3b/V2_steered_a003 \
  --inject-npz /models/anamnesis-extract/battery/a5_vectors_3b/a5_vectors.npz \
  --inject-key V2_L13 --inject-layer 13 \
  --inject-alpha 0.3276513576507568 --inject-alpha-frac 0.03 \
  --site-layer 14 --k 8 \
  --gen-stride 5 --n-gens "$NGENS" \
  --out-dir "$OUT" --device cuda:0 2>&1 | tee "$ROOT/logs/pathsig_legB.log"
