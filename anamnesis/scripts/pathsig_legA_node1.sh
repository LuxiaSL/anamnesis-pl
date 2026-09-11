#!/usr/bin/env bash
# LEG 1 (CPU, 0 GPU) — project the banked 8B v3 raw residual at L16 onto the per-layer
# calibration PCA. Node-side launcher; submitted through heimdall (see the report's job ids).
# Paths are node1-local by design: the `cd` fails fast if a 0-GPU job lands on the wrong node
# (HEIMDALL-OPERATING-GUIDE §4.7).
set -euo pipefail

ROOT=/home/athuser/luxi-files/anamnesis-pathsig
OUT=${OUT:-/models/anamnesis-extract/pathsig/8b_L16}
WORKERS=${WORKERS:-16}

cd "$ROOT"
mkdir -p "$ROOT/logs" "$OUT"
# shellcheck disable=SC1091
source /home/athuser/luxi-files/.venv-shared/bin/activate

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python -m anamnesis.scripts.pathsig_project_residual \
  --model 8b \
  --runs 8b_fat_01,8b_fat_ext \
  --layer 16 --k 8 \
  --runs-root /models/anamnesis-extract/runs \
  --calib-dir /models/anamnesis-extract/calibration/8b \
  --pca-file pca_model_corrected.pkl \
  --out-dir "$OUT" \
  --workers "$WORKERS" 2>&1 | tee "$ROOT/logs/pathsig_legA.log"

echo "PATHSIG_LEG1_COMPLETE"
