#!/usr/bin/env bash
# ANNEX 8B 2×2 — STAGE 1: the two autograd pulses (queue item 5, desk-authorized
# 2026-07-17; runs LAST in the queue). ⛔ GATED on Luxia's GPU word.
# Per PRICING-8b-2x2-2026-07-16.md (8B book frozen therein: .65/.80/.50):
#   pulse 1 = V4-8B (attention gradient @ L16, vmb_a5_build_v4_gradient --map-site 16)
#   pulse 2 = V7-8B raw entropy gradient (vmb_v4_b7b4_stage1 --site 16, site-parameterized
#             2026-07-17; Σ_L16 + V3_L16 banks verified on node)
# Then LOCAL CPU builds (b7 stage2 --site 16 → V7-8B + Rband-8B×3; RA-8B = unit(P[16:256]
# ·V4-8B) via annex_band_pass.py) + rider-1-style filings (anatomy + predictions BEFORE
# steering) → annex_8b_steer.sh (double-gated on that filing).
set -euo pipefail

: "${HEIMDALL_WORK_DIR:?}"
: "${HEIMDALL_VENV:?}"

REPO="$HOME/projects/anamnesis_exps"
MODEL=8b
MPATH=/models/llama-3.1-8b-instruct
RUNS=/models/anamnesis-extract/runs
STAGE0=$RUNS/vmb_stage0_8b
SIGMA=/models/anamnesis-extract/battery/arms/A5/a5_sigma_L16_8b.npz
VEC8B=/models/anamnesis-extract/battery/a5_vectors_8b/a5_vectors.npz
OUT=/models/anamnesis-extract/battery/annex/8b_pulses
BASE="source $HEIMDALL_VENV && cd $HEIMDALL_WORK_DIR && export PYTHONPATH=\$PWD/pipeline PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1"

submit() { # name, cmd, gpus, minutes, [after]  — hardened pattern (loud on failure)
  local AFTER=(); [[ -n ${5:-} ]] && AFTER=(--after "$5")
  local OUT_
  OUT_=$(heimdall submit "bash -c '$BASE && $2'" -n "$1" --node node1 -g "$3" -e "$4" \
    -w "$HEIMDALL_WORK_DIR" --env HF_HUB_OFFLINE=1 --max-retries 0 "${AFTER[@]}" 2>&1) \
    || { echo "SUBMIT FAILED for $1:" >&2; echo "$OUT_" >&2; exit 1; }
  echo "$OUT_" >&2
  grep -oE '[0-9a-f]{12}' <<< "$OUT_" | head -1
}

# self-deploy (the --site parameterization must be on the node)
rsync -a "$REPO/pipeline/anamnesis/scripts/" node1:luxi-files/anamnesis-pl/pipeline/anamnesis/scripts/ \
  --include='*.py' --exclude='ops/*' --exclude='__pycache__'
ssh node1 "mkdir -p $OUT"

P1=$(submit vmb-8b-v4-pulse \
  "python -u -m anamnesis.scripts.vmb_a5_build_v4_gradient --model $MODEL --model-path $MPATH --stage0-run $STAGE0 --out-dir $OUT --map-site 16 --n-gens 20" \
  1 30)
echo "vmb-8b-v4-pulse -> $P1"

P2=$(submit vmb-8b-b7-pulse \
  "python -u -m anamnesis.scripts.vmb_v4_b7b4_stage1 --model $MODEL --model-path $MPATH --stage0-run $STAGE0 --sigma $SIGMA --vectors $VEC8B --out-dir $OUT --site 16 --n-gens 20" \
  1 40 "$P1")
echo "vmb-8b-b7-pulse -> $P2 [after $P1 — same card, sequential]"

echo
echo "PULSES QUEUED. After they land:"
echo "  rsync -a node1:$OUT/ $REPO/outputs/battery/annex/8b_pulses/"
echo "  then LOCAL: b7 stage2 --site 16 (V7-8B + Rband-8B×3) + annex_band_pass.py (RA-8B)"
echo "  + rider-1 filing (PREDICTIONS-8b-rider1.md) BEFORE annex_8b_steer.sh."
