#!/usr/bin/env bash
# ANNEX probe-program PREREQUISITE (queue item 4, desk-authorized 2026-07-17):
# bank residual-Σ eigendecomps at L7 + L21 (L14 already banked) — one model load,
# one card, ~minutes. ⛔ GATED on Luxia's GPU word. Banked-for-reuse instrument
# infrastructure per PROGRAM-potential-probes-2026-07-16.md; the 12 probes themselves
# run later (each behind its own frozen response-class filing).
#
# The --vectors bank is incidental here (all its keys live at L14, so no screen rows
# emit at 7/21); the deliverable is the two a5_sigma_L{7,21}_3b.npz banks, pulled back
# local at the end for the CPU-side band-pass legs.
set -euo pipefail

: "${HEIMDALL_WORK_DIR:?}"
: "${HEIMDALL_VENV:?}"

REPO="$HOME/projects/anamnesis_exps"
MODEL=3b
MPATH=/models/llama-3.2-3b-instruct
RUNS=/models/anamnesis-extract/runs
STAGE0=$RUNS/vmb_stage0_3b
VEC=/models/anamnesis-extract/battery/annex/eosrep_vectors_3b/a5_vectors.npz
OUT=/models/anamnesis-extract/battery/arms/A5
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

# self-deploy (the multi-site --save-sigma-site change must be on the node)
rsync -a "$REPO/pipeline/anamnesis/scripts/" node1:luxi-files/anamnesis-pl/pipeline/anamnesis/scripts/ \
  --include='*.py' --exclude='ops/*' --exclude='__pycache__'

SIG=$(submit vmb-probe-sigma-l7l21 \
  "python -u -m anamnesis.scripts.vmb_a5_covariance_screen --model $MODEL --model-path $MPATH --stage0-run $STAGE0 --vectors $VEC --out-dir $OUT --sites 7,21 --save-sigma-site 7,21 --n-gens 60" \
  1 25)
echo "vmb-probe-sigma-l7l21 -> $SIG"
echo
echo "After it lands: rsync -a node1:$OUT/a5_sigma_L7_3b.npz node1:$OUT/a5_sigma_L21_3b.npz \\"
echo "  $REPO/outputs/battery/arms/A5/   (then the probe pulses + per-probe freezes are CPU-stageable)"
