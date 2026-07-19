#!/usr/bin/env bash
# ANNEX 8B 2×2 — STAGE 1 pulses, NODE2 variant (Heimdall, LOW PRIORITY; overnight).
# Pulls the two 8B prerequisites over IB first (stage0 manifest + the L16 vector bank —
# targeted files only, per NODE2-OPS rule 4). Pulses: V4-8B (map-site 16) + b7-8B
# (--site 16). Steer stays gated on the rider-1 filing (anatomy first, tomorrow-or-tonight
# by the annex after these land).
set -euo pipefail

: "${HEIMDALL_WORK_DIR:?}"
: "${HEIMDALL_VENV:?}"
PRIO=10

SHM=/dev/shm/luxi-anamnesis
DATA=$SHM/anamnesis-extract
M8B=/models/llama-3.1-8b-instruct
STAGE0=$DATA/runs/vmb_stage0_8b
SIGMA=$SHM/banks/sigma/a5_sigma_L16_8b.npz
VEC8B=$SHM/banks/a5_vectors_8b/a5_vectors.npz
OUT=$DATA/battery/annex/8b_pulses
IB="${NODE2_IB:?set NODE2_IB=<user>@<node2-ib-addr> (sanitized: account/IP not hardcoded)}"
BASE="source $HEIMDALL_VENV && cd $HEIMDALL_WORK_DIR && export PYTHONPATH=\$PWD/pipeline PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1"

submit() {
  local AFTER=(); [[ -n ${5:-} ]] && AFTER=(--after "$5")
  local OUT_
  OUT_=$(heimdall submit "bash -c '$BASE && $2'" -n "$1" --node node2 -g "$3" -e "$4" \
    -w "$HEIMDALL_WORK_DIR" --env HF_HUB_OFFLINE=1 --priority $PRIO --max-retries 2 \
    "${AFTER[@]}" 2>&1) \
    || { echo "SUBMIT FAILED for $1:" >&2; echo "$OUT_" >&2; exit 1; }
  echo "$OUT_" >&2
  grep -oE '[0-9a-f]{12}' <<< "$OUT_" | head -1
}

# targeted IB pulls (idempotent)
ssh node2 "mkdir -p $STAGE0 $SHM/banks/a5_vectors_8b $OUT && \
  rsync -a $IB:/models/anamnesis-extract/runs/vmb_stage0_8b/replay_manifest.json $STAGE0/ && \
  rsync -a $IB:/models/anamnesis-extract/battery/a5_vectors_8b/a5_vectors.npz $IB:/models/anamnesis-extract/battery/a5_vectors_8b/a5_vectors_stamps.json $SHM/banks/a5_vectors_8b/ && \
  ls $STAGE0/ $SHM/banks/a5_vectors_8b/"

P1=$(submit vmb-8b-v4-pulse-n2 \
  "python -u -m anamnesis.scripts.vmb_a5_build_v4_gradient --model 8b --model-path $M8B --stage0-run $STAGE0 --out-dir $OUT --map-site 16 --n-gens 20" \
  1 40)
echo "vmb-8b-v4-pulse-n2 -> $P1"

P2=$(submit vmb-8b-b7-pulse-n2 \
  "python -u -m anamnesis.scripts.vmb_v4_b7b4_stage1 --model 8b --model-path $M8B --stage0-run $STAGE0 --sigma $SIGMA --vectors $VEC8B --out-dir $OUT --site 16 --n-gens 20" \
  1 50 "$P1")
echo "vmb-8b-b7-pulse-n2 -> $P2 [after $P1]"

echo
echo "8B PULSES QUEUED at priority $PRIO. After landing: pull $OUT local, CPU builds"
echo "(b7 stage2 --site 16 + RA-8B band-pass) + rider-1 filing, THEN the steer chain."
echo "FINAL_JOB=$P2"
