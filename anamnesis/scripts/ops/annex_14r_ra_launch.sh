#!/usr/bin/env bash
# ANNEX 14r cell R-A STEERING BLOCK (staged session 8, 2026-07-16). ⛔ LUXIA-GATED (rider 5):
# do not run until Luxia's GPU word. Prereg-exact block, submit_b7_stage2 pattern with ONE
# object changed (the vector):
#
#   smoke    write-path gate: alpha=0 injection == hookless replay EXACTLY + dose-monotone
#            (vmb_smoke_write_hooks; exits 1 on fail -> chain halts)         [rider 3]
#   gen      RA_L14 x alpha{.03,.1,.3}, n=40/cell, inject L14, C§3 convention
#   rep      replay + signatures_v3 per cell (--inject-from-metadata)
#   entropy  logit-side entropy rows for RA cells (RA-3 channel); reference rows = the
#            BANKED 14j_leg2_entropy_V7_3b_corrected.json (echo, never recompute)
#
# Readout does NOT run on node (rider 2: banked analyzers only, run locally after pull-back).
# Seed namespace VMB14R-3B-{cell}: per-cell unique, convention-matched to the b7 grid.
# All legs --max-retries 0 (non-idempotent; ops-runbook law). Requires: HEIMDALL_WORK_DIR,
# HEIMDALL_VENV exported; heimdall CLI on PATH.
#
# Pull-back after the chain lands (run from repo root):
#   rsync -a node1:/models/anamnesis-extract/runs/vmb_14r_3b/ outputs/runs/vmb_14r_3b/
#   rsync -a node1:/models/anamnesis-extract/battery/annex/14r_ra_entropy_3b.json outputs/battery/annex/
set -euo pipefail

: "${HEIMDALL_WORK_DIR:?}"
: "${HEIMDALL_VENV:?}"

MODEL=3b
MPATH=/models/llama-3.2-3b-instruct
BATTERY=/models/anamnesis-extract/battery
RUNS=/models/anamnesis-extract/runs
CALIB=/models/anamnesis-extract/calibration/3b
STAGE0=$RUNS/vmb_stage0_3b
VEC_DIR=$BATTERY/annex/a5_vectors_3b_14r
RUN_ROOT=$RUNS/vmb_14r_3b
PROMPTS=pipeline/anamnesis/prompts/prompt_sets.json
KEY=RA_L14
LADDER=(0.03 0.1 0.3)
BASE="source $HEIMDALL_VENV && cd $HEIMDALL_WORK_DIR && export PYTHONPATH=\$PWD/pipeline PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1"

# ── staging: push the R-A vector bank to the node (idempotent, new files only) ──
ssh node1 "mkdir -p $VEC_DIR $RUN_ROOT"
rsync -a "$HOME/projects/anamnesis_exps/outputs/battery/annex/a5_vectors_3b_14r/" \
         "node1:$VEC_DIR/"
echo "staged R-A vector bank -> node1:$VEC_DIR/"

submit() { # name, cmd, gpus, minutes, [after]
  local AFTER=(); [[ -n ${5:-} ]] && AFTER=(--after "$5")
  heimdall submit "bash -c '$BASE && $2'" -n "$1" --node node1 -g "$3" -e "$4" \
    -w "$HEIMDALL_WORK_DIR" --env HF_HUB_OFFLINE=1 --max-retries 0 "${AFTER[@]}" \
    | grep -oE '[0-9a-f]{12}' | head -1
}

join_and() { # join args with ' && ' ("${arr[*]}" only honors IFS's FIRST char — real bug class)
  local out=$1; shift
  for c in "$@"; do out+=" && $c"; done
  printf '%s' "$out"
}

SMOKE=$(submit vmb-14r-ra-smoke \
  "python -u -m anamnesis.scripts.vmb_smoke_write_hooks --model $MODEL --model-path $MPATH --floor-run-dir $STAGE0 --calib-dir $CALIB --out /dev/shm/vmb_14r_smoke" \
  1 10)
echo "vmb-14r-ra-smoke -> $SMOKE"

GEN_CMDS=()
for A in "${LADDER[@]}"; do
  C=${KEY}_a${A}
  GEN_CMDS+=("python -u -m anamnesis.scripts.vmb_stage0_generate --model $MODEL --model-path $MPATH --prompts $PROMPTS --out-run-dir $RUN_ROOT/$C --gpus 0,1,2,3 --workers-per-gpu 4 --seeds-per-class 1 --limit 40 --seed-namespace VMB14R-3B-$C --inject-npz $VEC_DIR/a5_vectors.npz --inject-key $KEY --inject-layer 14 --inject-alpha-frac $A --inject-norms-json $VEC_DIR/a5_vectors_stamps.json")
done
GEN=$(submit vmb-14r-ra-gen "$(join_and "${GEN_CMDS[@]}")" 4 25 "$SMOKE")
echo "vmb-14r-ra-gen -> $GEN [after $SMOKE]"

REP_CMDS=()
for A in "${LADDER[@]}"; do
  C=${KEY}_a${A}
  REP_CMDS+=("python -u -m anamnesis.scripts.parallel_replay --model $MODEL --model-path $MPATH --run-dir $RUN_ROOT/$C --calib-dir $CALIB --manifest $RUN_ROOT/$C/replay_manifest.json --gpus 0,1 --workers-per-gpu 3 --no-raw --inject-from-metadata")
done
REP=$(submit vmb-14r-ra-rep "$(join_and "${REP_CMDS[@]}")" 2 25 "$GEN")
echo "vmb-14r-ra-rep -> $REP [after $GEN]"

CELLS="${KEY}_a0.03 ${KEY}_a0.1 ${KEY}_a0.3"
ENT=$(submit vmb-14r-ra-entropy \
  "python -u -m anamnesis.scripts.vmb_c3_entropy_replay --model $MODEL --model-path $MPATH --c3-run-dir $RUN_ROOT --cells $CELLS --null-prefixes RBAND --out-json $BATTERY/annex/14r_ra_entropy_3b.json" \
  1 25 "$REP")
echo "vmb-14r-ra-entropy -> $ENT [after $REP]"

echo
echo "CHAIN QUEUED: smoke -> gen -> rep -> entropy   (readout = LOCAL banked analyzers only)"
echo "cells land in $RUN_ROOT/; entropy JSON in $BATTERY/annex/"
