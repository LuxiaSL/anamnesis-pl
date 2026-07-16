#!/usr/bin/env bash
# ANNEX route-5 LAUNCH CHAIN (staged session 7, 2026-07-16). ⛔ RATIFICATION-GATED:
# do not run until the prereg cell is ratified. Fires the whole program in order:
#
#   smoke     bitwise gate vs the banked MT record (1 GPU, ~5 min) — MUST pass
#   killrung  short null search, budget/10, shufnull draw 0     — can kill cheap
#   cold      the primary: full-dim sep-CMA-ES on the dir0 gauge (10·d)
#   null_a/b  matched-budget null searches, shufnull draws 0/1  — mandatory
#   refine    variant (ii): init at V4, 1·d budget              — diagnostic
#
# Each leg = one Heimdall job, chained with --after (a failed leg halts the chain).
# w96 overnight config (Luxia 2026-07-16: box is ours overnight). Certification
# (the 160-gen behavioral cell) is NOT in this chain — it runs after a first-read
# of summary.json, as a separately ratified cell.
#
# Requires: HEIMDALL_WORK_DIR, HEIMDALL_VENV exported; heimdall CLI on PATH; the
# split + axis npz files rsynced to the node battery (staging step below).
set -euo pipefail

: "${HEIMDALL_WORK_DIR:?}"
: "${HEIMDALL_VENV:?}"

BATTERY=/models/anamnesis-extract/battery
RUNS=/models/anamnesis-extract/runs
SEARCH_ROOT="${SEARCH_ROOT:-$RUNS/annex_route5_3b}"
MANIFEST=$RUNS/vmb_stage0_3b/replay_manifest.json
SPLIT=$BATTERY/annex/annex_route5_split_3b.json
SEED="${SEED:-20260717}"
WPG="${WPG:-12}"   # w96 = 8 GPUs x 12
# STOP_AFTER_LEG lets a window fire only the cheap legs (smoke|killrung) and hold the
# expensive cold/null/refine legs for a later window (Luxia sequencing ruling 2026-07-16:
# smoke+killrung ride tonight; cold search absorbs the tail RESUMABLE). Default = full chain.
STOP_AFTER_LEG="${STOP_AFTER_LEG:-refine}"   # smoke|killrung|cold|null_a|null_b|refine
BASE="source $HEIMDALL_VENV && cd $HEIMDALL_WORK_DIR && export PYTHONPATH=\$PWD/pipeline PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1"
DRIVER="python -u -m anamnesis.scripts.annex_route5_driver --battery-root $BATTERY --runs-root $RUNS --manifest $MANIFEST --split-json $SPLIT --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu $WPG"

# ── staging: push the banked annex inputs to the node (idempotent) ──
ssh node1 "mkdir -p $BATTERY/annex $SEARCH_ROOT"
rsync -a "$HOME/projects/anamnesis_exps/outputs/battery/annex/annex_dir0_axis_3b.npz" \
         "$HOME/projects/anamnesis_exps/outputs/battery/annex/annex_dir0_shufnull_axis_3b.npz" \
         "$HOME/projects/anamnesis_exps/outputs/battery/annex/annex_route5_split_3b.json" \
         "node1:$BATTERY/annex/"
echo "staged annex inputs -> node1:$BATTERY/annex/"

submit() { # name, cmd, gpus, minutes, [after]
  local AFTER=(); [[ -n ${5:-} ]] && AFTER=(--after "$5")
  heimdall submit "bash -c '$BASE && $2'" -n "$1" --node node1 -g "$3" -e "$4" \
    -w "$HEIMDALL_WORK_DIR" --env HF_HUB_OFFLINE=1 "${AFTER[@]}" | grep -oE '[0-9a-f]{12}' | head -1
}

SMOKE=$(submit route5_smoke \
  "python -u -m anamnesis.scripts.annex_route5_worker --smoke --model 3b --model-path /models/llama-3.2-3b-instruct --calib-dir /models/anamnesis-extract/calibration/3b --battery-root $BATTERY --runs-root $RUNS" \
  1 15)
echo "route5_smoke -> $SMOKE"
[[ $STOP_AFTER_LEG == smoke ]] && { echo "STOP_AFTER_LEG=smoke -> chain halts (smoke only)"; exit 0; }

KILL=$(submit route5_killrung \
  "$DRIVER --mode killrung --null-draw 0 --seed $SEED --work-dir $SEARCH_ROOT/killrung" \
  8 60 "$SMOKE")
echo "route5_killrung -> $KILL [after $SMOKE]"
[[ $STOP_AFTER_LEG == killrung ]] && {
  echo "STOP_AFTER_LEG=killrung -> chain halts after legs 0-1 (cold/nulls/refine held for a later window)"
  echo "resume the tail with: STOP_AFTER_LEG=refine SEARCH_ROOT=$SEARCH_ROOT $0   (cold leg is --resume-safe)"
  exit 0
}

COLD=$(submit route5_cold \
  "$DRIVER --mode cold --seed $SEED --work-dir $SEARCH_ROOT/cold" \
  8 360 "$KILL")
echo "route5_cold -> $COLD [after $KILL]"

NULLA=$(submit route5_null_a \
  "$DRIVER --mode null --null-draw 0 --seed $SEED --work-dir $SEARCH_ROOT/null_a" \
  8 360 "$COLD")
echo "route5_null_a -> $NULLA [after $COLD]"

NULLB=$(submit route5_null_b \
  "$DRIVER --mode null --null-draw 1 --seed $SEED --work-dir $SEARCH_ROOT/null_b" \
  8 360 "$NULLA")
echo "route5_null_b -> $NULLB [after $NULLA]"

REFINE=$(submit route5_refine \
  "$DRIVER --mode refine --seed $SEED --work-dir $SEARCH_ROOT/refine" \
  8 60 "$NULLB")
echo "route5_refine -> $REFINE [after $NULLB]"

echo
echo "CHAIN QUEUED: smoke -> killrung -> cold -> null_a -> null_b -> refine"
echo "summaries land in $SEARCH_ROOT/<leg>/summary.json; history in history.jsonl"
echo "⚠ killrung CANNOT clear the route (under-run null = no evidence); it can only kill."
