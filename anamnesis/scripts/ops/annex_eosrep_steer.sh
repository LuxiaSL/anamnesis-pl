#!/usr/bin/env bash
# ANNEX EOS/rep follow-up block — STAGE B: the steering cells. ⛔ DOUBLE-GATED:
# (1) Luxia's GPU word; (2) PREDICTIONS-eosrep-block.md must exist (frozen AGAINST the
# stage-A substrate distribution — the freeze may TRIM the draft roster below; edit the
# MEMBERS arrays to match the frozen doc before firing).
#
# DRAFT roster (final at freeze):
#   UNCAPPED frame (max-new-tokens 2048): Rband1 x {+.1,+.3,-.1,-.3} (matched nulls both
#     signs) · Veos x {+.1,+.3,-.1,-.3} · Veos_perp x {+.3,-.3}
#   CAPPED frame (512, comparable to banked): Rband1 x {-.1,-.3} · Vrep x {-.1,-.3} ·
#     Vrep_perp x {+.1,-.1,-.3}
# Negative doses = negative --inject-alpha-frac (14n precedent). Expression column in-chain
# (standing rule). Readout: text instruments (length/EOS for uncapped; trigram/ttr for rep)
# + entropy rows + judge spot-checks per the frozen doc.
set -euo pipefail

: "${HEIMDALL_WORK_DIR:?}"
: "${HEIMDALL_VENV:?}"

REPO="$HOME/projects/anamnesis_exps"
PRED="$REPO/outputs/battery/annex/PREDICTIONS-eosrep-block.md"
[[ -f $PRED ]] || { echo "⛔ GATE: $PRED missing — freeze against the stage-A substrate first"; exit 1; }

MODEL=3b
MPATH=/models/llama-3.2-3b-instruct
RUNS=/models/anamnesis-extract/runs
CALIB=/models/anamnesis-extract/calibration/3b
RUN_ROOT=$RUNS/vmb_eosrep_3b
VEC_DIR=/models/anamnesis-extract/battery/annex/eosrep_vectors_3b
B7_VEC=/models/anamnesis-extract/battery/a5_vectors_3b_b7
PROMPTS=pipeline/anamnesis/prompts/prompt_sets.json
BATTERY=/models/anamnesis-extract/battery
BASE="source $HEIMDALL_VENV && cd $HEIMDALL_WORK_DIR && export PYTHONPATH=\$PWD/pipeline PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1"

# frame|vector_key|npz_dir|alpha_frac  (edit to the FROZEN roster before firing)
CELLS_SPEC=(
  "unc|Rband1_L14|$B7_VEC|0.1"   "unc|Rband1_L14|$B7_VEC|0.3"
  "unc|Rband1_L14|$B7_VEC|-0.1"  "unc|Rband1_L14|$B7_VEC|-0.3"
  "unc|Veos_L14|$VEC_DIR|0.1"    "unc|Veos_L14|$VEC_DIR|0.3"
  "unc|Veos_L14|$VEC_DIR|-0.1"   "unc|Veos_L14|$VEC_DIR|-0.3"
  "unc|Veos_perp_L14|$VEC_DIR|0.3"  "unc|Veos_perp_L14|$VEC_DIR|-0.3"
  "cap|Rband1_L14|$B7_VEC|-0.1"  "cap|Rband1_L14|$B7_VEC|-0.3"
  "cap|Vrep_L14|$VEC_DIR|-0.1"   "cap|Vrep_L14|$VEC_DIR|-0.3"
  "cap|Vrep_perp_L14|$VEC_DIR|0.1" "cap|Vrep_perp_L14|$VEC_DIR|-0.1" "cap|Vrep_perp_L14|$VEC_DIR|-0.3"
)

submit() { # name, cmd, gpus, minutes, [after]  — hardened pattern
  local AFTER=(); [[ -n ${5:-} ]] && AFTER=(--after "$5")
  local OUT_
  OUT_=$(heimdall submit "bash -c '$BASE && $2'" -n "$1" --node node1 -g "$3" -e "$4" \
    -w "$HEIMDALL_WORK_DIR" --env HF_HUB_OFFLINE=1 --max-retries 0 "${AFTER[@]}" 2>&1) \
    || { echo "SUBMIT FAILED for $1:" >&2; echo "$OUT_" >&2; exit 1; }
  echo "$OUT_" >&2
  grep -oE '[0-9a-f]{12}' <<< "$OUT_" | head -1
}

join_and() { local out=$1; shift; for c in "$@"; do out+=" && $c"; done; printf '%s' "$out"; }

cellname() { # vector_key, frac
  local F=${2#-}; [[ $2 == -* ]] && echo "${1}_an${F}" || echo "${1}_a${F}"
}

ssh node1 "mkdir -p $VEC_DIR $RUN_ROOT"
rsync -a "$REPO/outputs/battery/annex/eosrep_vectors_3b/" "node1:$VEC_DIR/"
rsync -a "$REPO/pipeline/anamnesis/scripts/annex_perp_vectors.py" node1:luxi-files/anamnesis-pl/pipeline/anamnesis/scripts/
echo "staged eosrep bank -> node1:$VEC_DIR/"

# per-frame multicell gen json (uncapped vs capped need different --max-new-tokens, and
# vectors come from TWO npz banks -> per-cell single-gen jobs would reload; instead: one
# multicell job per (frame, npz) group)
mkdir -p /tmp/claude-output
declare -A GROUPS=()
for spec in "${CELLS_SPEC[@]}"; do
  IFS='|' read -r FRAME KEY NPZDIR FRAC <<< "$spec"
  GROUPS["$FRAME|$NPZDIR"]=1
done

SMOKE=$(submit vmb-eosrep-smoke \
  "python -u -m anamnesis.scripts.vmb_smoke_write_hooks --model $MODEL --model-path $MPATH --floor-run-dir $RUNS/vmb_stage0_3b --calib-dir $CALIB --out /dev/shm/vmb_eosrep_smoke" \
  1 10)
echo "vmb-eosrep-smoke -> $SMOKE"

PREV=$SMOKE
GEN_IDS=()
for group in "${!GROUPS[@]}"; do
  IFS='|' read -r FRAME NPZDIR <<< "$group"
  MAXTOK=512; [[ $FRAME == unc ]] && MAXTOK=2048
  TAG=$(basename "$NPZDIR")
  JSONF=/tmp/claude-output/eosrep_gen_${FRAME}_${TAG}.json
  {
    echo '{"cells": ['
    FIRST=1
    for spec in "${CELLS_SPEC[@]}"; do
      IFS='|' read -r F K ND FR <<< "$spec"
      [[ $F == "$FRAME" && $ND == "$NPZDIR" ]] || continue
      C=$(cellname "$K" "$FR")_${FRAME}
      [[ $FIRST == 1 ]] && FIRST=0 || echo ','
      printf '  {"out_run_dir": "%s/%s", "seed_namespace": "VMBEOS-3B-%s", "inject_key": "%s", "inject_layer": 14, "inject_alpha_frac": %s}' \
        "$RUN_ROOT" "$C" "$C" "$K" "$FR"
    done
    echo ''; echo ']}'
  } > "$JSONF"
  python3 -c "import json; json.load(open('$JSONF'))"
  rsync -a "$JSONF" "node1:$BATTERY/annex/$(basename "$JSONF")"
  G=$(submit "vmb-eosrep-gen-$FRAME-$TAG" \
    "python -u -m anamnesis.scripts.vmb_a5_gen_multicell --model $MODEL --model-path $MPATH --prompts $PROMPTS --cells-json $BATTERY/annex/$(basename "$JSONF") --inject-npz $NPZDIR/a5_vectors.npz --inject-norms-json $NPZDIR/a5_vectors_stamps.json --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 4 --seeds-per-class 1 --limit 40 --max-new-tokens $MAXTOK" \
    8 45 "$PREV")
  echo "vmb-eosrep-gen-$FRAME-$TAG -> $G [after $PREV]"
  PREV=$G
  GEN_IDS+=("$G")
done

# replay (both columns) over ALL cells, one multicell json
REPJSON=/tmp/claude-output/eosrep_rep_cells.json
{
  echo '{"cells": ['
  FIRST=1
  for spec in "${CELLS_SPEC[@]}"; do
    IFS='|' read -r F K ND FR <<< "$spec"
    C=$(cellname "$K" "$FR")_${F}
    [[ $FIRST == 1 ]] && FIRST=0 || echo ','
    printf '  {"run_dir": "%s/%s", "manifest": "%s/%s/replay_manifest.json"}' \
      "$RUN_ROOT" "$C" "$RUN_ROOT" "$C"
  done
  echo ''; echo ']}'
} > "$REPJSON"
python3 -c "import json; json.load(open('$REPJSON'))"
rsync -a "$REPJSON" "node1:$BATTERY/annex/eosrep_rep_cells.json"

REP=$(submit vmb-eosrep-rep \
  "python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model $MODEL --model-path $MPATH --calib-dir $CALIB --cells-json $BATTERY/annex/eosrep_rep_cells.json --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 4 --no-raw --inject-from-metadata" \
  8 30 "$PREV")
echo "vmb-eosrep-rep -> $REP [after $PREV]"

EXPR=$(submit vmb-eosrep-expression \
  "python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model $MODEL --model-path $MPATH --calib-dir $CALIB --cells-json $BATTERY/annex/eosrep_rep_cells.json --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 4 --no-raw --sig-subdir signatures_v3_noinject" \
  8 30 "$REP")
echo "vmb-eosrep-expression -> $EXPR [after $REP]"

CELLS=""
for spec in "${CELLS_SPEC[@]}"; do
  IFS='|' read -r F K ND FR <<< "$spec"
  CELLS+="$(cellname "$K" "$FR")_${F} "
done
ENT=$(submit vmb-eosrep-entropy \
  "python -u -m anamnesis.scripts.vmb_c3_entropy_replay --model $MODEL --model-path $MPATH --c3-run-dir $RUN_ROOT --cells $CELLS --null-prefixes RBAND --out-json $BATTERY/annex/eosrep_entropy_3b.json" \
  1 40 "$PREV")
echo "vmb-eosrep-entropy -> $ENT [after $PREV, parallel to replays]"

echo
echo "STAGE-B QUEUED: smoke -> gen groups -> rep -> expression; entropy parallel."
