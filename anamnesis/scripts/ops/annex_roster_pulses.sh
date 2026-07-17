#!/usr/bin/env bash
# ANNEX in-shadow roster — STAGE 1: the three gradient pulses (ratified c93055b).
# ⛔ LUXIA-GATED: fires on her GPU word only. One 1-GPU job, ~15 min: margin + eos + repmass
# gradients over 20 banked stage-0 gens each (annex_potential_gradient, v4-recipe verbatim).
#
# AFTER this lands: pull back + build members + anatomy LOCALLY (CPU), then FILE the
# per-member rider-1 predictions BEFORE annex_roster_steer.sh may run:
#   rsync -a node1:/models/anamnesis-extract/battery/annex/roster_vectors_3b/ \
#         outputs/battery/annex/roster_vectors_3b/
#   (from pipeline/)  python -m anamnesis.scripts.annex_band_pass \
#     --gradients ../outputs/battery/annex/roster_vectors_3b/roster_gradients.npz \
#     --keys Gmargin_L14:Vconf_L14 Geos_L14:Veos_L14 Grep_L14:Vrep_L14 \
#     --sigma ../outputs/battery/arms/A5/a5_sigma_L14_3b.npz \
#     --stamps ../outputs/battery/a5_vectors_3b/a5_vectors_stamps.json \
#     --compare ../outputs/battery/a5_vectors_3b/a5_vectors.npz:V3_L14,V4_L14 \
#               ../outputs/battery/a5_vectors_3b_b7/a5_vectors.npz:V7_L14 \
#               ../outputs/battery/annex/a5_vectors_3b_14r/a5_vectors.npz:RA_L14 \
#     --out-dir ../outputs/battery/annex/roster_vectors_3b
set -euo pipefail

: "${HEIMDALL_WORK_DIR:?}"
: "${HEIMDALL_VENV:?}"

MODEL=3b
MPATH=/models/llama-3.2-3b-instruct
STAGE0=/models/anamnesis-extract/runs/vmb_stage0_3b
OUT=/models/anamnesis-extract/battery/annex/roster_vectors_3b
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

CMD=""
for F in margin eos repmass; do
  C="python -u -m anamnesis.scripts.annex_potential_gradient --model $MODEL --model-path $MPATH --stage0-run $STAGE0 --out-dir $OUT --functional $F --n-gens 20"
  CMD="${CMD:+$CMD && }$C"
done
J=$(submit annex-roster-pulses "$CMD" 1 20)
echo "annex-roster-pulses -> $J"
echo "NEXT (local, CPU): pull-back + annex_band_pass + PREDICTIONS filing — see header."
