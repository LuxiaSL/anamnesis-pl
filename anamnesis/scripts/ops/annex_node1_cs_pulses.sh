#!/usr/bin/env bash
# ANNEX control-surface tenancy — STAGE 1 on NODE1 (Luxia's word 2026-07-17: "launch
# everything on node1; 7 gpus open"). 7 pulses (incl. varentropy rider) + member build.
# NO STEERING IN THIS CHAIN — stage 2 is gated on the rider-1 filing + FROZEN grid.
set -euo pipefail

REPO="$HOME/projects/anamnesis_exps"
PROP="$REPO/outputs/battery/annex/PROPOSAL-control-surface-tenancy-2026-07-18.md"
grep -q "RATIFY-RECOMMENDED" "$PROP" || { echo "⛔ GATE: ratified proposal missing"; exit 1; }

: "${HEIMDALL_WORK_DIR:?}"
: "${HEIMDALL_VENV:?}"

MODEL=3b
MPATH=/models/llama-3.2-3b-instruct
RUNS=/models/anamnesis-extract/runs
BATTERY=/models/anamnesis-extract/battery
STAGE0=$RUNS/vmb_stage0_3b
VEC_DIR=$BATTERY/annex/cs_vectors_3b
SIG14=$BATTERY/arms/A5/a5_sigma_L14_3b.npz
B7BANK=$BATTERY/a5_vectors_3b_b7/a5_vectors.npz
EOSREP=$BATTERY/annex/eosrep_vectors_3b/a5_vectors.npz
VCONF=$BATTERY/annex/vconf_perp_3b/a5_vectors.npz
NORMS=$BATTERY/a5_vectors_3b/a5_vectors_stamps.json
BASE="source $HEIMDALL_VENV && cd $HEIMDALL_WORK_DIR && export PYTHONPATH=\$PWD/pipeline PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1"
FUNCS=(lexrarity copy selfrep tailmass wraprate freqrep varentropy)

submit() { # name, cmd, gpus, minutes, [after: SINGLE id only — never comma-join]
  local AFTER=(); [[ -n ${5:-} ]] && AFTER=(--after "$5")
  local OUT_
  OUT_=$(heimdall submit "bash -c '$BASE && $2'" -n "$1" --node node1 -g "$3" -e "$4" \
    -w "$HEIMDALL_WORK_DIR" --env HF_HUB_OFFLINE=1 --max-retries 2 "${AFTER[@]}" 2>&1) \
    || { echo "SUBMIT FAILED for $1:" >&2; echo "$OUT_" >&2; exit 1; }
  echo "$OUT_" >&2
  grep -oE '[0-9a-f]{12}' <<< "$OUT_" | head -1
}

# self-deploy code + the vconf bank (built node2-side; node1 lacks it)
rsync -a "$REPO/pipeline/anamnesis/scripts/" node1:luxi-files/anamnesis-pl/pipeline/anamnesis/scripts/ \
  --include='*.py' --exclude='ops/*' --exclude='__pycache__'
ssh node1 "mkdir -p $VEC_DIR $BATTERY/annex/vconf_perp_3b"
rsync -a "$REPO/outputs/battery/annex/vconf_perp_3b/" "node1:$BATTERY/annex/vconf_perp_3b/"

# ── seven pulses, ONE job, sequential inside (1 GPU; V7-recipe verbatim, L14/3B) ──
CMD=""
for F in "${FUNCS[@]}"; do
  [[ -n $CMD ]] && CMD+=" && "
  CMD+="python -u -m anamnesis.scripts.annex_cs_pulses --model $MODEL --model-path $MPATH --stage0-run $STAGE0 --out-dir $VEC_DIR --functional $F --n-gens 20 --map-site 14"
done
PULSES=$(submit vmb-cs-pulses "$CMD" 1 100)
echo "vmb-cs-pulses -> $PULSES (7 functionals sequential)"

MEM=$(submit vmb-cs-members \
  "python -u -m anamnesis.scripts.annex_cs_members --gradients $VEC_DIR/cs_gradients.npz --sigma-l14 $SIG14 --b7-npz $B7BANK --eosrep-npz $EOSREP --vconf-npz $VCONF --norms-json $NORMS --out-dir $VEC_DIR" \
  1 15 "$PULSES")
echo "vmb-cs-members -> $MEM [after $PULSES]"

cat <<EOF

STAGE 1 QUEUED (node1): pulses ($PULSES) -> members ($MEM).
WHEN COMPLETE, pull back for the freeze:
  rsync -a node1:$VEC_DIR/ $REPO/outputs/battery/annex/cs_vectors_3b/
THEN: file PREDICTIONS-cs-members-rider1.md (template; RESTATE the numeric lexfreq rule),
trim + FREEZE cs_cell_grid.json, and fire annex_node1_cs_steer.sh.
FINAL_JOB=$MEM
EOF
