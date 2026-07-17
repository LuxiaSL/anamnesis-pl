#!/usr/bin/env bash
# ANNEX control-surface tenancy — STAGE 1: six pulses + member build (node2, Heimdall).
# ⛔ FIRES ON LUXIA'S WORD (desk ruling a7fe7e2: RATIFY-RECOMMENDED, riders executed),
#    after session-12's GPU needs are met. Priority 10 — anything mainline preempts us.
# After this chain completes: pull back gradients/members/anatomy (echo at bottom),
# file PREDICTIONS-cs-members-rider1.md + freeze cs_cell_grid.json, THEN run
# annex_node2_cs_steer.sh. NO STEERING HAPPENS IN THIS CHAIN.
set -euo pipefail

REPO="$HOME/projects/anamnesis_exps"
PROP="$REPO/outputs/battery/annex/PROPOSAL-control-surface-tenancy-2026-07-18.md"
grep -q "RATIFY-RECOMMENDED" "$PROP" || { echo "⛔ GATE: ratified proposal missing"; exit 1; }

: "${HEIMDALL_WORK_DIR:?}"
: "${HEIMDALL_VENV:?}"
PRIO=10

SHM=/dev/shm/luxi-anamnesis
DATA=$SHM/anamnesis-extract
M3B=$SHM/models/llama-3.2-3b-instruct
STAGE0=$DATA/runs/vmb_stage0_3b
VEC_DIR=$SHM/banks/annex/cs_vectors_3b
SIG14=$SHM/banks/sigma/a5_sigma_L14_3b.npz
B7BANK=$SHM/banks/a5_vectors_3b_b7/a5_vectors.npz
EOSREP=$SHM/banks/annex/eosrep_vectors_3b/a5_vectors.npz
VCONF=$SHM/banks/annex/vconf_perp_3b/a5_vectors.npz
NORMS=$SHM/banks/a5_vectors_3b/a5_vectors_stamps.json
BASE="source $HEIMDALL_VENV && cd $HEIMDALL_WORK_DIR && export PYTHONPATH=\$PWD/pipeline PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1"
FUNCS=(lexrarity copy selfrep tailmass wraprate freqrep)

submit() { # name, cmd, gpus, minutes, [after: SINGLE id only — never comma-join]
  local AFTER=(); [[ -n ${5:-} ]] && AFTER=(--after "$5")
  local OUT_
  OUT_=$(heimdall submit "bash -c '$BASE && $2'" -n "$1" --node node2 -g "$3" -e "$4" \
    -w "$HEIMDALL_WORK_DIR" --env HF_HUB_OFFLINE=1 --priority $PRIO --max-retries 2 \
    "${AFTER[@]}" 2>&1) \
    || { echo "SUBMIT FAILED for $1:" >&2; echo "$OUT_" >&2; exit 1; }
  echo "$OUT_" >&2
  grep -oE '[0-9a-f]{12}' <<< "$OUT_" | head -1
}

# self-deploy current code
rsync -a "$REPO/pipeline/anamnesis/" node2:luxi-files/anamnesis-pl/pipeline/anamnesis/ \
  --exclude='__pycache__' --exclude='ops/*.sh'
ssh node2 "mkdir -p $VEC_DIR"

# ── six pulses, ONE job, sequential inside (1 GPU; V7-recipe verbatim, L14/3B) ──
CMD=""
for F in "${FUNCS[@]}"; do
  [[ -n $CMD ]] && CMD+=" && "
  CMD+="python -u -m anamnesis.scripts.annex_cs_pulses --model 3b --model-path $M3B --stage0-run $STAGE0 --out-dir $VEC_DIR --functional $F --n-gens 20 --map-site 14"
done
PULSES=$(submit vmb-cs-pulses "$CMD" 1 90)
echo "vmb-cs-pulses -> $PULSES (6 functionals sequential)"

# ── member build (CPU-class): band-pass Σ_L14 -> ⊥ vs V7 -> anatomy w/ cos-to-V7 ──
MEM=$(submit vmb-cs-members \
  "python -u -m anamnesis.scripts.annex_cs_members --gradients $VEC_DIR/cs_gradients.npz --sigma-l14 $SIG14 --b7-npz $B7BANK --eosrep-npz $EOSREP --vconf-npz $VCONF --norms-json $NORMS --out-dir $VEC_DIR" \
  1 15 "$PULSES")
echo "vmb-cs-members -> $MEM [after $PULSES]"

cat <<EOF

STAGE 1 QUEUED at priority $PRIO: pulses ($PULSES) -> members ($MEM).
WHEN COMPLETE, pull back for the freeze (leak predictions from cs_anatomy.json cos_to_V7
× the banked V7_L14 entropy effects of record):

  rsync -a node2:$VEC_DIR/ $REPO/outputs/battery/annex/cs_vectors_3b/

THEN: file PREDICTIONS-cs-members-rider1.md (template in annex/), trim + FREEZE
cs_cell_grid.json (status field), and fire annex_node2_cs_steer.sh.
EOF
