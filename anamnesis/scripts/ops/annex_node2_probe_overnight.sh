#!/usr/bin/env bash
# ANNEX probe program — the OVERNIGHT chain (node2, Heimdall, LOW PRIORITY so the
# loose-ends lane preempts cleanly; Luxia's standing word 2026-07-17 night).
# ⛔ GATED on PREDICTIONS-probe-program.md (all 12 response-class freezes filed BEFORE
# any pulse — the program's discipline rider).
#
# Chain: 4 pulse jobs (one per functional, 3 sites sequential inside each, 1 GPU)
#        -> members job (CPU-class: band-pass through per-site Σ + ⊥ at L14 + site nulls)
#        -> gen (18 cells x n=20 @ α=.1, multicell) -> state replay -> expression replay
#        -> entropy. All legs resume-aware => preemption-retry safe (--max-retries 2).
set -euo pipefail

REPO="$HOME/projects/anamnesis_exps"
PRED="$REPO/outputs/battery/annex/PREDICTIONS-probe-program.md"
[[ -f $PRED ]] || { echo "⛔ GATE: $PRED missing — freeze the PP book first"; exit 1; }

: "${HEIMDALL_WORK_DIR:?}"
: "${HEIMDALL_VENV:?}"
PRIO=10   # low — loose-ends lane preempts us, we retry

SHM=/dev/shm/luxi-anamnesis
DATA=$SHM/anamnesis-extract
M3B=$SHM/models/llama-3.2-3b-instruct
STAGE0=$DATA/runs/vmb_stage0_3b
VEC_DIR=$SHM/banks/annex/probe_vectors_3b
RUN_ROOT=$DATA/runs/vmb_probe_3b
SIG7=$DATA/battery/arms/A5/a5_sigma_L7_3b.npz
SIG14=$SHM/banks/sigma/a5_sigma_L14_3b.npz
SIG21=$DATA/battery/arms/A5/a5_sigma_L21_3b.npz
B7BANK=$SHM/banks/a5_vectors_3b_b7/a5_vectors.npz
NORMS=$SHM/banks/a5_vectors_3b/a5_vectors_stamps.json
PROMPTS=pipeline/anamnesis/prompts/prompt_sets.json
BASE="source $HEIMDALL_VENV && cd $HEIMDALL_WORK_DIR && export PYTHONPATH=\$PWD/pipeline PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1"
FUNCS=(attnent anchor vnorm gatemass)
SITES=(7 14 21)

submit() { # name, cmd, gpus, minutes, [after (comma-sep ok)]
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
ssh node2 "mkdir -p $VEC_DIR $RUN_ROOT"

# ── pulses: one job per functional (3 sites sequential inside) ──
PULSE_IDS=()
for F in "${FUNCS[@]}"; do
  CMD=""
  for S in "${SITES[@]}"; do
    [[ -n $CMD ]] && CMD+=" && "
    CMD+="python -u -m anamnesis.scripts.annex_probe_pulses --model 3b --model-path $M3B --stage0-run $STAGE0 --out-dir $VEC_DIR --functional $F --site $S --n-gens 20"
  done
  P=$(submit "vmb-probe-pulse-$F" "$CMD" 1 60)
  echo "vmb-probe-pulse-$F -> $P"
  PULSE_IDS+=("$P")
done
AFTER_PULSES=$(IFS=,; echo "${PULSE_IDS[*]}")

MEM=$(submit vmb-probe-members \
  "python -u -m anamnesis.scripts.annex_probe_members --gradients $VEC_DIR/probe_gradients.npz --sigma-l7 $SIG7 --sigma-l14 $SIG14 --sigma-l21 $SIG21 --v7-npz $B7BANK --norms-json $NORMS --out-dir $VEC_DIR" \
  1 15 "$AFTER_PULSES")
echo "vmb-probe-members -> $MEM [after all pulses]"

# ── cells-json: 12 raw + 4 perp(L14) + 2 site nulls, all α=.1 ──
mkdir -p /tmp/claude-output
GEN_JSON=/tmp/claude-output/probe_gen_cells.json
REP_JSON=/tmp/claude-output/probe_rep_cells.json
python3 - "$RUN_ROOT" "$GEN_JSON" "$REP_JSON" <<'PY'
import json, sys
run_root, gen_json, rep_json = sys.argv[1], sys.argv[2], sys.argv[3]
cells = []
for s in (7, 14, 21):
    for f in ("attnent", "anchor", "vnorm", "gatemass"):
        cells.append((f"P{f}_L{s}", s))
        if s == 14:
            cells.append((f"P{f}_perp_L14", 14))
cells += [("Rband1_L7", 7), ("Rband1_L21", 21)]
gen = {"cells": [{"out_run_dir": f"{run_root}/{k}_a0.1",
                  "seed_namespace": f"VMBPROBE-3B-{k}_a0.1",
                  "inject_key": k, "inject_layer": s, "inject_alpha_frac": 0.1}
                 for k, s in cells]}
rep = {"cells": [{"run_dir": f"{run_root}/{k}_a0.1",
                  "manifest": f"{run_root}/{k}_a0.1/replay_manifest.json"}
                 for k, s in cells]}
json.dump(gen, open(gen_json, "w")); json.dump(rep, open(rep_json, "w"))
print(f"{len(cells)} cells")
PY
rsync -a "$GEN_JSON" "$REP_JSON" node2:$SHM/

GEN=$(submit vmb-probe-gen \
  "python -u -m anamnesis.scripts.vmb_a5_gen_multicell --model 3b --model-path $M3B --prompts $PROMPTS --cells-json $SHM/probe_gen_cells.json --inject-npz $VEC_DIR/a5_vectors.npz --inject-norms-json $VEC_DIR/a5_vectors_stamps.json --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 4 --seeds-per-class 1 --limit 20" \
  8 40 "$MEM")
echo "vmb-probe-gen -> $GEN [after $MEM]"

REP=$(submit vmb-probe-rep \
  "python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model 3b --model-path $M3B --calib-dir $DATA/calibration/3b --cells-json $SHM/probe_rep_cells.json --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 4 --no-raw --inject-from-metadata" \
  8 30 "$GEN")
echo "vmb-probe-rep -> $REP [after $GEN]"

EXPR=$(submit vmb-probe-expression \
  "python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model 3b --model-path $M3B --calib-dir $DATA/calibration/3b --cells-json $SHM/probe_rep_cells.json --gpus 0,1,2,3,4,5,6,7 --workers-per-gpu 4 --no-raw --sig-subdir signatures_v3_noinject" \
  8 30 "$REP")
echo "vmb-probe-expression -> $EXPR [after $REP]"

CELLS=""
for s in 7 14 21; do for f in attnent anchor vnorm gatemass; do
  CELLS+="P${f}_L${s}_a0.1 "
  [[ $s == 14 ]] && CELLS+="P${f}_perp_L14_a0.1 "
done; done
CELLS+="Rband1_L7_a0.1 Rband1_L21_a0.1"
ENT=$(submit vmb-probe-entropy \
  "python -u -m anamnesis.scripts.vmb_c3_entropy_replay --model 3b --model-path $M3B --c3-run-dir $RUN_ROOT --cells $CELLS --null-prefixes RBAND --out-json $RUN_ROOT/probe_entropy_3b_node2.json" \
  1 40 "$GEN")
echo "vmb-probe-entropy -> $ENT [after $GEN, parallel to replays]"

echo
echo "OVERNIGHT PROBE CHAIN QUEUED at priority $PRIO: 4 pulse jobs -> members -> gen(18) -> rep -> expr; entropy parallel."
echo "FINAL_JOB=$EXPR"
