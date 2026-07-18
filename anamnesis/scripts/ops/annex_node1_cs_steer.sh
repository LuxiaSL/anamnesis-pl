#!/usr/bin/env bash
# ANNEX control-surface tenancy — STAGE 2 on NODE1: steer + two columns + entropy +
# readout. ⛔ GATED on (1) PREDICTIONS-cs-members-rider1.md filed, (2) cs_cell_grid.json
# FROZEN, (3) the member bank present node1-side (stage-1 output). 7 GPUs (GPU 0 = other
# user's job at launch time; Heimdall allocates the free set).
set -euo pipefail

REPO="$HOME/projects/anamnesis_exps"
ANNEX="$REPO/outputs/battery/annex"
PRED="$ANNEX/PREDICTIONS-cs-members-rider1.md"
GRID="$ANNEX/cs_cell_grid.json"
[[ -f $PRED ]] || { echo "⛔ GATE: $PRED missing — file the rider-1 freeze first"; exit 1; }
python3 - "$GRID" <<'PY' || exit 1
import json, sys
g = json.load(open(sys.argv[1]))
if not g.get("status", "").startswith("FROZEN"):
    print(f"⛔ GATE: grid not FROZEN (status: {g.get('status','?')[:60]}...)")
    raise SystemExit(1)
print(f"grid FROZEN ok: {len(g['cells'])} cells")
PY

: "${HEIMDALL_WORK_DIR:?}"
: "${HEIMDALL_VENV:?}"

MODEL=3b
MPATH=/models/llama-3.2-3b-instruct
RUNS=/models/anamnesis-extract/runs
BATTERY=/models/anamnesis-extract/battery
CALIB=/models/anamnesis-extract/calibration/3b
STAGE0=$RUNS/vmb_stage0_3b
VEC_DIR=$BATTERY/annex/cs_vectors_3b
RUN_ROOT=$RUNS/vmb_cs_3b
PROMPTS=pipeline/anamnesis/prompts/prompt_sets.json
BASE="source $HEIMDALL_VENV && cd $HEIMDALL_WORK_DIR && export PYTHONPATH=\$PWD/pipeline PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1"

ssh node1 "test -f $VEC_DIR/a5_vectors.npz" \
  || { echo "⛔ GATE: node1 $VEC_DIR/a5_vectors.npz missing — run stage 1 first"; exit 1; }
LOGFREQ_SHA=$(ssh node1 "python3 -c \"import json; print(json.load(open('$VEC_DIR/cs_gradients_stamps.json'))['Glex_L14']['logfreq_sha'])\"")
[[ -n $LOGFREQ_SHA ]] || { echo "⛔ GATE: could not read logfreq sha from stage-1 stamps"; exit 1; }
echo "logfreq sha of record: $LOGFREQ_SHA"

submit() { # name, cmd, gpus, minutes, [after: SINGLE id only — never comma-join]
  local AFTER=(); [[ -n ${5:-} ]] && AFTER=(--after "$5")
  local OUT_
  OUT_=$(heimdall submit "bash -c '$BASE && $2'" -n "$1" --node node1 -g "$3" -e "$4" \
    -w "$HEIMDALL_WORK_DIR" --env HF_HUB_OFFLINE=1 --max-retries 2 "${AFTER[@]}" 2>&1) \
    || { echo "SUBMIT FAILED for $1:" >&2; echo "$OUT_" >&2; exit 1; }
  echo "$OUT_" >&2
  grep -oE '[0-9a-f]{12}' <<< "$OUT_" | head -1
}

rsync -a "$REPO/pipeline/anamnesis/scripts/" node1:luxi-files/anamnesis-pl/pipeline/anamnesis/scripts/ \
  --include='*.py' --exclude='ops/*' --exclude='__pycache__'
ssh node1 "mkdir -p $RUN_ROOT"

mkdir -p /tmp/claude-output
GEN_JSON=/tmp/claude-output/cs_gen_cells.json
REP_JSON=/tmp/claude-output/cs_rep_cells.json
python3 - "$GRID" "$RUN_ROOT" "$GEN_JSON" "$REP_JSON" <<'PY'
import json, sys
grid, run_root, gen_json, rep_json = sys.argv[1:5]
g = json.load(open(grid))
gen = {"cells": [{"out_run_dir": f"{run_root}/{c['name']}",
                  "seed_namespace": f"{g['seed_namespace_prefix']}-{c['name']}",
                  "inject_key": c["key"], "inject_layer": g["inject_layer"],
                  "inject_alpha_frac": c["frac"]} for c in g["cells"]]}
rep = {"cells": [{"run_dir": f"{run_root}/{c['name']}",
                  "manifest": f"{run_root}/{c['name']}/replay_manifest.json"}
                 for c in g["cells"]]}
json.dump(gen, open(gen_json, "w"), indent=1)
json.dump(rep, open(rep_json, "w"), indent=1)
print(f"{len(g['cells'])} cells (n_per_cell={g['n_per_cell']})")
PY
rsync -a "$GEN_JSON" "node1:$BATTERY/annex/cs_gen_cells.json"
rsync -a "$REP_JSON" "node1:$BATTERY/annex/cs_rep_cells.json"

GEN=$(submit vmb-cs-gen \
  "python -u -m anamnesis.scripts.vmb_a5_gen_multicell --model $MODEL --model-path $MPATH --prompts $PROMPTS --cells-json $BATTERY/annex/cs_gen_cells.json --inject-npz $VEC_DIR/a5_vectors.npz --inject-norms-json $VEC_DIR/a5_vectors_stamps.json --gpus 0,1,2,3,4,5,6 --workers-per-gpu 4 --seeds-per-class 1 --limit 40" \
  7 150)
echo "vmb-cs-gen -> $GEN (the long pole: ~1480 gens on 7 GPUs)"

REP=$(submit vmb-cs-rep \
  "python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model $MODEL --model-path $MPATH --calib-dir $CALIB --cells-json $BATTERY/annex/cs_rep_cells.json --gpus 0,1,2,3,4,5,6 --workers-per-gpu 4 --no-raw --inject-from-metadata" \
  7 100 "$GEN")
echo "vmb-cs-rep -> $REP [after $GEN] (state column)"

EXPR=$(submit vmb-cs-expr \
  "python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model $MODEL --model-path $MPATH --calib-dir $CALIB --cells-json $BATTERY/annex/cs_rep_cells.json --gpus 0,1,2,3,4,5,6 --workers-per-gpu 4 --no-raw --sig-subdir signatures_v3_noinject" \
  7 100 "$REP")
echo "vmb-cs-expr -> $EXPR [after $REP] (expression column)"

CELLS=$(python3 -c "import json; print(' '.join(c['name'] for c in json.load(open('$GRID'))['cells']))")
ENT=$(submit vmb-cs-entropy \
  "python -u -m anamnesis.scripts.vmb_c3_entropy_replay --model $MODEL --model-path $MPATH --c3-run-dir $RUN_ROOT --cells $CELLS --null-prefixes RBAND --out-json $BATTERY/annex/cs_entropy_3b.json" \
  1 150 "$GEN")
echo "vmb-cs-entropy -> $ENT [after $GEN, parallel] (CS-5 scoring input)"

RDO=$(submit vmb-cs-readout \
  "python -u -m anamnesis.scripts.annex_cs_readout --run-root $RUN_ROOT --stage0-manifest $STAGE0/replay_manifest.json --tokenizer-path $MPATH --expect-logfreq-sha $LOGFREQ_SHA --out-readout $BATTERY/annex/cs_text_readout.json --out-perm $BATTERY/annex/cs_permutation_grid.json" \
  1 45 "$GEN")
echo "vmb-cs-readout -> $RDO [after $GEN] (instruments + permutation grid)"

cat <<EOF

STAGE 2 QUEUED (node1). FINAL_JOB=$EXPR
WHEN COMPLETE, pull back:
  rsync -a node1:$RUN_ROOT/ $REPO/outputs/battery/annex/vmb_cs_3b/
  rsync -a node1:$BATTERY/annex/cs_{entropy_3b,text_readout,permutation_grid}.json $REPO/outputs/battery/annex/
Then: annex_cs_leakscore (CS-5 scorecard) -> judge gates (3 clean subagent judges,
frozen questions) -> CS book into the RIDER 2 CS-1×CS-2 joint frame (+CS-6). Records
carry compute_node: node1.
EOF
