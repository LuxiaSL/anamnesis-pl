#!/usr/bin/env bash
# ANNEX two-actuators — NODE2 runner (per NODE2-OPS-2026-07-17.md; detached-proc,
# GPUs 0–6, node2 stamps, sync-off). ⛔ DOUBLE-GATED: (1) node2 validation gate PASSED
# (its gen smoke doubles as the parity gate for the f5b6b74 repetition-penalty plumbing);
# (2) PREDICTIONS-two-actuators.md exists (TA book frozen at staging).
# 3 SAMPLER cells (per-cell repetition_penalty, NO injection): rp0.85 · rp1.15 · rp1.30.
# Plain replay only — sampler cells' one signature column IS the expression column.
set -euo pipefail

REPO="$HOME/projects/anamnesis_exps"
PRED="$REPO/outputs/battery/annex/PREDICTIONS-two-actuators.md"
[[ -f $PRED ]] || { echo "⛔ GATE: $PRED missing — freeze the TA book first"; exit 1; }

SHM=/dev/shm/luxi-anamnesis
DATA=$SHM/anamnesis-extract
M3B=$SHM/models/llama-3.2-3b-instruct
RUN_ROOT=$DATA/runs/vmb_twoact_3b
GPUS=0,1,2,3,4,5,6
PENALTIES=("0.85|rp085" "1.15|rp115" "1.30|rp130")

mkdir -p /tmp/claude-output
GEN_JSON=/tmp/claude-output/twoact_node2_gen_cells.json
REP_JSON=/tmp/claude-output/twoact_node2_rep_cells.json
{
  echo '{"cells": ['
  FIRST=1
  for P in "${PENALTIES[@]}"; do
    IFS='|' read -r RP C <<< "$P"
    [[ $FIRST == 1 ]] && FIRST=0 || echo ','
    printf '  {"out_run_dir": "%s/%s", "seed_namespace": "VMBTWOACT-3B-%s", "inject_key": null, "repetition_penalty": %s}' \
      "$RUN_ROOT" "$C" "$C" "$RP"
  done
  echo ''; echo ']}'
} > "$GEN_JSON"
{
  echo '{"cells": ['
  FIRST=1
  for P in "${PENALTIES[@]}"; do
    IFS='|' read -r RP C <<< "$P"
    [[ $FIRST == 1 ]] && FIRST=0 || echo ','
    printf '  {"run_dir": "%s/%s", "manifest": "%s/%s/replay_manifest.json"}' \
      "$RUN_ROOT" "$C" "$RUN_ROOT" "$C"
  done
  echo ''; echo ']}'
} > "$REP_JSON"
python3 -c "import json; json.load(open('$GEN_JSON')); json.load(open('$REP_JSON')); print('cells-json ok')"
rsync -a "$GEN_JSON" "$REP_JSON" node2:$SHM/

ssh node2 "bash -s" <<EOF
set -euo pipefail
cd ~/luxi-files/anamnesis-pl
source ~/luxi-files/.venv-shared/bin/activate
export PYTHONPATH=\$PWD/pipeline PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1
LOG=~/luxi-files/annex_node2_twoact.log
{
echo "=== TWOACT NODE2 CHAIN \$(date -u +%FT%TZ) ==="
python -u -m anamnesis.scripts.vmb_a5_gen_multicell --model 3b --model-path $M3B \
  --prompts pipeline/anamnesis/prompts/prompt_sets.json --cells-json $SHM/twoact_node2_gen_cells.json \
  --gpus $GPUS --workers-per-gpu 4 --seeds-per-class 1 --limit 40
python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model 3b --model-path $M3B \
  --calib-dir $DATA/calibration/3b --cells-json $SHM/twoact_node2_rep_cells.json \
  --gpus $GPUS --workers-per-gpu 4 --no-raw
python - <<'PY'
import json, pathlib
for md in pathlib.Path("$RUN_ROOT").glob("*/metadata.json"):
    d = json.loads(md.read_text()); d["compute_node"] = "node2"
    md.write_text(json.dumps(d))
print("node2 stamps written")
PY
echo "=== TWOACT NODE2 CHAIN DONE \$(date -u +%FT%TZ) ==="
} 2>&1 | tee \$LOG
EOF

mkdir -p "$REPO/outputs/battery/annex/vmb_twoact_3b"
rsync -a node2:$RUN_ROOT/ "$REPO/outputs/battery/annex/vmb_twoact_3b/"
echo "TWOACT chain complete; artifacts synced to outputs/battery/annex/vmb_twoact_3b/"
echo "Readout next (local): matched-effect pairing by median trigram-rep, pick-the-actuator"
echo "GroupKFold vs eosrep expression columns, TA-4 clean-judge pass."
