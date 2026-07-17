#!/usr/bin/env bash
# ANNEX −Veos⊥ dose window — NODE2 runner (per NODE2-OPS-2026-07-17.md; detached-proc,
# GPUs 0–6, node2 stamps, sync-off). ⛔ DOUBLE-GATED: (1) node2 validation gate PASSED;
# (2) PREDICTIONS-eosdose-window.md exists (ED book frozen at staging).
# 5 cells UNCAPPED 2048: Veos_perp × {−.15,−.20,−.25} + Rband1 × {−.15,−.25} nulls.
set -euo pipefail

REPO="$HOME/projects/anamnesis_exps"
PRED="$REPO/outputs/battery/annex/PREDICTIONS-eosdose-window.md"
[[ -f $PRED ]] || { echo "⛔ GATE: $PRED missing — freeze the ED book first"; exit 1; }

SHM=/dev/shm/luxi-anamnesis
DATA=$SHM/anamnesis-extract
M3B=$SHM/models/llama-3.2-3b-instruct
BANK=$SHM/banks/annex/eosrep_vectors_3b
RUN_ROOT=$DATA/runs/vmb_eosdose_3b
GPUS=0,1,2,3,4,5,6
CELLS_SPEC=(
  "Veos_perp_L14|-0.15" "Veos_perp_L14|-0.2" "Veos_perp_L14|-0.25"
  "Rband1_L14|-0.15"    "Rband1_L14|-0.25"
)

cellname() { local F=${2#-}; [[ $2 == -* ]] && echo "${1}_an${F}" || echo "${1}_a${F}"; }

mkdir -p /tmp/claude-output
GEN_JSON=/tmp/claude-output/eosdose_node2_gen_cells.json
REP_JSON=/tmp/claude-output/eosdose_node2_rep_cells.json
{
  echo '{"cells": ['
  FIRST=1
  for spec in "${CELLS_SPEC[@]}"; do
    IFS='|' read -r K FR <<< "$spec"
    C=$(cellname "$K" "$FR")
    [[ $FIRST == 1 ]] && FIRST=0 || echo ','
    printf '  {"out_run_dir": "%s/%s", "seed_namespace": "VMBEOSD-3B-%s", "inject_key": "%s", "inject_layer": 14, "inject_alpha_frac": %s}' \
      "$RUN_ROOT" "$C" "$C" "$K" "$FR"
  done
  echo ''; echo ']}'
} > "$GEN_JSON"
{
  echo '{"cells": ['
  FIRST=1
  for spec in "${CELLS_SPEC[@]}"; do
    IFS='|' read -r K FR <<< "$spec"
    C=$(cellname "$K" "$FR")
    [[ $FIRST == 1 ]] && FIRST=0 || echo ','
    printf '  {"run_dir": "%s/%s", "manifest": "%s/%s/replay_manifest.json"}' \
      "$RUN_ROOT" "$C" "$RUN_ROOT" "$C"
  done
  echo ''; echo ']}'
} > "$REP_JSON"
python3 -c "import json; json.load(open('$GEN_JSON')); json.load(open('$REP_JSON')); print('cells-json ok')"
rsync -a "$GEN_JSON" "$REP_JSON" node2:$SHM/
CELLS=""; for spec in "${CELLS_SPEC[@]}"; do IFS='|' read -r K FR <<< "$spec"; CELLS+="$(cellname "$K" "$FR") "; done

ssh node2 "bash -s" <<EOF
set -euo pipefail
cd ~/luxi-files/anamnesis-pl
source ~/luxi-files/.venv-shared/bin/activate
export PYTHONPATH=\$PWD/pipeline PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1
LOG=~/luxi-files/annex_node2_eosdose.log
{
echo "=== EOSDOSE NODE2 CHAIN \$(date -u +%FT%TZ) (UNCAPPED 2048) ==="
python -u -m anamnesis.scripts.vmb_a5_gen_multicell --model 3b --model-path $M3B \
  --prompts pipeline/anamnesis/prompts/prompt_sets.json --cells-json $SHM/eosdose_node2_gen_cells.json \
  --inject-npz $BANK/a5_vectors.npz --inject-norms-json $BANK/a5_vectors_stamps.json \
  --gpus $GPUS --workers-per-gpu 4 --seeds-per-class 1 --limit 40 --max-new-tokens 2048
python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model 3b --model-path $M3B \
  --calib-dir $DATA/calibration/3b --cells-json $SHM/eosdose_node2_rep_cells.json \
  --gpus $GPUS --workers-per-gpu 4 --no-raw --inject-from-metadata
python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model 3b --model-path $M3B \
  --calib-dir $DATA/calibration/3b --cells-json $SHM/eosdose_node2_rep_cells.json \
  --gpus $GPUS --workers-per-gpu 4 --no-raw --sig-subdir signatures_v3_noinject
python -u -m anamnesis.scripts.vmb_c3_entropy_replay --model 3b --model-path $M3B \
  --c3-run-dir $RUN_ROOT --cells $CELLS --null-prefixes RBAND \
  --out-json $RUN_ROOT/eosdose_entropy_3b_node2.json
python - <<'PY'
import json, pathlib
for md in pathlib.Path("$RUN_ROOT").glob("*/metadata.json"):
    d = json.loads(md.read_text()); d["compute_node"] = "node2"
    md.write_text(json.dumps(d))
ej = pathlib.Path("$RUN_ROOT/eosdose_entropy_3b_node2.json")
d = json.loads(ej.read_text()); d["compute_node"] = "node2"; ej.write_text(json.dumps(d, indent=1))
print("node2 stamps written")
PY
echo "=== EOSDOSE NODE2 CHAIN DONE \$(date -u +%FT%TZ) ==="
} 2>&1 | tee \$LOG
EOF

mkdir -p "$REPO/outputs/battery/annex/vmb_eosdose_3b"
rsync -a node2:$RUN_ROOT/ "$REPO/outputs/battery/annex/vmb_eosdose_3b/"
echo "EOSDOSE chain complete; artifacts synced to outputs/battery/annex/vmb_eosdose_3b/"
echo "Readout next (local): length/rep vs in-chain Rband-uncapped envelope; ED book scoring."
