#!/usr/bin/env bash
# ANNEX Vconf⊥ cell — NODE2 runner (per NODE2-OPS-2026-07-17.md: detached-proc mode,
# GPUs 0–6 only, /dev/shm data root, node2 provenance stamps, sync-off at end).
# ⛔ DOUBLE-GATED: (1) the node2 validation gate must have PASSED (bitwise; see the
# gate record in the ledger); (2) PREDICTIONS-vconf-cell.md must exist (VC book, S9-2).
# Node-side chain smoke is REPLACED by the ride-level validation gate (named deviation:
# the gate's gen+replay smokes exercise the same hook/replay machinery).
set -euo pipefail

REPO="$HOME/projects/anamnesis_exps"
PRED="$REPO/outputs/battery/annex/PREDICTIONS-vconf-cell.md"
[[ -f $PRED ]] || { echo "⛔ GATE: $PRED missing"; exit 1; }

SHM=/dev/shm/luxi-anamnesis
DATA=$SHM/anamnesis-extract
M3B=$SHM/models/llama-3.2-3b-instruct
BANK=$SHM/banks/annex/vconf_perp_3b
RUN_ROOT=$DATA/runs/vmb_vconf_3b
GPUS=0,1,2,3,4,5,6
KEY=Vconf_perp_L14
FRACS=(0.1 0.3 -0.1 -0.3)

cellname() { local F=${1#-}; [[ $1 == -* ]] && echo "${KEY}_an${F}" || echo "${KEY}_a${F}"; }

mkdir -p /tmp/claude-output
GEN_JSON=/tmp/claude-output/vconf_node2_gen_cells.json
REP_JSON=/tmp/claude-output/vconf_node2_rep_cells.json
{
  echo '{"cells": ['
  FIRST=1
  for A in "${FRACS[@]}"; do
    C=$(cellname "$A")
    [[ $FIRST == 1 ]] && FIRST=0 || echo ','
    printf '  {"out_run_dir": "%s/%s", "seed_namespace": "VMBVCONF-3B-%s", "inject_key": "%s", "inject_layer": 14, "inject_alpha_frac": %s}' \
      "$RUN_ROOT" "$C" "$C" "$KEY" "$A"
  done
  echo ''; echo ']}'
} > "$GEN_JSON"
{
  echo '{"cells": ['
  FIRST=1
  for A in "${FRACS[@]}"; do
    C=$(cellname "$A")
    [[ $FIRST == 1 ]] && FIRST=0 || echo ','
    printf '  {"run_dir": "%s/%s", "manifest": "%s/%s/replay_manifest.json"}' \
      "$RUN_ROOT" "$C" "$RUN_ROOT" "$C"
  done
  echo ''; echo ']}'
} > "$REP_JSON"
python3 -c "import json; json.load(open('$GEN_JSON')); json.load(open('$REP_JSON')); print('cells-json ok')"
rsync -a "$GEN_JSON" "$REP_JSON" node2:$SHM/
CELLS=""; for A in "${FRACS[@]}"; do CELLS+="$(cellname "$A") "; done

ssh node2 "bash -s" <<EOF
set -euo pipefail
cd ~/luxi-files/anamnesis-pl
source ~/luxi-files/.venv-shared/bin/activate
export PYTHONPATH=\$PWD/pipeline PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1
LOG=~/luxi-files/annex_node2_vconf.log
{
echo "=== VCONF NODE2 CHAIN \$(date -u +%FT%TZ) ==="
python -u -m anamnesis.scripts.vmb_a5_gen_multicell --model 3b --model-path $M3B \
  --prompts pipeline/anamnesis/prompts/prompt_sets.json --cells-json $SHM/vconf_node2_gen_cells.json \
  --inject-npz $BANK/a5_vectors.npz --inject-norms-json $BANK/a5_vectors_stamps.json \
  --gpus $GPUS --workers-per-gpu 4 --seeds-per-class 1 --limit 40
python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model 3b --model-path $M3B \
  --calib-dir $DATA/calibration/3b --cells-json $SHM/vconf_node2_rep_cells.json \
  --gpus $GPUS --workers-per-gpu 4 --no-raw --inject-from-metadata
python -u -m anamnesis.scripts.vmb_a5_replay_multicell --model 3b --model-path $M3B \
  --calib-dir $DATA/calibration/3b --cells-json $SHM/vconf_node2_rep_cells.json \
  --gpus $GPUS --workers-per-gpu 4 --no-raw --sig-subdir signatures_v3_noinject
python -u -m anamnesis.scripts.vmb_c3_entropy_replay --model 3b --model-path $M3B \
  --c3-run-dir $RUN_ROOT --cells $CELLS --out-json $RUN_ROOT/vconf_entropy_3b_node2.json
python - <<'PY'
import json, pathlib
for md in pathlib.Path("$RUN_ROOT").glob("*/metadata.json"):
    d = json.loads(md.read_text()); d["compute_node"] = "node2"
    md.write_text(json.dumps(d))
ej = pathlib.Path("$RUN_ROOT/vconf_entropy_3b_node2.json")
d = json.loads(ej.read_text()); d["compute_node"] = "node2"; ej.write_text(json.dumps(d, indent=1))
print("node2 stamps written")
PY
echo "=== VCONF NODE2 CHAIN DONE \$(date -u +%FT%TZ) ==="
} 2>&1 | tee \$LOG
EOF

# sync-off (volatile /dev/shm rule): pull the whole run root local immediately
mkdir -p "$REPO/outputs/battery/annex/vmb_vconf_3b"
rsync -a node2:$RUN_ROOT/ "$REPO/outputs/battery/annex/vmb_vconf_3b/"
echo "VCONF chain complete; artifacts synced to outputs/battery/annex/vmb_vconf_3b/"
