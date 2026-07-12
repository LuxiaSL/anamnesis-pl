"""Arm A2 cell (ii): matched-token prefix-swap replay — prereg-vmb-v1 §2c A2.

The SAME realized continuation is replayed under a SWAPPED system prompt: for each
swap pair (system_mode → execution_mode), take pure-execution-mode continuations and
rebuild input_ids = tokenize(chat(system=swap_system_prompt, user=same_user_prompt))
+ same generated token ids. The native replay (original prompt) is the condition's
own signature from the corpus replay pass — so only the swapped-prefix variants run here.

Prediction (§2c): near-floor overall; any residue CONFINED to prompt-region-lookback /
attention families ("the prompt is in the cache; carriage without execution should read
as lookback-only"). The faithfulness floor is bitwise-zero (Stage-0 measured), so the
addendum-2026-07-12b replay ruler applies: deltas reported in SEED-floor units,
visibility bar = 0.1× the cell's stochastic-floor median.

Output layout: <out-dir>/replay_manifest.json (synthetic gids), signatures_v3/,
prefix_swap_index.json mapping sig → (swap label, source condition, source gen id).

Usage (per model, after vmb_a2_generate + its corpus replay):
    PYTHONPATH=pipeline python -m anamnesis.scripts.vmb_a2_prefix_replay \
        --model 3b --model-path /models/llama-3.2-3b-instruct \
        --a2-root /models/anamnesis-extract/runs \
        --calib-dir /models/anamnesis-extract/calibration/3b \
        --gpus 0,1,2,3 --per-swap 20
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer

from anamnesis.config import MODEL_PRESETS
from anamnesis.modes.prompt_swap import PROMPT_SWAP_PAIRS
from anamnesis.scripts.vmb_stage0_generate import VMB_CANONICAL_DATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--a2-root", type=Path, required=True,
                    help="Dir holding the vmb_a2_<model>_<cond> run dirs")
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--per-swap", type=int, default=20,
                    help="Continuations per swap pair (topic-stratified: seed_idx 0 per topic)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path)
    out_dir = args.a2_root / f"vmb_a2_{args.model}_prefix_swap"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: dict[str, dict] = {}
    index: list[dict] = []
    gid_counter = 0
    for pair in PROMPT_SWAP_PAIRS:
        # Source condition = PURE EXECUTION mode (its native replay already exists)
        src_dir = args.a2_root / f"vmb_a2_{args.model}_pure_{pair.execution_mode}"
        manifest = json.loads((src_dir / "replay_manifest.json").read_text())
        md = json.loads((src_dir / "metadata.json").read_text())
        gens = {int(g["generation_id"]): g for g in md["generations"]}

        # topic-stratified: seed_idx 0 per topic → gid = topic_idx * k + 0; k inferred
        k = max(int(g["repetition"]) for g in md["generations"]) + 1
        chosen = [t * k for t in range(min(args.per_swap, 20))]

        for src_gid in chosen:
            g = gens[src_gid]
            e = manifest["entries"][str(src_gid)]
            gen_ids = e["input_ids"][e["prompt_length"]:]
            # Rebuild the prompt with the SWAP system prompt, same user prompt
            messages = [
                {"role": "system", "content": pair.get_system_prompt()},
                {"role": "user", "content": g["user_prompt"]},
            ]
            res = tok.apply_chat_template(messages, add_generation_prompt=True,
                                          date_string=VMB_CANONICAL_DATE)
            if hasattr(res, "keys"):
                res = res["input_ids"]
            new_prompt_ids = list(res)
            new_ids = new_prompt_ids + list(gen_ids)
            entries[str(gid_counter)] = {"input_ids": new_ids,
                                         "prompt_length": len(new_prompt_ids),
                                         "n_gen": len(gen_ids)}
            index.append({"sig": f"gen_{gid_counter:03d}", "swap": pair.label,
                          "source_condition": f"pure_{pair.execution_mode}",
                          "source_gen_id": src_gid, "topic_idx": g["topic_idx"],
                          "native_sig_dir": str(src_dir / "signatures_v3"),
                          "native_sig": f"gen_{src_gid:03d}"})
            gid_counter += 1

    (out_dir / "replay_manifest.json").write_text(
        json.dumps({"entries": entries, "n_ok": len(entries), "n_flagged": 0, "flagged": []}))
    (out_dir / "prefix_swap_index.json").write_text(json.dumps(index, indent=1))
    logger.info(f"{len(entries)} prefix-swap replays staged → {out_dir}")
    if args.dry_run:
        return

    env = {**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "."),
           "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    cmd = [sys.executable, "-u", "-m", "anamnesis.scripts.parallel_replay",
           "--model", args.model, "--model-path", args.model_path,
           "--run-dir", str(out_dir), "--calib-dir", str(args.calib_dir),
           "--manifest", str(out_dir / "replay_manifest.json"),
           "--gpus", args.gpus, "--workers-per-gpu", "2", "--no-raw"]
    rc = subprocess.run(cmd, env=env).returncode
    n = len(list((out_dir / "signatures_v3").glob("gen_*.npz")))
    logger.info(f"prefix-swap replay rc={rc}; {n}/{len(entries)} signatures")


if __name__ == "__main__":
    main()
