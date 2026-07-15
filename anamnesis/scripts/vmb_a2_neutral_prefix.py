"""A2-ii — length-matched NEUTRAL-prefix control (session-5 item 5; P5 pre-declared 2026-07-14).

The A2 prefix-swap residue (0.16–0.30× seed floor, attention/lookback, 3 families) is EMBARGOED
behind a length confound: the swapped system prompt differs in TOKEN LENGTH from native, so the
residue could be a prompt-length KV-cache artifact, not instruction-content carriage. This control
replays the SAME continuations under a content-NEUTRAL system prompt, token-length-matched to each
swap variant EXACTLY (target = the banked swap entry's prompt_length → perfect per-(swap,gid)
pairing). Neutral content = a generic assistant instruction padded with a neutral single-token
filler to hit the target length (content-neutral by construction; it carries no mode semantics).

Analyzer `vmb_a2ii_control.py` compares swap-residue vs neutral-residue per continuation (Wilcoxon,
BH). Un-embargo iff swap EXCEEDS neutral on the flagged attention cells (frozen in the declaration).
First-read → outer loop; nothing stamped. Output: vmb_a2_<model>_neutral_prefix (swap index schema).
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
from anamnesis.scripts.vmb_stage0_generate import VMB_CANONICAL_DATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NEUTRAL_BASE = "You are a helpful assistant."
FILLER = " and"   # a common single-token filler; content-neutral length padding


def _measure(tok, sys_content: str, user: str) -> tuple[int, list[int]]:
    res = tok.apply_chat_template(
        [{"role": "system", "content": sys_content}, {"role": "user", "content": user}],
        add_generation_prompt=True, date_string=VMB_CANONICAL_DATE)
    ids = list(res["input_ids"] if hasattr(res, "keys") else res)
    return len(ids), ids


def _neutral_ids(tok, user: str, target: int) -> tuple[list[int], int]:
    """Build a neutral prompt whose total prompt_length == target (best-effort exact)."""
    n_base, _ = _measure(tok, NEUTRAL_BASE, user)
    n = max(0, target - n_base)
    ids = None
    for _ in range(8):
        L, ids = _measure(tok, NEUTRAL_BASE + FILLER * n, user)
        if L == target:
            return ids, L
        n = max(0, n + (target - L))
    return ids, len(ids)   # best-effort; caller records actual


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--a2-root", type=Path, required=True)
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path)
    swap_dir = args.a2_root / f"vmb_a2_{args.model}_prefix_swap"
    out_dir = args.a2_root / f"vmb_a2_{args.model}_neutral_prefix"
    out_dir.mkdir(parents=True, exist_ok=True)

    swap_index = json.loads((swap_dir / "prefix_swap_index.json").read_text())
    swap_manifest = json.loads((swap_dir / "replay_manifest.json").read_text())["entries"]
    md_cache: dict[str, dict] = {}

    def user_prompt(entry) -> str:
        src_dir = Path(entry["native_sig_dir"]).parent
        if str(src_dir) not in md_cache:
            md = json.loads((src_dir / "metadata.json").read_text())
            md_cache[str(src_dir)] = {int(g["generation_id"]): g for g in md["generations"]}
        return md_cache[str(src_dir)][int(entry["source_gen_id"])]["user_prompt"]

    entries: dict[str, dict] = {}
    index: list[dict] = []
    n_exact = 0
    for e in swap_index:
        gid = int(e["sig"].split("_")[1])
        sw = swap_manifest[str(gid)]
        target_plen = int(sw["prompt_length"])
        gen_ids = sw["input_ids"][sw["prompt_length"]:]
        np_ids, actual = _neutral_ids(tok, user_prompt(e), target_plen)
        n_exact += int(actual == target_plen)
        entries[str(gid)] = {"input_ids": np_ids + list(gen_ids),
                             "prompt_length": actual, "n_gen": len(gen_ids),
                             "target_plen": target_plen}
        index.append({**e, "neutral_prompt_length": actual, "target_plen": target_plen})

    (out_dir / "replay_manifest.json").write_text(
        json.dumps({"entries": entries, "n_ok": len(entries), "n_flagged": 0, "flagged": []}))
    # same index schema the analyzer keys on (sig, swap, source_gen_id, native_sig_dir, native_sig)
    (out_dir / "prefix_swap_index.json").write_text(json.dumps(index, indent=1))
    logger.info(f"{len(entries)} neutral-prefix replays staged → {out_dir} "
                f"({n_exact}/{len(entries)} hit target length EXACTLY)")
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
    logger.info(f"neutral-prefix replay rc={rc}; {n}/{len(entries)} signatures")


if __name__ == "__main__":
    main()
