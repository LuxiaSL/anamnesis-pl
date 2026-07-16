"""Bitwise confidence gate for the multi-cell gen path: run 2 cells (DIFFERENT layers,
to exercise the per-cell hook re-attach) via the OLD single-cell launcher and the NEW
multi-cell launcher, then diff every gen_record byte-for-byte. PASS = the load-once path
is provably identical to the reload-per-cell path.

Usage: python -m anamnesis.scripts.vmb_a5_multicell_smoke --model 8b --model-path <p> \
    --npz <a5_vectors.npz> --norms <stamps.json> --scratch /tmp/mc_smoke
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROMPTS = "pipeline/anamnesis/prompts/prompt_sets.json"
LIMIT, SEEDS, TOK = 6, 1, 16   # tiny + fast; identity holds at any size


def run(cmd, env=None):
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"cmd failed rc={r.returncode}: {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--norms", required=True)
    ap.add_argument("--scratch", type=Path, required=True)
    # two cells at DIFFERENT layers (default: 8B map L16 + sweep L20)
    ap.add_argument("--cellA", default="V3_L16:16:0.1")   # key:layer:frac
    ap.add_argument("--cellB", default="V3_L20:20:0.1")
    args = ap.parse_args()

    cells = []
    for spec in (args.cellA, args.cellB):
        key, layer, frac = spec.split(":")
        cells.append({"key": key, "layer": int(layer), "frac": float(frac),
                      "ns": f"MCSMOKE-{args.model.upper()}-{key}"})

    old_root = args.scratch / "old"
    new_root = args.scratch / "new"

    # ── OLD: one single-cell invocation per cell (reloads model each) ──
    for c in cells:
        run([sys.executable, "-m", "anamnesis.scripts.vmb_stage0_generate",
             "--model", args.model, "--model-path", args.model_path, "--prompts", PROMPTS,
             "--out-run-dir", str(old_root / c["key"]), "--gpus", "0", "--workers-per-gpu", "2",
             "--seeds-per-class", str(SEEDS), "--limit", str(LIMIT), "--max-new-tokens", str(TOK),
             "--seed-namespace", c["ns"], "--inject-npz", args.npz, "--inject-key", c["key"],
             "--inject-layer", str(c["layer"]), "--inject-alpha-frac", str(c["frac"]),
             "--inject-norms-json", args.norms])

    # ── NEW: one multi-cell invocation for both cells (model loaded once) ──
    cells_json = args.scratch / "cells.json"
    cells_json.write_text(json.dumps({"cells": [
        {"out_run_dir": str(new_root / c["key"]), "seed_namespace": c["ns"],
         "inject_key": c["key"], "inject_layer": c["layer"], "inject_alpha_frac": c["frac"]}
        for c in cells]}))
    run([sys.executable, "-m", "anamnesis.scripts.vmb_a5_gen_multicell",
         "--model", args.model, "--model-path", args.model_path, "--prompts", PROMPTS,
         "--cells-json", str(cells_json), "--gpus", "0", "--workers-per-gpu", "2",
         "--seeds-per-class", str(SEEDS), "--limit", str(LIMIT), "--max-new-tokens", str(TOK),
         "--inject-npz", args.npz, "--inject-norms-json", args.norms])

    # ── DIFF every gen_record byte-for-byte ──
    all_ok = True
    for c in cells:
        od = old_root / c["key"] / "gen_records"
        nd = new_root / c["key"] / "gen_records"
        of = sorted(od.glob("gen_*.json"))
        ok = True
        for f in of:
            nb = nd / f.name
            if not nb.exists() or f.read_bytes() != nb.read_bytes():
                ok = False
        print(f"cell {c['key']} (L{c['layer']}): {len(of)} gens, "
              f"{'BITWISE-IDENTICAL' if ok else 'MISMATCH'}")
        all_ok &= ok and len(of) == LIMIT
    print("MULTICELL_SMOKE:", "PASS" if all_ok else "FAIL")
    if not all_ok:
        raise SystemExit("multicell bitwise smoke FAILED")


if __name__ == "__main__":
    main()
