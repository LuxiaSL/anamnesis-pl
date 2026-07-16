"""Bitwise confidence gate for multi-cell REPLAY: replay a few gens of 2 existing banked
cells (DIFFERENT injection layers, to exercise per-cell hook re-attach) via the OLD
per-cell parallel_replay and the NEW multi-cell path into separate sig-subdirs, then diff
every signature (.npz + .json) byte-for-byte. PASS = load-once replay is provably
identical to reload-per-cell replay.

Uses inject_from_metadata (each cell's own a5_injection). Writes to sig_smoke_old /
sig_smoke_new so it never touches the real signatures_v3.

Usage: python -m anamnesis.scripts.vmb_a5_replay_multicell_smoke --model 8b \
    --model-path <p> --calib-dir <c> --cellA <run>/V3_L16_L16_a0.1 \
    --cellB <run>/V3_L20_L20_a0.1 --gen-ids 0 1 10 11
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

OLD, NEW = "sig_smoke_old", "sig_smoke_new"


def run(cmd):
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"cmd failed rc={r.returncode}: {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--cellA", type=Path, required=True)
    ap.add_argument("--cellB", type=Path, required=True)
    ap.add_argument("--gen-ids", type=int, nargs="+", default=[0, 1, 10, 11])
    args = ap.parse_args()
    cells = [args.cellA, args.cellB]
    gids = [str(g) for g in args.gen_ids]

    # ── OLD: one parallel_replay per cell (reloads model+calib each) ──
    for c in cells:
        run([sys.executable, "-m", "anamnesis.scripts.parallel_replay",
             "--model", args.model, "--model-path", args.model_path,
             "--run-dir", str(c), "--calib-dir", args.calib_dir,
             "--manifest", str(c / "replay_manifest.json"), "--gpus", "0",
             "--workers-per-gpu", "2", "--no-raw", "--no-resume",
             "--sig-subdir", OLD, "--inject-from-metadata", "--gen-ids", *gids])

    # ── NEW: one multi-cell replay for both cells (model+calib loaded once) ──
    cells_json = args.cellA.parent / "_replay_smoke_cells.json"
    cells_json.write_text(json.dumps({"cells": [
        {"run_dir": str(c), "manifest": str(c / "replay_manifest.json"),
         "gen_ids": args.gen_ids} for c in cells]}))
    run([sys.executable, "-m", "anamnesis.scripts.vmb_a5_replay_multicell",
         "--model", args.model, "--model-path", args.model_path, "--calib-dir", args.calib_dir,
         "--cells-json", str(cells_json), "--gpus", "0", "--workers-per-gpu", "2",
         "--no-raw", "--no-resume", "--sig-subdir", NEW, "--inject-from-metadata"])

    # ── DIFF every signature byte-for-byte (.npz + .json) ──
    all_ok = True
    for c in cells:
        od, nd = c / OLD, c / NEW
        cell_ok = True
        for g in args.gen_ids:
            for ext in ("npz", "json"):
                of, nf = od / f"gen_{g:03d}.{ext}", nd / f"gen_{g:03d}.{ext}"
                if not (of.exists() and nf.exists() and of.read_bytes() == nf.read_bytes()):
                    cell_ok = False
        print(f"cell {c.name}: {len(args.gen_ids)} gens, "
              f"{'BITWISE-IDENTICAL' if cell_ok else 'MISMATCH'}")
        all_ok &= cell_ok
    print("REPLAY_MULTICELL_SMOKE:", "PASS" if all_ok else "FAIL")
    if not all_ok:
        raise SystemExit("replay multicell bitwise smoke FAILED")


if __name__ == "__main__":
    main()
