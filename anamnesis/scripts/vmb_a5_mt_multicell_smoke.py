"""Bitwise confidence gate for the MT multicell path (P3, canonical-ops 2026-07-16):
run 2 gate-PASS MT cells x a few gens through vmb_a5_mt_launch via the OLD
per-cell legacy-loop and the NEW one-cells-json multicell path, into separate
sig-subdirs, then diff every signature (.npz + .json) byte-for-byte. PASS =
load-once MT replay is provably identical to reload-per-cell MT replay.

Usage: python -m anamnesis.scripts.vmb_a5_mt_multicell_smoke --model 3b \
    --model-path <p> --stage0-run <runs>/vmb_stage0_3b --calib-dir <c> \
    --vectors-dir <battery>/a5_vectors_3b --gate-report <report.json> \
    --out-root /tmp/mt_smoke --gen-ids 0 1 10 11
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

OLD, NEW = "sig_smoke_old", "sig_smoke_new"


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"cmd failed rc={r.returncode}: {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--vectors-dir", type=Path, required=True)
    ap.add_argument("--gate-report", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--workers-per-gpu", type=int, default=2)
    ap.add_argument("--gen-ids", type=int, nargs="+", default=[0, 1, 10, 11])
    ap.add_argument("--n-cells", type=int, default=2)
    args = ap.parse_args()

    report = json.loads(args.gate_report.read_text())
    pass_cells = [c for c, i in sorted(report["cells"].items()) if i["PASS"]]
    if len(pass_cells) < args.n_cells:
        raise SystemExit(f"need {args.n_cells} gate-PASS cells, report has {len(pass_cells)}")
    cells = pass_cells[: args.n_cells]
    # subset report so both paths run exactly these cells
    sub = {"cells": {c: report["cells"][c] for c in cells}}
    sub_report = args.out_root / "_smoke_gate_report.json"
    args.out_root.mkdir(parents=True, exist_ok=True)
    sub_report.write_text(json.dumps(sub))

    base = [sys.executable, "-m", "anamnesis.scripts.vmb_a5_mt_launch",
            "--model", args.model, "--model-path", args.model_path,
            "--stage0-run", str(args.stage0_run), "--calib-dir", str(args.calib_dir),
            "--vectors-dir", str(args.vectors_dir), "--gate-report", str(sub_report),
            "--out-root", str(args.out_root), "--gpus", args.gpus,
            "--workers-per-gpu", str(args.workers_per_gpu),
            "--gen-ids", *[str(g) for g in args.gen_ids]]

    # OLD: per-cell legacy loop (reloads model per cell)
    run(base + ["--path", "legacy-loop", "--sig-subdir", OLD, "--no-resume"])
    # NEW: one cells-json through the multicell path (loads once per worker)
    run(base + ["--path", "multicell", "--sig-subdir", NEW, "--no-resume"])

    # DIFF every signature byte-for-byte (.npz + .json)
    all_ok = True
    for cell in cells:
        od, nd = args.out_root / cell / OLD, args.out_root / cell / NEW
        cell_ok = True
        for g in args.gen_ids:
            for ext in ("npz", "json"):
                of, nf = od / f"gen_{g:03d}.{ext}", nd / f"gen_{g:03d}.{ext}"
                if not (of.exists() and nf.exists() and of.read_bytes() == nf.read_bytes()):
                    cell_ok = False
        print(f"cell {cell}: {len(args.gen_ids)} gens, "
              f"{'BITWISE-IDENTICAL' if cell_ok else 'MISMATCH'}")
        all_ok &= cell_ok
    print("MT_MULTICELL_SMOKE:", "PASS" if all_ok else "FAIL")
    if not all_ok:
        raise SystemExit("MT multicell bitwise smoke FAILED")


if __name__ == "__main__":
    main()
