"""A5 matched-token cell driver: for every on-policy-gate PASS cell, replay the
A4-convention banked continuations (2 per class x 80 classes) under injection.

Reads the gate report (vmb_a5_onpolicy_gate output) at RUN time; FAIL cells are
skipped and logged — the gate value ships in the record either way. Deltas are
vs the banked UNSTEERED Stage-0 signatures of the same gens (12b seed-floor
ruler; bitwise determinism makes them pure effect).

Path of record (canonical-ops 2026-07-16, backlog #2 closed): ONE cells-json with
per-cell explicit injection specs through vmb_a5_replay_multicell — the model
loads once per worker and loops every PASS cell (the old per-cell parallel_replay
loop reloaded the model per cell, and now also trips the path-of-record guard).
--path legacy-loop keeps the old behavior for the bitwise parity smoke ONLY.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GEN_IDS = [k * 10 + s for k in range(80) for s in (0, 1)]  # A4 convention
MAP_SITE = 14


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--vectors-dir", type=Path, required=True)
    ap.add_argument("--gate-report", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--workers-per-gpu", type=int, default=8)
    ap.add_argument("--gen-ids", type=int, nargs="+", default=None,
                    help="override the A4-convention gen ids (smoke subsets)")
    ap.add_argument("--sig-subdir", default="signatures_v3")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--path", choices=["multicell", "legacy-loop"], default="multicell",
                    help="legacy-loop = the old per-cell parallel_replay loop; kept "
                         "ONLY for the bitwise parity smoke (reloads model per cell)")
    args = ap.parse_args()

    report = json.loads(args.gate_report.read_text())
    npz = args.vectors_dir / "a5_vectors.npz"
    gen_ids = args.gen_ids if args.gen_ids is not None else GEN_IDS
    manifest = args.stage0_run / "replay_manifest.json"

    cells, skipped = [], 0
    for cell, info in sorted(report["cells"].items()):
        if not info["PASS"]:
            logger.warning(f"SKIP {cell}: gate FAIL (agreement {info['agreement_mean']:.3f})")
            skipped += 1
            continue
        key = cell.rsplit("_a", 1)[0]
        site = int(info.get("site", MAP_SITE))
        out_dir = args.out_root / cell
        out_dir.mkdir(parents=True, exist_ok=True)
        cells.append({
            "cell": cell, "key": key, "site": site,
            "alpha_abs": float(info["alpha_abs"]),
            "alpha_frac": float(info["alpha_frac"]),
            "agreement": float(info["agreement_mean"]),
            "out_dir": out_dir,
        })

    if not cells:
        logger.warning(f"no PASS cells in {args.gate_report} — nothing to run")
        return

    if args.path == "legacy-loop":
        # OLD path: one parallel_replay per cell (model reload per cell). Kept for
        # the bitwise smoke; --single-cell-ok waives the path-of-record guard for
        # this deliberate case.
        ran = failed = 0
        for c in cells:
            cmd = [sys.executable, "-u", "-m", "anamnesis.scripts.parallel_replay",
                   "--single-cell-ok",
                   "--model", args.model, "--model-path", args.model_path,
                   "--run-dir", str(c["out_dir"]), "--calib-dir", str(args.calib_dir),
                   "--manifest", str(manifest),
                   "--gpus", args.gpus, "--workers-per-gpu", str(args.workers_per_gpu),
                   "--no-raw", "--sig-subdir", args.sig_subdir,
                   "--inject-npz", str(npz), "--inject-key", c["key"],
                   "--inject-layer", str(c["site"]),
                   "--inject-alpha", str(c["alpha_abs"]),
                   "--inject-alpha-frac", str(c["alpha_frac"]),
                   "--gen-ids", *[str(g) for g in gen_ids]]
            if args.no_resume:
                cmd.append("--no-resume")
            logger.info(f"RUN {c['cell']} (agreement {c['agreement']:.3f}) [legacy-loop]")
            if subprocess.run(cmd).returncode != 0:
                logger.error(f"{c['cell']}: replay launcher rc!=0")
                failed += 1
            else:
                ran += 1
        logger.info(f"matched-token cells: {ran} ran, {skipped} gate-skipped, "
                    f"{failed} failed [legacy-loop]")
        sys.exit(1 if failed else 0)

    # NEW path of record: one multicell invocation, model loads once per worker.
    cells_json = args.out_root / "_mt_cells.json"
    cells_json.write_text(json.dumps({"cells": [
        {"run_dir": str(c["out_dir"]), "manifest": str(manifest),
         "gen_ids": list(gen_ids),
         "inject_npz": str(npz), "inject_key": c["key"],
         "inject_layer": c["site"], "inject_alpha": c["alpha_abs"],
         "inject_alpha_frac": c["alpha_frac"]}
        for c in cells]}, indent=1))
    for c in cells:
        logger.info(f"QUEUED {c['cell']} (agreement {c['agreement']:.3f})")
    cmd = [sys.executable, "-u", "-m", "anamnesis.scripts.vmb_a5_replay_multicell",
           "--model", args.model, "--model-path", args.model_path,
           "--calib-dir", str(args.calib_dir), "--cells-json", str(cells_json),
           "--gpus", args.gpus, "--workers-per-gpu", str(args.workers_per_gpu),
           "--no-raw", "--sig-subdir", args.sig_subdir]
    if args.no_resume:
        cmd.append("--no-resume")
    rc = subprocess.run(cmd).returncode
    logger.info(f"matched-token cells: {len(cells)} queued via multicell (rc={rc}), "
                f"{skipped} gate-skipped")
    sys.exit(rc)


if __name__ == "__main__":
    main()
