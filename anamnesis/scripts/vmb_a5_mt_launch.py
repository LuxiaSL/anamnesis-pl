"""A5 matched-token cell driver: for every on-policy-gate PASS cell, replay the
A4-convention banked continuations (2 per class x 80 classes) under injection.

Reads the gate report (vmb_a5_onpolicy_gate output) at RUN time; FAIL cells are
skipped and logged — the gate value ships in the record either way. Deltas are
vs the banked UNSTEERED Stage-0 signatures of the same gens (12b seed-floor
ruler; bitwise determinism makes them pure effect).
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
    args = ap.parse_args()

    report = json.loads(args.gate_report.read_text())
    npz = args.vectors_dir / "a5_vectors.npz"
    ran = skipped = failed = 0
    for cell, info in sorted(report["cells"].items()):
        if not info["PASS"]:
            logger.warning(f"SKIP {cell}: gate FAIL (agreement {info['agreement_mean']:.3f})")
            skipped += 1
            continue
        key = cell.rsplit("_a", 1)[0]
        out_dir = args.out_root / cell
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "-m", "anamnesis.scripts.parallel_replay",
               "--model", args.model, "--model-path", args.model_path,
               "--run-dir", str(out_dir), "--calib-dir", str(args.calib_dir),
               "--manifest", str(args.stage0_run / "replay_manifest.json"),
               "--gpus", args.gpus, "--workers-per-gpu", str(args.workers_per_gpu),
               "--no-raw",
               "--inject-npz", str(npz), "--inject-key", key,
               "--inject-layer", str(MAP_SITE),
               "--inject-alpha", str(info["alpha_abs"]),
               "--inject-alpha-frac", str(info["alpha_frac"]),
               "--gen-ids", *[str(g) for g in GEN_IDS]]
        logger.info(f"RUN {cell} (agreement {info['agreement_mean']:.3f})")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            logger.error(f"{cell}: replay launcher rc={rc}")
            failed += 1
        else:
            ran += 1
    logger.info(f"matched-token cells: {ran} ran, {skipped} gate-skipped, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
