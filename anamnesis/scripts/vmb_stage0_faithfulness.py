"""vmb Stage-0 faithfulness-floor driver (prereg-vmb-v1 §5, addendum 2026-07-12a item 3).

20 continuations per model × 10 replays each, STRATIFIED:
  - 4 replays on a single PINNED device  → within-device component (pure replay determinism)
  - 6 replays spread across other devices → cross-device component (operational jitter)

Continuation selection (addendum item 6): the seed-index-0 generation of each topic,
one per topic, 5 per task stratum via stratum = topic_idx // 5 (deterministic).
Floor gid layout: gid = (stratum·20 + topic)·10 + 0.

Mechanics: builds a SYNTHETIC replay manifest where replay instance r of continuation c
gets gen_id = c·10 + r (all ten entries share the same input_ids), then invokes
run_replay_extraction once per device with exactly the gen-ids scheduled there
(CUDA_VISIBLE_DEVICES pinned). Signatures land in <out-dir>/signatures_v3; raw is NOT
banked (--no-raw: the jitter lives in the signatures; the base gen's raw is banked by
the stochastic replay pass). A replay_index.json maps each signature to
(continuation_id, replay_idx, device) for floors.compute_faithfulness_floors.

Usage (node1):
    PYTHONPATH=pipeline python -m anamnesis.scripts.vmb_stage0_faithfulness \
        --model 3b --model-path /models/llama-3.2-3b-instruct \
        --floor-run-dir /models/anamnesis-extract/runs/vmb_stage0_3b \
        --calib-dir /models/anamnesis-extract/calibration/3b \
        --out-dir /models/anamnesis-extract/runs/vmb_stage0_3b/faithfulness \
        --pinned-gpu 0 --spread-gpus 1,2,3
"""
from __future__ import annotations

import argparse

from anamnesis.scripts._gpu import resolve_physical_gpus
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from anamnesis.config import MODEL_PRESETS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N_TOPICS = 20
SEEDS_PER_CLASS = 10
N_REPLAYS = 10
N_PINNED = 4          # within-device component
# remaining N_REPLAYS - N_PINNED spread across --spread-gpus round-robin


def select_continuations(floor_manifest: dict) -> dict[int, dict]:
    """continuation_id (0..19 = topic_idx) → source manifest entry (seed-0 gen per topic)."""
    entries = floor_manifest["entries"]
    out: dict[int, dict] = {}
    for topic_idx in range(N_TOPICS):
        stratum_idx = topic_idx // 5          # 5 topics per stratum, deterministic
        gid = (stratum_idx * N_TOPICS + topic_idx) * SEEDS_PER_CLASS + 0
        e = entries.get(str(gid))
        if e is None:
            raise KeyError(f"floor manifest missing gid {gid} (topic {topic_idx}, stratum {stratum_idx})")
        out[topic_idx] = e
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="vmb Stage-0 faithfulness floors (stratified replays)")
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--floor-run-dir", type=Path, required=True,
                    help="Stage-0 stochastic floor run dir (holds replay_manifest.json)")
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pinned-gpu", default="0")
    ap.add_argument("--spread-gpus", default="1,2,3",
                    help="Devices for the cross-device component (round-robin)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    floor_manifest = json.loads((args.floor_run_dir / "replay_manifest.json").read_text())
    conts = select_continuations(floor_manifest)

    # ── Synthetic manifest: replay instance gid = continuation·10 + replay_idx ──
    entries: dict[str, dict] = {}
    index: list[dict] = []
    pinned = resolve_physical_gpus([args.pinned_gpu.strip()])[0]
    args.pinned_gpu = pinned
    spread = resolve_physical_gpus(
        [g.strip() for g in args.spread_gpus.split(",") if g.strip()])
    device_of: dict[int, str] = {}
    for cont_id, src in sorted(conts.items()):
        for r in range(N_REPLAYS):
            gid = cont_id * N_REPLAYS + r
            entries[str(gid)] = dict(src)
            device = args.pinned_gpu if r < N_PINNED else spread[(r - N_PINNED) % len(spread)]
            device_of[gid] = device
            index.append({"sig": f"gen_{gid:03d}", "continuation_id": cont_id,
                          "replay_idx": r, "device": f"gpu{device}",
                          "component": "within" if r < N_PINNED else "cross"})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "replay_manifest.json").write_text(
        json.dumps({"entries": entries, "n_ok": len(entries), "n_flagged": 0, "flagged": []}))
    (args.out_dir / "replay_index.json").write_text(json.dumps(index, indent=1))
    logger.info(f"{len(conts)} continuations × {N_REPLAYS} replays = {len(entries)} instances; "
                f"pinned gpu{args.pinned_gpu} ×{N_PINNED}, spread {spread}")

    if args.dry_run:
        for e in index[:12]:
            logger.info(f"  {e}")
        return

    # ── One worker per device, given exactly its scheduled gen-ids ──
    by_device: dict[str, list[int]] = {}
    for gid, dev in device_of.items():
        by_device.setdefault(dev, []).append(gid)

    procs = []
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    python = sys.executable
    for dev, gids in sorted(by_device.items()):
        cmd = [python, "-m", "anamnesis.scripts.run_replay_extraction",
               "--model", args.model, "--model-path", args.model_path,
               "--run-dir", str(args.out_dir), "--calib-dir", str(args.calib_dir),
               "--manifest", str(args.out_dir / "replay_manifest.json"),
               "--gen-ids", *[str(g) for g in sorted(gids)],
               "--no-raw", "--label", f"faith-gpu{dev}"]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": dev,
               "PYTHONPATH": os.environ.get("PYTHONPATH", "."),
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
        log_path = log_dir / f"faith_gpu{dev}.log"
        fh = open(log_path, "w")
        procs.append((dev, subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT), fh))
        logger.info(f"  device gpu{dev}: {len(gids)} replays → {log_path}")

    t0 = time.time()
    fails = 0
    for dev, proc, fh in procs:
        rc = proc.wait()
        fh.close()
        if rc != 0:
            fails += 1
            logger.error(f"device gpu{dev} worker exited rc={rc}")
    n_sigs = len(list((args.out_dir / "signatures_v3").glob("gen_*.npz")))
    logger.info(f"faithfulness replays done in {time.time()-t0:.0f}s ({fails} failed workers); "
                f"{n_sigs}/{len(entries)} signatures")


if __name__ == "__main__":
    main()
