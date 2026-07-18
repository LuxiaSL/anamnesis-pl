"""Build the checkpoints-json for a Part-C arm's dense-grid battery replay (anamnesis s12).

Emits the {"checkpoints":[{"label","adapter_path","run_dir"}, ...]} schema consumed by
run_replay_multickpt.py (LoRA-adapter swap path — cell-3b SGD, cell-4 DPO arms). Enumerates
checkpoint-XXXX + final under an arm's checkpoint dir and points each at a per-step run_dir
under the cohort run-root, mirroring the reference cohort layout:

    /models/anamnesis-extract/runs/vmb_a6cohort_qwen/<arm>/step-XXXX/signatures_v3

⚠ Full-FT (cell-3a) checkpoints are FULL-WEIGHT — NOT adapter-swappable. They use the
load-per-checkpoint path (run_replay_extraction.py with --model-path=<checkpoint>), NOT this
swap driver. This builder refuses a dir whose checkpoints lack adapter_config.json.

Usage:
    python -m anamnesis.scripts.build_partc_replay_cells \
        --ckpt-dir /dev/shm/luxi-anamnesis/partc/cell4/checkpoints/qwen_cat_dpo_r16_s0 \
        --arm cat_dpo_r16_s0 \
        --run-root /models/anamnesis-extract/runs/vmb_a6cohort_qwen \
        --out cells_cat_dpo.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--arm", required=True, help="cohort subdir label, e.g. cat_dpo_r16_s0")
    ap.add_argument("--run-root", type=Path,
                    default=Path("/models/anamnesis-extract/runs/vmb_a6cohort_qwen"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--include-final", action="store_true", default=True)
    args = ap.parse_args()

    ckpts = []
    steps = sorted(
        [d for d in args.ckpt_dir.glob("checkpoint-*") if d.is_dir()],
        key=lambda d: int(re.search(r"checkpoint-(\d+)", d.name).group(1)))
    if args.include_final and (args.ckpt_dir / "final").is_dir():
        steps.append(args.ckpt_dir / "final")

    if not steps:
        raise SystemExit(f"no checkpoint-*/final under {args.ckpt_dir}")

    for d in steps:
        if not (d / "adapter_config.json").exists():
            raise SystemExit(
                f"{d} has no adapter_config.json — this is a FULL-WEIGHT checkpoint. "
                f"Use the load-per-checkpoint path (run_replay_extraction.py --model-path=<ckpt>), "
                f"NOT run_replay_multickpt (the swap driver only handles LoRA adapters).")
        label = d.name  # checkpoint-0075 | final
        run_dir = args.run_root / args.arm / label.replace("checkpoint-", "step-")
        ckpts.append({"label": f"{args.arm}-{label}",
                      "adapter_path": str(d),
                      "run_dir": str(run_dir)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"checkpoints": ckpts}, indent=1))
    print(f"wrote {len(ckpts)} checkpoints -> {args.out}")
    for c in ckpts:
        print(f"  {c['label']:32s} -> {c['run_dir']}")


if __name__ == "__main__":
    main()
