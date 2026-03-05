#!/usr/bin/env python3
"""Run the unified analysis gauntlet on signature data.

Usage:
    python -m anamnesis.scripts.run_unified_analysis --run 8b_baseline
    python -m anamnesis.scripts.run_unified_analysis --run 8b_baseline --resume
    python -m anamnesis.scripts.run_unified_analysis --run 8b_baseline --all-reps
    python -m anamnesis.scripts.run_unified_analysis --run 8b_baseline --skip 8 9
    python -m anamnesis.scripts.run_unified_analysis --sig-dir path/to/signatures --run my_run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# Known run configurations
KNOWN_RUNS: dict[str, Path] = {
    "8b_baseline": Path("outputs/runs/run_8b_baseline/signatures"),
    "3b_run4": Path(os.environ.get(
        "ANAMNESIS_LEGACY_DATA", "phase_0"
    )) / "outputs" / "runs" / "run4_format_controlled" / "signatures",
    "8b_v2": Path("outputs/runs/8b_fat_01/signatures_v2"),
    "3b_v2": Path("outputs/runs/3b_fat_01/signatures_v2"),
}

# Addon directories for runs with split feature computation
KNOWN_ADDONS: dict[str, list[Path]] = {
    "8b_v2": [
        Path("outputs/runs/8b_fat_01/signatures_v2_addon"),
        Path("outputs/runs/8b_fat_01/signatures_v2_contrastive"),
    ],
    "3b_v2": [
        Path("outputs/runs/3b_fat_01/signatures_v2_contrastive"),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified analysis runner")
    parser.add_argument(
        "--run", type=str, required=True,
        help=f"Run name. Known: {list(KNOWN_RUNS.keys())}. Or use --sig-dir for custom path.",
    )
    parser.add_argument(
        "--sig-dir", type=str, default=None,
        help="Custom signature directory (overrides --run lookup).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Custom output directory.",
    )
    parser.add_argument(
        "--all-reps", action="store_true",
        help="Use all repetitions (core_only=False).",
    )
    parser.add_argument(
        "--skip", type=int, nargs="+", default=[],
        help="Section numbers to skip (1-10).",
    )
    parser.add_argument(
        "--addon-dirs", type=str, nargs="+", default=None,
        help="Additional signature directories to merge (for split feature sets).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint — skip sections that already have results.",
    )
    parser.add_argument(
        "--modes", type=str, default=None,
        help="Comma-separated list of modes to include (e.g. 'linear,socratic,contrastive,dialectical,analogical').",
    )
    args = parser.parse_args()

    # Resolve signature directory
    if args.sig_dir:
        sig_dir = Path(args.sig_dir)
    elif args.run in KNOWN_RUNS:
        sig_dir = KNOWN_RUNS[args.run]
    else:
        print(f"Unknown run '{args.run}'. Use --sig-dir or one of: {list(KNOWN_RUNS.keys())}")
        sys.exit(1)

    if not sig_dir.exists():
        print(f"Signature directory not found: {sig_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None

    # Resolve addon directories
    addon_dirs: list[Path] | None = None
    if args.addon_dirs:
        addon_dirs = [Path(d) for d in args.addon_dirs]
    elif args.run in KNOWN_ADDONS:
        addon_dirs = KNOWN_ADDONS[args.run]

    # Parse mode filter
    mode_filter: list[str] | None = None
    if args.modes:
        mode_filter = [m.strip() for m in args.modes.split(",")]
        print(f"Mode filter: {mode_filter}")

    # Auto-resolve run name for known runs with addons when using mode filter
    # This lets --run 8b_v2 --modes ... use the right sig dir and addons
    run_name = args.run
    if mode_filter and not args.output_dir:
        # Auto-suffix output dir so we don't overwrite the full results
        n_modes = len(mode_filter)
        output_dir = Path(f"outputs/analysis/{run_name}_{n_modes}way")

    from anamnesis.analysis.unified_runner import run_full_analysis

    run_full_analysis(
        signature_dir=sig_dir,
        run_name=run_name,
        output_dir=output_dir,
        core_only=not args.all_reps,
        skip_sections=set(args.skip) if args.skip else None,
        resume=args.resume,
        addon_dirs=addon_dirs,
        mode_filter=mode_filter,
    )


if __name__ == "__main__":
    main()
