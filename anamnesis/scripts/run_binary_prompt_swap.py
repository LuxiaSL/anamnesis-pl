#!/usr/bin/env python3
"""Binary prompt-swap confound test.

For each swap type (e.g. swap_socratic→linear), trains a binary classifier
on the two relevant modes from core data, then predicts on swap samples.

If swap samples are classified as the execution mode → signal is execution-based.
If classified as the system-prompt mode → signal is confounded by prompt.

This matches the original protocol for apples-to-apples comparison.

Usage:
    python -m anamnesis.scripts.run_binary_prompt_swap --run 8b_v2
    python -m anamnesis.scripts.run_binary_prompt_swap --run 3b_v2
    python -m anamnesis.scripts.run_binary_prompt_swap --run 8b_v2 --run 3b_v2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from anamnesis.analysis.geometric_trio.data_loader import (
    TIER_KEYS,
    TIER_GROUPS,
    BASELINE_TIERS,
    ENGINEERED_TIERS,
    load_run4,
)
from anamnesis.config import RUNS


def _parse_swap_mode(mode_str: str) -> tuple[str, str] | None:
    """Parse 'swap_A→B' into (system_prompt_mode, execution_mode)."""
    match = re.match(r"swap_(\w+)→(\w+)", mode_str)
    if match:
        return match.group(1), match.group(2)
    return None


def _load_swap_samples(
    signature_dir: Path,
    addon_dirs: list[Path] | None,
) -> tuple[list[dict], dict[str, list[NDArray]]]:
    """Load all swap samples with their features from primary + addon dirs.

    Returns
    -------
    swap_info : list[dict]
        Per-sample metadata (system_prompt_mode, execution_mode, swap_name, etc.)
    swap_tier_features : dict[str, list[NDArray]]
        Per-tier feature arrays for swap samples, only tiers with complete coverage.
    """
    all_npz_files = sorted(signature_dir.glob("gen_*.npz"))
    if not all_npz_files:
        raise FileNotFoundError(f"No npz files in {signature_dir}")

    # Identify swap samples
    swap_info: list[dict] = []
    swap_npz_paths: list[Path] = []
    for npz_path in all_npz_files:
        json_path = npz_path.with_suffix(".json")
        if not json_path.exists():
            continue
        with open(json_path) as f:
            meta = json.load(f)
        parsed = _parse_swap_mode(meta.get("mode", ""))
        if parsed:
            swap_info.append({
                "file_stem": npz_path.stem,
                "system_prompt_mode": parsed[0],
                "execution_mode": parsed[1],
                "swap_name": meta["mode"],
                "topic": meta.get("topic", ""),
            })
            swap_npz_paths.append(npz_path)

    if not swap_info:
        raise ValueError(f"No prompt-swap samples found in {signature_dir}")

    # Load features from primary dir
    individual_tiers = list(TIER_KEYS.keys())
    swap_tier_features: dict[str, list[NDArray | None]] = {}

    for i, npz_path in enumerate(swap_npz_paths):
        npz_data = np.load(npz_path, allow_pickle=True)
        for tier_name in individual_tiers:
            npz_key = TIER_KEYS.get(tier_name, "")
            if npz_key and npz_key in npz_data.files:
                if tier_name not in swap_tier_features:
                    swap_tier_features[tier_name] = [None] * len(swap_npz_paths)
                swap_tier_features[tier_name][i] = npz_data[npz_key]

    # Merge addon features
    if addon_dirs:
        for addon_dir in addon_dirs:
            addon_path = Path(addon_dir)
            if not addon_path.exists():
                continue
            for i in range(len(swap_npz_paths)):
                stem = swap_info[i]["file_stem"]
                addon_npz = addon_path / f"{stem}.npz"
                if addon_npz.exists():
                    addon_data = np.load(addon_npz, allow_pickle=True)
                    for tier_name in individual_tiers:
                        npz_key = TIER_KEYS.get(tier_name, "")
                        if npz_key and npz_key in addon_data.files:
                            if tier_name not in swap_tier_features:
                                swap_tier_features[tier_name] = [None] * len(swap_npz_paths)
                            if swap_tier_features[tier_name][i] is None:
                                swap_tier_features[tier_name][i] = addon_data[npz_key]

    # Keep only tiers with complete coverage
    complete: dict[str, list[NDArray]] = {}
    for t, arrays in swap_tier_features.items():
        if all(a is not None for a in arrays):
            complete[t] = arrays  # type: ignore[assignment]

    # Build composites
    for group_name, members in TIER_GROUPS.items():
        available_members = [m for m in members if m in complete]
        if available_members:
            complete[group_name] = [
                np.concatenate([complete[m][j] for m in available_members])
                for j in range(len(swap_npz_paths))
            ]

    return swap_info, complete


def run_binary_prompt_swap(
    run_name: str,
    signature_dir: Path,
    addon_dirs: list[Path] | None = None,
) -> dict:
    """Run binary prompt-swap test for all swap types and all tiers.

    For each swap type (e.g. swap_socratic→linear):
    1. Load core data filtered to the two relevant modes
    2. Train binary RF per tier on these two modes
    3. Predict on swap samples
    4. Report: how many classified as execution mode vs system-prompt mode

    Returns structured results per swap type × per tier.
    """
    # Load core data (all modes, for slicing)
    core_data = load_run4(
        signature_dir=signature_dir,
        core_only=True,
        addon_dirs=addon_dirs,
    )

    # Load swap samples
    swap_info, swap_features = _load_swap_samples(signature_dir, addon_dirs)

    # Group swaps by type
    swap_types: dict[str, list[int]] = {}
    for i, info in enumerate(swap_info):
        swap_types.setdefault(info["swap_name"], []).append(i)

    print(f"\n  Swap types found: {list(swap_types.keys())}")
    print(f"  Total swap samples: {len(swap_info)}")

    # Discover tiers to test
    test_tiers: list[str] = []
    for tier in BASELINE_TIERS + ENGINEERED_TIERS:
        if tier in core_data.tier_features and tier in swap_features:
            test_tiers.append(tier)
    for group in TIER_GROUPS:
        if group in core_data.group_features and group in swap_features:
            test_tiers.append(group)

    print(f"  Tiers to test: {test_tiers}")

    results: dict = {
        "run_name": run_name,
        "n_swap_samples": len(swap_info),
        "swap_types": list(swap_types.keys()),
        "per_swap_type": {},
        "aggregate": {},
    }

    # ── Per swap type ──
    for swap_name, indices in sorted(swap_types.items()):
        parsed = _parse_swap_mode(swap_name)
        if not parsed:
            continue
        sys_mode, exec_mode = parsed

        print(f"\n  --- {swap_name} (sys={sys_mode}, exec={exec_mode}, n={len(indices)}) ---")

        # Binary core data: filter to just these two modes
        mode_mask = (core_data.modes == sys_mode) | (core_data.modes == exec_mode)
        y_core = core_data.modes[mode_mask]
        n_per_mode = {sys_mode: int(np.sum(y_core == sys_mode)),
                      exec_mode: int(np.sum(y_core == exec_mode))}

        swap_type_results: dict = {
            "system_prompt_mode": sys_mode,
            "execution_mode": exec_mode,
            "n_swap": len(indices),
            "n_training_per_mode": n_per_mode,
            "per_tier": {},
        }

        for tier_name in test_tiers:
            try:
                X_all = core_data.get_tier(tier_name)
            except KeyError:
                continue

            X_core = X_all[mode_mask]
            X_swap = np.stack([swap_features[tier_name][i] for i in indices], axis=0)

            scaler = StandardScaler()
            X_core_scaled = scaler.fit_transform(X_core)
            X_swap_scaled = scaler.transform(X_swap)

            clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
            clf.fit(X_core_scaled, y_core)

            predictions = clf.predict(X_swap_scaled)
            probas = clf.predict_proba(X_swap_scaled)
            class_labels = clf.classes_

            # How many classified as execution mode?
            n_exec = int(np.sum(predictions == exec_mode))
            n_sys = int(np.sum(predictions == sys_mode))
            n_total = len(predictions)

            # Mean probability of execution mode class
            exec_class_idx = list(class_labels).index(exec_mode)
            mean_p_exec = float(np.mean(probas[:, exec_class_idx]))

            swap_type_results["per_tier"][tier_name] = {
                "n_execution": n_exec,
                "n_system": n_sys,
                "n_total": n_total,
                "pct_execution": float(n_exec / n_total),
                "mean_p_execution": mean_p_exec,
                "predictions": predictions.tolist(),
            }

            label = "EXEC" if n_exec > n_sys else ("SYS" if n_sys > n_exec else "TIE")
            print(f"    {tier_name:<25} {n_exec}/{n_total} exec  P(exec)={mean_p_exec:.3f}  [{label}]")

        results["per_swap_type"][swap_name] = swap_type_results

    # ── Aggregate across all swap types ──
    print(f"\n  === AGGREGATE (n={len(swap_info)}) ===")
    aggregate: dict = {}
    for tier_name in test_tiers:
        total_exec = 0
        total_sys = 0
        total_n = 0
        all_p_exec: list[float] = []

        for swap_name, swap_data in results["per_swap_type"].items():
            tier_data = swap_data.get("per_tier", {}).get(tier_name)
            if tier_data:
                total_exec += tier_data["n_execution"]
                total_sys += tier_data["n_system"]
                total_n += tier_data["n_total"]
                all_p_exec.append(tier_data["mean_p_execution"])

        if total_n > 0:
            aggregate[tier_name] = {
                "n_execution": total_exec,
                "n_system": total_sys,
                "n_total": total_n,
                "pct_execution": float(total_exec / total_n),
                "mean_p_execution": float(np.mean(all_p_exec)),
                "signal_type": (
                    "execution_based" if total_exec > total_sys * 1.5
                    else "system_prompt_based" if total_sys > total_exec * 1.5
                    else "ambiguous"
                ),
            }
            label = aggregate[tier_name]["signal_type"].upper()
            print(f"    {tier_name:<25} {total_exec}/{total_n} exec  "
                  f"P(exec)={aggregate[tier_name]['mean_p_execution']:.3f}  [{label}]")

    results["aggregate"] = aggregate
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Binary prompt-swap confound test")
    parser.add_argument(
        "--run", type=str, nargs="+", required=True,
        help=f"Run name(s). Known: {list(RUNS.keys())}",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/analysis",
        help="Base output directory.",
    )
    args = parser.parse_args()

    all_results: dict = {}

    for run_name in args.run:
        if run_name not in RUNS:
            print(f"Unknown run: {run_name}. Known: {list(RUNS.keys())}")
            continue

        print(f"\n{'='*60}")
        print(f"BINARY PROMPT-SWAP: {run_name}")
        print(f"{'='*60}")

        sig_dir = RUNS[run_name].signature_dir
        addon_dirs = list(RUNS[run_name].addon_dirs) if RUNS[run_name].addon_dirs else None

        results = run_binary_prompt_swap(
            run_name=run_name,
            signature_dir=sig_dir,
            addon_dirs=addon_dirs,
        )
        all_results[run_name] = results

    # Save results
    output_path = Path(args.output_dir) / "binary_prompt_swap_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
