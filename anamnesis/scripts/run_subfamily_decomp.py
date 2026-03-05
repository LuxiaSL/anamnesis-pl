#!/usr/bin/env python3
"""Sub-family decomposition: run RF classification on feature sub-families.

Splits each feature family into sub-families based on signal type,
runs independent RF classification on each, and reports per-sub-family accuracy.

This answers: within temporal_dynamics, is the signal from T2-derived or T2.5-derived
features? Within attention_flow, is it sysprompt_mass or recency_bias?

Usage:
    python -m anamnesis.scripts.run_subfamily_decomp --run 8b_v2
    python -m anamnesis.scripts.run_subfamily_decomp --run 8b_v2 --modes linear,socratic,contrastive,dialectical,analogical
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from anamnesis.analysis.geometric_trio.data_loader import (
    TIER_KEYS,
    TIER_GROUPS,
    load_run4,
)

# ── Known runs (duplicated for independence) ──

KNOWN_RUNS: dict[str, Path] = {
    "8b_v2": Path("outputs/runs/8b_fat_01/signatures_v2"),
    "3b_v2": Path("outputs/runs/3b_fat_01/signatures_v2"),
}
KNOWN_ADDONS: dict[str, list[Path]] = {
    "8b_v2": [
        Path("outputs/runs/8b_fat_01/signatures_v2_addon"),
        Path("outputs/runs/8b_fat_01/signatures_v2_contrastive"),
    ],
    "3b_v2": [
        Path("outputs/runs/3b_fat_01/signatures_v2_contrastive"),
    ],
}


# ── Sub-family definitions ──

def _classify_td_feature(name: str) -> str:
    """Classify a temporal_dynamics feature into T2-derived or T2.5-derived."""
    # td_L{N}_{signal}_{operator}
    parts = name.split("_")
    if len(parts) < 3:
        return "unknown"
    signal = parts[2]
    if signal in ("attn",):
        return "td_T2_attn_entropy"
    elif signal in ("head",):
        return "td_T2_head_agreement"
    elif signal in ("key",):
        if len(parts) >= 4 and parts[3] == "drift":
            return "td_T2.5_key_drift"
        elif len(parts) >= 4 and parts[3] == "novelty":
            return "td_T2.5_key_novelty"
        return "td_T2.5_key"
    elif signal in ("lookback",):
        return "td_T2.5_lookback_ratio"
    return "unknown"


def _classify_td_operator(name: str) -> str:
    """Classify a temporal_dynamics feature by its temporal operator."""
    if "_w0_" in name:
        return "w0"
    elif "_w1_" in name:
        return "w1"
    elif "_w2_" in name:
        return "w2"
    elif "_w3_" in name:
        return "w3"
    elif "_dominant_freq" in name:
        return "stft"
    elif "_spectral_centroid" in name:
        return "stft"
    elif "_bandwidth" in name:
        return "stft"
    elif "_band_energy" in name:
        return "stft"
    return "other"


def _classify_af_feature(name: str) -> str:
    """Classify an attention_flow feature into sub-family.

    Names: attn_flow_L{N}_{signal}_{operator}
    """
    # Strip prefix: attn_flow_L{N}_ → get signal
    match = re.match(r"attn_flow_L\d+_(.+?)_(mean|std|w\d+_|dominant_|spectral_|bandwidth|low_|mid_|high_|decay)", name)
    if not match:
        # Try simpler patterns
        if "sysprompt_mass" in name:
            return "af_sysprompt_mass"
        if "sysprompt_decay" in name:
            return "af_sysprompt_decay"
        if "recency_bias" in name:
            return "af_recency_bias"
        if "region_sysprompt" in name:
            return "af_region_sysprompt"
        if "region_early" in name:
            return "af_region_early_gen"
        if "region_mid" in name:
            return "af_region_mid_gen"
        if "region_recent" in name:
            return "af_region_recent"
        if "head_diversity_recency" in name:
            return "af_head_diversity_recency"
        if "head_diversity_sysprompt" in name:
            return "af_head_diversity_sysprompt"
        return "af_unknown"

    signal = match.group(1)
    if "sysprompt_mass" in signal:
        return "af_sysprompt_mass"
    if "sysprompt_decay" in signal:
        return "af_sysprompt_decay"
    if "recency_bias" in signal:
        return "af_recency_bias"
    if "region_sysprompt" in signal:
        return "af_region_sysprompt"
    if "region_early" in signal:
        return "af_region_early_gen"
    if "region_mid" in signal:
        return "af_region_mid_gen"
    if "region_recent" in signal:
        return "af_region_recent"
    if "head_diversity_recency" in signal:
        return "af_head_diversity_recency"
    if "head_diversity_sysprompt" in signal:
        return "af_head_diversity_sysprompt"
    return "af_unknown"


def _classify_gf_feature(name: str) -> str:
    """Classify a gate_features feature into sub-family.

    Names: gate_L{N}_{signal}_{operator} or gate_cross_layer_*
    """
    if "cross_layer" in name or "layer_agreement" in name or "layer_sparsity_diversity" in name:
        return "gf_cross_layer"

    # gate_L{N}_{signal}_{operator}
    match = re.match(r"gate_L\d+_(\w+?)_", name)
    if match:
        signal = match.group(1)
        return f"gf_{signal}"

    # Fallback
    if "sparsity" in name:
        return "gf_sparsity"
    if "drift" in name:
        return "gf_drift"
    if "eff_dim" in name:
        return "gf_eff_dim"
    if "topk" in name:
        return "gf_topk_overlap"
    return "gf_unknown"


def _classify_cp_feature(name: str) -> str:
    """Classify a contrastive_projection feature by temporal position.

    Names: cp_L{N}_t{N}_d{N}
    """
    match = re.match(r"cp_L\d+_(t\d+)_d\d+", name)
    if match:
        return f"cp_{match.group(1)}"
    return "cp_unknown"


# ── Sub-family extraction ──

SUBFAMILY_CLASSIFIERS: dict[str, callable] = {
    "temporal_dynamics": _classify_td_feature,
    "attention_flow": _classify_af_feature,
    "gate_features": _classify_gf_feature,
    "contrastive_projection": _classify_cp_feature,
}

# Also define coarser groupings for temporal_dynamics
TD_COARSE_GROUPS = {
    "td_T2": ["td_T2_attn_entropy", "td_T2_head_agreement"],
    "td_T2.5": ["td_T2.5_key_drift", "td_T2.5_key_novelty", "td_T2.5_lookback_ratio"],
}

TD_OPERATOR_GROUPS = {
    "td_w0_only": ["w0"],
    "td_w0_w1": ["w0", "w1"],
    "td_windowed": ["w0", "w1", "w2", "w3"],
    "td_stft_only": ["stft"],
}


def get_feature_names(
    signature_dir: Path,
    addon_dirs: list[Path] | None,
    tier_name: str,
) -> list[str] | None:
    """Get feature names for a tier from the first npz file."""
    # Check primary dir
    npz_files = sorted(signature_dir.glob("gen_*.npz"))
    if not npz_files:
        return None

    for npz_path in npz_files:
        npz = np.load(npz_path, allow_pickle=True)
        json_path = npz_path.with_suffix(".json")
        if not json_path.exists():
            continue

        # Check if this npz has the tier
        npz_key = TIER_KEYS.get(tier_name, f"features_{tier_name}")
        if npz_key in npz.files:
            names = npz.get("feature_names")
            if names is not None:
                with open(json_path) as f:
                    meta = json.load(f)
                slices = meta.get("tier_slices", {})
                slice_key = npz_key.replace("features_", "")
                if slice_key in slices:
                    start, end = slices[slice_key]
                    return [str(n) for n in names[start:end]]
            # If no slices, all names might be for this tier
            return [str(n) for n in names]

    # Check addon dirs
    if addon_dirs:
        for addon_dir in addon_dirs:
            addon_files = sorted(Path(addon_dir).glob("gen_*.npz"))
            for npz_path in addon_files:
                npz = np.load(npz_path, allow_pickle=True)
                npz_key = TIER_KEYS.get(tier_name, f"features_{tier_name}")
                if npz_key in npz.files:
                    names = npz.get("feature_names")
                    if names is not None:
                        json_path = npz_path.with_suffix(".json")
                        if json_path.exists():
                            with open(json_path) as f:
                                meta = json.load(f)
                            slices = meta.get("tier_slices", {})
                            slice_key = npz_key.replace("features_", "")
                            if slice_key in slices:
                                start, end = slices[slice_key]
                                return [str(n) for n in names[start:end]]
                        return [str(n) for n in names]

    return None


def run_rf_on_subset(
    X: NDArray[np.float32],
    y: NDArray,
    feature_mask: NDArray[np.bool_],
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Run RF classification on a feature subset."""
    X_sub = X[:, feature_mask]
    n_features = int(feature_mask.sum())

    if n_features == 0:
        return {"accuracy": 0.0, "n_features": 0, "error": "no features"}

    scaler = StandardScaler()
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_accs = []
    for train_idx, test_idx in cv.split(X_sub, y):
        X_train = scaler.fit_transform(X_sub[train_idx])
        X_test = scaler.transform(X_sub[test_idx])
        clf.fit(X_train, y[train_idx])
        fold_accs.append(float(clf.score(X_test, y[test_idx])))

    return {
        "accuracy": float(np.mean(fold_accs)),
        "std": float(np.std(fold_accs)),
        "fold_accs": fold_accs,
        "n_features": n_features,
    }


def decompose_family(
    data,
    tier_name: str,
    feature_names: list[str],
    classifier_fn,
) -> dict[str, dict]:
    """Decompose a tier into sub-families and run RF on each."""
    X = data.get_tier(tier_name)
    y = data.modes

    # Classify each feature
    subfamilies: dict[str, list[int]] = {}
    for i, name in enumerate(feature_names):
        sf = classifier_fn(name)
        subfamilies.setdefault(sf, []).append(i)

    results: dict[str, dict] = {}

    # Run RF on each sub-family
    for sf_name, indices in sorted(subfamilies.items()):
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[indices] = True
        result = run_rf_on_subset(X, y, mask)
        results[sf_name] = result
        print(f"    {sf_name:<30} {result['accuracy']:.1%} ({result['n_features']} features)")

    # Run on full family for comparison
    full_mask = np.ones(X.shape[1], dtype=bool)
    full_result = run_rf_on_subset(X, y, full_mask)
    results["_full_family"] = full_result
    print(f"    {'_full_family':<30} {full_result['accuracy']:.1%} ({full_result['n_features']} features)")

    return results


def decompose_td_coarse(data, feature_names: list[str]) -> dict[str, dict]:
    """Decompose temporal_dynamics into T2-derived vs T2.5-derived."""
    X = data.get_tier("temporal_dynamics")
    y = data.modes

    # Classify features
    feature_classes = [_classify_td_feature(n) for n in feature_names]

    results: dict[str, dict] = {}

    for group_name, member_classes in TD_COARSE_GROUPS.items():
        indices = [i for i, fc in enumerate(feature_classes) if fc in member_classes]
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[indices] = True
        result = run_rf_on_subset(X, y, mask)
        results[group_name] = result
        print(f"    {group_name:<30} {result['accuracy']:.1%} ({result['n_features']} features)")

    return results


def decompose_td_operators(data, feature_names: list[str]) -> dict[str, dict]:
    """Decompose temporal_dynamics by temporal operator type."""
    X = data.get_tier("temporal_dynamics")
    y = data.modes

    feature_ops = [_classify_td_operator(n) for n in feature_names]

    results: dict[str, dict] = {}

    for group_name, allowed_ops in TD_OPERATOR_GROUPS.items():
        indices = [i for i, fo in enumerate(feature_ops) if fo in allowed_ops]
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[indices] = True
        result = run_rf_on_subset(X, y, mask)
        results[group_name] = result
        print(f"    {group_name:<30} {result['accuracy']:.1%} ({result['n_features']} features)")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Sub-family decomposition")
    parser.add_argument("--run", type=str, nargs="+", required=True)
    parser.add_argument("--modes", type=str, default=None)
    args = parser.parse_args()

    mode_filter = None
    if args.modes:
        mode_filter = [m.strip() for m in args.modes.split(",")]

    all_results: dict = {}

    for run_name in args.run:
        sig_dir = KNOWN_RUNS.get(run_name)
        if not sig_dir:
            print(f"Unknown run: {run_name}")
            continue

        addon_dirs = KNOWN_ADDONS.get(run_name)

        print(f"\n{'='*60}")
        print(f"SUB-FAMILY DECOMPOSITION: {run_name}")
        if mode_filter:
            print(f"  Mode filter: {mode_filter}")
        print(f"{'='*60}")

        data = load_run4(
            signature_dir=sig_dir,
            core_only=True,
            addon_dirs=addon_dirs,
            mode_filter=mode_filter,
        )
        print(f"  {data.n_samples} samples, {len(data.unique_modes)} modes")

        run_results: dict = {}

        # ── Temporal dynamics decomposition ──
        td_names = get_feature_names(sig_dir, addon_dirs, "temporal_dynamics")
        if td_names and "temporal_dynamics" in data.tier_features:
            print(f"\n  --- temporal_dynamics ({len(td_names)} features) ---")

            print(f"\n  By signal type:")
            run_results["td_by_signal"] = decompose_family(
                data, "temporal_dynamics", td_names, _classify_td_feature,
            )

            print(f"\n  T2-derived vs T2.5-derived (coarse):")
            run_results["td_coarse"] = decompose_td_coarse(data, td_names)

            print(f"\n  By temporal operator:")
            run_results["td_by_operator"] = decompose_td_operators(data, td_names)

        # ── Attention flow decomposition ──
        af_names = get_feature_names(sig_dir, addon_dirs, "attention_flow")
        if af_names and "attention_flow" in data.tier_features:
            print(f"\n  --- attention_flow ({len(af_names)} features) ---")
            print(f"\n  By signal type:")
            run_results["af_by_signal"] = decompose_family(
                data, "attention_flow", af_names, _classify_af_feature,
            )

        # ── Gate features decomposition ──
        gf_names = get_feature_names(sig_dir, addon_dirs, "gate_features")
        if gf_names and "gate_features" in data.tier_features:
            print(f"\n  --- gate_features ({len(gf_names)} features) ---")
            print(f"\n  By signal type:")
            run_results["gf_by_signal"] = decompose_family(
                data, "gate_features", gf_names, _classify_gf_feature,
            )

        # ── Contrastive projection decomposition ──
        cp_names = get_feature_names(sig_dir, addon_dirs, "contrastive_projection")
        if cp_names and "contrastive_projection" in data.tier_features:
            print(f"\n  --- contrastive_projection ({len(cp_names)} features) ---")
            print(f"\n  By temporal position:")
            run_results["cp_by_temporal"] = decompose_family(
                data, "contrastive_projection", cp_names, _classify_cp_feature,
            )

        all_results[run_name] = run_results

    # Save
    suffix = "_5way" if mode_filter else ""
    output_path = Path(f"outputs/analysis/subfamily_decomp{suffix}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
