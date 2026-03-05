#!/usr/bin/env python3
"""Complementarity, resolution, and redundancy analysis across feature families.

Reads existing results.json files and computes:
1. Cross-run consistency (same tiers across different runs)
2. Resolution matrix (hard vs easy pair performance per tier)
3. Complementarity matrix (correlation of pairwise accuracy profiles)
4. Sub-family feature importance breakdown
5. Confusion matrix analysis

Usage:
    python -m anamnesis.scripts.analyze_complementarity
    python -m anamnesis.scripts.analyze_complementarity --include-5way
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ── Configuration ──

ANALYSIS_DIR = Path("outputs/analysis")

# Run configurations
RUNS: dict[str, Path] = {
    "8b_baseline": ANALYSIS_DIR / "8b_baseline" / "results.json",
    "3b_run4": ANALYSIS_DIR / "3b_run4" / "results.json",
    "8b_v2": ANALYSIS_DIR / "8b_v2" / "results.json",
    "3b_v2": ANALYSIS_DIR / "3b_v2" / "results.json",
}

OPTIONAL_RUNS: dict[str, Path] = {
    "8b_v2_5way": ANALYSIS_DIR / "8b_v2_5way" / "results.json",
    "3b_v2_5way": ANALYSIS_DIR / "3b_v2_5way" / "results.json",
}

# Mode pair classification
ORIGINAL_5_MODES = {"linear", "socratic", "contrastive", "dialectical", "analogical"}
EASY_3_MODES = {"compressed", "structured", "associative"}

# Sub-family definitions (signal name → conceptual origin)
SUBFAMILIES: dict[str, dict[str, str]] = {
    "temporal_dynamics": {
        "attn_entropy": "T2-temporal",
        "head_agreement": "T2-temporal",
        "key_drift": "T2.5-temporal",
        "key_novelty": "T2.5-temporal",
        "lookback_ratio": "T2.5-temporal",
    },
    "attention_flow": {
        "sysprompt_mass": "prompt-tracking",
        "sysprompt_decay_rate": "prompt-tracking",
        "recency_bias": "attention-dynamics",
        "region_sysprompt": "region-decomp",
        "region_early_gen": "region-decomp",
        "region_mid_gen": "region-decomp",
        "region_recent": "region-decomp",
        "head_diversity_recency": "head-diversity",
        "head_diversity_sysprompt": "head-diversity",
    },
    "residual_trajectory": {
        "velocity_norm": "trajectory-speed",
        "direction_change": "trajectory-curvature",
        "directness": "trajectory-shape",
        "acceleration_norm": "trajectory-dynamics",
    },
    "gate_features": {
        "sparsity": "gate-activation",
        "drift": "gate-dynamics",
        "eff_dim": "gate-geometry",
        "topk_overlap": "gate-stability",
        "layer_agreement": "gate-cross-layer",
        "layer_sparsity_diversity": "gate-cross-layer",
    },
    "contrastive_projection": {
        "t0": "cp-t0-prompt",
        "t1": "cp-t1-early",
        "t2": "cp-t2-mid",
        "t3": "cp-t3-late",
        "t4": "cp-t4-final",
    },
}


def load_results(path: Path) -> dict | None:
    """Load a results.json file, return None if not found."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _get_pair_name(m1: str, m2: str) -> str:
    """Canonical pair name (alphabetical order)."""
    return f"{min(m1,m2)}_vs_{max(m1,m2)}"


def _classify_pair(pair: str) -> str:
    """Classify a mode pair as 'hard', 'easy', or 'cross'."""
    modes = pair.split("_vs_")
    if len(modes) != 2:
        return "unknown"
    m1, m2 = modes
    if m1 in ORIGINAL_5_MODES and m2 in ORIGINAL_5_MODES:
        return "hard"
    if m1 in EASY_3_MODES or m2 in EASY_3_MODES:
        if m1 in EASY_3_MODES and m2 in EASY_3_MODES:
            return "easy-easy"
        return "cross"
    return "unknown"


# ── Analysis 1: Cross-Run Consistency ──

def analyze_consistency(results: dict[str, dict]) -> dict:
    """Compare baseline tier RF accuracy across runs with same modes."""
    print("\n" + "=" * 70)
    print("1. CROSS-RUN CONSISTENCY")
    print("=" * 70)

    baseline_tiers = ["T1", "T2", "T2.5", "T3", "T2+T2.5", "combined"]
    comparisons: list[dict] = []

    # 8B: baseline (5-way) vs v2_5way (5-way on v2 data)
    pairs = [
        ("8b_baseline", "8b_v2_5way", "8B: baseline vs v2 (5-mode)"),
        ("3b_run4", "3b_v2_5way", "3B: run4 vs v2 (5-mode)"),
    ]

    for run_a, run_b, label in pairs:
        r_a = results.get(run_a)
        r_b = results.get(run_b)
        if not r_a or not r_b:
            print(f"\n  {label}: SKIPPED (missing {run_a if not r_a else run_b})")
            continue

        print(f"\n  {label}")
        clf_a = r_a.get("classification", {})
        clf_b = r_b.get("classification", {})

        comp = {"label": label, "tiers": {}}
        for tier in baseline_tiers:
            acc_a = clf_a.get(tier, {}).get("rf_5way", {}).get("accuracy")
            acc_b = clf_b.get(tier, {}).get("rf_5way", {}).get("accuracy")
            if acc_a is not None and acc_b is not None:
                diff = acc_b - acc_a
                flag = " *** DIVERGENT" if abs(diff) > 0.05 else ""
                print(f"    {tier:<12} {acc_a:.1%} → {acc_b:.1%}  (Δ={diff:+.1%}){flag}")
                comp["tiers"][tier] = {
                    "run_a": acc_a, "run_b": acc_b,
                    "diff": diff, "divergent": abs(diff) > 0.05,
                }

        comparisons.append(comp)

    return {"comparisons": comparisons}


# ── Analysis 2: Resolution Matrix ──

def analyze_resolution(results: dict[str, dict]) -> dict:
    """Pairwise accuracy by difficulty category for each tier."""
    print("\n" + "=" * 70)
    print("2. RESOLUTION MATRIX (hard vs easy pairs)")
    print("=" * 70)

    output: dict = {}

    for run_name in ["8b_v2", "3b_v2"]:
        r = results.get(run_name)
        if not r:
            continue

        print(f"\n  --- {run_name} ---")
        clf = r.get("classification", {})

        # Get all tiers that have pairwise data
        tier_resolution: dict[str, dict] = {}
        all_tiers = sorted(k for k, v in clf.items()
                          if isinstance(v, dict) and "pairwise_binary" in v)

        for tier in all_tiers:
            pw = clf[tier].get("pairwise_binary", {})
            if not pw:
                continue

            buckets: dict[str, list[float]] = {
                "hard": [], "cross": [], "easy-easy": [],
            }

            for pair, data in pw.items():
                if not isinstance(data, dict) or "accuracy" not in data:
                    continue
                cat = _classify_pair(pair)
                if cat in buckets:
                    buckets[cat].append(data["accuracy"])

            tier_resolution[tier] = {
                cat: {
                    "mean": float(np.mean(accs)) if accs else None,
                    "min": float(np.min(accs)) if accs else None,
                    "max": float(np.max(accs)) if accs else None,
                    "n_pairs": len(accs),
                }
                for cat, accs in buckets.items()
            }

        # Print table
        print(f"  {'Tier':<25} {'Hard (mean)':<12} {'Cross (mean)':<12} {'Easy (mean)':<12} {'Hard min':<10}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
        for tier in all_tiers:
            tr = tier_resolution.get(tier, {})
            hard = tr.get("hard", {})
            cross = tr.get("cross", {})
            easy = tr.get("easy-easy", {})
            h_mean = f"{hard.get('mean', 0):.1%}" if hard.get("mean") else "N/A"
            c_mean = f"{cross.get('mean', 0):.1%}" if cross.get("mean") else "N/A"
            e_mean = f"{easy.get('mean', 0):.1%}" if easy.get("mean") else "N/A"
            h_min = f"{hard.get('min', 0):.1%}" if hard.get("min") else "N/A"
            print(f"  {tier:<25} {h_mean:<12} {c_mean:<12} {e_mean:<12} {h_min:<10}")

        output[run_name] = tier_resolution

    return output


# ── Analysis 3: Complementarity Matrix ──

def analyze_complementarity(results: dict[str, dict]) -> dict:
    """Correlation of pairwise accuracy profiles between tiers."""
    print("\n" + "=" * 70)
    print("3. COMPLEMENTARITY MATRIX (accuracy profile correlation)")
    print("=" * 70)

    output: dict = {}

    for run_name in ["8b_v2"]:  # Focus on 8B
        r = results.get(run_name)
        if not r:
            continue

        print(f"\n  --- {run_name} ---")
        clf = r.get("classification", {})

        # Build accuracy vectors per tier
        tiers_with_pw = [k for k, v in clf.items()
                        if isinstance(v, dict) and "pairwise_binary" in v
                        and k not in ("combined_v2",)]  # Skip perfect tiers

        # Get union of all pairs
        all_pairs: set[str] = set()
        for tier in tiers_with_pw:
            pw = clf[tier].get("pairwise_binary", {})
            all_pairs.update(pw.keys())
        all_pairs_sorted = sorted(all_pairs)

        # Build accuracy matrix: tiers × pairs
        acc_matrix = np.zeros((len(tiers_with_pw), len(all_pairs_sorted)))
        for i, tier in enumerate(tiers_with_pw):
            pw = clf[tier].get("pairwise_binary", {})
            for j, pair in enumerate(all_pairs_sorted):
                data = pw.get(pair, {})
                acc_matrix[i, j] = data.get("accuracy", 0.5) if isinstance(data, dict) else 0.5

        # Focus on HARD pairs only — easy/hard structure dominates raw correlation
        hard_indices = [j for j, pair in enumerate(all_pairs_sorted)
                       if _classify_pair(pair) == "hard"]

        if len(tiers_with_pw) > 1 and hard_indices:
            hard_matrix = acc_matrix[:, hard_indices]

            # Check for zero-variance tiers (100% on all hard pairs)
            valid_tiers = []
            valid_indices = []
            for i, tier in enumerate(tiers_with_pw):
                if np.std(hard_matrix[i]) > 1e-10:
                    valid_tiers.append(tier)
                    valid_indices.append(i)
                else:
                    print(f"  (skipping {tier} — zero variance on hard pairs)")

            if len(valid_tiers) > 1:
                valid_matrix = hard_matrix[valid_indices]
                corr = np.corrcoef(valid_matrix)

                print(f"\n  Hard-pair accuracy profile correlation "
                      f"({len(hard_indices)} hard pairs, lower = more complementary):")
                print(f"  {'Tier A':<25} {'Tier B':<25} {'r':<8} {'Interpretation'}")
                print(f"  {'-'*25} {'-'*25} {'-'*8} {'-'*20}")

                pairs_by_corr: list[tuple[float, str, str]] = []
                for i in range(len(valid_tiers)):
                    for j in range(i + 1, len(valid_tiers)):
                        r_val = corr[i, j]
                        if np.isnan(r_val):
                            continue
                        pairs_by_corr.append((r_val, valid_tiers[i], valid_tiers[j]))

                pairs_by_corr.sort(key=lambda x: x[0])

                for r_val, t_a, t_b in pairs_by_corr:
                    interp = ("COMPLEMENTARY" if r_val < 0.3
                              else "MODERATE" if r_val < 0.6
                              else "REDUNDANT")
                    print(f"  {t_a:<25} {t_b:<25} {r_val:+.3f}  {interp}")

            # Also show per-hard-pair accuracy table for individual tiers
            individual_tiers = [t for t in tiers_with_pw
                               if t not in ("T2+T2.5", "combined", "engineered",
                                           "combined_v2", "T2+T2.5+engineered")]
            hard_pair_names = [all_pairs_sorted[j] for j in hard_indices]

            print(f"\n  Per-hard-pair accuracy (individual tiers):")
            header = f"  {'Pair':<28}"
            for t in individual_tiers:
                header += f" {t[:8]:<9}"
            print(header)
            print(f"  {'-'*28}" + f" {'-'*9}" * len(individual_tiers))

            for j_idx, j in enumerate(hard_indices):
                pair_name = all_pairs_sorted[j].replace("_vs_", "/")
                row = f"  {pair_name:<28}"
                for i, tier in enumerate(tiers_with_pw):
                    if tier in individual_tiers:
                        row += f" {acc_matrix[i, j]:.1%}    "
                print(row)

            output[run_name] = {
                "tiers": tiers_with_pw,
                "hard_pairs": hard_pair_names,
                "pairs_by_correlation": [
                    {"tier_a": t_a, "tier_b": t_b, "r": float(r_val)}
                    for r_val, t_a, t_b in pairs_by_corr
                ] if len(valid_tiers) > 1 else [],
            }

    return output


# ── Analysis 4: Sub-Family Feature Importance ──

def analyze_subfamily_importance(results: dict[str, dict]) -> dict:
    """Group feature importance by sub-family within each tier."""
    print("\n" + "=" * 70)
    print("4. SUB-FAMILY FEATURE IMPORTANCE")
    print("=" * 70)

    output: dict = {}

    for run_name in ["8b_v2"]:
        r = results.get(run_name)
        if not r:
            continue

        print(f"\n  --- {run_name} ---")
        abl = r.get("tier_ablation", {})
        top_feats = abl.get("top_features_rf", [])

        if not isinstance(top_feats, list) or not top_feats:
            print("  No feature importance data available")
            continue

        # Group by family
        family_importance: dict[str, float] = {}
        subfam_importance: dict[str, float] = {}

        for feat in top_feats:
            name = feat.get("name", "")
            imp = feat.get("importance", 0)

            # Determine family from prefix
            family = _feature_to_family(name)
            family_importance[family] = family_importance.get(family, 0) + imp

            # Determine sub-family
            subfam = _feature_to_subfamily(name)
            subfam_importance[subfam] = subfam_importance.get(subfam, 0) + imp

        print(f"\n  Top feature families (by cumulative importance in top-{len(top_feats)}):")
        for family, imp in sorted(family_importance.items(), key=lambda x: -x[1]):
            print(f"    {family:<30} {imp:.4f}")

        print(f"\n  Top sub-families:")
        for subfam, imp in sorted(subfam_importance.items(), key=lambda x: -x[1])[:15]:
            print(f"    {subfam:<35} {imp:.4f}")

        # Also check per-tier top features
        for tier_key in ["top_features_rf_t2t25"]:
            tier_feats = abl.get(tier_key, [])
            if isinstance(tier_feats, list) and tier_feats:
                print(f"\n  {tier_key} sub-families:")
                tier_subfam: dict[str, float] = {}
                for feat in tier_feats[:20]:
                    name = feat.get("name", "")
                    imp = feat.get("importance", 0)
                    subfam = _feature_to_subfamily(name)
                    tier_subfam[subfam] = tier_subfam.get(subfam, 0) + imp
                for subfam, imp in sorted(tier_subfam.items(), key=lambda x: -x[1])[:10]:
                    print(f"      {subfam:<35} {imp:.4f}")

        output[run_name] = {
            "family_importance": family_importance,
            "subfam_importance": subfam_importance,
        }

    return output


def _feature_to_family(name: str) -> str:
    """Map a feature name to its parent family."""
    prefix = name.split("_")[0] if "_" in name else name
    mapping = {
        "cp": "contrastive_projection",
        "af": "attention_flow",
        "td": "temporal_dynamics",
        "rt": "residual_trajectory",
        "gf": "gate_features",
    }
    if prefix in mapping:
        return mapping[prefix]

    # T1/T2/T2.5/T3 features don't have a clean prefix
    # Check for known T-tier patterns
    for tier_prefix in ["act_norm", "logit", "token", "delta",
                        "attn_entropy", "head_agree", "residual",
                        "lookback", "key_drift", "key_novelty", "epoch",
                        "pca_"]:
        if name.startswith(tier_prefix):
            if tier_prefix in ("lookback", "key_drift", "key_novelty", "epoch"):
                return "T2.5"
            elif tier_prefix in ("attn_entropy", "head_agree", "residual"):
                return "T2"
            elif tier_prefix.startswith("pca"):
                return "T3"
            else:
                return "T1"

    return f"unknown({name[:20]})"


def _feature_to_subfamily(name: str) -> str:
    """Map a feature name to its sub-family."""
    parts = name.split("_")

    # contrastive_projection: cp_L{N}_t{N}_d{N}
    if parts[0] == "cp":
        if len(parts) >= 3:
            layer = parts[1]  # L8, L16, etc.
            temporal = parts[2]  # t0-t4
            return f"cp_{temporal}"
        return "cp_unknown"

    # temporal_dynamics: td_L{N}_{signal}_{operator}
    if parts[0] == "td":
        if len(parts) >= 3:
            signal = parts[2]  # attn, head, key, lookback
            if signal in ("attn",):
                return "td_attn_entropy"
            elif signal in ("head",):
                return "td_head_agreement"
            elif signal in ("key",):
                if len(parts) >= 4:
                    return f"td_key_{parts[3]}"  # key_drift, key_novelty
                return "td_key"
            elif signal in ("lookback",):
                return "td_lookback_ratio"
        return "td_unknown"

    # attention_flow: af_L{N}_{signal}
    if parts[0] == "af":
        if len(parts) >= 3:
            signal = parts[2]
            if signal in ("sysprompt",):
                if len(parts) >= 4 and parts[3] == "decay":
                    return "af_sysprompt_decay"
                return "af_sysprompt_mass"
            elif signal in ("recency",):
                return "af_recency_bias"
            elif signal in ("region",):
                if len(parts) >= 4:
                    return f"af_region_{parts[3]}"
                return "af_region"
            elif signal in ("head",):
                return "af_head_diversity"
        return "af_unknown"

    # residual_trajectory: rt_L{N}_{signal}
    if parts[0] == "rt":
        if len(parts) >= 3:
            return f"rt_{parts[2]}"
        return "rt_unknown"

    # gate_features: gf_{signal} or gf_L{N}_{signal}
    if parts[0] == "gf":
        if len(parts) >= 2:
            if parts[1].startswith("L"):
                if len(parts) >= 3:
                    return f"gf_{parts[2]}"
                return "gf_unknown"
            return f"gf_{parts[1]}"
        return "gf_unknown"

    return f"other({name[:20]})"


# ── Analysis 5: Confusion Matrix Overlap ──

def analyze_confusion_overlap(results: dict[str, dict]) -> dict:
    """Compare confusion patterns across tiers."""
    print("\n" + "=" * 70)
    print("5. CONFUSION PATTERN ANALYSIS")
    print("=" * 70)

    output: dict = {}

    for run_name in ["8b_v2"]:
        r = results.get(run_name)
        if not r:
            continue

        print(f"\n  --- {run_name} ---")
        clf = r.get("classification", {})

        # Extract confusion matrices
        tiers_with_cm: dict[str, Any] = {}
        for tier_name, tier_data in clf.items():
            if not isinstance(tier_data, dict):
                continue
            cm = tier_data.get("rf_5way", {}).get("confusion_matrix")
            if cm and isinstance(cm, list):
                tiers_with_cm[tier_name] = np.array(cm)

        if not tiers_with_cm:
            print("  No confusion matrices available")
            continue

        # For each tier, identify hardest confusions
        print(f"\n  Hardest confusions per tier (lowest pairwise accuracy):")
        for tier, cm in sorted(tiers_with_cm.items()):
            if cm.shape[0] < 3:
                continue
            modes_list = sorted(clf.get(tier, {}).get("rf_5way", {}).get("class_labels", []))
            if not modes_list or len(modes_list) != cm.shape[0]:
                continue

            # Find lowest off-diagonal (normalized by row sum)
            norm_cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            worst_pairs = []
            for i in range(cm.shape[0]):
                for j in range(cm.shape[0]):
                    if i != j:
                        # Low diagonal = high confusion
                        pair_acc = (norm_cm[i, i] + norm_cm[j, j]) / 2
                        worst_pairs.append((pair_acc, modes_list[i], modes_list[j]))

            worst_pairs.sort()
            if worst_pairs:
                wp = worst_pairs[0]
                print(f"    {tier:<25} hardest: {wp[1]}-{wp[2]} ({wp[0]:.1%} diag)")

        output[run_name] = {"tiers_analyzed": list(tiers_with_cm.keys())}

    return output


# ── Analysis 6: Tier Inversion Check ──

def analyze_tier_inversion(results: dict[str, dict]) -> dict:
    """Check T2.5 > T2 > T1 inversion across runs."""
    print("\n" + "=" * 70)
    print("6. TIER INVERSION CHECK (T2.5 > T2 > T1)")
    print("=" * 70)

    output: dict = {}
    check_runs = ["3b_run4", "8b_baseline", "8b_v2", "3b_v2",
                   "8b_v2_5way", "3b_v2_5way"]

    for run_name in check_runs:
        r = results.get(run_name)
        if not r:
            continue

        clf = r.get("classification", {})
        t1_acc = clf.get("T1", {}).get("rf_5way", {}).get("accuracy")
        t2_acc = clf.get("T2", {}).get("rf_5way", {}).get("accuracy")
        t25_acc = clf.get("T2.5", {}).get("rf_5way", {}).get("accuracy")
        t3_acc = clf.get("T3", {}).get("rf_5way", {}).get("accuracy")

        if all(x is not None for x in [t1_acc, t2_acc, t25_acc]):
            inversion = t25_acc > t2_acc > t1_acc
            status = "YES" if inversion else "no"
            n_modes = r.get("n_samples", 0) // 20  # assuming 20 topics
            print(f"  {run_name:<18} T1={t1_acc:.1%}  T2={t2_acc:.1%}  T2.5={t25_acc:.1%}  "
                  f"T3={t3_acc:.1%}  inversion={status}  ({n_modes} modes)")

            output[run_name] = {
                "T1": t1_acc, "T2": t2_acc, "T2.5": t25_acc, "T3": t3_acc,
                "inversion": inversion, "n_modes": n_modes,
            }

    return output


# ── Analysis 7: New Features Value-Add ──

def analyze_value_add(results: dict[str, dict]) -> dict:
    """Compare new feature families vs baseline on 5-mode subset."""
    print("\n" + "=" * 70)
    print("7. NEW FEATURE VALUE-ADD (5-mode, vs baseline)")
    print("=" * 70)

    output: dict = {}

    pairs = [
        ("8b_baseline", "8b_v2_5way", "8B"),
        ("3b_run4", "3b_v2_5way", "3B"),
    ]

    for baseline_name, v2_name, model in pairs:
        r_base = results.get(baseline_name)
        r_v2 = results.get(v2_name)
        if not r_base or not r_v2:
            print(f"\n  {model}: SKIPPED (missing data)")
            continue

        print(f"\n  --- {model} ---")
        clf_base = r_base.get("classification", {})
        clf_v2 = r_v2.get("classification", {})

        # Baseline composites
        base_t2t25 = clf_base.get("T2+T2.5", {}).get("rf_5way", {}).get("accuracy")
        base_combined = clf_base.get("combined", {}).get("rf_5way", {}).get("accuracy")

        # V2 individual new families
        new_families = ["residual_trajectory", "attention_flow", "gate_features",
                       "temporal_dynamics", "contrastive_projection"]
        v2_composites = ["engineered", "T2+T2.5+engineered", "combined_v2"]

        print(f"\n  Baseline:     T2+T2.5={base_t2t25:.1%}  combined={base_combined:.1%}")

        print(f"\n  New families (v2 data, 5-mode):")
        for fam in new_families:
            acc = clf_v2.get(fam, {}).get("rf_5way", {}).get("accuracy")
            if acc is not None:
                delta = acc - (base_t2t25 or 0)
                print(f"    {fam:<25} {acc:.1%}  (vs baseline T2+T2.5: {delta:+.1%})")

        print(f"\n  V2 composites:")
        for comp in v2_composites:
            acc = clf_v2.get(comp, {}).get("rf_5way", {}).get("accuracy")
            if acc is not None:
                delta = acc - (base_combined or 0)
                print(f"    {comp:<25} {acc:.1%}  (vs baseline combined: {delta:+.1%})")

        output[model] = {
            "baseline_t2t25": base_t2t25,
            "baseline_combined": base_combined,
        }

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Complementarity analysis")
    parser.add_argument("--include-5way", action="store_true",
                       help="Include 5-way subset results if available.")
    args = parser.parse_args()

    # Load all results
    all_runs = dict(RUNS)
    if args.include_5way:
        all_runs.update(OPTIONAL_RUNS)
    else:
        # Auto-include if they exist
        for name, path in OPTIONAL_RUNS.items():
            if path.exists():
                all_runs[name] = path

    results: dict[str, dict] = {}
    for name, path in all_runs.items():
        r = load_results(path)
        if r:
            results[name] = r
            print(f"  Loaded: {name}")
        else:
            print(f"  Missing: {name}")

    print(f"\n  Total runs loaded: {len(results)}")

    # Run all analyses
    report: dict = {}
    report["consistency"] = analyze_consistency(results)
    report["resolution"] = analyze_resolution(results)
    report["complementarity"] = analyze_complementarity(results)
    report["subfamily_importance"] = analyze_subfamily_importance(results)
    report["confusion"] = analyze_confusion_overlap(results)
    report["tier_inversion"] = analyze_tier_inversion(results)
    report["value_add"] = analyze_value_add(results)

    # Save
    output_path = ANALYSIS_DIR / "complementarity_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n\nReport saved to {output_path}")


if __name__ == "__main__":
    main()
