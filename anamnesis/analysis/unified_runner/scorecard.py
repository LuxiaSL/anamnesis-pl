"""Section 10: Prediction scorecard — evaluate pre-registered 8B predictions."""

from __future__ import annotations

from pydantic import BaseModel


def _safe_get(d: dict, *keys, default=None):
    """Nested dict access with fallback."""
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return default
    return d


def _as_dict(value: object) -> dict:
    """Return a dict view of a section result.

    Sections typed with pydantic models during Phase 3a are dumped to a
    dict so the legacy ``_safe_get``-based scorecard logic continues to
    work unchanged. Subsection schemas (Phase 3a part viii) will replace
    this with attribute access.
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    return value if isinstance(value, dict) else {}


def run_scorecard(all_results: dict) -> dict:
    """
    Evaluate 9 pre-registered predictions against computed results.

    Predictions from research/notes/8b-predictions-preregistration.md
    """
    scorecard: list[dict] = []

    # ── Prediction 1: CCGP = 1.0 ──
    ccgp_data = all_results.get("ccgp", {})
    ccgp_scores = []
    for key, variant in ccgp_data.get("variants", {}).items():
        if isinstance(variant, dict) and "ccgp_score" in variant:
            ccgp_scores.append(variant["ccgp_score"])

    min_ccgp = min(ccgp_scores) if ccgp_scores else None
    p1_outcome = "CONFIRMED" if min_ccgp is not None and min_ccgp >= 1.0 else (
        "PARTIAL" if min_ccgp is not None and min_ccgp >= 0.89 else "WRONG"
    )
    scorecard.append({
        "prediction": "1. CCGP = 1.0",
        "confidence": "95%",
        "importance": "HIGH",
        "outcome": p1_outcome,
        "metric": f"min CCGP across variants = {min_ccgp}",
        "surprise_threshold": "CCGP < 0.89",
    })

    # ── Prediction 2: Centroid topology preserved ──
    topo = all_results.get("topology", {}).get("topology_summary", {})
    euc_topo = topo.get("euclidean", {})
    nearest = euc_topo.get("nearest_pair", "")
    outgroup_ratio = euc_topo.get("analogical_outgroup_ratio", 0)

    # Expected: (contrastive,dialectical) or (linear,socratic) as nearest, analogical outgroup
    expected_pairs = {"contrastive-dialectical", "dialectical-contrastive",
                      "linear-socratic", "socratic-linear"}
    has_expected_near = nearest in expected_pairs
    has_outgroup = outgroup_ratio > 1.2

    p2_outcome = "CONFIRMED" if has_expected_near and has_outgroup else (
        "PARTIAL" if has_expected_near or has_outgroup else "WRONG"
    )
    scorecard.append({
        "prediction": "2. Centroid topology preserved",
        "confidence": "85%",
        "importance": "HIGH",
        "outcome": p2_outcome,
        "metric": f"nearest={nearest}, outgroup_ratio={outgroup_ratio:.2f}",
        "detail": f"expected_near={has_expected_near}, outgroup={has_outgroup}",
    })

    # ── Prediction 3: T1/T2/T2.5 ID convergence ──
    id_data = all_results.get("intrinsic_dimension", {})
    convergence = id_data.get("tier_convergence", {})
    max_diff = convergence.get("max_pairwise_diff")

    p3_outcome = "CONFIRMED" if max_diff is not None and max_diff < 4.0 else (
        "PARTIAL" if max_diff is not None and max_diff < 6.0 else "WRONG"
    )
    scorecard.append({
        "prediction": "3. T1/T2/T2.5 ID convergence (within ±2)",
        "confidence": "75%",
        "importance": "HIGH",
        "outcome": p3_outcome,
        "metric": f"max pairwise diff = {max_diff}",
        "values": convergence.get("values", {}),
    })

    # ── Prediction 4: Per-mode ID ordering preserved ──
    per_mode = id_data.get("per_mode", {})
    mode_ids = {}
    for mode, mdata in per_mode.items():
        if isinstance(mdata, dict):
            mid = mdata.get("dadapy_id")
            if isinstance(mid, (int, float)):
                mode_ids[mode] = mid

    # Expected: linear < contrastive < dialectical < socratic ≈ analogical
    expected_order = ["linear", "contrastive", "dialectical", "socratic", "analogical"]
    if len(mode_ids) >= 5:
        actual_order = sorted(mode_ids.keys(), key=lambda m: mode_ids[m])
        # Check if general trend holds (allow ±1 position)
        n_in_order = sum(
            1 for i, m in enumerate(expected_order)
            if m in actual_order and abs(actual_order.index(m) - i) <= 1
        )
        p4_outcome = "CONFIRMED" if n_in_order >= 4 else (
            "PARTIAL" if n_in_order >= 3 else "WRONG"
        )
    else:
        p4_outcome = "INSUFFICIENT_DATA"
        actual_order = []

    scorecard.append({
        "prediction": "4. Per-mode ID ordering: linear < contrastive < dialectical < socratic ≈ analogical",
        "confidence": "70%",
        "importance": "MEDIUM",
        "outcome": p4_outcome,
        "expected_order": expected_order,
        "actual_order": actual_order,
        "mode_ids": mode_ids,
    })

    # ── Prediction 5: T3 remains elevated ──
    t3_id = _safe_get(id_data, "global", "T3", "dadapy_id")
    t1_id = _safe_get(id_data, "global", "T1", "dadapy_id")
    t2_id = _safe_get(id_data, "global", "T2", "dadapy_id")

    if all(isinstance(x, (int, float)) for x in [t3_id, t1_id, t2_id]):
        mean_other = (t1_id + t2_id) / 2
        elevated = t3_id > mean_other + 5
        p5_outcome = "CONFIRMED" if elevated else (
            "PARTIAL" if t3_id > mean_other + 2 else "WRONG"
        )
    else:
        p5_outcome = "INSUFFICIENT_DATA"
        mean_other = None

    scorecard.append({
        "prediction": "5. T3 remains elevated relative to T1/T2/T2.5",
        "confidence": "80%",
        "importance": "MEDIUM",
        "outcome": p5_outcome,
        "t3_id": t3_id,
        "mean_t1_t2": mean_other,
    })

    # ── Prediction 6: T2.5 load-bearing (tier inversion) ──
    ablation = all_results.get("tier_ablation", {})
    tier_inversion = ablation.get("tier_inversion_t25_gt_t2_gt_t1", None)
    per_tier = ablation.get("per_tier_accuracy", {})
    removal_cost = ablation.get("leave_one_tier_out", {})

    t25_cost = _safe_get(removal_cost, "T2.5", "cost_of_removal")
    t2_cost = _safe_get(removal_cost, "T2", "cost_of_removal")
    t1_cost = _safe_get(removal_cost, "T1", "cost_of_removal")

    p6_outcome = "CONFIRMED" if tier_inversion else (
        "PARTIAL" if per_tier.get("T2.5", 0) >= per_tier.get("T1", 0) else "WRONG"
    )
    scorecard.append({
        "prediction": "6. T2.5 load-bearing (T2.5 > T2 > T1 accuracy)",
        "confidence": "70%",
        "importance": "MEDIUM",
        "outcome": p6_outcome,
        "tier_inversion_holds": tier_inversion,
        "per_tier_accuracy": per_tier,
        "removal_costs": {"T1": t1_cost, "T2": t2_cost, "T2.5": t25_cost},
    })

    # ── Prediction 7: 5-way accuracy ~67-73% ──
    clf_data = _as_dict(all_results.get("classification"))
    t2t25_acc = _safe_get(clf_data, "T2+T2.5", "rf_5way", "accuracy")
    combined_acc = _safe_get(clf_data, "combined", "rf_5way", "accuracy")

    best_acc = max(t2t25_acc or 0, combined_acc or 0)
    p7_outcome = "CONFIRMED" if 0.57 <= best_acc <= 0.83 else (
        "PARTIAL" if 0.50 <= best_acc <= 0.90 else "WRONG"
    )
    scorecard.append({
        "prediction": "7. 5-way accuracy ~67-73%",
        "confidence": "50%",
        "importance": "LOW",
        "outcome": p7_outcome,
        "t2t25_accuracy": t2t25_acc,
        "combined_accuracy": combined_acc,
    })

    # ── Prediction 8: Hard pairs improve more ──
    pairwise = _safe_get(clf_data, "T2+T2.5", "pairwise_binary", default={})
    hard_pairs = ["linear_vs_socratic", "linear_vs_contrastive", "contrastive_vs_socratic"]
    easy_pairs = ["analogical_vs_contrastive", "analogical_vs_dialectical",
                  "analogical_vs_linear", "analogical_vs_socratic"]

    hard_accs = [pairwise.get(p, {}).get("accuracy", 0) for p in hard_pairs if p in pairwise]
    easy_accs = [pairwise.get(p, {}).get("accuracy", 0) for p in easy_pairs if p in pairwise]

    if hard_accs and easy_accs:
        # Compare against 3B baselines (hardcoded from the 3B experiment)
        # 3B hard pairs were ~72.5-80%, easy pairs ~87.5-100%
        mean_hard = float(sum(hard_accs) / len(hard_accs))
        mean_easy = float(sum(easy_accs) / len(easy_accs))
        p8_outcome = "NOTED"  # Hard to evaluate without direct 3B comparison data
    else:
        mean_hard = mean_easy = None
        p8_outcome = "INSUFFICIENT_DATA"

    scorecard.append({
        "prediction": "8. Hard pairs improve more than easy pairs",
        "confidence": "55%",
        "importance": "LOW",
        "outcome": p8_outcome,
        "mean_hard_pair_accuracy": mean_hard,
        "mean_easy_pair_accuracy": mean_easy,
        "all_pairwise": {k: v.get("accuracy") for k, v in pairwise.items()} if pairwise else {},
    })

    # ── Prediction 9: Delta-hyperbolicity slight decrease ──
    topo_data = all_results.get("topology", {})
    delta_rel = _safe_get(topo_data, "gromov_delta_euclidean", "delta_relative")

    if isinstance(delta_rel, (int, float)):
        # 3B was 0.154
        p9_outcome = "CONFIRMED" if delta_rel < 0.154 else (
            "PARTIAL" if delta_rel < 0.20 else "WRONG"
        )
    else:
        p9_outcome = "INSUFFICIENT_DATA"

    scorecard.append({
        "prediction": "9. Delta-hyperbolicity slight decrease (more tree-like)",
        "confidence": "50%",
        "importance": "LOW",
        "outcome": p9_outcome,
        "delta_rel_8b": delta_rel,
        "delta_rel_3b": 0.154,
    })

    # Summary
    outcomes = [s["outcome"] for s in scorecard]
    summary = {
        "confirmed": outcomes.count("CONFIRMED"),
        "partial": outcomes.count("PARTIAL"),
        "wrong": outcomes.count("WRONG"),
        "noted": outcomes.count("NOTED"),
        "insufficient": outcomes.count("INSUFFICIENT_DATA"),
        "total": len(scorecard),
    }

    return {"predictions": scorecard, "summary": summary}
