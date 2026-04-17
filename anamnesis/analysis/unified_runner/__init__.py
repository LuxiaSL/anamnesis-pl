"""Unified analysis runner for Anamnesis signature data.

Performs the full analysis gauntlet on any set of extraction signatures:
  1. Data integrity & descriptive statistics
  2. Classification (5-way mode discrimination)
  3. Tier ablation & feature importance
  4. Intrinsic dimension profiling
  5. CCGP (cross-condition generalization)
  6. Topology & hyperbolicity
  7. Silhouette & clustering
  8. Contrastive projection (optional, requires torch)
  9. Semantic independence (optional, requires sentence-transformers)
  10. Prediction scorecard
  11. Manifold geometry (curvature, geodesic, anisotropy)

Supports checkpoint-based resume: saves results after each section completes.
On restart, detects which sections already have results and skips them.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .data_loading import AnalysisData, load_analysis_data
from .utils import clean_for_json

# Section number → results key mapping
SECTION_KEYS: dict[int, str] = {
    1: "integrity",
    2: "classification",
    3: "tier_ablation",
    4: "intrinsic_dimension",
    5: "ccgp",
    6: "topology",
    7: "clustering",
    8: "contrastive",
    9: "semantic",
    10: "scorecard",
    11: "manifold_geometry",
}

SECTION_NAMES: dict[int, str] = {
    1: "Data Integrity",
    2: "Classification",
    3: "Tier Ablation",
    4: "Intrinsic Dimension",
    5: "CCGP",
    6: "Topology",
    7: "Clustering",
    8: "Contrastive Projection",
    9: "Semantic Independence",
    10: "Prediction Scorecard",
    11: "Manifold Geometry",
}


def _save_checkpoint(results: dict, output_dir: Path) -> None:
    """Save current results as checkpoint."""
    checkpoint_path = output_dir / "results.json"
    with open(checkpoint_path, "w") as f:
        json.dump(clean_for_json(results), f, indent=2)


def _load_checkpoint(output_dir: Path) -> dict | None:
    """Load existing checkpoint if present."""
    checkpoint_path = output_dir / "results.json"
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _detect_completed_sections(checkpoint: dict) -> set[int]:
    """Detect which sections have results in a checkpoint."""
    completed = set()
    for section_num, key in SECTION_KEYS.items():
        if key in checkpoint and checkpoint[key] is not None:
            # Check it's not just an error stub
            val = checkpoint[key]
            if isinstance(val, dict) and val.get("error"):
                continue
            completed.add(section_num)
    return completed


def run_full_analysis(
    signature_dir: Path | str,
    run_name: str,
    output_dir: Path | str | None = None,
    core_only: bool = True,
    skip_sections: set[int] | None = None,
    resume: bool = False,
    addon_dirs: list[Path | str] | None = None,
    mode_filter: list[str] | None = None,
) -> dict:
    """Run the complete analysis gauntlet.

    Parameters
    ----------
    signature_dir : Path
        Directory containing gen_NNN.npz + gen_NNN.json files.
    run_name : str
        Label for this run (e.g. "8b_baseline").
    output_dir : Path, optional
        Where to save results. Defaults to outputs/analysis/{run_name}/.
    core_only : bool
        If True, use one rep per topic-mode pair.
    skip_sections : set of int, optional
        Section numbers to skip (1-11).
    resume : bool
        If True, load existing checkpoint and skip completed sections.
    addon_dirs : list[Path], optional
        Additional directories with feature arrays to merge.
    mode_filter : list[str], optional
        If provided, only include samples whose mode is in this list.

    Returns
    -------
    dict with all results.
    """
    skip = set(skip_sections) if skip_sections else set()

    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "outputs" / "analysis" / run_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    print("=" * 60)
    print(f"UNIFIED ANALYSIS: {run_name}")
    print("=" * 60)

    # Handle resume
    results: dict = {}
    if resume:
        checkpoint = _load_checkpoint(output_dir)
        if checkpoint is not None:
            completed = _detect_completed_sections(checkpoint)
            if completed:
                print(f"\nResuming from checkpoint. Completed sections: {sorted(completed)}")
                results = checkpoint
                skip = skip | completed
            else:
                print("\nCheckpoint found but no completed sections. Starting fresh.")
        else:
            print("\nNo checkpoint found. Starting fresh.")

    # Base metadata (preserve from checkpoint if resuming, else set fresh)
    if "run_name" not in results:
        results["run_name"] = run_name
        results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        results["core_only"] = core_only
    results["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Load data
    print("\nLoading data...")
    load_text = 9 not in skip
    data = load_analysis_data(
        signature_dir=signature_dir,
        run_name=run_name,
        core_only=core_only,
        load_text=load_text,
        addon_dirs=addon_dirs,
        mode_filter=mode_filter,
    )
    results["n_samples"] = data.n_samples
    print(f"  {data.n_samples} samples, {len(data.unique_modes)} modes, "
          f"{len(data.unique_topics)} topics")

    # Track timing per section
    section_times: dict[str, float] = results.get("section_times", {})

    # ── Section 1: Data integrity ──
    if 1 not in skip:
        print("\n--- Section 1: Data Integrity ---")
        from .integrity import run_integrity_checks
        t0 = time.perf_counter()
        results["integrity"] = run_integrity_checks(data)
        section_times["integrity"] = time.perf_counter() - t0
        print(f"  Done ({section_times['integrity']:.1f}s)")
        _save_checkpoint(results, output_dir)

        if not results["integrity"].get("all_clean", False):
            print("  WARNING: NaN/Inf detected in features!")
    else:
        print("\n--- Section 1: Data Integrity --- SKIPPED")

    # ── Section 2: Classification ──
    if 2 not in skip:
        print("\n--- Section 2: Classification ---")
        from .classification import run_classification
        t0 = time.perf_counter()
        results["classification"] = run_classification(data)
        section_times["classification"] = time.perf_counter() - t0
        print(f"  Done ({section_times['classification']:.1f}s)")
        _save_checkpoint(results, output_dir)
    else:
        print("\n--- Section 2: Classification --- SKIPPED")

    # ── Section 3: Tier ablation ──
    if 3 not in skip:
        print("\n--- Section 3: Tier Ablation ---")
        from .tier_ablation import run_tier_ablation
        t0 = time.perf_counter()
        results["tier_ablation"] = run_tier_ablation(data)
        section_times["tier_ablation"] = time.perf_counter() - t0
        print(f"  Done ({section_times['tier_ablation']:.1f}s)")
        _save_checkpoint(results, output_dir)
    else:
        print("\n--- Section 3: Tier Ablation --- SKIPPED")

    # ── Section 4: Intrinsic dimension ──
    if 4 not in skip:
        print("\n--- Section 4: Intrinsic Dimension ---")
        from .geometry import run_intrinsic_dimension
        t0 = time.perf_counter()
        results["intrinsic_dimension"] = run_intrinsic_dimension(data)
        section_times["intrinsic_dimension"] = time.perf_counter() - t0
        print(f"  Done ({section_times['intrinsic_dimension']:.1f}s)")
        _save_checkpoint(results, output_dir)
    else:
        print("\n--- Section 4: Intrinsic Dimension --- SKIPPED")

    # ── Section 5: CCGP ──
    if 5 not in skip:
        print("\n--- Section 5: CCGP ---")
        from .geometry import run_ccgp
        t0 = time.perf_counter()
        results["ccgp"] = run_ccgp(data)
        section_times["ccgp"] = time.perf_counter() - t0
        print(f"  Done ({section_times['ccgp']:.1f}s)")
        _save_checkpoint(results, output_dir)
    else:
        print("\n--- Section 5: CCGP --- SKIPPED")

    # ── Section 6: Topology & hyperbolicity ──
    if 6 not in skip:
        print("\n--- Section 6: Topology ---")
        from .geometry import run_topology
        t0 = time.perf_counter()
        results["topology"] = run_topology(data)
        section_times["topology"] = time.perf_counter() - t0
        print(f"  Done ({section_times['topology']:.1f}s)")
        _save_checkpoint(results, output_dir)
    else:
        print("\n--- Section 6: Topology --- SKIPPED")

    # ── Section 7: Clustering ──
    if 7 not in skip:
        print("\n--- Section 7: Clustering ---")
        from .clustering import run_clustering
        t0 = time.perf_counter()
        results["clustering"] = run_clustering(data)
        section_times["clustering"] = time.perf_counter() - t0
        print(f"  Done ({section_times['clustering']:.1f}s)")
        _save_checkpoint(results, output_dir)
    else:
        print("\n--- Section 7: Clustering --- SKIPPED")

    # ── Section 8: Contrastive projection ──
    if 8 not in skip:
        print("\n--- Section 8: Contrastive Projection ---")
        from .contrastive import run_contrastive
        t0 = time.perf_counter()
        results["contrastive"] = run_contrastive(data)
        section_times["contrastive"] = time.perf_counter() - t0
        print(f"  Done ({section_times['contrastive']:.1f}s)")
        _save_checkpoint(results, output_dir)
    else:
        print("\n--- Section 8: Contrastive Projection --- SKIPPED")

    # ── Section 9: Semantic independence ──
    if 9 not in skip:
        print("\n--- Section 9: Semantic Independence ---")
        from .semantic import run_semantic
        t0 = time.perf_counter()
        results["semantic"] = run_semantic(
            data,
            signature_dir=signature_dir,
            addon_dirs=addon_dirs,
        )
        section_times["semantic"] = time.perf_counter() - t0
        print(f"  Done ({section_times['semantic']:.1f}s)")
        _save_checkpoint(results, output_dir)
    else:
        print("\n--- Section 9: Semantic Independence --- SKIPPED")

    # ── Section 10: Prediction scorecard ──
    # Scorecard always re-runs (it evaluates results from other sections)
    if 10 not in skip:
        print("\n--- Section 10: Prediction Scorecard ---")
        from .scorecard import run_scorecard
        results["scorecard"] = run_scorecard(results)

    # ── Section 11: Manifold Geometry ──
    if 11 not in skip:
        print("\n--- Section 11: Manifold Geometry ---")
        from .geometry import run_manifold_geometry
        t0 = time.perf_counter()
        results["manifold_geometry"] = run_manifold_geometry(data)
        section_times["manifold_geometry"] = time.perf_counter() - t0
        print(f"  Done ({section_times['manifold_geometry']:.1f}s)")
        _save_checkpoint(results, output_dir)
    else:
        print("\n--- Section 11: Manifold Geometry --- SKIPPED")

    # Save final timing
    results["section_times"] = section_times

    # Final save
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(clean_for_json(results), f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary
    _print_summary(results)

    return results


def _print_summary(results: dict) -> None:
    """Print key findings to stdout."""
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # Classification
    clf = results.get("classification", {})
    # Show all tiers that have results
    reported_tiers = ["T2+T2.5", "combined", "engineered", "combined_v2",
                      "T2+T2.5+engineered"]
    for tier in reported_tiers:
        acc = clf.get(tier, {}).get("rf_5way", {}).get("accuracy")
        if acc is not None:
            print(f"\n  5-way RF ({tier}): {acc:.1%}")

    # CV stability
    for tier in ["T2+T2.5", "combined_v2", "combined"]:
        stab = clf.get(tier, {}).get("cv_stability", {})
        if stab and "median" in stab:
            print(f"  CV stability ({tier}): median={stab['median']:.1%}, "
                  f"95% CI=[{stab['ci_lo']:.1%}, {stab['ci_hi']:.1%}]")

    # Permutation test
    for tier in ["T2+T2.5", "combined_v2", "combined"]:
        perm = clf.get(tier, {}).get("permutation_test", {})
        if perm and "p_value" in perm:
            print(f"  Permutation p ({tier}): {perm['p_value']}")

    # Tier ablation
    ablation = results.get("tier_ablation", {})
    ranking = ablation.get("tier_ranking", [])
    if ranking:
        rank_str = " > ".join(f"{r['tier']}({r['accuracy']:.0%})" for r in ranking)
        print(f"\n  Tier ranking: {rank_str}")
        inversion = ablation.get("tier_inversion_t25_gt_t2_gt_t1")
        print(f"  T2.5 > T2 > T1 inversion: {inversion}")

    # ID
    id_data = results.get("intrinsic_dimension", {})
    if id_data.get("global"):
        print("\n  Intrinsic dimension:")
        for tier in ["T1", "T2", "T2.5", "T3", "T2+T2.5"]:
            tid = id_data.get("global", {}).get(tier, {}).get("dadapy_id")
            if isinstance(tid, (int, float)):
                print(f"    {tier}: {tid:.1f}")
        conv = id_data.get("tier_convergence", {})
        if conv and "max_pairwise_diff" in conv:
            print(f"  Tier convergence (max diff): {conv['max_pairwise_diff']:.1f}")

    # CCGP
    ccgp = results.get("ccgp", {}).get("summary", {})
    if ccgp:
        print(f"\n  CCGP: min={ccgp.get('min_ccgp')}, all_perfect={ccgp.get('all_perfect')}")

    # Topology
    topo = results.get("topology", {}).get("topology_summary", {}).get("euclidean", {})
    if topo:
        print(f"\n  Topology: nearest={topo.get('nearest_pair')}, "
              f"outgroup_ratio={topo.get('analogical_outgroup_ratio', 0):.2f}")

    delta = results.get("topology", {}).get("gromov_delta_euclidean", {})
    if delta:
        print(f"  Delta-hyperbolicity: delta_rel={delta.get('delta_relative', 0):.3f}")

    # Semantic orthogonality
    semantic = results.get("semantic", {})
    per_tier_sem = semantic.get("per_tier_semantic", {})
    if per_tier_sem:
        print(f"\n  Semantic orthogonality ({len(per_tier_sem)} tiers tested):")
        tfidf_acc = semantic.get("tfidf_classification", {}).get("rf", {}).get("accuracy")
        if tfidf_acc is not None:
            print(f"    TF-IDF surface baseline: {tfidf_acc:.1%}")
        for tier_name, tier_data in per_tier_sem.items():
            if isinstance(tier_data, dict) and "error" not in tier_data:
                clf_acc = tier_data.get("classification", {}).get("rf", {}).get("accuracy")
                mantel_r = tier_data.get("mantel_tfidf_cosine", {}).get("r")
                r2_med = tier_data.get("text_to_compute_r2", {}).get("median_r2")
                n_sub = tier_data.get("per_mode_surface_vs_compute", {}).get("n_sub_semantic", "?")
                parts = []
                if clf_acc is not None:
                    parts.append(f"RF={clf_acc:.1%}")
                if mantel_r is not None:
                    parts.append(f"Mantel r={mantel_r:.3f}")
                if r2_med is not None:
                    parts.append(f"R²={r2_med:.3f}")
                parts.append(f"sub-semantic modes={n_sub}")
                print(f"    {tier_name}: {', '.join(parts)}")

    # Scorecard
    sc = results.get("scorecard", {})
    if sc:
        summary = sc.get("summary", {})
        print(f"\n  Prediction scorecard: "
              f"{summary.get('confirmed', 0)} confirmed, "
              f"{summary.get('partial', 0)} partial, "
              f"{summary.get('wrong', 0)} wrong")
        for pred in sc.get("predictions", []):
            print(f"    {pred['prediction']}: {pred['outcome']}")

    # Section times
    times = results.get("section_times", {})
    if times:
        total = sum(times.values())
        print(f"\n  Total analysis time: {total:.0f}s")
        for name, elapsed in sorted(times.items(), key=lambda x: x[1], reverse=True):
            print(f"    {name}: {elapsed:.1f}s")
