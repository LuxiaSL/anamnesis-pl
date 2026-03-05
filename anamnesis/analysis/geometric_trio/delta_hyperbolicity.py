"""
Delta-hyperbolicity analysis of Run 4 computational signatures.

Computes Gromov's four-point delta on pairwise distance matrices to
assess whether the signature space has tree-like (hyperbolic) structure.

Low delta → tree-like hierarchy, natural resolution scales
High delta → more complex geometry, not naturally hierarchical

Pre-registered predictions:
  P1. Delta is moderate (not strongly hyperbolic, not purely Euclidean)
  P2. More tree-like in T2+T2.5 space than combined space
  P3. Analogical forms a deep outgroup branch
  P4. Linear-socratic cluster together (shallow separation)
  P5. Contrastive-dialectical on separate branch from linear-socratic
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform

from .data_loader import Run4Data, load_run4

TIERS_AND_GROUPS = ["T1", "T2", "T2.5", "T3", "T2+T2.5", "combined"]


@dataclass
class HyperbolicityResult:
    """Delta-hyperbolicity results for one feature space."""
    feature_space: str
    n_samples: int
    ambient_dim: int
    # Gromov delta (absolute and relative)
    delta: float  # absolute delta
    delta_relative: float  # delta / diameter (normalized)
    diameter: float  # max pairwise distance
    # Distribution of per-quadruple deltas
    delta_mean: float
    delta_median: float
    delta_std: float
    delta_95th: float
    # Number of quadruples sampled
    n_quadruples: int
    # Per-mode centroid distances
    centroid_distances: dict[tuple[str, str], float]
    # Tree topology test: relative distances suggest which hierarchy
    topology_notes: str


@dataclass
class HyperbolicityResults:
    """Complete delta-hyperbolicity analysis."""
    results: dict[str, HyperbolicityResult]
    predictions: dict[str, dict]


def _gromov_delta_exact(D: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
    """
    Compute exact Gromov delta from a distance matrix.

    For each quadruple (i,j,k,l), compute the three sums of paired distances:
      S1 = d(i,j) + d(k,l)
      S2 = d(i,k) + d(j,l)
      S3 = d(i,l) + d(j,k)
    Sort them. delta_ijkl = (largest - second largest) / 2.
    Gromov delta = max over all quadruples.

    For N=100, C(100,4) ≈ 3.9M quadruples — feasible but slow in pure Python.
    Uses vectorized approach on sampled quadruples.

    Returns (delta, all_deltas_array).
    """
    n = D.shape[0]

    if n <= 4:
        # Direct computation
        if n < 4:
            return 0.0, np.array([0.0])
        i, j, k, l = 0, 1, 2, 3
        s1 = D[i, j] + D[k, l]
        s2 = D[i, k] + D[j, l]
        s3 = D[i, l] + D[j, k]
        sums = sorted([s1, s2, s3])
        delta = (sums[2] - sums[1]) / 2
        return delta, np.array([delta])

    # For N=100, exact is ~3.9M quadruples.
    # Sample if too large, otherwise compute all.
    max_quadruples = 500_000
    from itertools import combinations as comb

    n_total = n * (n - 1) * (n - 2) * (n - 3) // 24  # C(n, 4)

    if n_total <= max_quadruples:
        # Exact: enumerate all quadruples
        quads = np.array(list(comb(range(n), 4)), dtype=np.int32)
    else:
        # Random sample
        rng = np.random.default_rng(42)
        quads = np.zeros((max_quadruples, 4), dtype=np.int32)
        for q in range(max_quadruples):
            quads[q] = rng.choice(n, size=4, replace=False)

    # Vectorized delta computation
    i, j, k, l = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]
    s1 = D[i, j] + D[k, l]
    s2 = D[i, k] + D[j, l]
    s3 = D[i, l] + D[j, k]

    sums = np.stack([s1, s2, s3], axis=1)
    sums.sort(axis=1)
    deltas = (sums[:, 2] - sums[:, 1]) / 2

    return float(np.max(deltas)), deltas


def _compute_centroid_distances(
    X: NDArray[np.float32],
    modes: NDArray,
) -> dict[tuple[str, str], float]:
    """Compute pairwise Euclidean distances between mode centroids."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    unique_modes = sorted(set(modes))
    centroids = {}
    for mode in unique_modes:
        mask = modes == mode
        centroids[mode] = X_std[mask].mean(axis=0)

    distances: dict[tuple[str, str], float] = {}
    for i, m1 in enumerate(unique_modes):
        for m2 in unique_modes[i + 1:]:
            d = float(np.linalg.norm(centroids[m1] - centroids[m2]))
            distances[(m1, m2)] = d

    return distances


def _assess_topology(
    centroid_dists: dict[tuple[str, str], float],
) -> str:
    """Assess hierarchical topology from centroid distances."""
    # Find nearest and farthest pairs
    sorted_pairs = sorted(centroid_dists.items(), key=lambda x: x[1])
    nearest = sorted_pairs[0]
    farthest = sorted_pairs[-1]

    notes = []
    notes.append(f"Nearest pair: {nearest[0]} (d={nearest[1]:.2f})")
    notes.append(f"Farthest pair: {farthest[0]} (d={farthest[1]:.2f})")

    # Check specific predictions
    lin_soc = centroid_dists.get(("linear", "socratic"),
                                 centroid_dists.get(("socratic", "linear")))
    con_dia = centroid_dists.get(("contrastive", "dialectical"),
                                 centroid_dists.get(("dialectical", "contrastive")))
    if lin_soc and con_dia:
        notes.append(f"Linear-socratic distance: {lin_soc:.2f}")
        notes.append(f"Contrastive-dialectical distance: {con_dia:.2f}")

    # Analogical distances (should be large — outgroup)
    anal_dists = {k: v for k, v in centroid_dists.items() if "analogical" in k}
    if anal_dists:
        mean_anal = np.mean(list(anal_dists.values()))
        non_anal = {k: v for k, v in centroid_dists.items() if "analogical" not in k}
        mean_non_anal = np.mean(list(non_anal.values())) if non_anal else 0
        notes.append(f"Mean analogical distance: {mean_anal:.2f}")
        notes.append(f"Mean non-analogical distance: {mean_non_anal:.2f}")
        notes.append(f"Analogical outgroup ratio: {mean_anal / max(mean_non_anal, 1e-6):.2f}")

    return "\n    ".join(notes)


def run_hyperbolicity_single(
    X: NDArray[np.float32],
    modes: NDArray,
    feature_space: str,
) -> HyperbolicityResult:
    """Compute delta-hyperbolicity for a single feature space."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X).astype(np.float64)

    # Pairwise distance matrix
    D = squareform(pdist(X_std, metric="euclidean"))
    diameter = float(np.max(D))

    # Gromov delta
    delta, all_deltas = _gromov_delta_exact(D)
    delta_relative = delta / diameter if diameter > 0 else 0.0

    # Centroid distances
    centroid_dists = _compute_centroid_distances(X, modes)
    topology = _assess_topology(centroid_dists)

    return HyperbolicityResult(
        feature_space=feature_space,
        n_samples=X.shape[0],
        ambient_dim=X.shape[1],
        delta=delta,
        delta_relative=delta_relative,
        diameter=diameter,
        delta_mean=float(np.mean(all_deltas)),
        delta_median=float(np.median(all_deltas)),
        delta_std=float(np.std(all_deltas)),
        delta_95th=float(np.percentile(all_deltas, 95)),
        n_quadruples=len(all_deltas),
        centroid_distances=centroid_dists,
        topology_notes=topology,
    )


def run_hyperbolicity(data: Run4Data) -> HyperbolicityResults:
    """Run delta-hyperbolicity across all feature spaces."""
    results: dict[str, HyperbolicityResult] = {}

    for tier in TIERS_AND_GROUPS:
        X = data.get_tier(tier)
        print(f"\n{'='*60}")
        print(f"Delta-hyperbolicity: {tier} (D={X.shape[1]})")
        print(f"{'='*60}")

        result = run_hyperbolicity_single(X, data.modes, tier)
        results[tier] = result

        print(f"  delta = {result.delta:.4f} (relative: {result.delta_relative:.4f})")
        print(f"  diameter = {result.diameter:.2f}")
        print(f"  delta distribution: mean={result.delta_mean:.4f}, "
              f"median={result.delta_median:.4f}, "
              f"95th={result.delta_95th:.4f}")
        print(f"  n_quadruples = {result.n_quadruples:,}")
        print(f"  Topology: {result.topology_notes}")

    predictions = _evaluate_predictions(results)

    return HyperbolicityResults(results=results, predictions=predictions)


def _evaluate_predictions(results: dict[str, HyperbolicityResult]) -> dict[str, dict]:
    """Evaluate pre-registered delta-hyperbolicity predictions."""
    predictions: dict[str, dict] = {}

    t2t25 = results.get("T2+T2.5")
    combined = results.get("combined")

    # P1: Delta is moderate
    if t2t25:
        # "Moderate" = relative delta in [0.1, 0.4]
        predictions["P1_moderate_delta"] = {
            "prediction": "Relative delta in [0.1, 0.4] for T2+T2.5",
            "actual_relative": t2t25.delta_relative,
            "actual_absolute": t2t25.delta,
            "in_range": 0.1 <= t2t25.delta_relative <= 0.4,
            "assessment": (
                "strongly hyperbolic" if t2t25.delta_relative < 0.1 else
                "moderate" if t2t25.delta_relative < 0.4 else
                "not tree-like"
            ),
        }

    # P2: T2+T2.5 more tree-like than combined
    if t2t25 and combined:
        predictions["P2_T2T25_more_treelike"] = {
            "prediction": "T2+T2.5 relative delta < combined relative delta",
            "T2+T2.5_delta_rel": t2t25.delta_relative,
            "combined_delta_rel": combined.delta_relative,
            "confirmed": t2t25.delta_relative < combined.delta_relative,
        }

    # P3: Analogical is outgroup
    if t2t25:
        anal_dists = {k: v for k, v in t2t25.centroid_distances.items()
                      if "analogical" in k}
        non_anal = {k: v for k, v in t2t25.centroid_distances.items()
                    if "analogical" not in k}
        if anal_dists and non_anal:
            mean_anal = np.mean(list(anal_dists.values()))
            mean_non_anal = np.mean(list(non_anal.values()))
            predictions["P3_analogical_outgroup"] = {
                "prediction": "Mean analogical centroid distance > mean non-analogical",
                "mean_analogical": float(mean_anal),
                "mean_non_analogical": float(mean_non_anal),
                "ratio": float(mean_anal / max(mean_non_anal, 1e-6)),
                "confirmed": mean_anal > mean_non_anal,
            }

    # P4: Linear-socratic nearby
    if t2t25:
        ls = t2t25.centroid_distances.get(("linear", "socratic"))
        if ls is not None:
            all_dists = list(t2t25.centroid_distances.values())
            predictions["P4_linear_socratic_nearby"] = {
                "prediction": "Linear-socratic distance < median pairwise distance",
                "linear_socratic_dist": ls,
                "median_dist": float(np.median(all_dists)),
                "confirmed": ls < float(np.median(all_dists)),
                "rank": sorted(all_dists).index(ls) + 1,
                "of": len(all_dists),
            }

    # P5: Contrastive-dialectical on separate branch
    if t2t25:
        cd = t2t25.centroid_distances.get(("contrastive", "dialectical"))
        ls = t2t25.centroid_distances.get(("linear", "socratic"))
        if cd is not None and ls is not None:
            # Check cross-branch distances
            cl = t2t25.centroid_distances.get(("contrastive", "linear"))
            cs = t2t25.centroid_distances.get(("contrastive", "socratic"))
            dl = t2t25.centroid_distances.get(("dialectical", "linear"))
            ds = t2t25.centroid_distances.get(("dialectical", "socratic"))
            cross_dists = [d for d in [cl, cs, dl, ds] if d is not None]
            within_dists = [d for d in [cd, ls] if d is not None]
            predictions["P5_two_branches"] = {
                "prediction": "Cross-branch distances > within-branch distances",
                "contrastive_dialectical": cd,
                "linear_socratic": ls,
                "mean_cross": float(np.mean(cross_dists)) if cross_dists else None,
                "mean_within": float(np.mean(within_dists)) if within_dists else None,
                "confirmed": (float(np.mean(cross_dists)) > float(np.mean(within_dists))
                              if cross_dists and within_dists else False),
            }

    return predictions


def save_results(results: HyperbolicityResults, output_dir: Path) -> None:
    """Save delta-hyperbolicity results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    output: dict = {"results": {}, "predictions": results.predictions}

    for tier, result in results.results.items():
        tier_out = {
            "feature_space": result.feature_space,
            "n_samples": result.n_samples,
            "ambient_dim": result.ambient_dim,
            "delta": result.delta,
            "delta_relative": result.delta_relative,
            "diameter": result.diameter,
            "delta_mean": result.delta_mean,
            "delta_median": result.delta_median,
            "delta_std": result.delta_std,
            "delta_95th": result.delta_95th,
            "n_quadruples": result.n_quadruples,
            "centroid_distances": {
                f"{k[0]}__{k[1]}": v
                for k, v in result.centroid_distances.items()
            },
            "topology_notes": result.topology_notes,
        }
        output["results"][tier] = tier_out

    # Convert numpy types for JSON serialization
    def _clean(obj: object) -> object:
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    output = _clean(output)

    with open(output_dir / "delta_hyperbolicity_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_dir / 'delta_hyperbolicity_results.json'}")


if __name__ == "__main__":
    print("Loading Run 4 data...")
    data = load_run4()
    print(f"Loaded {data.n_samples} samples")

    print("\nRunning delta-hyperbolicity analysis...")
    results = run_hyperbolicity(data)

    # Print predictions
    print("\n" + "=" * 60)
    print("PREDICTION OUTCOMES")
    print("=" * 60)
    for name, pred in results.predictions.items():
        print(f"\n{name}:")
        for k, v in pred.items():
            print(f"  {k}: {v}")

    # Save
    output_dir = Path(__file__).parent / "results"
    save_results(results, output_dir)
