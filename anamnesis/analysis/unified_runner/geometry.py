"""Sections 4-6: Intrinsic dimension, CCGP, topology & hyperbolicity."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import cosine, pdist, squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .data_loading import AnalysisData
from .utils import ALL_TIERS, standardize, remove_constant, get_available_tiers


# ──────────────────────────────────────────────────────────────
# Section 4: Intrinsic dimension
# ──────────────────────────────────────────────────────────────

def run_intrinsic_dimension(data: AnalysisData) -> dict:
    """Intrinsic dimension profiling with TwoNN, bootstrap CIs, GRIDE."""
    try:
        from dadapy import Data as DADAData
    except ImportError:
        return {"error": "dadapy not installed"}

    try:
        from skdim.id import TwoNN as SkdimTwoNN
        has_skdim = True
    except ImportError:
        has_skdim = False

    seeds = [42, 123, 777]
    results: dict = {"global": {}, "per_mode": {}, "gride": {}}

    # Global ID per tier
    available_tiers, _ = get_available_tiers(data)
    for tier in available_tiers:
        print(f"    ID: {tier}")
        X = data.get_tier(tier)
        X_std = standardize(X)
        X_clean = remove_constant(X_std)

        tier_result: dict = {"n_features_clean": X_clean.shape[1]}

        # DADApy point estimate
        try:
            dada = DADAData(X_clean)
            dada.compute_id_2NN()
            tier_result["dadapy_id"] = float(dada.intrinsic_dim)
            tier_result["dadapy_err"] = float(dada.intrinsic_dim_err)
        except Exception as e:
            tier_result["dadapy_id"] = f"ERROR: {e}"

        # skdim cross-check
        if has_skdim:
            try:
                est = SkdimTwoNN()
                est.fit(X_clean)
                tier_result["skdim_id"] = float(est.dimension_)
            except Exception as e:
                tier_result["skdim_id"] = f"ERROR: {e}"

        # Bootstrap CIs across seeds
        boot_by_seed: dict = {}
        for seed in seeds:
            rng = np.random.default_rng(seed)
            boot_ids = []
            n = X_clean.shape[0]
            subsample_size = max(8, int(n * 0.8))
            for _ in range(200):
                idx = rng.choice(n, size=subsample_size, replace=False)
                try:
                    dada_b = DADAData(X_clean[idx])
                    dada_b.compute_id_2NN()
                    if dada_b.intrinsic_dim is not None:
                        boot_ids.append(float(dada_b.intrinsic_dim))
                except Exception:
                    continue
            if boot_ids:
                boot_by_seed[seed] = {
                    "mean": float(np.mean(boot_ids)),
                    "std": float(np.std(boot_ids)),
                    "ci_lo": float(np.percentile(boot_ids, 2.5)),
                    "ci_hi": float(np.percentile(boot_ids, 97.5)),
                    "n_successful": len(boot_ids),
                }
        tier_result["bootstrap_by_seed"] = boot_by_seed

        results["global"][tier] = tier_result

    # Per-mode ID (T2+T2.5 only)
    print("    Per-mode ID (T2+T2.5)...")
    X_full = data.get_tier("T2+T2.5")
    for mode in data.unique_modes:
        mask = data.mode_mask(mode)
        X_mode = X_full[mask]
        X_std = standardize(X_mode)
        X_clean = remove_constant(X_std)

        mode_result: dict = {"n_samples": int(np.sum(mask))}
        try:
            dada = DADAData(X_clean)
            dada.compute_id_2NN()
            mode_result["dadapy_id"] = float(dada.intrinsic_dim)
        except Exception as e:
            mode_result["dadapy_id"] = f"ERROR: {e}"

        if has_skdim:
            try:
                est = SkdimTwoNN()
                est.fit(X_clean)
                mode_result["skdim_id"] = float(est.dimension_)
            except Exception as e:
                mode_result["skdim_id"] = f"ERROR: {e}"

        # Bootstrap per mode
        boot_ids_all = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            n = X_clean.shape[0]
            subsample_size = max(6, int(n * 0.8))
            for _ in range(200):
                idx = rng.choice(n, size=subsample_size, replace=False)
                try:
                    dada_b = DADAData(X_clean[idx])
                    dada_b.compute_id_2NN()
                    if dada_b.intrinsic_dim is not None:
                        boot_ids_all.append(float(dada_b.intrinsic_dim))
                except Exception:
                    continue
        if boot_ids_all:
            mode_result["bootstrap_mean"] = float(np.mean(boot_ids_all))
            mode_result["bootstrap_std"] = float(np.std(boot_ids_all))
            mode_result["bootstrap_ci"] = [
                float(np.percentile(boot_ids_all, 2.5)),
                float(np.percentile(boot_ids_all, 97.5)),
            ]

        results["per_mode"][mode] = mode_result

    # GRIDE multiscale (T2+T2.5)
    print("    GRIDE multiscale...")
    try:
        X_full_std = standardize(data.get_tier("T2+T2.5"))
        X_full_clean = remove_constant(X_full_std)
        dada_g = DADAData(X_full_clean)
        dada_g.compute_id_2NN()
        scales = [2, 4, 8, 16, 32, 64]
        valid_scales = [s for s in scales if s < X_full_clean.shape[0] // 2]
        if valid_scales:
            ids_scale, errs_scale, _ = dada_g.return_id_scaling_gride(
                range_max=max(valid_scales) + 1,
            )
            results["gride"] = {
                "ids": [float(x) for x in ids_scale],
                "errors": [float(x) for x in errs_scale],
            }
    except Exception as e:
        results["gride"] = {"error": str(e)}

    # Tier convergence metric
    global_ids = {}
    for tier in ["T1", "T2", "T2.5"]:
        tid = results["global"].get(tier, {}).get("dadapy_id")
        if isinstance(tid, (int, float)):
            global_ids[tier] = tid

    if len(global_ids) == 3:
        vals = list(global_ids.values())
        results["tier_convergence"] = {
            "max_pairwise_diff": float(max(vals) - min(vals)),
            "converged_within_2": (max(vals) - min(vals)) < 4.0,
            "values": global_ids,
        }

    return results


# ──────────────────────────────────────────────────────────────
# Section 5: CCGP
# ──────────────────────────────────────────────────────────────

def _generate_topic_folds(
    topics: list[str], n_folds: int, rng: np.random.Generator,
) -> list[tuple[list[str], list[str]]]:
    topics_arr = np.array(topics)
    n = len(topics_arr)
    fold_size = n // n_folds
    perm = rng.permutation(n)
    folds = []
    for i in range(n_folds):
        test_idx = perm[i * fold_size : (i + 1) * fold_size]
        train_idx = np.setdiff1d(perm, test_idx)
        folds.append((topics_arr[train_idx].tolist(), topics_arr[test_idx].tolist()))
    return folds


def _ccgp_variant(
    X: NDArray, modes: NDArray, topics: NDArray,
    n_folds: int, seed: int, clf_name: str,
) -> dict:
    """Run CCGP with specific classifier/fold/seed."""
    rng = np.random.default_rng(seed)
    unique_modes = sorted(set(modes))
    unique_topics = sorted(set(topics))

    topic_folds = _generate_topic_folds(unique_topics, n_folds=n_folds, rng=rng)

    # 5-way multiclass
    accs = []
    per_mode_correct: dict[str, int] = {m: 0 for m in unique_modes}
    per_mode_total: dict[str, int] = {m: 0 for m in unique_modes}

    for train_topics, test_topics in topic_folds:
        train_mask = np.isin(topics, train_topics)
        test_mask = np.isin(topics, test_topics)
        X_train, y_train = X[train_mask], modes[train_mask]
        X_test, y_test = X[test_mask], modes[test_mask]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        n_modes = len(unique_modes)
        if clf_name == "knn3":
            clf = KNeighborsClassifier(n_neighbors=min(3, len(X_train) // n_modes))
        elif clf_name == "knn5":
            clf = KNeighborsClassifier(n_neighbors=min(5, len(X_train) // n_modes))
        elif clf_name == "linearsvc":
            clf = LinearSVC(max_iter=5000, dual="auto")
        elif clf_name == "rf":
            clf = RandomForestClassifier(n_estimators=100, random_state=seed)
        else:
            raise ValueError(f"Unknown classifier: {clf_name}")

        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        accs.append(float(np.mean(y_pred == y_test)))

        for mode in unique_modes:
            mm = y_test == mode
            per_mode_correct[mode] += int(np.sum(y_pred[mm] == mode))
            per_mode_total[mode] += int(np.sum(mm))

    per_mode_recall = {
        m: per_mode_correct[m] / max(per_mode_total[m], 1) for m in unique_modes
    }

    # Binary dichotomies
    dichotomies = []
    n = len(unique_modes)
    for k in range(1, n // 2 + 1):
        for group_a in combinations(unique_modes, k):
            group_b = tuple(m for m in unique_modes if m not in group_a)
            if k == n // 2 and group_a > group_b:
                continue
            dichotomies.append((group_a, group_b))

    n_decodable = 0
    dich_details = []
    for group_a, group_b in dichotomies:
        fold_accs = []
        for train_topics, test_topics in topic_folds:
            train_mask = np.isin(topics, train_topics)
            test_mask = np.isin(topics, test_topics)
            mode_mask = np.isin(modes, group_a + group_b)
            train_full = train_mask & mode_mask
            test_full = test_mask & mode_mask

            if np.sum(train_full) < 4 or np.sum(test_full) < 2:
                continue

            X_tr, X_te = X[train_full], X[test_full]
            y_tr = np.isin(modes[train_full], group_a).astype(int)
            y_te = np.isin(modes[test_full], group_a).astype(int)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            if clf_name == "knn3":
                clf = KNeighborsClassifier(n_neighbors=min(3, max(1, len(X_tr) // 2)))
            elif clf_name == "knn5":
                clf = KNeighborsClassifier(n_neighbors=min(5, max(1, len(X_tr) // 2)))
            elif clf_name == "linearsvc":
                clf = LinearSVC(max_iter=5000, dual="auto")
            elif clf_name == "rf":
                clf = RandomForestClassifier(n_estimators=100, random_state=seed)
            else:
                raise ValueError(clf_name)

            clf.fit(X_tr_s, y_tr)
            fold_accs.append(float(np.mean(clf.predict(X_te_s) == y_te)))

        mean_acc = float(np.mean(fold_accs)) if fold_accs else 0.0
        decodable = mean_acc > 0.65
        if decodable:
            n_decodable += 1
        dich_details.append({
            "group_a": list(group_a),
            "group_b": list(group_b),
            "mean_accuracy": mean_acc,
            "decodable": decodable,
        })

    return {
        "multiclass_mean": float(np.mean(accs)) if accs else 0.0,
        "multiclass_fold_accs": accs,
        "per_mode_recall": per_mode_recall,
        "n_decodable": n_decodable,
        "n_dichotomies": len(dichotomies),
        "ccgp_score": n_decodable / max(len(dichotomies), 1),
        "dichotomies": dich_details,
    }


def run_ccgp(data: AnalysisData) -> dict:
    """Run CCGP with multiple variants."""
    results: dict = {"variants": {}}
    tier = "T2+T2.5"
    X = data.get_tier(tier)

    # Primary: kNN k=3, multiple seeds
    for seed in [42, 123, 777]:
        key = f"knn3_seed{seed}_5fold"
        print(f"    CCGP: {key}")
        results["variants"][key] = _ccgp_variant(
            X, data.modes, data.topics, n_folds=5, seed=seed, clf_name="knn3",
        )

    # Different classifiers
    for clf_name in ["knn5", "linearsvc", "rf"]:
        key = f"{clf_name}_seed42_5fold"
        print(f"    CCGP: {key}")
        results["variants"][key] = _ccgp_variant(
            X, data.modes, data.topics, n_folds=5, seed=42, clf_name=clf_name,
        )

    # Different fold counts
    for n_folds in [4, 10, 20]:
        key = f"knn3_seed42_{n_folds}fold"
        print(f"    CCGP: {key}")
        results["variants"][key] = _ccgp_variant(
            X, data.modes, data.topics, n_folds=n_folds, seed=42, clf_name="knn3",
        )

    # Other tiers for reference
    for tier_name in ["T2", "T2.5", "combined"]:
        X_t = data.get_tier(tier_name)
        key = f"knn3_seed42_5fold_{tier_name}"
        print(f"    CCGP: {key}")
        results["variants"][key] = _ccgp_variant(
            X_t, data.modes, data.topics, n_folds=5, seed=42, clf_name="knn3",
        )

    # Summary
    ccgp_scores = [v["ccgp_score"] for v in results["variants"].values()]
    results["summary"] = {
        "min_ccgp": float(min(ccgp_scores)),
        "max_ccgp": float(max(ccgp_scores)),
        "all_perfect": all(s == 1.0 for s in ccgp_scores),
    }

    return results


# ──────────────────────────────────────────────────────────────
# Section 6: Topology & hyperbolicity
# ──────────────────────────────────────────────────────────────

def _compute_centroids(
    X: NDArray, modes: NDArray, metric: str = "euclidean",
) -> tuple[dict[str, NDArray], dict[tuple[str, str], float]]:
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    unique_modes = sorted(set(modes))
    centroids = {m: X_std[modes == m].mean(axis=0) for m in unique_modes}

    distances: dict[tuple[str, str], float] = {}
    for i, m1 in enumerate(unique_modes):
        for m2 in unique_modes[i + 1:]:
            if metric == "euclidean":
                d = float(np.linalg.norm(centroids[m1] - centroids[m2]))
            elif metric == "cosine":
                d = float(cosine(centroids[m1], centroids[m2]))
            elif metric == "manhattan":
                d = float(np.sum(np.abs(centroids[m1] - centroids[m2])))
            else:
                raise ValueError(f"Unknown metric: {metric}")
            distances[(m1, m2)] = d

    return centroids, distances


def _hierarchical_clustering(
    dists: dict[tuple[str, str], float], modes: list[str], method: str = "average",
) -> str:
    n = len(modes)
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            key = (modes[i], modes[j])
            if key not in dists:
                key = (modes[j], modes[i])
            condensed.append(dists[key])

    Z = linkage(condensed, method=method)
    root = to_tree(Z)

    def _to_newick(node) -> str:
        if node.is_leaf():
            return modes[node.id]
        return f"({_to_newick(node.get_left())},{_to_newick(node.get_right())})"

    return _to_newick(root)


def run_topology(data: AnalysisData) -> dict:
    """Run centroid topology and delta-hyperbolicity analysis."""
    tier = "T2+T2.5"
    X = data.get_tier(tier)
    modes_list = sorted(set(data.modes))
    results: dict = {"tier": tier}

    # Centroid distances in 3 metrics
    metric_dists: dict[str, dict[tuple[str, str], float]] = {}
    for metric in ["euclidean", "cosine", "manhattan"]:
        _, dists = _compute_centroids(X, data.modes, metric)
        metric_dists[metric] = dists
        results[f"{metric}_centroid_distances"] = {
            f"{k[0]}__{k[1]}": v for k, v in dists.items()
        }

    # Hierarchical clustering
    results["hierarchical_clustering"] = {}
    for metric_name, dists in metric_dists.items():
        methods = ["single", "average", "complete"]
        if metric_name == "euclidean":
            methods.append("ward")
        for method in methods:
            key = f"{metric_name}_{method}"
            try:
                results["hierarchical_clustering"][key] = _hierarchical_clustering(
                    dists, modes_list, method,
                )
            except Exception as e:
                results["hierarchical_clustering"][key] = f"ERROR: {e}"

    # Topology summary
    topology_summary: dict = {}
    for metric_name, dists in metric_dists.items():
        sorted_pairs = sorted(dists.items(), key=lambda x: x[1])
        nearest = sorted_pairs[0]
        farthest = sorted_pairs[-1]

        anal_dists = {k: v for k, v in dists.items() if "analogical" in k}
        non_anal = {k: v for k, v in dists.items() if "analogical" not in k}
        mean_anal = float(np.mean(list(anal_dists.values()))) if anal_dists else 0
        mean_non_anal = float(np.mean(list(non_anal.values()))) if non_anal else 1

        topology_summary[metric_name] = {
            "nearest_pair": f"{nearest[0][0]}-{nearest[0][1]}",
            "nearest_dist": nearest[1],
            "farthest_pair": f"{farthest[0][0]}-{farthest[0][1]}",
            "farthest_dist": farthest[1],
            "analogical_outgroup_ratio": mean_anal / max(mean_non_anal, 1e-10),
        }
    results["topology_summary"] = topology_summary

    # Gromov delta-hyperbolicity (Euclidean)
    print("    Delta-hyperbolicity (Euclidean)...")
    X_std = standardize(X).astype(np.float64)
    D = squareform(pdist(X_std, metric="euclidean"))
    diameter = float(np.max(D))

    rng = np.random.default_rng(42)
    n_pts = D.shape[0]
    n_quads = min(500_000, n_pts * (n_pts - 1) * (n_pts - 2) * (n_pts - 3) // 24)
    quads = np.zeros((n_quads, 4), dtype=np.int32)
    for q in range(n_quads):
        quads[q] = rng.choice(n_pts, size=4, replace=False)

    i, j, k, l = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]
    s1 = D[i, j] + D[k, l]
    s2 = D[i, k] + D[j, l]
    s3 = D[i, l] + D[j, k]
    sums = np.stack([s1, s2, s3], axis=1)
    sums.sort(axis=1)
    deltas = (sums[:, 2] - sums[:, 1]) / 2

    results["gromov_delta_euclidean"] = {
        "delta_max": float(np.max(deltas)),
        "delta_relative": float(np.max(deltas)) / diameter if diameter > 0 else 0,
        "delta_mean": float(np.mean(deltas)),
        "delta_median": float(np.median(deltas)),
        "diameter": diameter,
        "n_quadruples": n_quads,
    }

    return results


# ── Section 6b: Manifold Geometry (tangent-space, geodesic distortion, curvature) ──


def run_manifold_geometry(data: AnalysisData) -> dict:
    """Manifold geometry analyses on T2+T2.5: tangent-space alignment,
    geodesic-vs-euclidean distortion, and scale-dependent curvature.

    These characterize the shape of the mode manifold beyond what
    intrinsic dimension and hyperbolicity capture.
    """
    from scipy.linalg import subspace_angles
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap
    from sklearn.neighbors import NearestNeighbors

    X = data.get_tier("T2+T2.5")
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    y = data.modes

    results: dict = {}

    # ── 1. Local tangent-space alignment across modes ──
    print("    Local tangent-space alignment...")
    try:
        n_components = min(10, X_std.shape[1])
        mode_subspaces: dict[str, NDArray] = {}

        for mode in data.unique_modes:
            mask = data.mode_mask(mode)
            X_mode = X_std[mask]
            if X_mode.shape[0] < n_components + 1:
                continue
            pca = PCA(n_components=n_components)
            pca.fit(X_mode)
            mode_subspaces[mode] = pca.components_.T  # columns = principal directions

        # Pairwise principal angles between mode subspaces
        tangent_angles: dict[str, dict] = {}
        modes_with_subspaces = sorted(mode_subspaces.keys())
        for i, m1 in enumerate(modes_with_subspaces):
            for m2 in modes_with_subspaces[i + 1:]:
                angles = subspace_angles(mode_subspaces[m1], mode_subspaces[m2])
                angles_deg = np.degrees(angles)
                tangent_angles[f"{m1}_vs_{m2}"] = {
                    "mean_angle_deg": float(np.mean(angles_deg)),
                    "max_angle_deg": float(np.max(angles_deg)),
                    "min_angle_deg": float(np.min(angles_deg)),
                    "angles_deg": angles_deg.tolist(),
                }

        # Variance explained per mode (how concentrated is each mode's structure?)
        mode_variance: dict[str, dict] = {}
        for mode in data.unique_modes:
            mask = data.mode_mask(mode)
            X_mode = X_std[mask]
            if X_mode.shape[0] < n_components + 1:
                continue
            pca = PCA(n_components=n_components)
            pca.fit(X_mode)
            mode_variance[mode] = {
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "cumulative_5": float(np.sum(pca.explained_variance_ratio_[:5])),
                "cumulative_10": float(np.sum(pca.explained_variance_ratio_[:10])),
            }

        results["tangent_space"] = {
            "pairwise_angles": tangent_angles,
            "mode_variance_explained": mode_variance,
            "n_components": n_components,
        }
    except Exception as e:
        results["tangent_space"] = {"error": str(e)}

    # ── 2. Geodesic-vs-Euclidean distortion (Isomap) ──
    print("    Geodesic-vs-Euclidean distortion (Isomap)...")
    try:
        n_neighbors_iso = min(10, X_std.shape[0] - 1)
        n_components_iso = min(10, X_std.shape[1], X_std.shape[0] - 1)
        isomap = Isomap(
            n_neighbors=n_neighbors_iso,
            n_components=n_components_iso,
        )
        isomap.fit(X_std)

        # Geodesic distance matrix from Isomap's kernel
        D_geo = isomap.dist_matrix_

        # Euclidean distance matrix
        D_euc = np.sqrt(np.sum((X_std[:, None] - X_std[None, :]) ** 2, axis=2))

        # Distortion: geodesic / euclidean (for non-zero euclidean)
        mask_nonzero = D_euc > 1e-10
        np.fill_diagonal(mask_nonzero, False)
        distortion = np.where(mask_nonzero, D_geo / D_euc, np.nan)

        # Overall distortion stats
        valid_distortion = distortion[~np.isnan(distortion)]
        overall = {
            "mean_distortion": float(np.mean(valid_distortion)),
            "std_distortion": float(np.std(valid_distortion)),
            "max_distortion": float(np.max(valid_distortion)),
            "median_distortion": float(np.median(valid_distortion)),
        }

        # Per-mode: average distortion for pairs within vs between modes
        within_dists: list[float] = []
        between_dists: list[float] = []
        per_mode_distortion: dict[str, dict] = {}

        for mode in data.unique_modes:
            mask_mode = data.mode_mask(mode)
            idx_mode = np.where(mask_mode)[0]
            mode_dists = []
            for ii in range(len(idx_mode)):
                for jj in range(ii + 1, len(idx_mode)):
                    a, b = idx_mode[ii], idx_mode[jj]
                    if mask_nonzero[a, b]:
                        d = float(distortion[a, b])
                        mode_dists.append(d)
                        within_dists.append(d)

            if mode_dists:
                per_mode_distortion[mode] = {
                    "mean": float(np.mean(mode_dists)),
                    "std": float(np.std(mode_dists)),
                }

        # Between-mode distortions (sample to keep tractable)
        rng = np.random.default_rng(42)
        for _ in range(min(5000, len(y) * len(y))):
            a, b = rng.choice(len(y), size=2, replace=False)
            if y[a] != y[b] and mask_nonzero[a, b]:
                between_dists.append(float(distortion[a, b]))

        results["geodesic_distortion"] = {
            "overall": overall,
            "per_mode": per_mode_distortion,
            "within_mode_mean": float(np.mean(within_dists)) if within_dists else None,
            "between_mode_mean": float(np.mean(between_dists)) if between_dists else None,
            "isomap_n_neighbors": n_neighbors_iso,
            "reconstruction_error": float(isomap.reconstruction_error()),
        }
    except Exception as e:
        results["geodesic_distortion"] = {"error": str(e)}

    # ── 3. Scale-dependent curvature proxies ──
    print("    Scale-dependent curvature...")
    try:
        scales = [k for k in [5, 10, 20, 50] if k < X_std.shape[0]]
        curvature_results: dict = {"scales": scales, "per_scale": {}}

        for k in scales:
            nn = NearestNeighbors(n_neighbors=k + 1)
            nn.fit(X_std)
            _, indices = nn.kneighbors(X_std)
            knn_indices = indices[:, 1:]  # exclude self

            # For each point: local PCA on k neighbors, reconstruction error
            recon_errors: list[float] = []
            for i in range(len(X_std)):
                neighbors = X_std[knn_indices[i]]
                centered = neighbors - np.mean(neighbors, axis=0)

                # Use min of (k-1, n_features) components
                n_comp_local = min(k - 1, centered.shape[1])
                if n_comp_local < 1:
                    continue

                pca_local = PCA(n_components=n_comp_local)
                pca_local.fit(centered)

                # Use ID-estimated components (or top 5) for reconstruction
                n_use = min(5, n_comp_local)
                var_explained = float(np.sum(pca_local.explained_variance_ratio_[:n_use]))
                recon_error = 1.0 - var_explained
                recon_errors.append(recon_error)

            recon_arr = np.array(recon_errors)

            # Per-mode curvature
            per_mode_curv: dict[str, float] = {}
            for mode in data.unique_modes:
                mask_mode = data.mode_mask(mode)
                mode_errors = recon_arr[mask_mode[:len(recon_arr)]]
                if len(mode_errors) > 0:
                    per_mode_curv[mode] = float(np.mean(mode_errors))

            curvature_results["per_scale"][k] = {
                "mean_curvature": float(np.mean(recon_arr)),
                "std_curvature": float(np.std(recon_arr)),
                "per_mode": per_mode_curv,
            }

        results["curvature"] = curvature_results
    except Exception as e:
        results["curvature"] = {"error": str(e)}

    # ── 4. Persistent homology (optional) ──
    try:
        import ripser
        print("    Persistent homology (ripser)...")

        # Subsample for tractability
        rng = np.random.default_rng(42)
        n_sub = min(100, X_std.shape[0])
        sub_idx = rng.choice(X_std.shape[0], size=n_sub, replace=False)
        X_sub = X_std[sub_idx]

        diagrams = ripser.ripser(X_sub, maxdim=2)["dgms"]

        betti: dict[str, dict] = {}
        for dim, dgm in enumerate(diagrams):
            finite_mask = np.isfinite(dgm[:, 1])
            lifetimes = dgm[finite_mask, 1] - dgm[finite_mask, 0]
            betti[f"H{dim}"] = {
                "n_features": len(dgm),
                "n_finite": int(np.sum(finite_mask)),
                "mean_lifetime": float(np.mean(lifetimes)) if len(lifetimes) > 0 else 0.0,
                "max_lifetime": float(np.max(lifetimes)) if len(lifetimes) > 0 else 0.0,
                "median_lifetime": float(np.median(lifetimes)) if len(lifetimes) > 0 else 0.0,
            }

        results["persistent_homology"] = {
            "betti_numbers": betti,
            "n_subsampled": n_sub,
        }
    except ImportError:
        results["persistent_homology"] = {
            "error": "ripser not installed — pip install ripser for persistent homology",
        }
    except Exception as e:
        results["persistent_homology"] = {"error": str(e)}

    return results
