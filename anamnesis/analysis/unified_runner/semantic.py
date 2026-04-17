"""Section 9: Semantic independence — compute vs text content signal.

Tests semantic orthogonality of compute features:
- Per-tier orthogonality battery (Mantel, R², per-mode surface vs compute)
- Prompt-swap confound test (train on core, predict on prompt-swap samples)
- TF-IDF and SBERT surface baselines
- Contrastive projection comparison
- Shuffle controls and retrieval analysis
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from .data_loading import AnalysisData

logger = logging.getLogger(__name__)


def _embed_texts_tfidf(texts: list[str], n_components: int = 100) -> NDArray:
    """TF-IDF + SVD embedding of texts."""
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    X_tfidf = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=min(n_components, X_tfidf.shape[1] - 1), random_state=42)
    return svd.fit_transform(X_tfidf).astype(np.float32)


def _embed_texts_sbert(texts: list[str]) -> NDArray | None:
    """Sentence-BERT embedding (optional dependency)."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)
    except ImportError:
        return None
    except Exception:
        return None


def _classify_condition(
    X: NDArray, y: NDArray, topics: NDArray,
    clf_name: str = "rf", seed: int = 42,
) -> dict:
    """Classify with GroupKFold by topic."""
    unique_topics = sorted(set(topics))
    topic_to_id = {t: i for i, t in enumerate(unique_topics)}
    groups = np.array([topic_to_id[t] for t in topics])

    n_folds = min(5, len(unique_topics))
    gkf = GroupKFold(n_splits=n_folds)
    fold_accs = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])

        if clf_name == "rf":
            clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
        elif clf_name == "knn":
            clf = KNeighborsClassifier(n_neighbors=3)
        else:
            raise ValueError(clf_name)

        clf.fit(X_tr, y[train_idx])
        fold_accs.append(float(accuracy_score(y[test_idx], clf.predict(X_te))))

    return {
        "accuracy": float(np.mean(fold_accs)),
        "fold_accuracies": fold_accs,
    }


def _mantel_test(
    D_compute: NDArray, D_semantic: NDArray, n_permutations: int = 1000, seed: int = 42,
) -> dict:
    """Mantel test between two distance matrices."""
    from scipy.spatial.distance import squareform

    # Extract upper triangle
    n = D_compute.shape[0]
    idx = np.triu_indices(n, k=1)
    x = D_compute[idx]
    y = D_semantic[idx]

    observed_r = float(np.corrcoef(x, y)[0, 1])

    rng = np.random.default_rng(seed)
    null_rs = []
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        D_perm = D_semantic[np.ix_(perm, perm)]
        y_perm = D_perm[idx]
        null_rs.append(float(np.corrcoef(x, y_perm)[0, 1]))

    null_arr = np.array(null_rs)
    p_value = float(np.mean(null_arr >= observed_r))

    return {
        "r": observed_r,
        "p_value": max(p_value, 1.0 / (n_permutations + 1)),
        "null_mean": float(np.mean(null_arr)),
        "null_std": float(np.std(null_arr)),
    }


def _text_to_compute_r2(
    X_semantic: NDArray, X_compute: NDArray, topics: NDArray,
) -> dict:
    """Ridge regression: predict each compute feature from semantic embedding."""
    unique_topics = sorted(set(topics))
    topic_to_id = {t: i for i, t in enumerate(unique_topics)}
    groups = np.array([topic_to_id[t] for t in topics])

    n_folds = min(5, len(unique_topics))
    gkf = GroupKFold(n_splits=n_folds)

    # Predict each compute feature
    n_compute = X_compute.shape[1]
    r2_per_feature = np.full(n_compute, np.nan)

    for feat_idx in range(n_compute):
        y_feat = X_compute[:, feat_idx]
        fold_r2s = []

        for train_idx, test_idx in gkf.split(X_semantic, y_feat, groups):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_semantic[train_idx])
            X_te = scaler.transform(X_semantic[test_idx])

            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
            ridge.fit(X_tr, y_feat[train_idx])
            y_pred = ridge.predict(X_te)

            ss_res = np.sum((y_feat[test_idx] - y_pred) ** 2)
            ss_tot = np.sum((y_feat[test_idx] - np.mean(y_feat[test_idx])) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-10)
            fold_r2s.append(r2)

        r2_per_feature[feat_idx] = float(np.mean(fold_r2s))

    return {
        "median_r2": float(np.median(r2_per_feature)),
        "mean_r2": float(np.mean(r2_per_feature)),
        "n_features_r2_above_01": int(np.sum(r2_per_feature > 0.1)),
        "n_features_r2_above_0": int(np.sum(r2_per_feature > 0)),
        "best_r2": float(np.max(r2_per_feature)),
        "worst_r2": float(np.min(r2_per_feature)),
    }


def _contrastive_topic_heldout(
    X_tfidf: NDArray, X_compute: NDArray,
    X_sbert: NDArray | None,
    y: NDArray, topics: NDArray, seed: int = 42,
) -> dict:
    """Compare contrastive projection (MLP + kNN) across feature types, topic-heldout.

    This is the decisive test: TF-IDF collapses under topic-heldout contrastive
    projection while compute features generalize.
    """
    try:
        from .contrastive import _train_contrastive_mlp, _embed, _build_topic_folds
    except ImportError:
        return {"error": "contrastive module not available (torch missing?)"}

    folds = _build_topic_folds(topics, n_folds=5, seed=seed)

    conditions: dict[str, NDArray] = {
        "tfidf": StandardScaler().fit_transform(X_tfidf),
        "compute_t2t25": StandardScaler().fit_transform(X_compute),
    }
    if X_sbert is not None:
        conditions["sbert"] = StandardScaler().fit_transform(X_sbert)
        conditions["combined_compute_sbert"] = StandardScaler().fit_transform(
            np.concatenate([X_compute, X_sbert], axis=1)
        )

    results: dict = {}
    for cond_name, X in conditions.items():
        fold_accs = []
        fold_sils = []
        train_accs = []

        for train_mask, test_mask in folds:
            try:
                model = _train_contrastive_mlp(X[train_mask], y[train_mask])
                emb_train = _embed(model, X[train_mask])
                emb_test = _embed(model, X[test_mask])

                knn = KNeighborsClassifier(n_neighbors=3)
                knn.fit(emb_train, y[train_mask])

                # Test accuracy
                y_pred_test = knn.predict(emb_test)
                fold_accs.append(float(np.mean(y_pred_test == y[test_mask])))

                # Train accuracy (to detect overfitting / train-test gap)
                y_pred_train = knn.predict(emb_train)
                train_accs.append(float(np.mean(y_pred_train == y[train_mask])))

                # Test silhouette
                if len(set(y[test_mask])) > 1:
                    from sklearn.metrics import silhouette_score
                    fold_sils.append(float(silhouette_score(emb_test, y[test_mask])))
            except Exception as e:
                fold_accs.append(0.0)
                train_accs.append(0.0)

        test_acc = float(np.mean(fold_accs)) if fold_accs else 0.0
        train_acc = float(np.mean(train_accs)) if train_accs else 0.0

        results[cond_name] = {
            "test_knn_accuracy": test_acc,
            "train_knn_accuracy": train_acc,
            "train_test_gap": train_acc - test_acc,
            "silhouette": float(np.mean(fold_sils)) if fold_sils else None,
            "fold_accs": fold_accs,
        }

    return results


def _per_mode_surface_vs_compute(
    X_surface: NDArray, X_compute: NDArray,
    y: NDArray, topics: NDArray, seed: int = 42,
) -> dict:
    """Per-mode recall comparison: TF-IDF/surface vs compute features.

    For each mode, compute the recall (true positive rate) under topic-heldout
    CV for both surface and compute classifiers. The gap per mode is the key
    diagnostic: modes where compute >> surface are sub-semantic.
    """
    unique_modes = sorted(set(y))
    unique_topics = sorted(set(topics))
    topic_to_id = {t: i for i, t in enumerate(unique_topics)}
    groups = np.array([topic_to_id[t] for t in topics])

    n_folds = min(5, len(unique_topics))
    gkf = GroupKFold(n_splits=n_folds)

    # Accumulate per-mode correct/total across folds for each feature set
    def _accumulate_per_mode(X: NDArray) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {
            m: {"correct": 0, "total": 0} for m in unique_modes
        }
        for train_idx, test_idx in gkf.split(X, y, groups):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])

            clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
            clf.fit(X_tr, y[train_idx])
            y_pred = clf.predict(X_te)

            for mode in unique_modes:
                mask = y[test_idx] == mode
                counts[mode]["total"] += int(mask.sum())
                counts[mode]["correct"] += int((y_pred[mask] == mode).sum())
        return counts

    surface_counts = _accumulate_per_mode(X_surface)
    compute_counts = _accumulate_per_mode(X_compute)

    per_mode: dict[str, dict[str, float]] = {}
    for mode in unique_modes:
        s_total = surface_counts[mode]["total"]
        s_correct = surface_counts[mode]["correct"]
        c_total = compute_counts[mode]["total"]
        c_correct = compute_counts[mode]["correct"]

        s_recall = s_correct / max(s_total, 1)
        c_recall = c_correct / max(c_total, 1)

        per_mode[mode] = {
            "surface_recall": float(s_recall),
            "compute_recall": float(c_recall),
            "gap_compute_minus_surface": float(c_recall - s_recall),
            "sub_semantic": c_recall > s_recall + 0.05,
            "surface_n": s_total,
            "compute_n": c_total,
        }

    # Summary
    sub_semantic_modes = [m for m, d in per_mode.items() if d["sub_semantic"]]
    surface_dominant_modes = [
        m for m, d in per_mode.items()
        if d["gap_compute_minus_surface"] < -0.05
    ]

    return {
        "per_mode": per_mode,
        "n_sub_semantic": len(sub_semantic_modes),
        "sub_semantic_modes": sub_semantic_modes,
        "n_surface_dominant": len(surface_dominant_modes),
        "surface_dominant_modes": surface_dominant_modes,
    }


def _parse_swap_mode(mode_str: str) -> tuple[str, str] | None:
    """Parse 'swap_A→B' into (system_prompt_mode, execution_mode).

    Returns None if not a swap mode.
    """
    # Match patterns like 'swap_socratic→linear' or 'swap_dialectical→contrastive'
    match = re.match(r"swap_(\w+)→(\w+)", mode_str)
    if match:
        return match.group(1), match.group(2)
    return None


def _run_prompt_swap_confound(
    data: AnalysisData,
    signature_dir: Path,
    addon_dirs: list[Path] | None = None,
) -> dict:
    """Prompt-swap confound test: train on core set, predict on prompt-swap samples.

    For each tier, train a classifier on the core (non-swap) samples, then
    predict on swap samples. The key question per tier:
    - Does the classifier follow the system prompt mode (confound)?
    - Or does it follow the execution/directive mode (genuine signal)?

    This is the definitive T3 deconfound at 8B. T3 captures t0 (prompt encoding),
    so if T3 follows the system prompt, it's a confound. If it follows execution,
    the t0 geometry genuinely reflects execution intent.
    """
    from ..geometric_trio.data_loader import TIER_KEYS, BASELINE_TIERS, ENGINEERED_TIERS

    # Load ALL samples (including swaps) — need to bypass core_only
    all_npz_files = sorted(signature_dir.glob("gen_*.npz"))
    if not all_npz_files:
        return {"error": "No npz files in signature dir"}

    # Find swap samples
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
        return {"error": "No prompt-swap samples found"}

    # Discover available tiers from the core data
    run4 = data.run4
    test_tiers: list[str] = []
    for tier in BASELINE_TIERS + ENGINEERED_TIERS:
        if tier in run4.tier_features:
            test_tiers.append(tier)
    # Also test key composites
    for group in ["T2+T2.5", "combined_v2"]:
        if group in run4.group_features:
            test_tiers.append(group)

    # Load swap sample features — primary dir first, then merge addons
    # Only require individual tiers (not composites) from primary npz
    individual_test_tiers = [t for t in test_tiers if t in TIER_KEYS]
    swap_tier_features: dict[str, list[NDArray]] = {}
    loaded_swap_indices: list[int] = list(range(len(swap_npz_paths)))

    # Pass 1: load whatever is available from primary npz files
    for i, npz_path in enumerate(swap_npz_paths):
        npz_data = np.load(npz_path, allow_pickle=True)
        for tier_name in individual_test_tiers:
            npz_key = TIER_KEYS.get(tier_name, "")
            if npz_key and npz_key in npz_data.files:
                swap_tier_features.setdefault(tier_name, [None] * len(swap_npz_paths))
                swap_tier_features[tier_name][i] = npz_data[npz_key]

    # Pass 2: merge addon features
    if addon_dirs:
        for addon_dir in (addon_dirs or []):
            addon_path = Path(addon_dir)
            if not addon_path.exists():
                continue
            for i in range(len(swap_npz_paths)):
                stem = swap_info[i]["file_stem"]
                addon_npz = addon_path / f"{stem}.npz"
                if addon_npz.exists():
                    addon_data = np.load(addon_npz, allow_pickle=True)
                    for tier_name in individual_test_tiers:
                        npz_key = TIER_KEYS.get(tier_name, "")
                        if npz_key and npz_key in addon_data.files:
                            swap_tier_features.setdefault(tier_name, [None] * len(swap_npz_paths))
                            if swap_tier_features[tier_name][i] is None:
                                swap_tier_features[tier_name][i] = addon_data[npz_key]

    # Filter to tiers where all samples were loaded
    complete_tiers = {
        t for t, arrays in swap_tier_features.items()
        if all(a is not None for a in arrays)
    }
    swap_tier_features = {
        t: arrays for t, arrays in swap_tier_features.items()
        if t in complete_tiers
    }

    # Build composite features for swap samples from individual tier arrays
    for group in ["T2+T2.5", "combined_v2"]:
        if group in test_tiers and group not in swap_tier_features:
            from ..geometric_trio.data_loader import TIER_GROUPS
            members = TIER_GROUPS.get(group, [])
            available_members = [m for m in members if m in complete_tiers]
            if available_members:
                swap_tier_features[group] = [
                    np.concatenate([swap_tier_features[m][j] for m in available_members])
                    for j in range(len(swap_npz_paths))
                ]

    if not complete_tiers:
        return {"error": "Could not load any complete tier features for swap samples"}

    sys_modes = np.array([s["system_prompt_mode"] for s in swap_info])
    exec_modes = np.array([s["execution_mode"] for s in swap_info])

    results: dict = {
        "n_swap_samples": len(swap_info),
        "swap_types": list(set(s["swap_name"] for s in swap_info)),
        "per_tier": {},
    }

    # For each tier: train on core samples, predict on swap samples
    for tier_name in test_tiers:
        if tier_name not in swap_tier_features or not swap_tier_features[tier_name]:
            continue
        if len(swap_tier_features[tier_name]) != len(swap_info):
            continue

        try:
            X_train = data.get_tier(tier_name)
        except KeyError:
            continue

        X_swap = np.stack(swap_tier_features[tier_name], axis=0)
        y_train = data.modes

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_swap_scaled = scaler.transform(X_swap)

        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
        clf.fit(X_train_scaled, y_train)

        predictions = clf.predict(X_swap_scaled)
        probas = clf.predict_proba(X_swap_scaled)
        class_labels = clf.classes_

        # Score: does classifier follow system_prompt_mode or execution_mode?
        follows_system = int(np.sum(predictions == sys_modes))
        follows_execution = int(np.sum(predictions == exec_modes))
        follows_neither = int(len(predictions) - follows_system - follows_execution)

        # Per-swap-type breakdown
        per_swap_type: dict[str, dict] = {}
        for swap_name in set(s["swap_name"] for s in swap_info):
            mask = np.array([s["swap_name"] == swap_name for s in swap_info])
            n_this = int(mask.sum())
            preds_this = predictions[mask]
            sys_this = sys_modes[mask]
            exec_this = exec_modes[mask]

            per_swap_type[swap_name] = {
                "n": n_this,
                "follows_system_prompt": int(np.sum(preds_this == sys_this)),
                "follows_execution": int(np.sum(preds_this == exec_this)),
                "follows_neither": int(n_this - np.sum(preds_this == sys_this) - np.sum(preds_this == exec_this)),
                "predictions": predictions[mask].tolist(),
                "system_modes": sys_this.tolist(),
                "execution_modes": exec_this.tolist(),
            }

        tier_result = {
            "n_features": int(X_train.shape[1]),
            "follows_system_prompt": follows_system,
            "follows_execution": follows_execution,
            "follows_neither": follows_neither,
            "pct_system": float(follows_system / len(predictions)),
            "pct_execution": float(follows_execution / len(predictions)),
            "signal_type": (
                "execution_based" if follows_execution > follows_system * 1.5
                else "system_prompt_based" if follows_system > follows_execution * 1.5
                else "ambiguous"
            ),
            "per_swap_type": per_swap_type,
        }

        results["per_tier"][tier_name] = tier_result

    return results


def _get_semantic_test_tiers(data: AnalysisData) -> list[str]:
    """Discover which tiers/groups to test for semantic independence.

    Returns all individual tiers present + key composites.
    Every tier gets its own orthogonality assessment — critical for validating
    that each feature family contributes execution-based, not content-based, signal.
    """
    from ..geometric_trio.data_loader import BASELINE_TIERS, ENGINEERED_TIERS

    run4 = data.run4
    tiers: list[str] = []

    # All individual baseline tiers (T1, T2, T2.5, T3)
    for tier in BASELINE_TIERS:
        if tier in run4.tier_features:
            tiers.append(tier)

    # All individual engineered families
    for tier in ENGINEERED_TIERS:
        if tier in run4.tier_features:
            tiers.append(tier)

    # Key composites
    for group in ["T2+T2.5", "engineered", "combined_v2", "T2+T2.5+engineered"]:
        if group in run4.group_features:
            tiers.append(group)

    return tiers


def run_semantic(
    data: AnalysisData,
    signature_dir: Path | str | None = None,
    addon_dirs: list[Path | str] | None = None,
) -> dict:
    """Run semantic independence analyses.

    Tests each available compute tier against surface/semantic baselines:
    - TF-IDF classification (topic-heldout)
    - SBERT classification (topic-heldout)
    - Mantel test (distance correlation between compute and semantic)
    - Text-to-compute R² (can text predict compute features?)
    - Per-mode surface vs compute decomposition
    - Prompt-swap confound test (if swap samples exist)
    - Contrastive projection comparison
    - Shuffle controls
    - Retrieval analysis

    Parameters
    ----------
    data : AnalysisData
        Core analysis data (excluding prompt-swap samples).
    signature_dir : Path, optional
        Signature directory for loading prompt-swap samples.
    addon_dirs : list[Path], optional
        Addon directories for loading prompt-swap tier features.
    """
    if data.generated_texts is None or all(t == "" for t in data.generated_texts):
        return {"error": "No generated text available"}

    results: dict = {}
    y = data.modes
    topics = data.topics

    # ── Semantic baselines (computed once) ──
    print("    TF-IDF surface baseline...")
    X_tfidf = _embed_texts_tfidf(data.generated_texts)
    results["tfidf_classification"] = {
        "rf": _classify_condition(X_tfidf, y, topics, clf_name="rf"),
        "knn": _classify_condition(X_tfidf, y, topics, clf_name="knn"),
        "dims": X_tfidf.shape[1],
    }

    print("    Sentence-BERT embeddings...")
    X_sbert = _embed_texts_sbert(data.generated_texts)
    if X_sbert is not None:
        results["sbert_classification"] = {
            "rf": _classify_condition(X_sbert, y, topics, clf_name="rf"),
            "knn": _classify_condition(X_sbert, y, topics, clf_name="knn"),
            "dims": X_sbert.shape[1],
        }
    else:
        results["sbert_classification"] = {"error": "sentence-transformers not available"}

    # ── Per-tier semantic orthogonality battery ──
    # Test each individual new family + key composites
    test_tiers = _get_semantic_test_tiers(data)
    print(f"    Testing semantic orthogonality for {len(test_tiers)} tiers: {test_tiers}")

    from scipy.spatial.distance import pdist, squareform

    X_tfidf_std = StandardScaler().fit_transform(X_tfidf)
    X_sbert_std = StandardScaler().fit_transform(X_sbert) if X_sbert is not None else None
    semantic_emb = X_sbert if X_sbert is not None else X_tfidf

    per_tier_semantic: dict[str, dict] = {}

    for tier_name in test_tiers:
        print(f"      Tier: {tier_name}...")
        try:
            X_compute = data.get_tier(tier_name)
        except KeyError:
            per_tier_semantic[tier_name] = {"error": f"tier {tier_name} not found"}
            continue

        tier_result: dict = {
            "n_features": int(X_compute.shape[1]),
        }

        # Classification accuracy (topic-heldout)
        tier_result["classification"] = {
            "rf": _classify_condition(X_compute, y, topics, clf_name="rf"),
            "knn": _classify_condition(X_compute, y, topics, clf_name="knn"),
        }

        # Mantel test vs TF-IDF and SBERT
        X_comp_std = StandardScaler().fit_transform(X_compute)
        for dist_metric in ["cosine"]:
            D_compute = squareform(pdist(X_comp_std, metric=dist_metric))
            D_tfidf = squareform(pdist(X_tfidf_std, metric=dist_metric))
            tier_result[f"mantel_tfidf_{dist_metric}"] = _mantel_test(D_compute, D_tfidf)

            if X_sbert_std is not None:
                D_sbert = squareform(pdist(X_sbert_std, metric=dist_metric))
                tier_result[f"mantel_sbert_{dist_metric}"] = _mantel_test(D_compute, D_sbert)

        # Text-to-compute R² (can semantic features predict this tier's features?)
        tier_result["text_to_compute_r2"] = _text_to_compute_r2(
            semantic_emb, X_compute, topics,
        )

        # Per-mode surface vs compute decomposition
        tier_result["per_mode_surface_vs_compute"] = _per_mode_surface_vs_compute(
            X_tfidf, X_compute, y, topics,
        )

        # Shuffle controls
        tier_result["shuffle_controls"] = _run_shuffle_controls(X_compute, y, topics)

        per_tier_semantic[tier_name] = tier_result

    results["per_tier_semantic"] = per_tier_semantic

    # ── Legacy top-level keys (T2+T2.5, for backward compatibility) ──
    t2t25_results = per_tier_semantic.get("T2+T2.5", {})
    X_compute_main = data.get_tier("T2+T2.5")
    results["compute_classification"] = t2t25_results.get("classification", {})

    # Combined (compute + semantic) — for T2+T2.5
    if X_sbert is not None:
        X_combined = np.concatenate([X_compute_main, X_sbert], axis=1)
        results["combined_classification"] = {
            "rf": _classify_condition(X_combined, y, topics, clf_name="rf"),
            "knn": _classify_condition(X_combined, y, topics, clf_name="knn"),
            "dims": X_combined.shape[1],
        }

        # Semantic + noise control (dimensionality matching)
        rng = np.random.default_rng(42)
        noise = rng.standard_normal((X_sbert.shape[0], X_compute_main.shape[1])).astype(np.float32)
        X_semantic_noise = np.concatenate([X_sbert, noise], axis=1)
        results["semantic_noise_classification"] = {
            "rf": _classify_condition(X_semantic_noise, y, topics, clf_name="rf"),
            "knn": _classify_condition(X_semantic_noise, y, topics, clf_name="knn"),
            "dims": X_semantic_noise.shape[1],
        }

    # Mantel — backward compat top-level keys from T2+T2.5
    results["mantel_tfidf"] = t2t25_results.get("mantel_tfidf_cosine", {})
    results["mantel_sbert"] = t2t25_results.get("mantel_sbert_cosine", {})
    results["mantel_tfidf_cosine"] = t2t25_results.get("mantel_tfidf_cosine", {})
    results["mantel_sbert_cosine"] = t2t25_results.get("mantel_sbert_cosine", {})

    # Also do euclidean for T2+T2.5 (backward compat)
    X_comp_std = StandardScaler().fit_transform(X_compute_main)
    D_compute_euc = squareform(pdist(X_comp_std, metric="euclidean"))
    D_tfidf_euc = squareform(pdist(X_tfidf_std, metric="euclidean"))
    results["mantel_tfidf_euclidean"] = _mantel_test(D_compute_euc, D_tfidf_euc)
    if X_sbert_std is not None:
        D_sbert_euc = squareform(pdist(X_sbert_std, metric="euclidean"))
        results["mantel_sbert_euclidean"] = _mantel_test(D_compute_euc, D_sbert_euc)

    # Text-to-compute R² — backward compat
    results["text_to_compute_r2"] = t2t25_results.get("text_to_compute_r2", {})

    # Per-mode surface vs compute — backward compat
    results["per_mode_surface_vs_compute"] = t2t25_results.get(
        "per_mode_surface_vs_compute", {},
    )

    # Shuffle controls — backward compat
    results["shuffle_controls"] = t2t25_results.get("shuffle_controls", {})

    # ── Cross-tier analyses (run once) ──
    # Topic-heldout contrastive projection comparison
    print("    Contrastive projection comparison (topic-heldout)...")
    results["contrastive_projection_comparison"] = _contrastive_topic_heldout(
        X_tfidf, X_compute_main, X_sbert, y, topics,
    )

    # Retrieval analysis
    print("    Retrieval analysis...")
    results["retrieval"] = _run_retrieval_analysis(
        X_compute_main, X_tfidf, X_sbert, y, topics,
    )

    # ── Prompt-swap confound test ──
    # Train on core set, predict on prompt-swap samples.
    # The definitive T3 deconfound: does T3 follow system prompt or execution?
    if signature_dir is not None:
        print("    Prompt-swap confound test...")
        try:
            addon_paths = [Path(d) for d in addon_dirs] if addon_dirs else None
            results["prompt_swap_confound"] = _run_prompt_swap_confound(
                data, Path(signature_dir), addon_dirs=addon_paths,
            )
            ps = results["prompt_swap_confound"]
            if "per_tier" in ps:
                n_swap = ps.get("n_swap_samples", 0)
                print(f"      {n_swap} swap samples, {len(ps['per_tier'])} tiers tested")
                for tier_name, tier_data in ps["per_tier"].items():
                    sig_type = tier_data.get("signal_type", "?")
                    pct_exec = tier_data.get("pct_execution", 0)
                    pct_sys = tier_data.get("pct_system", 0)
                    print(f"        {tier_name}: {sig_type} "
                          f"(exec={pct_exec:.0%}, sys={pct_sys:.0%})")
            elif "error" in ps:
                print(f"      Skipped: {ps['error']}")
        except Exception as e:
            logger.warning(f"Prompt-swap confound test failed: {e}")
            results["prompt_swap_confound"] = {"error": str(e)}
    else:
        results["prompt_swap_confound"] = {"error": "signature_dir not provided"}

    return results


def _run_shuffle_controls(
    X_compute: NDArray, y: NDArray, topics: NDArray, seed: int = 42,
) -> dict:
    """Shuffle controls for compute features — rules out dimensionality artifacts.

    Within-topic shuffle: shuffle compute rows within each topic group.
    If mode signal is real within topics, this should drop to ~chance.
    Global shuffle: shuffle all compute rows. Absolute null baseline.
    """
    rng = np.random.default_rng(seed)
    results: dict = {}

    # Within-topic shuffle
    X_within_shuffled = X_compute.copy()
    for topic in sorted(set(topics)):
        topic_mask = topics == topic
        topic_indices = np.where(topic_mask)[0]
        perm = rng.permutation(len(topic_indices))
        X_within_shuffled[topic_indices] = X_compute[topic_indices[perm]]

    results["within_topic_shuffle"] = _classify_condition(
        X_within_shuffled, y, topics, clf_name="rf", seed=seed,
    )

    # Global shuffle
    X_global_shuffled = X_compute[rng.permutation(len(X_compute))]
    results["global_shuffle"] = _classify_condition(
        X_global_shuffled, y, topics, clf_name="rf", seed=seed,
    )

    return results


def _run_retrieval_analysis(
    X_compute: NDArray, X_tfidf: NDArray, X_sbert: NDArray | None,
    y: NDArray, topics: NDArray, k: int = 5,
) -> dict:
    """kNN mode-match retrieval accuracy in raw feature space (no projection).

    For each sample, find its k nearest neighbors and check what fraction
    share the same mode. Compare across feature types.
    """
    from sklearn.neighbors import NearestNeighbors

    results: dict = {}

    feature_sets: dict[str, NDArray] = {
        "compute_t2t25": StandardScaler().fit_transform(X_compute),
        "tfidf": StandardScaler().fit_transform(X_tfidf),
    }
    if X_sbert is not None:
        feature_sets["sbert"] = StandardScaler().fit_transform(X_sbert)
        feature_sets["combined_compute_sbert"] = StandardScaler().fit_transform(
            np.concatenate([X_compute, X_sbert], axis=1),
        )

    neighbor_indices: dict[str, NDArray] = {}

    for name, X in feature_sets.items():
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nn.fit(X)
        _, indices = nn.kneighbors(X)
        # Exclude self (first neighbor)
        knn_indices = indices[:, 1:]
        neighbor_indices[name] = knn_indices

        # Mode-match accuracy: fraction of neighbors sharing the query's mode
        mode_matches = np.array([
            np.mean(y[knn_indices[i]] == y[i]) for i in range(len(y))
        ])

        # Per-mode breakdown
        per_mode: dict[str, float] = {}
        for mode in sorted(set(y)):
            mask = y == mode
            per_mode[mode] = float(np.mean(mode_matches[mask]))

        results[name] = {
            "mean_mode_match": float(np.mean(mode_matches)),
            "std_mode_match": float(np.std(mode_matches)),
            "per_mode": per_mode,
            "k": k,
        }

    # Jaccard overlap: do compute and semantic neighbors agree?
    if "compute_t2t25" in neighbor_indices and "tfidf" in neighbor_indices:
        compute_nn = neighbor_indices["compute_t2t25"]
        tfidf_nn = neighbor_indices["tfidf"]
        jaccards = []
        for i in range(len(y)):
            c_set = set(compute_nn[i])
            t_set = set(tfidf_nn[i])
            intersection = len(c_set & t_set)
            union = len(c_set | t_set)
            jaccards.append(intersection / max(union, 1))
        results["jaccard_compute_tfidf"] = {
            "mean": float(np.mean(jaccards)),
            "std": float(np.std(jaccards)),
        }

    if "compute_t2t25" in neighbor_indices and "sbert" in neighbor_indices:
        compute_nn = neighbor_indices["compute_t2t25"]
        sbert_nn = neighbor_indices["sbert"]
        jaccards = []
        for i in range(len(y)):
            c_set = set(compute_nn[i])
            s_set = set(sbert_nn[i])
            intersection = len(c_set & s_set)
            union = len(c_set | s_set)
            jaccards.append(intersection / max(union, 1))
        results["jaccard_compute_sbert"] = {
            "mean": float(np.mean(jaccards)),
            "std": float(np.std(jaccards)),
        }

    return results
