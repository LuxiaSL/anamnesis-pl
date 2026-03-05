"""Section 7: Silhouette scores and clustering visualization."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

from .data_loading import AnalysisData
from .utils import ALL_TIERS, get_available_tiers


def run_clustering(data: AnalysisData) -> dict:
    """Silhouette analysis, K-Means ARI, and dimensionality reduction."""
    results: dict = {"silhouette_by_tier": {}, "per_mode_silhouette": {}}

    available_tiers, _ = get_available_tiers(data)
    # Silhouette per tier (mode labels) — compute both cosine (Phase 0 standard) and euclidean
    for tier in available_tiers:
        X = data.get_tier(tier)
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        tier_sil: dict = {}
        for metric in ["cosine", "euclidean"]:
            try:
                score = float(silhouette_score(X_std, data.modes, metric=metric))
            except Exception as e:
                score = f"ERROR: {e}"
            tier_sil[f"mode_silhouette_{metric}"] = score

        # Backward compat: top-level key points to cosine (Phase 0 standard)
        tier_sil["mode_silhouette"] = tier_sil["mode_silhouette_cosine"]

        # Also topic silhouette (should be low — mode signal should not correlate with topic)
        try:
            topic_score = float(silhouette_score(X_std, data.topics, metric="cosine"))
        except Exception:
            topic_score = None
        tier_sil["topic_silhouette"] = topic_score

        results["silhouette_by_tier"][tier] = tier_sil

    # Per-mode silhouette decomposition (T2+T2.5)
    X_key = data.get_tier("T2+T2.5")
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_key)

    # Per-mode silhouette with both metrics
    for metric in ["cosine", "euclidean"]:
        key = f"per_mode_silhouette_{metric}"
        results[key] = {}
        try:
            sample_sils = silhouette_samples(X_std, data.modes, metric=metric)
            for mode in data.unique_modes:
                mask = data.mode_mask(mode)
                mode_sils = sample_sils[mask]
                results[key][mode] = {
                    "mean": float(np.mean(mode_sils)),
                    "std": float(np.std(mode_sils)),
                    "min": float(np.min(mode_sils)),
                    "max": float(np.max(mode_sils)),
                    "n_negative": int(np.sum(mode_sils < 0)),
                }
        except Exception as e:
            results[key]["error"] = str(e)

    # Backward compat: top-level key points to cosine
    results["per_mode_silhouette"] = results["per_mode_silhouette_cosine"]

    # K-Means ARI
    results["kmeans_ari"] = {}
    for tier in ["T2+T2.5", "combined"]:
        X = data.get_tier(tier)
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        # Map mode labels to integers for ARI
        mode_to_int = {m: i for i, m in enumerate(data.unique_modes)}
        y_true = np.array([mode_to_int[m] for m in data.modes])

        try:
            km = KMeans(n_clusters=len(data.unique_modes), random_state=42, n_init=10)
            y_pred = km.fit_predict(X_std)
            ari = float(adjusted_rand_score(y_true, y_pred))
        except Exception as e:
            ari = f"ERROR: {e}"

        results["kmeans_ari"][tier] = ari

    # UMAP / t-SNE embeddings (save coordinates for plotting)
    results["embeddings"] = {}

    # t-SNE (always available via sklearn)
    from sklearn.manifold import TSNE

    X_key_std = StandardScaler().fit_transform(data.get_tier("T2+T2.5"))
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, data.n_samples - 1))
        coords = tsne.fit_transform(X_key_std)
        results["embeddings"]["tsne_t2t25"] = {
            "coords": coords.tolist(),
            "modes": data.modes.tolist(),
            "topics": data.topics.tolist(),
        }
    except Exception as e:
        results["embeddings"]["tsne_t2t25"] = {"error": str(e)}

    # UMAP (optional)
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, data.n_samples - 1))
        coords = reducer.fit_transform(X_key_std)
        results["embeddings"]["umap_t2t25"] = {
            "coords": coords.tolist(),
            "modes": data.modes.tolist(),
            "topics": data.topics.tolist(),
        }
    except ImportError:
        results["embeddings"]["umap_t2t25"] = {"error": "umap not installed"}
    except Exception as e:
        results["embeddings"]["umap_t2t25"] = {"error": str(e)}

    return results
