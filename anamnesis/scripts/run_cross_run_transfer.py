#!/usr/bin/env python3
"""Cross-run functional transfer at 8B — paper Run 2 ↔ paper Run 3.

Replicates the Phase-0.5 cross-run transfer experiment
(`phase_0/scripts/run_cross_run_transfer.py`) at 8B scale, using data from
two 8B runs extracted with the same pipeline + same calibration:

  train run (paper Run 3, format-controlled)
      = run_8b_baseline
      modes: linear, analogical, socratic, contrastive, dialectical
  test run  (paper Run 2, format-free process modes)
      = run_8b_r2_equivalent
      modes: associative, compressed, deliberative, pedagogical, structured

Pre-registered functional pairs (mapped from 3B findings):
  - deliberative ↔ dialectical  (propose-challenge-revise)
  - pedagogical  ↔ socratic      (interactive explanation)
  - associative  ↔ analogical    (connection-driven, non-sequential)
  - structured   ↔ linear        (sequential exposition)
  - compressed   → ? wildcard (no Run-3 equivalent)
  - contrastive  → ? wildcard (no Run-2 equivalent, in reverse)

3B baselines for comparison:
  - Forward  (train R3, embed R2): pedagogical → socratic = 76%
  - Reverse  (train R2, embed R3): dialectical → deliberative = 85%
  - LDA silhouette of R4 in R3's LDA space: -0.156 (destructive interference)
  - LDA silhouette of R3 in its own LDA space: 0.438

Outputs `{run_name}/results.json` matching the Phase-0.5 structure for
direct numerical comparison.

Usage:
    python -m anamnesis.scripts.run_cross_run_transfer
    python -m anamnesis.scripts.run_cross_run_transfer --n-seeds 10
    python -m anamnesis.scripts.run_cross_run_transfer \\
        --train-run run_8b_baseline --test-run run_8b_r2_equivalent
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]

# ── Repo-root path inference ──────────────────────────────────────────────────
# This script lives at `pipeline/anamnesis/scripts/run_cross_run_transfer.py`.
# Output data lives at `<repo_root>/outputs/`.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../pipeline/anamnesis
PIPELINE_ROOT = PACKAGE_ROOT.parent  # .../pipeline
REPO_ROOT = PIPELINE_ROOT.parent  # .../anamnesis_exps
OUTPUTS_BASE = REPO_ROOT / "outputs"


# ── Pre-registered mappings ───────────────────────────────────────────────────

# Paper Run 2 → Paper Run 3 (forward = train R3, embed R2)
FORWARD_MAPPING: dict[str, str] = {
    "deliberative": "dialectical",
    "pedagogical": "socratic",
    "associative": "analogical",
    "structured": "linear",
}
FORWARD_WILDCARD = "compressed"  # no predicted Run-3 equivalent

# Paper Run 3 → Paper Run 2 (reverse = train R2, embed R3)
REVERSE_MAPPING: dict[str, str] = {
    "dialectical": "deliberative",
    "socratic": "pedagogical",
    "analogical": "associative",
    "linear": "structured",
}
REVERSE_WILDCARD = "contrastive"  # no predicted Run-2 equivalent


# ── MLP components (inlined from phase_0/scripts/run_contrastive_projection.py) ──

class ProjectionNet(nn.Module):
    """Contrastive projection network: input → hidden → bottleneck → L2-norm."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, drop: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return nn.functional.normalize(out, p=2, dim=1)


def mine_triplets(
    labels: NDArray[np.int64],
    rng: np.random.RandomState,
    n_triplets: int = 300,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Fully random triplet mining: random anchor, random positive, random negative."""
    anchors, positives, negatives = [], [], []
    for _ in range(n_triplets):
        anchor_idx = rng.randint(len(labels))
        anchor_label = labels[anchor_idx]
        same = np.where(labels == anchor_label)[0]
        same = same[same != anchor_idx]
        if len(same) == 0:
            continue
        pos_idx = rng.choice(same)
        diff = np.where(labels != anchor_label)[0]
        neg_idx = rng.choice(diff)
        anchors.append(anchor_idx)
        positives.append(pos_idx)
        negatives.append(neg_idx)
    return np.array(anchors), np.array(positives), np.array(negatives)


def train_full_data_mlp(
    X: F32,
    y: NDArray[np.int64],
    bottleneck_dim: int = 32,
    n_epochs: int = 200,
    lr: float = 1e-3,
    margin: float = 1.0,
    dropout: float = 0.5,
    weight_decay: float = 1e-3,
    torch_seed: int = 42,
) -> tuple[ProjectionNet, float]:
    """Train contrastive MLP on full dataset; return trained model."""
    torch.manual_seed(torch_seed)
    rng = np.random.RandomState(torch_seed)

    X_t = torch.tensor(X, dtype=torch.float32)
    model = ProjectionNet(X.shape[1], hidden_dim=256, output_dim=bottleneck_dim, drop=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin)

    final_loss = 0.0
    model.train()
    for _ in range(n_epochs):
        a_idx, p_idx, n_idx = mine_triplets(y, rng, n_triplets=300)
        if len(a_idx) < 10:
            continue
        emb = model(X_t)
        loss = triplet_loss_fn(emb[a_idx], emb[p_idx], emb[n_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    return model, final_loss


def embed_with_model(model: ProjectionNet, X: F32) -> F32:
    """Embed data through a frozen model."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        emb = model(X_t).numpy()
    return emb.astype(np.float32)


# ── Data loading ──────────────────────────────────────────────────────────────

# Parses layer tags like "_L13_" or trailing "_L31" from feature names.
_LAYER_RE = re.compile(r"_L(\d+)(?:_|$)")


def load_feature_names(
    run_name: str,
    outputs_base: Path = OUTPUTS_BASE,
) -> NDArray[np.str_]:
    """Read `feature_names` from the first .npz in a run's signatures dir.

    Used for building feature-subset indices before calling load_run_features.
    """
    sig_dir = outputs_base / "runs" / run_name / "signatures"
    first = next(sig_dir.glob("gen_*.npz"), None)
    if first is None:
        raise FileNotFoundError(f"No .npz files in {sig_dir}")
    sig = np.load(first, allow_pickle=True)
    if "feature_names" not in sig.files:
        raise KeyError(f"`feature_names` not in {first.name}")
    return np.array(sig["feature_names"], copy=True)


def build_layer_indices(
    feature_names: NDArray[np.str_],
    layers: set[int] | None = None,
    include_unlayered: bool = False,
) -> list[int]:
    """Return indices of features whose layer-tag is in ``layers``.

    Parameters
    ----------
    feature_names : array of feature names (one per feature dim).
    layers : set of layer indices to keep. If None, all layer-tagged features
        are kept (filters only by whether a tag exists).
    include_unlayered : if True, also include features with no ``_Ln`` tag
        (typically T1 global logit statistics).
    """
    indices: list[int] = []
    for i, name in enumerate(feature_names):
        m = _LAYER_RE.search(str(name))
        if m is not None:
            lyr = int(m.group(1))
            if layers is None or lyr in layers:
                indices.append(i)
        elif include_unlayered:
            indices.append(i)
    return indices


def load_run_features(
    run_name: str,
    outputs_base: Path = OUTPUTS_BASE,
    feature_key: str | Sequence[str] = "features",
    feature_indices: Sequence[int] | None = None,
) -> tuple[F32, NDArray[np.int64], list[str], NDArray[np.str_]]:
    """Load per-generation features and mode labels from a pipeline run directory.

    Reads `{outputs_base}/runs/{run_name}/signatures/gen_*.json` (for mode labels)
    and matching `.npz` (for feature vectors).

    Parameters
    ----------
    run_name : pipeline run directory name.
    outputs_base : root of ``outputs/`` tree.
    feature_key : str or sequence of str. If a single string, uses that .npz key
        directly (back-compat default ``"features"``). If a sequence, concatenates
        the arrays from each key in the given order.
    feature_indices : optional list/array of column indices to keep. Applied
        AFTER any tier concatenation, so indices are into the concatenated
        feature vector.

    Returns
    -------
    X : (n, d) float32
    y : (n,) int64 (mode labels encoded)
    mode_names : list[str] (in label-order)
    labels_str : (n,) str (raw mode names, for debugging)
    """
    sig_dir = outputs_base / "runs" / run_name / "signatures"
    if not sig_dir.is_dir():
        raise FileNotFoundError(f"Signatures dir not found: {sig_dir}")

    feature_keys: list[str] = (
        [feature_key] if isinstance(feature_key, str) else list(feature_key)
    )

    feature_list: list[F32] = []
    mode_labels: list[str] = []
    for json_path in sorted(sig_dir.glob("gen_*.json")):
        npz_path = json_path.with_suffix(".npz")
        if not npz_path.exists():
            logger.warning(f"  Missing .npz for {json_path.name} — skipping")
            continue
        with open(json_path) as f:
            meta = json.load(f)
        sig = np.load(npz_path, allow_pickle=True)
        parts: list[F32] = []
        for key in feature_keys:
            if key not in sig.files:
                raise KeyError(
                    f"Feature key {key!r} not found in {npz_path.name}. "
                    f"Available: {list(sig.files)}"
                )
            parts.append(sig[key].astype(np.float32))
        combined = (
            np.concatenate(parts, axis=0).astype(np.float32) if len(parts) > 1 else parts[0]
        )
        feature_list.append(combined)
        mode_labels.append(meta["mode"])

    if not feature_list:
        raise RuntimeError(f"No feature vectors loaded from {sig_dir}")

    X = np.stack(feature_list).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if feature_indices is not None:
        idx = np.asarray(feature_indices, dtype=np.int64)
        if idx.size == 0:
            raise ValueError("feature_indices is empty")
        X = X[:, idx]

    labels_str = np.array(mode_labels)

    le = LabelEncoder()
    y = le.fit_transform(labels_str).astype(np.int64)
    mode_names = list(le.classes_)
    logger.info(
        f"[{run_name}] loaded n={X.shape[0]} samples, d={X.shape[1]} features, "
        f"modes={mode_names} (key={feature_key!r}, "
        f"sliced={feature_indices is not None})"
    )
    return X, y, mode_names, labels_str


# ── Centroid + mapping logic ──────────────────────────────────────────────────

def compute_centroids(
    emb: F32, y: NDArray[np.int64], mode_names: list[str]
) -> dict[str, F32]:
    return {name: emb[y == i].mean(axis=0) for i, name in enumerate(mode_names)}


def compute_similarity_matrix(
    row_centroids: dict[str, F32],
    col_centroids: dict[str, F32],
    row_order: list[str],
    col_order: list[str],
) -> F32:
    """Cosine similarity matrix: rows × cols."""
    sim = np.zeros((len(row_order), len(col_order)), dtype=np.float32)
    for i, r in enumerate(row_order):
        for j, c in enumerate(col_order):
            a, b = row_centroids[r], col_centroids[c]
            norm = np.linalg.norm(a) * np.linalg.norm(b) + 1e-10
            sim[i, j] = float(np.dot(a, b) / norm)
    return sim


def compute_nearest_centroid_mapping(
    emb_test: F32,
    y_test: NDArray[np.int64],
    modes_test: list[str],
    centroids_train: dict[str, F32],
    modes_train: list[str],
) -> NDArray[np.int64]:
    """For each test sample, find nearest train-mode centroid. Returns assignment matrix."""
    stack = np.stack([centroids_train[m] for m in modes_train])
    stack = stack / (np.linalg.norm(stack, axis=1, keepdims=True) + 1e-10)
    # emb_test is already L2-normalized (ProjectionNet does it).
    sims = emb_test @ stack.T  # (n_test, n_train_modes)
    nearest = sims.argmax(axis=1)
    assignment = np.zeros((len(modes_test), len(modes_train)), dtype=np.int64)
    for i, _name in enumerate(modes_test):
        mask = y_test == i
        for train_idx in nearest[mask]:
            assignment[i, train_idx] += 1
    return assignment


def evaluate_mapping(
    assignment: NDArray[np.int64],
    modes_test: list[str],
    modes_train: list[str],
    predicted: dict[str, str],
) -> dict[str, Any]:
    """Score mapping against pre-registered predictions."""
    total_mapped = 0
    correct_mapped = 0
    per_mode: dict[str, dict[str, Any]] = {}
    for i, test_name in enumerate(modes_test):
        dist = {modes_train[j]: int(assignment[i, j]) for j in range(len(modes_train))}
        n_total = int(assignment[i].sum())
        actual = modes_train[int(assignment[i].argmax())] if n_total > 0 else None
        if test_name not in predicted:
            per_mode[test_name] = {
                "predicted": None,
                "actual_nearest": actual,
                "samples": n_total,
                "assignment_distribution": dist,
            }
            continue
        pred = predicted[test_name]
        pred_idx = modes_train.index(pred)
        n_correct = int(assignment[i, pred_idx])
        total_mapped += n_total
        correct_mapped += n_correct
        per_mode[test_name] = {
            "predicted": pred,
            "actual_nearest": actual,
            "correct": n_correct,
            "total": n_total,
            "accuracy": float(n_correct / n_total) if n_total > 0 else 0.0,
            "match": actual == pred,
            "assignment_distribution": dist,
        }
    overall = float(correct_mapped / total_mapped) if total_mapped > 0 else 0.0
    modes_correct = sum(1 for v in per_mode.values() if v.get("match", False))
    return {
        "overall_accuracy": overall,
        "correct_mapped": correct_mapped,
        "total_mapped": total_mapped,
        "modes_correct": modes_correct,
        "modes_total": len(predicted),
        "chance_accuracy": 1.0 / len(modes_train),
        "per_mode": per_mode,
    }


# ── Transfer analysis (multi-seed) ────────────────────────────────────────────

def run_transfer_multi_seed(
    X_train: F32,
    y_train: NDArray[np.int64],
    modes_train: list[str],
    X_test: F32,
    y_test: NDArray[np.int64],
    modes_test: list[str],
    predicted: dict[str, str],
    wildcard_mode: str | None,
    direction_label: str,
    n_seeds: int = 10,
    bottleneck_dim: int = 32,
    n_epochs: int = 200,
) -> dict[str, Any]:
    """Train MLP on train data, embed test data, score mapping. Aggregate across seeds."""
    logger.info(f"=== Transfer [{direction_label}] ({n_seeds} seeds) ===")

    all_acc: list[float] = []
    all_modes_correct: list[int] = []
    per_mode_accs: dict[str, list[float]] = {name: [] for name in predicted}
    all_sim_matrices: list[F32] = []
    all_assignments: list[NDArray[np.int64]] = []
    wildcard_assignments: dict[str, int] | None = None
    if wildcard_mode and wildcard_mode in modes_test:
        wildcard_assignments = {name: 0 for name in modes_train}

    for seed_i in range(n_seeds):
        torch_seed = seed_i * 7 + 42
        model, final_loss = train_full_data_mlp(
            X_train, y_train, bottleneck_dim=bottleneck_dim,
            n_epochs=n_epochs, torch_seed=torch_seed,
        )
        emb_train = embed_with_model(model, X_train)
        emb_test = embed_with_model(model, X_test)
        train_centroids = compute_centroids(emb_train, y_train, modes_train)
        test_centroids = compute_centroids(emb_test, y_test, modes_test)
        sim = compute_similarity_matrix(test_centroids, train_centroids, modes_test, modes_train)
        all_sim_matrices.append(sim)
        assignment = compute_nearest_centroid_mapping(
            emb_test, y_test, modes_test, train_centroids, modes_train,
        )
        all_assignments.append(assignment)
        scored = evaluate_mapping(assignment, modes_test, modes_train, predicted)
        all_acc.append(scored["overall_accuracy"])
        all_modes_correct.append(scored["modes_correct"])
        for name, info in scored["per_mode"].items():
            if name in per_mode_accs and "accuracy" in info:
                per_mode_accs[name].append(info["accuracy"])
        if wildcard_assignments is not None and wildcard_mode in modes_test:
            wc_idx = modes_test.index(wildcard_mode)
            wc_nearest = modes_train[int(assignment[wc_idx].argmax())]
            wildcard_assignments[wc_nearest] += 1
        logger.info(
            f"  seed {seed_i}: acc={scored['overall_accuracy']:.2%}, "
            f"modes_correct={scored['modes_correct']}/{scored['modes_total']}, "
            f"loss={final_loss:.4f}"
        )

    median_sim = np.median(np.stack(all_sim_matrices), axis=0)
    mean_assignment = np.mean(np.stack(all_assignments).astype(np.float64), axis=0)
    results: dict[str, Any] = {
        "direction": direction_label,
        "n_seeds": n_seeds,
        "mapping_accuracy_mean": float(np.mean(all_acc)),
        "mapping_accuracy_median": float(np.median(all_acc)),
        "mapping_accuracy_std": float(np.std(all_acc)),
        "mapping_accuracy_per_seed": [float(a) for a in all_acc],
        "modes_correct_mean": float(np.mean(all_modes_correct)),
        "modes_correct_per_seed": list(map(int, all_modes_correct)),
        "chance_accuracy": 1.0 / len(modes_train),
        "per_mode_accuracy": {
            name: {
                "mean": float(np.mean(accs)) if accs else float("nan"),
                "std": float(np.std(accs)) if accs else float("nan"),
                "per_seed": [float(a) for a in accs],
            }
            for name, accs in per_mode_accs.items()
        },
        "median_similarity_matrix": median_sim.tolist(),
        "mean_assignment_matrix": mean_assignment.tolist(),
        "similarity_rows": modes_test,
        "similarity_cols": modes_train,
        "predicted_mapping": predicted,
    }
    if wildcard_assignments is not None:
        results["wildcard_mode"] = wildcard_mode
        results["wildcard_assignments"] = wildcard_assignments

    logger.info(
        f"  aggregate [{direction_label}]: "
        f"acc={results['mapping_accuracy_mean']:.2%} ± {results['mapping_accuracy_std']:.2%} "
        f"(chance={results['chance_accuracy']:.2%})"
    )
    for name, info in results["per_mode_accuracy"].items():
        logger.info(f"    {name} → {predicted[name]}: {info['mean']:.2%} ± {info['std']:.2%}")
    if wildcard_assignments is not None:
        logger.info(f"  {wildcard_mode} → {wildcard_assignments}")

    return results


# ── LDA direction test ────────────────────────────────────────────────────────

def run_lda_direction_test(
    X_r2: F32,
    y_r2: NDArray[np.int64],
    modes_r2: list[str],
    X_r3: F32,
    y_r3: NDArray[np.int64],
    modes_r3: list[str],
) -> dict[str, Any]:
    """Fit LDA on R2, project R3 onto its discriminant directions.

    If R3 silhouette in R2's LDA space is near-zero/negative, the linear
    directions separating R2 carry no R3 mode information — the cross-run
    signal is manifold-encoded, not linearly shared.
    """
    logger.info("=== LDA direction projection test ===")
    scaler = StandardScaler()
    X_r2_s = scaler.fit_transform(X_r2)
    X_r3_s = scaler.transform(X_r3)

    lda = LinearDiscriminantAnalysis()
    X_r2_proj = lda.fit_transform(X_r2_s, y_r2)
    X_r3_proj = lda.transform(X_r3_s)

    sil_r2_own = float(silhouette_score(X_r2_proj, y_r2, metric="cosine"))
    sil_r3_in_r2 = float(silhouette_score(X_r3_proj, y_r3, metric="cosine"))
    sil_r3_samples = silhouette_samples(X_r3_proj, y_r3, metric="cosine")
    per_mode_r3_sil = {
        name: float(sil_r3_samples[y_r3 == i].mean()) for i, name in enumerate(modes_r3)
    }

    knn = KNeighborsClassifier(n_neighbors=3, metric="cosine")
    knn.fit(X_r2_proj, y_r2)
    r3_predictions = knn.predict(X_r3_proj)
    cross_matrix = np.zeros((len(modes_r3), len(modes_r2)), dtype=np.int64)
    for i, _ in enumerate(modes_r3):
        mask = y_r3 == i
        for pred in r3_predictions[mask]:
            cross_matrix[i, int(pred)] += 1

    logger.info(f"  R2 sil in own LDA: {sil_r2_own:.4f}")
    logger.info(f"  R3 sil in R2 LDA:  {sil_r3_in_r2:.4f}")
    for name, sil in sorted(per_mode_r3_sil.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"    R3 per-mode sil: {name}: {sil:.4f}")

    return {
        "r2_sil_own_lda": sil_r2_own,
        "r3_sil_in_r2_lda": sil_r3_in_r2,
        "r3_per_mode_sil": per_mode_r3_sil,
        "cross_prediction_matrix": cross_matrix.tolist(),
        "cross_prediction_rows": modes_r3,
        "cross_prediction_cols": modes_r2,
        "projection_dim": int(X_r2_proj.shape[1]),
        "interpretation": (
            "If R3 sil in R2 LDA is near zero or negative, the linear "
            "directions separating R2 modes carry no R3 mode information — "
            "confirms 'directions vs manifolds' dissociation."
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-run functional transfer at 8B (paper R2 ↔ R3)"
    )
    parser.add_argument(
        "--train-run", type=str, default="run_8b_baseline",
        help="Pipeline run to train the MLP on (default: run_8b_baseline = paper R3)",
    )
    parser.add_argument(
        "--test-run", type=str, default="run_8b_r2_equivalent",
        help="Pipeline run to embed into the trained MLP (default: run_8b_r2_equivalent = paper R2)",
    )
    parser.add_argument(
        "--output-run", type=str, default="8b_cross_run_transfer",
        help="Subdirectory under outputs/analysis/ to write results.json",
    )
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--bottleneck-dim", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument(
        "--feature-key", type=str, default="features",
        help="NPZ key for feature vectors (default: 'features' = full baseline tiers)",
    )
    args = parser.parse_args()

    output_dir = OUTPUTS_BASE / "analysis" / args.output_run
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # --- Load data ---
    # train_run plays the role of Phase-0.5 "Run 4" (format-controlled = paper R3)
    X_r3, y_r3, modes_r3, _ = load_run_features(args.train_run, feature_key=args.feature_key)
    # test_run plays the role of Phase-0.5 "Run 3" (process modes = paper R2)
    X_r2, y_r2, modes_r2, _ = load_run_features(args.test_run, feature_key=args.feature_key)

    if X_r3.shape[1] != X_r2.shape[1]:
        raise RuntimeError(
            f"Feature dimensions must match (same pipeline, same calibration): "
            f"{args.train_run}={X_r3.shape[1]}, {args.test_run}={X_r2.shape[1]}"
        )

    results: dict[str, Any] = {
        "experiment": "8b_cross_run_transfer",
        "train_run": args.train_run,
        "test_run": args.test_run,
        "feature_key": args.feature_key,
        "r3_n_samples": int(X_r3.shape[0]),
        "r2_n_samples": int(X_r2.shape[0]),
        "n_features": int(X_r3.shape[1]),
        "r3_modes": modes_r3,
        "r2_modes": modes_r2,
        "forward_mapping": FORWARD_MAPPING,
        "reverse_mapping": REVERSE_MAPPING,
        "three_b_reference": {
            "forward_pedagogical_to_socratic": 0.76,
            "reverse_dialectical_to_deliberative": 0.85,
            "lda_r4_sil_in_r3_lda": -0.156,
            "lda_r3_sil_own": 0.438,
        },
    }

    # --- Forward transfer (train R3, embed R2) ---
    scaler_fwd = StandardScaler()
    X_r3_fwd = scaler_fwd.fit_transform(X_r3)
    X_r2_fwd = scaler_fwd.transform(X_r2)
    fwd = run_transfer_multi_seed(
        X_train=X_r3_fwd.astype(np.float32), y_train=y_r3, modes_train=modes_r3,
        X_test=X_r2_fwd.astype(np.float32), y_test=y_r2, modes_test=modes_r2,
        predicted=FORWARD_MAPPING, wildcard_mode=FORWARD_WILDCARD,
        direction_label=f"forward (train {args.train_run} → embed {args.test_run})",
        n_seeds=args.n_seeds, bottleneck_dim=args.bottleneck_dim, n_epochs=args.n_epochs,
    )
    results["transfer_forward"] = fwd

    # --- Reverse transfer (train R2, embed R3) ---
    scaler_rev = StandardScaler()
    X_r2_rev = scaler_rev.fit_transform(X_r2)
    X_r3_rev = scaler_rev.transform(X_r3)
    rev = run_transfer_multi_seed(
        X_train=X_r2_rev.astype(np.float32), y_train=y_r2, modes_train=modes_r2,
        X_test=X_r3_rev.astype(np.float32), y_test=y_r3, modes_test=modes_r3,
        predicted=REVERSE_MAPPING, wildcard_mode=REVERSE_WILDCARD,
        direction_label=f"reverse (train {args.test_run} → embed {args.train_run})",
        n_seeds=args.n_seeds, bottleneck_dim=args.bottleneck_dim, n_epochs=args.n_epochs,
    )
    results["transfer_reverse"] = rev

    # --- LDA direction test ---
    lda_results = run_lda_direction_test(
        X_r2.astype(np.float32), y_r2, modes_r2,
        X_r3.astype(np.float32), y_r3, modes_r3,
    )
    results["lda_direction_test"] = lda_results

    # --- Wrap ---
    elapsed = time.time() - t_start
    results["elapsed_seconds"] = float(elapsed)

    out_path = output_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✓ Saved → {out_path}")
    logger.info(f"Elapsed: {elapsed:.1f}s")

    # --- Headline summary ---
    logger.info("\n" + "=" * 60)
    logger.info("HEADLINE: 8B vs 3B (Phase 0.5)")
    logger.info("=" * 60)
    fwd_pedsoc = fwd["per_mode_accuracy"].get("pedagogical", {}).get("mean", float("nan"))
    rev_diadel = rev["per_mode_accuracy"].get("dialectical", {}).get("mean", float("nan"))
    logger.info(
        f"  Forward pedagogical → socratic:    "
        f"8B={fwd_pedsoc:.2%}  3B=76.00%  Δ={fwd_pedsoc - 0.76:+.2%}"
    )
    logger.info(
        f"  Reverse dialectical → deliberative: "
        f"8B={rev_diadel:.2%}  3B=85.00%  Δ={rev_diadel - 0.85:+.2%}"
    )
    logger.info(
        f"  LDA R3 sil in R2 LDA space:        "
        f"8B={lda_results['r3_sil_in_r2_lda']:+.4f}  3B=-0.1564"
    )
    logger.info(
        f"  LDA R2 sil in own LDA (sanity):    "
        f"8B={lda_results['r2_sil_own_lda']:+.4f}  3B=+0.4381"
    )


if __name__ == "__main__":
    main()
