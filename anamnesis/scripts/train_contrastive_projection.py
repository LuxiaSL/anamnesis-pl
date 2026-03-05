"""Train contrastive projection model on raw hidden states.

One-time training per model scale. Trains an MLP with TripletMarginLoss
to project raw hidden states into a mode-discriminative embedding space.

Saves weights as numpy arrays (.npz) for pure-numpy inference during
feature extraction — no torch dependency at inference time.

Usage:
    python -m anamnesis.scripts.train_contrastive_projection \
        --raw-dir outputs/runs/8b_fat_01/raw_tensors/ \
        --metadata-dir outputs/runs/8b_fat_01/signatures/ \
        --output-path outputs/calibration/llama31_8b/contrastive_projection.npz \
        --model 8b

    # With positional correction:
    python -m anamnesis.scripts.train_contrastive_projection \
        --raw-dir outputs/runs/8b_fat_01/raw_tensors/ \
        --metadata-dir outputs/runs/8b_fat_01/signatures/ \
        --output-path outputs/calibration/llama31_8b/contrastive_projection.npz \
        --model 8b --positional-means outputs/calibration/llama31_8b/positional_means.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]

# Model presets (same as feature_pipeline.py)
_MODEL_PRESETS = {
    "8b": {
        "hidden_dim_model": 4096,
        "layer_indices": [8, 16, 20, 24, 28],
    },
    "3b": {
        "hidden_dim_model": 3072,
        "layer_indices": [7, 14, 18, 21, 24],
    },
}


def _load_training_data(
    raw_dir: Path,
    metadata_dir: Path,
    layer_indices: list[int],
    temporal_samples: int = 5,
    positional_means_path: Path | None = None,
    exclude_prompt_swap: bool = True,
) -> tuple[F32, NDArray, list[str]]:
    """Load hidden states and mode labels for training.

    Parameters
    ----------
    raw_dir : Path
        Directory with raw tensor npz files.
    metadata_dir : Path
        Directory with metadata json files (for mode labels).
    layer_indices : list[int]
        Which transformer layers to sample hidden states from.
    temporal_samples : int
        Number of evenly-spaced time points per generation.
    positional_means_path : Path, optional
        Path to positional means for correction.
    exclude_prompt_swap : bool
        Whether to exclude prompt-swap generations from training.

    Returns
    -------
    X : array [N, hidden_dim] — hidden states from all (gen, layer, time) combinations
    y : array [N] — mode labels (repeated for each layer/time sample)
    gen_ids : list[str] — generation IDs for tracking
    """
    from anamnesis.extraction.raw_saver import list_raw_tensor_ids, load_raw_tensors

    positional_means: F32 | None = None
    if positional_means_path and positional_means_path.exists():
        pm_data = np.load(positional_means_path)
        positional_means = pm_data["positional_means"].astype(np.float32)
        logger.info(f"Loaded positional means: {positional_means.shape}")

    gen_ids_all = list_raw_tensor_ids(raw_dir)
    logger.info(f"Found {len(gen_ids_all)} raw tensor files")

    all_hidden: list[F32] = []
    all_labels: list[str] = []
    all_gen_ids: list[str] = []
    skipped = 0

    for gen_id in gen_ids_all:
        # Load metadata for mode label
        json_path = metadata_dir / f"gen_{gen_id:03d}.json"
        if not json_path.exists():
            skipped += 1
            continue
        try:
            with open(json_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue

        mode = meta.get("mode", "")
        condition = meta.get("condition", "standard")

        # Skip prompt-swap generations
        if exclude_prompt_swap and (
            condition.startswith("prompt_swap") or mode.startswith("swap_")
        ):
            continue

        # Load raw tensors
        try:
            data = load_raw_tensors(gen_id, raw_dir)
        except Exception as e:
            logger.warning(f"Failed to load gen_{gen_id:03d}: {e}")
            skipped += 1
            continue

        T = len(data.hidden_states)
        if T < 2:
            skipped += 1
            continue

        # Inject positional means if loaded
        if positional_means is not None:
            data.positional_means = positional_means

        # Compute temporal sample indices
        if temporal_samples == 1:
            t_indices = [0]
        else:
            t_indices = [int(round(i * (T - 1) / (temporal_samples - 1)))
                         for i in range(temporal_samples)]

        for l_idx in layer_indices:
            arr_idx = l_idx + 1  # hidden_states[t][0] = embedding
            for t in t_indices:
                if t >= T:
                    continue
                h = data.hidden_states[t][arr_idx].copy().astype(np.float32)

                # Positional correction
                if positional_means is not None:
                    position = data.prompt_length + t
                    if arr_idx < positional_means.shape[0] and position < positional_means.shape[1]:
                        h = h - positional_means[arr_idx, position]

                all_hidden.append(h)
                all_labels.append(mode)
                all_gen_ids.append(f"gen_{gen_id:03d}_L{l_idx}_t{t}")

    if skipped > 0:
        logger.info(f"Skipped {skipped} generations (missing metadata or errors)")

    X = np.stack(all_hidden, axis=0)  # [N, hidden_dim]
    y = np.array(all_labels)

    logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} dims")
    mode_counts = {}
    for label in y:
        mode_counts[label] = mode_counts.get(label, 0) + 1
    for mode, count in sorted(mode_counts.items()):
        logger.info(f"  {mode}: {count} samples")

    return X, y, all_gen_ids


def train_projection(
    X: F32,
    y: NDArray,
    hidden_dim: int = 256,
    bottleneck_dim: int = 32,
    n_epochs: int = 300,
    lr: float = 1e-3,
    margin: float = 1.0,
    weight_decay: float = 1e-3,
    batch_triplets: int = 512,
    val_fraction: float = 0.2,
    seed: int = 42,
    standardize: bool = True,
) -> dict[str, F32]:
    """Train contrastive projection MLP and return numpy weights.

    Parameters
    ----------
    X : array [N, input_dim]
    y : array [N] — string mode labels
    hidden_dim : int
        MLP hidden layer size.
    bottleneck_dim : int
        Output embedding dimension.
    n_epochs : int
        Training epochs.
    lr : float
        Learning rate.
    margin : float
        Triplet loss margin.
    weight_decay : float
        L2 regularization.
    batch_triplets : int
        Number of triplets per epoch.
    val_fraction : float
        Fraction of data for validation.
    seed : int
        Random seed.
    standardize : bool
        Whether to standardize inputs before training.

    Returns
    -------
    Dict with numpy weight arrays: w1, b1, w2, b2, scaler_mean, scaler_scale.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        raise ImportError(
            "PyTorch is required for training contrastive projection. "
            "Install with: pip install torch"
        )

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Train/val split (by generation, not by sample)
    # Group samples by generation prefix (gen_XXX)
    unique_labels = sorted(set(y))
    n_samples = len(X)

    # Simple random split
    perm = rng.permutation(n_samples)
    n_val = max(1, int(n_samples * val_fraction))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    # Optional standardization
    scaler_mean: F32 | None = None
    scaler_scale: F32 | None = None
    if standardize:
        scaler_mean = X[train_idx].mean(axis=0).astype(np.float32)
        scaler_scale = X[train_idx].std(axis=0).astype(np.float32)
        scaler_scale = np.where(scaler_scale < 1e-12, 1.0, scaler_scale)
        X_scaled = ((X - scaler_mean) / scaler_scale).astype(np.float32)
    else:
        X_scaled = X

    X_train = X_scaled[train_idx]
    y_train = y[train_idx]
    X_val = X_scaled[val_idx]
    y_val = y[val_idx]

    input_dim = X_train.shape[1]
    logger.info(
        f"Training: {len(X_train)} samples, validation: {len(X_val)} samples, "
        f"input_dim={input_dim}, hidden={hidden_dim}, bottleneck={bottleneck_dim}"
    )

    # Build model
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_dim, bottleneck_dim),
    )
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    triplet_loss = nn.TripletMarginLoss(margin=margin)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    label_to_idx: dict[str, list[int]] = {l: [] for l in unique_labels}
    for i, l in enumerate(y_train):
        label_to_idx[l].append(i)
    # Filter out labels with < 2 samples
    valid_labels = [l for l in unique_labels if len(label_to_idx[l]) >= 2]

    best_val_acc = 0.0
    best_weights: dict[str, F32] = {}
    patience = 50
    no_improve = 0

    for epoch in range(n_epochs):
        # Sample triplets
        anchors, positives, negatives = [], [], []
        for _ in range(batch_triplets):
            label = valid_labels[rng.integers(len(valid_labels))]
            a_idx, p_idx = rng.choice(label_to_idx[label], size=2, replace=False)
            neg_label = valid_labels[rng.integers(len(valid_labels))]
            while neg_label == label:
                neg_label = valid_labels[rng.integers(len(valid_labels))]
            n_idx = rng.choice(label_to_idx[neg_label])
            anchors.append(a_idx)
            positives.append(p_idx)
            negatives.append(n_idx)

        if not anchors:
            continue

        a = nn.functional.normalize(model(X_train_t[anchors]), dim=1)
        p = nn.functional.normalize(model(X_train_t[positives]), dim=1)
        n = nn.functional.normalize(model(X_train_t[negatives]), dim=1)

        loss = triplet_loss(a, p, n)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                X_val_t = torch.tensor(X_val, dtype=torch.float32)
                emb_train = nn.functional.normalize(model(X_train_t), dim=1).numpy()
                emb_val = nn.functional.normalize(model(X_val_t), dim=1).numpy()

            # kNN accuracy on validation
            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
            knn.fit(emb_train, y_train)
            val_acc = float(knn.score(emb_val, y_val))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
                # Save best weights
                state = model.state_dict()
                best_weights = {
                    "w1": state["0.weight"].numpy().copy(),
                    "b1": state["0.bias"].numpy().copy(),
                    "w2": state["3.weight"].numpy().copy(),
                    "b2": state["3.bias"].numpy().copy(),
                }
            else:
                no_improve += 10

            if (epoch + 1) % 50 == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{n_epochs}: "
                    f"loss={loss.item():.4f}, val_kNN={val_acc:.3f} "
                    f"(best={best_val_acc:.3f})"
                )

            model.train()

            if no_improve >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    if not best_weights:
        # Never validated — save final weights
        state = model.state_dict()
        best_weights = {
            "w1": state["0.weight"].numpy().copy(),
            "b1": state["0.bias"].numpy().copy(),
            "w2": state["3.weight"].numpy().copy(),
            "b2": state["3.bias"].numpy().copy(),
        }

    logger.info(f"Best validation kNN accuracy: {best_val_acc:.3f}")

    result = best_weights.copy()
    if scaler_mean is not None:
        result["scaler_mean"] = scaler_mean
        result["scaler_scale"] = scaler_scale
    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Train contrastive projection on raw hidden states",
    )
    parser.add_argument(
        "--raw-dir", type=Path, required=True,
        help="Directory containing raw tensor npz files",
    )
    parser.add_argument(
        "--metadata-dir", type=Path, default=None,
        help="Directory with metadata json files (default: signatures/ next to raw_dir)",
    )
    parser.add_argument(
        "--output-path", type=Path, required=True,
        help="Where to save trained projection model (.npz)",
    )
    parser.add_argument(
        "--model", choices=list(_MODEL_PRESETS.keys()), required=True,
        help="Model preset for layer selection",
    )
    parser.add_argument(
        "--positional-means", type=Path, default=None,
        help="Path to positional means npz for correction",
    )
    parser.add_argument(
        "--temporal-samples", type=int, default=5,
        help="Number of temporal samples per generation (default: 5)",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=256,
        help="MLP hidden dimension (default: 256)",
    )
    parser.add_argument(
        "--bottleneck-dim", type=int, default=32,
        help="Output embedding dimension (default: 32)",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=300,
        help="Training epochs (default: 300)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    preset = _MODEL_PRESETS[args.model]

    if args.metadata_dir is None:
        args.metadata_dir = args.raw_dir.parent / "signatures"

    logger.info(f"Model: {args.model}")
    logger.info(f"Layers: {preset['layer_indices']}")
    logger.info(f"Temporal samples: {args.temporal_samples}")

    # Load training data
    X, y, gen_ids = _load_training_data(
        raw_dir=args.raw_dir,
        metadata_dir=args.metadata_dir,
        layer_indices=preset["layer_indices"],
        temporal_samples=args.temporal_samples,
        positional_means_path=args.positional_means,
    )

    if len(X) == 0:
        logger.error("No training data loaded!")
        sys.exit(1)

    # Train
    weights = train_projection(
        X, y,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        n_epochs=args.n_epochs,
        seed=args.seed,
    )

    # Save
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_path, **weights)
    size_kb = args.output_path.stat().st_size / 1024
    logger.info(f"Saved projection model: {args.output_path} ({size_kb:.1f} KB)")

    # Quick verification: load back and test inference
    from anamnesis.extraction.feature_families.contrastive_projection import (
        ContrastiveProjectionInference,
    )
    model = ContrastiveProjectionInference.load(args.output_path)
    test_emb = model.project(X[0])
    logger.info(
        f"Verification: input dim={X.shape[1]}, output dim={len(test_emb)}, "
        f"norm={np.linalg.norm(test_emb):.4f}"
    )


if __name__ == "__main__":
    main()
