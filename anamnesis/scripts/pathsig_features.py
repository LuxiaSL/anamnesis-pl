"""Shared helper: banked projected paths → path-signature design matrices (+ the null).

Used by both reads (E1 8B mode stake, E2 resolver cell). Deliberately thin — every number it
produces comes out of ``extraction.feature_families.path_signature``; this module only
marshals ``[T, k]`` arrays into it and stacks the rows.

The projection has already happened (``pathsig_project_residual.py`` on the node, or the
in-job projection in ``pathsig_s51_regen.py``), so the basis handed to the family here is the
identity on the already-projected coordinates. That is not a shortcut around the family's
"never fit a basis" rule — the real basis is the banked per-layer calibration PCA, applied
upstream and recorded in the bank's sidecar; routing through ``ProjectionBasis`` keeps the
augmentation / integration / permutation-null code paths identical to the family's own.

**Exploratory. Nothing computed here is quotable pre-first-read (C§8).**
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from anamnesis.extraction.feature_families.path_signature import (
    PathSignatureConfig,
    ProjectionBasis,
    feature_names,
    signature_features_from_path,
)

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]


class PathBank(BaseModel):
    """A loaded ``paths_*.npz`` — variable-length ``[T, k]`` trajectories, ragged-packed."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    paths: np.ndarray = Field(description="[sum_T, k] float32, concatenated in gen_ids order")
    offsets: np.ndarray = Field(description="[n+1] int64 row offsets into paths")
    gen_ids: np.ndarray
    lengths: np.ndarray
    prompt_lengths: np.ndarray
    source: str = ""
    variant: str = "pc"

    @property
    def n(self) -> int:
        return int(self.gen_ids.shape[0])

    @property
    def k_max(self) -> int:
        return int(self.paths.shape[1])

    def path(self, i: int) -> F64:
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return np.ascontiguousarray(self.paths[a:b].astype(np.float64))

    @classmethod
    def load(cls, npz_path: Path | str, variant: Literal["pc", "nopc"] = "pc") -> "PathBank":
        p = Path(npz_path)
        if not p.exists():
            raise FileNotFoundError(f"path bank not found: {p}")
        z = np.load(p)
        key = "paths_pc" if variant == "pc" else "paths_nopc"
        if key not in z.files:
            raise KeyError(f"{p}: no '{key}' (has {sorted(z.files)})")
        bank = cls(
            paths=np.asarray(z[key]),
            offsets=np.asarray(z["offsets"], dtype=np.int64),
            gen_ids=np.asarray(z["gen_ids"], dtype=np.int64),
            lengths=np.asarray(z["lengths"], dtype=np.int64),
            prompt_lengths=np.asarray(z["prompt_lengths"], dtype=np.int64),
            source=str(p), variant=variant,
        )
        if int(bank.offsets[-1]) != int(bank.paths.shape[0]):
            raise ValueError(f"{p}: offsets tail {bank.offsets[-1]} != rows {bank.paths.shape[0]}")
        if bank.offsets.shape[0] != bank.n + 1:
            raise ValueError(f"{p}: offsets/gen_ids length mismatch")
        return bank

    @classmethod
    def from_list(cls, paths: list[NDArray[Any]], gen_ids: list[int],
                  prompt_lengths: list[int] | None = None, source: str = "",
                  variant: str = "pc") -> "PathBank":
        if not paths:
            raise ValueError("from_list: empty path list")
        lengths = [int(np.asarray(p).shape[0]) for p in paths]
        offsets = np.zeros(len(paths) + 1, dtype=np.int64)
        np.cumsum(np.asarray(lengths, dtype=np.int64), out=offsets[1:])
        return cls(
            paths=np.concatenate([np.asarray(p, dtype=np.float32) for p in paths], axis=0),
            offsets=offsets,
            gen_ids=np.asarray(gen_ids, dtype=np.int64),
            lengths=np.asarray(lengths, dtype=np.int64),
            prompt_lengths=np.asarray(prompt_lengths if prompt_lengths is not None
                                      else [0] * len(paths), dtype=np.int64),
            source=source, variant=variant,
        )


def identity_basis(k: int, label: str = "pcaA") -> ProjectionBasis:
    """Identity over already-projected coordinates (see module docstring)."""
    return ProjectionBasis(components=np.eye(k, dtype=np.float64), mean=None, label=label)


def signature_matrix(
    bank: PathBank,
    *,
    layer: int,
    k: int,
    level: Literal[1, 2],
    time_augment: bool = True,
    permutation_seed: int | None = None,
) -> tuple[F32, list[str], list[int]]:
    """Stack one design matrix over every path in ``bank``.

    Returns ``(X [n_ok, P], names, kept_indices)``. A path too short for the requested level
    is DROPPED and reported through ``kept_indices`` — never zero-filled (the family's
    ``on_short_path='raise'`` default is respected and caught here, once, at the stacking
    layer where the caller can see the count).
    """
    if k > bank.k_max:
        raise ValueError(f"k={k} > banked rank {bank.k_max} ({bank.source})")
    cfg = PathSignatureConfig(
        layer_indices=(layer,), n_components=k, level=level, time_augment=time_augment,
        permute_increments=permutation_seed is not None,
        permutation_seed=permutation_seed,
        basis_label="pcaA",
    )
    basis = identity_basis(k)
    names = feature_names(cfg)
    rows: list[F64] = []
    kept: list[int] = []
    for i in range(bank.n):
        X = bank.path(i)[:, :k]
        try:
            feats, _T = signature_features_from_path(X, basis, cfg)
        except Exception:  # noqa: BLE001 — short/degenerate path: drop, count, never impute
            continue
        rows.append(feats)
        kept.append(i)
    if not rows:
        raise RuntimeError(f"signature_matrix: every path failed ({bank.source})")
    return np.stack(rows).astype(np.float32), names, kept


def null_matrices(
    bank: PathBank,
    *,
    layer: int,
    k: int,
    level: Literal[1, 2],
    seeds: tuple[int, ...],
    time_augment: bool = True,
) -> list[tuple[F32, list[int]]]:
    """The increment-permutation null (spec §2) at >=3 seeds, on REAL banked paths."""
    if len(seeds) < 3:
        raise ValueError(f"spec §2 asks for >=3 shuffle seeds per cell; got {len(seeds)}")
    out: list[tuple[F32, list[int]]] = []
    for s in seeds:
        X, _names, kept = signature_matrix(
            bank, layer=layer, k=k, level=level, time_augment=time_augment,
            permutation_seed=int(s),
        )
        out.append((X, kept))
    return out
