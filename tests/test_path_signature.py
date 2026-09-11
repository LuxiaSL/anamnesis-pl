"""Tests for the path-signature feature family (level-2 log-signature).

The module carries its own battery (``--selftest``); this file runs it under pytest and adds
the checks that matter most as regressions:

  * the increment-permutation null's asymmetry (level-1 invariant, level-2 destroyed) — the
    spec's primary control, and the one thing that proves the implementation;
  * feature_map classification of every emitted name, plus NO regression in the classification
    of the existing frozen name corpus (the new METHOD value and the lvl1/lvl2 tokens must not
    reclassify anything that already exists).

CPU only; no banked data, no GPU, no network.
"""

from __future__ import annotations

import numpy as np
import pytest

from anamnesis.analysis.feature_map import FeatureMap, Method, Source, classify
from anamnesis.extraction.feature_families.path_signature import (
    ArrayPathSource,
    MalformedPathError,
    PathSignatureConfig,
    ProjectionBasis,
    ProjectionBasisBank,
    ShortPathError,
    extract_null_battery,
    extract_path_signature_from_source,
    feature_names,
    level1_terms,
    level2_area_matrix,
    log_signature_level2,
    permute_increments,
    selftest,
    signature_features_from_path,
    time_augment_path,
)


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(20260911)


@pytest.fixture(scope="module")
def basis(rng: np.random.Generator) -> ProjectionBasis:
    Q, _ = np.linalg.qr(rng.standard_normal((48, 16)))
    return ProjectionBasis(components=Q.T.copy(), mean=rng.standard_normal(48), label="pcaA")


@pytest.fixture(scope="module")
def path(rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0.0, 1.0, 140)
    latent = np.stack(
        [np.sin(3 * np.pi * t), np.cos(2 * np.pi * t) * t, t ** 2, np.exp(-3 * t)], axis=1,
    )
    mix = rng.standard_normal((4, 48)) / np.sqrt(48)
    return latent @ mix + 0.02 * rng.standard_normal((140, 48)) + 5.0


def test_module_selftest_passes() -> None:
    """The in-module battery is the contract; keep it green under CI too."""
    assert selftest() is True


# ── the null ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", [1, 2, 3, 17, 99])
def test_permutation_null_level1_invariant_level2_moves(
    path: np.ndarray, basis: ProjectionBasis, seed: int,
) -> None:
    cfg = PathSignatureConfig(layer_indices=(16,), n_components=4, time_augment=True)
    real, _ = signature_features_from_path(path, basis, cfg)
    null, _ = signature_features_from_path(
        path, basis, cfg.model_copy(
            update={"permute_increments": True, "permutation_seed": seed},
        ),
    )
    n1 = cfg.n_level1
    assert np.max(np.abs(null[:n1] - real[:n1])) <= 1e-10      # invariant by construction
    assert np.max(np.abs(null[n1:] - real[n1:])) > 1e-3 * np.max(np.abs(real[n1:]))


def test_permutation_null_is_reproducible(path: np.ndarray, basis: ProjectionBasis) -> None:
    proj = basis.project(path, 4)
    assert np.array_equal(permute_increments(proj, 5), permute_increments(proj, 5))
    assert not np.array_equal(permute_increments(proj, 5), permute_increments(proj, 6))


def test_unseeded_null_is_refused() -> None:
    with pytest.raises(ValueError):
        PathSignatureConfig(permute_increments=True)


def test_null_battery_requires_three_seeds(
    path: np.ndarray, basis: ProjectionBasis,
) -> None:
    src = ArrayPathSource({0: {16: path}})
    bank = ProjectionBasisBank.single(basis)
    cfg = PathSignatureConfig(layer_indices=(16,))
    with pytest.raises(ValueError):
        extract_null_battery(src, 0, bank, cfg, [1, 2])
    battery = extract_null_battery(src, 0, bank, cfg, [1, 2, 3])
    real = extract_path_signature_from_source(src, 0, bank, cfg)
    assert all(b.feature_names == real.feature_names for b in battery)


# ── the math ──────────────────────────────────────────────────────────────────


def test_straight_line_has_zero_area(rng: np.random.Generator) -> None:
    direction = rng.standard_normal(6)
    line = np.outer(np.linspace(0.0, 3.0, 90), direction) + rng.standard_normal(6)
    assert np.max(np.abs(level2_area_matrix(line))) <= 1e-10


def test_unit_square_area_is_one() -> None:
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=np.float64)
    A = level2_area_matrix(square)
    assert abs(A[0, 1] - 1.0) <= 1e-12
    assert np.max(np.abs(A + A.T)) <= 1e-14        # antisymmetric


def test_areas_are_translation_invariant(path: np.ndarray, basis: ProjectionBasis,
                                         rng: np.random.Generator) -> None:
    Y = basis.project(path, 5)
    shift = rng.standard_normal(Y.shape[1]) * 10.0
    assert np.max(np.abs(level2_area_matrix(Y) - level2_area_matrix(Y + shift))) <= 1e-8


def test_chen_concatenation_consistency(path: np.ndarray, basis: ProjectionBasis) -> None:
    Y = basis.project(path, 5)
    m = 61
    left, right = Y[: m + 1], Y[m:]
    L_l, L_r = level1_terms(left), level1_terms(right)
    bracket = 0.5 * (np.outer(L_l, L_r) - np.outer(L_r, L_l))
    expected = level2_area_matrix(left) + level2_area_matrix(right) + bracket
    assert np.max(np.abs(level2_area_matrix(Y) - expected)) <= 1e-8
    assert np.max(np.abs(level1_terms(Y) - (L_l + L_r))) <= 1e-10


def test_time_augmentation_adds_unit_clock(path: np.ndarray, basis: ProjectionBasis) -> None:
    Y = basis.project(path, 4)
    aug = time_augment_path(Y)
    assert aug.shape == (Y.shape[0], 5)
    assert aug[0, -1] == 0.0 and aug[-1, -1] == 1.0


@pytest.mark.parametrize(
    "k,aug,n1,n2",
    [(4, True, 5, 10), (4, False, 4, 6), (8, True, 9, 36), (8, False, 8, 28)],
)
def test_arity_matches_spec(path: np.ndarray, basis: ProjectionBasis,
                            k: int, aug: bool, n1: int, n2: int) -> None:
    cfg = PathSignatureConfig(layer_indices=(16,), n_components=k, time_augment=aug)
    feats, _ = signature_features_from_path(path, basis, cfg)
    names = feature_names(cfg)
    assert cfg.n_level1 == n1 and cfg.n_level2 == n2
    assert len(feats) == len(names) == n1 + n2


# ── failure modes ─────────────────────────────────────────────────────────────


def test_nan_path_raises() -> None:
    with pytest.raises(MalformedPathError):
        level1_terms(np.array([[1.0, np.nan], [2.0, 3.0]]))


def test_short_path_raises_by_default(path: np.ndarray, basis: ProjectionBasis) -> None:
    src = ArrayPathSource({0: {16: path[:2]}})
    bank = ProjectionBasisBank.single(basis)
    with pytest.raises(ShortPathError):
        extract_path_signature_from_source(src, 0, bank, PathSignatureConfig(layer_indices=(16,)))


def test_missing_layer_raises(path: np.ndarray, basis: ProjectionBasis) -> None:
    src = ArrayPathSource({0: {8: path}})
    bank = ProjectionBasisBank.single(basis)
    with pytest.raises(KeyError):
        extract_path_signature_from_source(src, 0, bank, PathSignatureConfig(layer_indices=(16,)))


def test_level_must_be_1_or_2(path: np.ndarray) -> None:
    with pytest.raises(ValueError):
        log_signature_level2(path, level=3)


# ── classification ────────────────────────────────────────────────────────────


def test_every_emitted_name_classifies() -> None:
    cfg = PathSignatureConfig(layer_indices=(0, 14, 16, 27), n_components=8, time_augment=True)
    fm = FeatureMap(feature_names(cfg), n_layers=32)
    assert fm.unclassified() == []
    assert {t.source for t in fm.tags} == {Source.residual}
    assert {t.method for t in fm.tags} == {Method.iterated_integral}
    assert {t.family for t in fm.tags} == {"path_signature"}


def test_lvl1_static_lvl2_dynamic() -> None:
    cfg = PathSignatureConfig(layer_indices=(16,), n_components=4)
    for name in feature_names(cfg):
        tag = classify(name, n_layers=32)
        assert tag.dynamic is ("_lvl2_" in name)


def test_no_regression_on_existing_name_shapes() -> None:
    """The new METHOD value and lvl1/lvl2 tokens must not touch any pre-existing name."""
    legacy = [
        "res_traj_L16_velocity_norm_mean", "res_traj_L16_direction_change_std",
        "activation_norm_L16_mean", "delta_norm_L16_mean", "pca_L16_c0",
        "attn_flow_L16_recency_bias", "gate_L16_sparsity_mean", "kv_L16_key_spread",
        "cache_L16_sink_mass", "spectral_L16_fiedler", "attn_entropy_L16_mean",
        "logit_entropy_mean", "value_L16_eff_dim", "qk_L16_self_align",
        "xrt_L16_alloc_entropy_mean", "ph_L16_h3_entropy",
    ]
    for name in legacy:
        tag = classify(name, n_layers=32)
        assert tag.method != Method.iterated_integral, name
        assert tag.source != Source.unknown, name
        assert tag.method != Method.unknown, name
