"""Typed result schemas for unified_runner sections.

Each ``run_<section>()`` function returns an instance of the corresponding
model below. The JSON written by ``run_full_analysis`` under
``outputs/analysis/<run>/results.json`` stays structurally identical —
``clean_for_json`` remains in the write path and handles NaN/Inf scrubbing
and numpy coercion for anything that slips through.

Backward-compat rules (Phase 3a):
- ``extra="forbid"`` on every model — schema drift surfaces loudly.
- Success-path fields stay required; error-path fields are ``Optional``
  with default ``None`` so the previous ``{"error": "..."}`` stub shape
  round-trips via ``model_dump(exclude_none=True)``.
- Numpy arrays are never stored on models; section runners call ``.tolist()``
  (or ``float()`` / ``int()``) before handing data to the schema.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

_FORBID = ConfigDict(extra="forbid")


# ─────────────────────────────────────────────────────────────────────────
# Section 1: Data integrity
# ─────────────────────────────────────────────────────────────────────────


class NanInfCount(BaseModel):
    """NaN/Inf counts for a single tier. ``error`` populated only when the
    tier could not be loaded (in which case ``nan``/``inf`` are sentinel -1).
    """

    model_config = _FORBID

    nan: int
    inf: int
    error: str | None = None


class TierVarianceReport(BaseModel):
    """Per-tier variance diagnostics: constant/near-constant feature counts."""

    model_config = _FORBID

    n_features: int
    n_constant: int
    n_near_constant: int


class LengthByModeStats(BaseModel):
    """Generation-length stats for a single mode (includes median)."""

    model_config = _FORBID

    mean: float
    std: float
    min: int
    max: int
    median: float


class LengthOverallStats(BaseModel):
    """Overall generation-length stats (no median — matches legacy shape)."""

    model_config = _FORBID

    mean: float
    std: float
    min: int
    max: int


class TierValueRange(BaseModel):
    """Per-tier feature-value range summary."""

    model_config = _FORBID

    global_mean: float
    global_std: float
    global_min: float
    global_max: float
    feature_mean_range: list[float] = Field(
        description="Two-element list [min, max] of per-feature means.",
    )


class IntegrityResult(BaseModel):
    """Section 1 result: data integrity + descriptive statistics."""

    model_config = _FORBID

    n_samples: int
    n_modes: int
    n_topics: int
    modes: list[str]
    topics: list[str]
    samples_per_mode: dict[str, int]
    samples_per_topic: dict[str, int]
    balanced: bool
    tier_dims: dict[str, int]
    total_features: int
    nan_inf: dict[str, NanInfCount]
    all_clean: bool
    variance_report: dict[str, TierVarianceReport]
    length_by_mode: dict[str, LengthByModeStats] | None = None
    length_overall: LengthOverallStats | None = None
    value_ranges: dict[str, TierValueRange]


# ─────────────────────────────────────────────────────────────────────────
# Section 2: Classification
# ─────────────────────────────────────────────────────────────────────────


class ClassifierAccuracyResult(BaseModel):
    """Basic classifier CV output used by ``linear_probe`` and each pair in
    ``pairwise_binary``: accuracy + per-fold accuracies."""

    model_config = _FORBID

    accuracy: float
    fold_accuracies: list[float]


class ClassifierWithConfusionResult(BaseModel):
    """RF CV output with confusion matrix.

    Used by ``rf_5way`` (always succeeds) and ``rf_4way_no_analogical``
    (returns just ``{"error": ...}`` when analogical is absent).
    """

    model_config = _FORBID

    accuracy: float | None = None
    fold_accuracies: list[float] | None = None
    confusion_matrix: list[list[int]] | None = None
    labels: list[str] | None = None
    error: str | None = None


class TopicHeldoutResult(BaseModel):
    """GroupKFold-by-topic CV output."""

    model_config = _FORBID

    accuracy: float
    fold_accuracies: list[float]
    n_groups: int


class CVStabilityResult(BaseModel):
    """Multi-seed RF stability distribution (key tiers only)."""

    model_config = _FORBID

    mean: float
    median: float
    std: float
    ci_lo: float
    ci_hi: float
    min: float
    max: float
    all_accuracies: list[float]
    n_seeds: int


class PermutationTestResult(BaseModel):
    """Label-permutation null distribution (key tiers only)."""

    model_config = _FORBID

    observed_accuracy: float
    p_value: float
    null_mean: float
    null_std: float
    null_max: float
    null_p95: float
    null_p99: float
    n_permutations: int


class PerModeLengthStats(BaseModel):
    """Generation-length stats for the length-only confound baseline."""

    model_config = _FORBID

    mean: float
    std: float
    min: float
    max: float


class LengthOnlyResult(BaseModel):
    """Length-only confound baseline (Section 2).

    Two shapes exist on disk:
    - Error path (no length metadata): ``{"accuracy": null, "error": "..."}``.
    - Success path: adds ``fold_accuracies``, ``confusion_matrix``, ``labels``,
      ``per_mode_lengths``.

    The custom serializer keeps these wire shapes intact. ``accuracy`` is
    explicitly emitted even when ``None`` to preserve the error-stub shape.
    """

    model_config = _FORBID

    accuracy: float | None = None
    fold_accuracies: list[float] | None = None
    confusion_matrix: list[list[int]] | None = None
    labels: list[str] | None = None
    per_mode_lengths: dict[str, PerModeLengthStats] | None = None
    error: str | None = None

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, Any]:
        if self.error is not None:
            return {"accuracy": self.accuracy, "error": self.error}
        out: dict[str, Any] = {"accuracy": self.accuracy}
        if self.fold_accuracies is not None:
            out["fold_accuracies"] = self.fold_accuracies
        if self.confusion_matrix is not None:
            out["confusion_matrix"] = self.confusion_matrix
        if self.labels is not None:
            out["labels"] = self.labels
        if self.per_mode_lengths is not None:
            out["per_mode_lengths"] = {
                k: v.model_dump(mode="json") for k, v in self.per_mode_lengths.items()
            }
        return out


class TierClassificationResult(BaseModel):
    """Per-tier classification battery.

    ``cv_stability`` / ``permutation_test`` are populated only for key
    composite tiers (``T2+T2.5``, ``combined``, ``combined_v2``, ...);
    other tiers omit them on the wire.
    """

    model_config = _FORBID

    rf_5way: ClassifierWithConfusionResult
    topic_heldout: TopicHeldoutResult
    linear_probe: ClassifierAccuracyResult
    pairwise_binary: dict[str, ClassifierAccuracyResult]
    rf_4way_no_analogical: ClassifierWithConfusionResult
    cv_stability: CVStabilityResult | None = None
    permutation_test: PermutationTestResult | None = None


class ClassificationResult(BaseModel):
    """Section 2 result: per-tier classification + length-only confound.

    Wire format stores dynamic tier keys at the top level (e.g. ``T1``,
    ``T2+T2.5``, ``combined``) alongside ``length_only``. Internally we
    gather the tier entries into ``by_tier`` so consumers can use
    attribute access (``result.by_tier[tier].rf_5way.accuracy``). The
    validator reshapes wire → internal; the serializer reshapes back.
    """

    model_config = _FORBID

    by_tier: dict[str, TierClassificationResult]
    length_only: LengthOnlyResult | None = None

    @model_validator(mode="before")
    @classmethod
    def _reshape_in(cls, data: Any) -> Any:
        if isinstance(data, dict) and "by_tier" not in data:
            flat = dict(data)
            length_only = flat.pop("length_only", None)
            return {"by_tier": flat, "length_only": length_only}
        return data

    @model_serializer(mode="plain")
    def _reshape_out(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            tier: tier_result.model_dump(mode="json", exclude_none=True)
            for tier, tier_result in self.by_tier.items()
        }
        if self.length_only is not None:
            out["length_only"] = self.length_only.model_dump(mode="json")
        return out


# ─────────────────────────────────────────────────────────────────────────
# Section 3: Tier ablation
# ─────────────────────────────────────────────────────────────────────────


class PairwiseTierCombo(BaseModel):
    """Accuracy of two baseline tiers concatenated."""

    model_config = _FORBID

    accuracy: float
    n_features: int
    individual_max: float
    gain_over_best_individual: float


class TripleTierCombo(BaseModel):
    """Accuracy of three baseline tiers concatenated."""

    model_config = _FORBID

    accuracy: float
    n_features: int
    best_pairwise_subset: float
    gain_over_best_pair: float


class CrossGroupAblation(BaseModel):
    """Accuracy of a baseline composite + a single engineered family."""

    model_config = _FORBID

    accuracy: float
    n_features: int
    baseline_accuracy: float
    engineered_alone: float
    gain_over_baseline: float


class LeaveOneOutEntry(BaseModel):
    """Per-tier leave-one-out accuracy + cost of removal."""

    model_config = _FORBID

    accuracy_without: float
    cost_of_removal: float


class TierRankingEntry(BaseModel):
    """One row of the tier-ranking table (tier name + its standalone accuracy)."""

    model_config = _FORBID

    tier: str
    accuracy: float


class FeatureImportanceEntry(BaseModel):
    """One row of a feature-importance table (RF or LR top-N)."""

    model_config = _FORBID

    name: str
    importance: float


class StdVsMeanResult(BaseModel):
    """*_std vs *_mean feature RF accuracy split.

    Success path sets ``n_std_features``, ``n_mean_features`` plus the
    two accuracies + comparator. Error path sets ``error``, ``n_names``,
    ``n_features``. The two shapes have disjoint keys.
    """

    model_config = _FORBID

    n_std_features: int | None = None
    n_mean_features: int | None = None
    std_accuracy: float | None = None
    mean_accuracy: float | None = None
    std_beats_mean: bool | None = None
    n_names: int | None = None
    n_features: int | None = None
    error: str | None = None


class PerTopicEffectSize(BaseModel):
    """Per-topic Cohen's d (success) or an error stub (too few samples /
    insufficient pairs). Success and error keys do not overlap.
    """

    model_config = _FORBID

    cohens_d: float | None = None
    mean_within: float | None = None
    mean_between: float | None = None
    n_within_pairs: int | None = None
    n_between_pairs: int | None = None
    n_samples: int | None = None
    n: int | None = None
    error: str | None = None


class CohensDPerTopicResult(BaseModel):
    """Cohen's d summary across topics.

    When no topics produce a successful d, the summary fields are ``null``
    on the wire (not absent). The custom serializer preserves this shape.
    """

    model_config = _FORBID

    per_topic: dict[str, PerTopicEffectSize]
    mean_d: float | None
    median_d: float | None
    std_d: float | None
    min_d: float | None
    max_d: float | None
    all_positive: bool | None
    n_topics: int

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, Any]:
        return {
            "per_topic": {
                k: v.model_dump(mode="json", exclude_none=True)
                for k, v in self.per_topic.items()
            },
            "mean_d": self.mean_d,
            "median_d": self.median_d,
            "std_d": self.std_d,
            "min_d": self.min_d,
            "max_d": self.max_d,
            "all_positive": self.all_positive,
            "n_topics": self.n_topics,
        }


class TierAblationResult(BaseModel):
    """Section 3 result: tier ablation + feature importance.

    Several fields are present only for v2 runs (``cross_group_ablation``,
    ``top_features_rf``, etc.). ``top_features_rf_combined`` is a legacy
    key from pre-``feature_importance_composite`` snapshots and is
    preserved for round-trip of older baseline runs.
    """

    model_config = _FORBID

    per_tier_accuracy: dict[str, float]
    pairwise_tier_combinations: dict[str, PairwiseTierCombo]
    triple_tier_combinations: dict[str, TripleTierCombo] | None = None
    cross_group_ablation: dict[str, CrossGroupAblation] | None = None
    cross_group_baseline: str | None = None
    leave_one_tier_out: dict[str, LeaveOneOutEntry]
    leave_one_out_baseline_accuracy: float | None = None
    tier_ranking: list[TierRankingEntry]
    tier_inversion_t25_gt_t2_gt_t1: bool
    top_features_rf: list[FeatureImportanceEntry] | None = None
    top_features_lr: list[FeatureImportanceEntry] | None = None
    feature_importance_composite: str | None = None
    top_features_rf_t2t25: list[FeatureImportanceEntry]
    top_features_lr_t2t25: list[FeatureImportanceEntry]
    top_features_rf_combined: list[FeatureImportanceEntry] | None = Field(
        default=None,
        description="Legacy top-features key from pre-v2 baseline snapshots.",
    )
    tier_contribution_ratio: dict[str, float]
    std_vs_mean: StdVsMeanResult
    cohens_d_per_topic: CohensDPerTopicResult


# ─────────────────────────────────────────────────────────────────────────
# Section 7: Clustering
# ─────────────────────────────────────────────────────────────────────────


class TierSilhouette(BaseModel):
    """Per-tier silhouette scores.

    ``mode_silhouette_cosine`` / ``_euclidean`` / ``mode_silhouette`` are
    floats on success; they fall back to ``"ERROR: ..."`` strings when
    silhouette_score raises (e.g. degenerate cluster set).
    ``topic_silhouette`` is float or None.
    """

    model_config = _FORBID

    mode_silhouette_cosine: float | str
    mode_silhouette_euclidean: float | str
    mode_silhouette: float | str
    topic_silhouette: float | None


class PerModeSilhouetteStats(BaseModel):
    """Per-mode silhouette distribution summary."""

    model_config = _FORBID

    mean: float
    std: float
    min: float
    max: float
    n_negative: int


class EmbeddingResult(BaseModel):
    """2-D embedding (t-SNE or UMAP) for plotting.

    Success path stores ``coords`` (n×2 float), ``modes``, ``topics``.
    Error path stores only ``error`` (e.g. UMAP not installed).
    """

    model_config = _FORBID

    coords: list[list[float]] | None = None
    modes: list[str] | None = None
    topics: list[str] | None = None
    error: str | None = None


class ClusteringResult(BaseModel):
    """Section 7 result: silhouette + K-Means ARI + 2-D embeddings.

    Both ``per_mode_silhouette`` (top-level backward-compat key that
    aliases the cosine variant) and the explicit cosine/euclidean
    variants are preserved. When silhouette_samples raises, the
    per-mode dict contains a top-level ``"error": "..."`` key alongside
    any mode entries — the dict type here is ``dict[str, Any]`` so
    those heterogeneous entries round-trip cleanly.
    """

    model_config = _FORBID

    silhouette_by_tier: dict[str, TierSilhouette]
    per_mode_silhouette: dict[str, Any]
    per_mode_silhouette_cosine: dict[str, Any]
    per_mode_silhouette_euclidean: dict[str, Any]
    kmeans_ari: dict[str, float | str]
    embeddings: dict[str, EmbeddingResult]


# ─────────────────────────────────────────────────────────────────────────
# Section 8: Contrastive projection
# ─────────────────────────────────────────────────────────────────────────


class ContrastiveTierResult(BaseModel):
    """Top-level per-tier contrastive result (T2+T2.5 / combined)."""

    model_config = _FORBID

    knn_accuracy_mean: float
    knn_accuracy_std: float
    knn_fold_accs: list[float]
    silhouette_mean: float | None


class ContrastiveAblationEntry(BaseModel):
    """Single tier entry inside ``contrastive.tier_ablation.individual``
    or the combined/T2+T2.5 sub-keys of ``tier_ablation``."""

    model_config = _FORBID

    knn_accuracy: float
    knn_std: float
    silhouette: float | None
    n_features: int


class ContrastivePairwiseEntry(BaseModel):
    """Pairwise contrastive ablation entry (adds best-individual baseline)."""

    model_config = _FORBID

    knn_accuracy: float
    knn_std: float
    silhouette: float | None
    n_features: int
    best_individual_knn: float
    gain_over_best_individual: float


class ContrastiveSuperAdditivity(BaseModel):
    """T2+T2.5 super-additivity summary.

    Keys like ``T2.5_alone`` and ``T2+T2.5_pair`` aren't legal Python
    identifiers, so the model exposes them as ``T2_5_alone`` etc. A
    custom validator + serializer translates to/from the on-disk names.
    """

    model_config = _FORBID

    T2_alone: float
    T2_5_alone: float
    T2_T2_5_pair: float
    best_individual: float
    gain: float
    combined_knn: float
    T2_T2_5_beats_combined: bool

    _ON_DISK_RENAMES: ClassVar[dict[str, str]] = {
        "T2.5_alone": "T2_5_alone",
        "T2+T2.5_pair": "T2_T2_5_pair",
        "T2+T2.5_beats_combined": "T2_T2_5_beats_combined",
    }

    @model_validator(mode="before")
    @classmethod
    def _from_disk(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {cls._ON_DISK_RENAMES.get(k, k): v for k, v in data.items()}
        return data

    @model_serializer(mode="plain")
    def _to_disk(self) -> dict[str, Any]:
        return {
            "T2_alone": self.T2_alone,
            "T2.5_alone": self.T2_5_alone,
            "T2+T2.5_pair": self.T2_T2_5_pair,
            "best_individual": self.best_individual,
            "gain": self.gain,
            "combined_knn": self.combined_knn,
            "T2+T2.5_beats_combined": self.T2_T2_5_beats_combined,
        }


class ContrastiveTierAblation(BaseModel):
    """Contrastive MLP tier ablation bundle.

    ``T2+T2.5`` on the wire becomes ``T2_T2_5`` in Python; translated by
    the validator + serializer pair below.
    """

    model_config = _FORBID

    individual: dict[str, ContrastiveAblationEntry]
    pairwise: dict[str, ContrastivePairwiseEntry]
    T2_T2_5: ContrastiveAblationEntry
    combined: ContrastiveAblationEntry
    super_additivity: ContrastiveSuperAdditivity

    @model_validator(mode="before")
    @classmethod
    def _from_disk(cls, data: Any) -> Any:
        if isinstance(data, dict) and "T2+T2.5" in data:
            out = dict(data)
            out["T2_T2_5"] = out.pop("T2+T2.5")
            return out
        return data

    @model_serializer(mode="plain")
    def _to_disk(self) -> dict[str, Any]:
        return {
            "individual": {
                k: v.model_dump(mode="json", exclude_none=True)
                for k, v in self.individual.items()
            },
            "pairwise": {
                k: v.model_dump(mode="json", exclude_none=True)
                for k, v in self.pairwise.items()
            },
            "T2+T2.5": self.T2_T2_5.model_dump(mode="json", exclude_none=True),
            "combined": self.combined.model_dump(mode="json", exclude_none=True),
            "super_additivity": self.super_additivity.model_dump(mode="json"),
        }


class CapacitySweepEntry(BaseModel):
    """One hidden-dim row in the capacity sweep."""

    model_config = _FORBID

    knn_accuracy: float
    silhouette: float | None


class LinearBaselineEntry(BaseModel):
    """LDA / NCA linear projection baseline."""

    model_config = _FORBID

    knn_accuracy: float
    knn_std: float
    silhouette: float | None
    n_components: int


class ContrastiveResult(BaseModel):
    """Section 8 result: contrastive projection (MLP + triplet loss).

    Top-level fields are all ``Optional`` so the PyTorch-missing error
    stub (``{"error": "..."}``) round-trips cleanly. The ``T2+T2.5`` key
    is renamed to ``T2_T2_5`` via validator/serializer for the same
    reason as ContrastiveTierAblation.
    """

    model_config = _FORBID

    T2_T2_5: ContrastiveTierResult | None = None
    combined: ContrastiveTierResult | None = None
    capacity_sweep: dict[str, CapacitySweepEntry] | None = None
    tier_ablation: ContrastiveTierAblation | None = None
    linear_baselines: dict[str, LinearBaselineEntry] | None = None
    error: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_disk(cls, data: Any) -> Any:
        if isinstance(data, dict) and "T2+T2.5" in data:
            out = dict(data)
            out["T2_T2_5"] = out.pop("T2+T2.5")
            return out
        return data

    @model_serializer(mode="plain")
    def _to_disk(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.T2_T2_5 is not None:
            out["T2+T2.5"] = self.T2_T2_5.model_dump(mode="json", exclude_none=True)
        if self.combined is not None:
            out["combined"] = self.combined.model_dump(mode="json", exclude_none=True)
        if self.capacity_sweep is not None:
            out["capacity_sweep"] = {
                k: v.model_dump(mode="json", exclude_none=True)
                for k, v in self.capacity_sweep.items()
            }
        if self.tier_ablation is not None:
            out["tier_ablation"] = self.tier_ablation.model_dump(mode="json")
        if self.linear_baselines is not None:
            out["linear_baselines"] = {
                k: v.model_dump(mode="json", exclude_none=True)
                for k, v in self.linear_baselines.items()
            }
        if self.error is not None:
            out["error"] = self.error
        return out


# ─────────────────────────────────────────────────────────────────────────
# Section 4: Intrinsic dimension
# ─────────────────────────────────────────────────────────────────────────


class BootstrapStats(BaseModel):
    """Per-seed bootstrap distribution of dadapy TwoNN IDs."""

    model_config = _FORBID

    mean: float
    std: float
    ci_lo: float
    ci_hi: float
    n_successful: int


class GlobalTierIDResult(BaseModel):
    """ID metrics for one tier of the full dataset.

    ``dadapy_id`` / ``skdim_id`` fall back to ``"ERROR: ..."`` strings
    when the respective estimator raises. ``dadapy_err`` is set only on
    dadapy success. ``bootstrap_by_seed`` keys are stringified ints.
    """

    model_config = _FORBID

    n_features_clean: int
    dadapy_id: float | str | None = None
    dadapy_err: float | None = None
    skdim_id: float | str | None = None
    bootstrap_by_seed: dict[str, BootstrapStats]


class PerModeIDResult(BaseModel):
    """ID metrics for one mode on the T2+T2.5 feature set."""

    model_config = _FORBID

    n_samples: int
    dadapy_id: float | str | None = None
    skdim_id: float | str | None = None
    bootstrap_mean: float | None = None
    bootstrap_std: float | None = None
    bootstrap_ci: list[float] | None = None


class GRIDEResult(BaseModel):
    """Multiscale GRIDE estimator output or an error stub."""

    model_config = _FORBID

    ids: list[float] | None = None
    errors: list[float] | None = None
    error: str | None = None


class TierConvergenceResult(BaseModel):
    """T1/T2/T2.5 ID convergence summary."""

    model_config = _FORBID

    max_pairwise_diff: float
    converged_within_2: bool
    values: dict[str, float]


class IntrinsicDimensionResult(BaseModel):
    """Section 4 result.

    All fields optional so the top-level "dadapy not installed" error
    stub (``{"error": "..."}``) round-trips cleanly.
    """

    model_config = _FORBID

    global_: dict[str, GlobalTierIDResult] | None = None
    per_mode: dict[str, PerModeIDResult] | None = None
    gride: GRIDEResult | None = None
    tier_convergence: TierConvergenceResult | None = None
    error: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_disk(cls, data: Any) -> Any:
        if isinstance(data, dict) and "global" in data:
            out = dict(data)
            out["global_"] = out.pop("global")
            return out
        return data

    @model_serializer(mode="plain")
    def _to_disk(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.global_ is not None:
            out["global"] = {
                k: v.model_dump(mode="json", exclude_none=True)
                for k, v in self.global_.items()
            }
        if self.per_mode is not None:
            out["per_mode"] = {
                k: v.model_dump(mode="json", exclude_none=True)
                for k, v in self.per_mode.items()
            }
        if self.gride is not None:
            out["gride"] = self.gride.model_dump(mode="json", exclude_none=True)
        if self.tier_convergence is not None:
            out["tier_convergence"] = self.tier_convergence.model_dump(mode="json")
        if self.error is not None:
            out["error"] = self.error
        return out


# ─────────────────────────────────────────────────────────────────────────
# Section 5: CCGP
# ─────────────────────────────────────────────────────────────────────────


class CCGPDichotomy(BaseModel):
    """Single binary dichotomy evaluation within a CCGP variant."""

    model_config = _FORBID

    group_a: list[str]
    group_b: list[str]
    mean_accuracy: float
    decodable: bool


class CCGPVariant(BaseModel):
    """One CCGP variant (classifier × seed × fold-count × optional tier)."""

    model_config = _FORBID

    multiclass_mean: float
    multiclass_fold_accs: list[float]
    per_mode_recall: dict[str, float]
    n_decodable: int
    n_dichotomies: int
    ccgp_score: float
    dichotomies: list[CCGPDichotomy]


class CCGPSummary(BaseModel):
    """CCGP score summary across all variants."""

    model_config = _FORBID

    min_ccgp: float
    max_ccgp: float
    all_perfect: bool


class CCGPResult(BaseModel):
    """Section 5 result: CCGP across classifier/seed/fold variants."""

    model_config = _FORBID

    variants: dict[str, CCGPVariant]
    summary: CCGPSummary


# ─────────────────────────────────────────────────────────────────────────
# Section 6: Topology & hyperbolicity
# ─────────────────────────────────────────────────────────────────────────


class TopologyMetricSummary(BaseModel):
    """Nearest/farthest centroid-pair summary for one distance metric."""

    model_config = _FORBID

    nearest_pair: str
    nearest_dist: float
    farthest_pair: str
    farthest_dist: float
    analogical_outgroup_ratio: float


class GromovDeltaResult(BaseModel):
    """Gromov delta-hyperbolicity diagnostics (Euclidean)."""

    model_config = _FORBID

    delta_max: float
    delta_relative: float
    delta_mean: float
    delta_median: float
    diameter: float
    n_quadruples: int


class TopologyResult(BaseModel):
    """Section 6 result: centroid distances, hierarchical clustering,
    topology summary, and delta-hyperbolicity.

    ``hierarchical_clustering`` values are Newick-string trees on success
    or ``"ERROR: ..."`` strings on failure.
    """

    model_config = _FORBID

    tier: str
    euclidean_centroid_distances: dict[str, float]
    cosine_centroid_distances: dict[str, float]
    manhattan_centroid_distances: dict[str, float]
    hierarchical_clustering: dict[str, str]
    topology_summary: dict[str, TopologyMetricSummary]
    gromov_delta_euclidean: GromovDeltaResult


# ─────────────────────────────────────────────────────────────────────────
# Section 11: Manifold geometry
# ─────────────────────────────────────────────────────────────────────────


class TangentAngles(BaseModel):
    """Principal angles between two mode subspaces."""

    model_config = _FORBID

    mean_angle_deg: float
    max_angle_deg: float
    min_angle_deg: float
    angles_deg: list[float]


class ModeVarianceExplained(BaseModel):
    """PCA explained-variance-ratio summary for one mode."""

    model_config = _FORBID

    explained_variance_ratio: list[float]
    cumulative_5: float
    cumulative_10: float


class TangentSpaceResult(BaseModel):
    """Local tangent-space alignment + per-mode variance explained.

    Returns an error stub on PCA failure; otherwise populates all three
    success fields.
    """

    model_config = _FORBID

    pairwise_angles: dict[str, TangentAngles] | None = None
    mode_variance_explained: dict[str, ModeVarianceExplained] | None = None
    n_components: int | None = None
    error: str | None = None


class GeodesicOverall(BaseModel):
    """Overall geodesic-vs-Euclidean distortion stats."""

    model_config = _FORBID

    mean_distortion: float
    std_distortion: float
    max_distortion: float
    median_distortion: float


class GeodesicPerMode(BaseModel):
    """Per-mode geodesic distortion (within-mode pairs only)."""

    model_config = _FORBID

    mean: float
    std: float


class GeodesicDistortionResult(BaseModel):
    """Isomap geodesic / Euclidean distortion diagnostics."""

    model_config = _FORBID

    overall: GeodesicOverall | None = None
    per_mode: dict[str, GeodesicPerMode] | None = None
    within_mode_mean: float | None = None
    between_mode_mean: float | None = None
    isomap_n_neighbors: int | None = None
    reconstruction_error: float | None = None
    error: str | None = None


class CurvatureScaleEntry(BaseModel):
    """Local-PCA reconstruction-error proxy for curvature at one scale."""

    model_config = _FORBID

    mean_curvature: float
    std_curvature: float
    per_mode: dict[str, float]


class CurvatureResult(BaseModel):
    """Scale-dependent curvature proxies.

    ``per_scale`` keys are stringified integers matching ``scales``.
    """

    model_config = _FORBID

    scales: list[int] | None = None
    per_scale: dict[str, CurvatureScaleEntry] | None = None
    error: str | None = None


class BettiNumberEntry(BaseModel):
    """Persistent-homology summary for one dimension (H0/H1/H2)."""

    model_config = _FORBID

    n_features: int
    n_finite: int
    mean_lifetime: float
    max_lifetime: float
    median_lifetime: float


class PersistentHomologyResult(BaseModel):
    """Persistent homology via ripser (or error stub)."""

    model_config = _FORBID

    betti_numbers: dict[str, BettiNumberEntry] | None = None
    n_subsampled: int | None = None
    error: str | None = None


class ManifoldGeometryResult(BaseModel):
    """Section 11 result: tangent space, geodesic distortion,
    curvature proxies, and persistent homology."""

    model_config = _FORBID

    tangent_space: TangentSpaceResult
    geodesic_distortion: GeodesicDistortionResult
    curvature: CurvatureResult
    persistent_homology: PersistentHomologyResult


# ─────────────────────────────────────────────────────────────────────────
# Section 9: Semantic independence
# ─────────────────────────────────────────────────────────────────────────


class ClassificationScore(BaseModel):
    """Per-classifier CV output inside a SemanticClassifierBundle."""

    model_config = _FORBID

    accuracy: float
    fold_accuracies: list[float]


class SemanticClassifierBundle(BaseModel):
    """Bundle of rf + knn accuracies for a given feature set.

    ``dims`` is present for tfidf / sbert / combined / semantic_noise
    (which record the feature-set dimensionality). Newer v2 runs omit
    ``dims`` for compute_classification (it pulls from
    per_tier_semantic.classification, which never set it).
    """

    model_config = _FORBID

    rf: ClassificationScore | None = None
    knn: ClassificationScore | None = None
    dims: int | None = None
    error: str | None = None


class MantelResult(BaseModel):
    """Mantel test distance-matrix correlation."""

    model_config = _FORBID

    r: float
    p_value: float
    null_mean: float
    null_std: float


class TextToComputeR2(BaseModel):
    """Ridge R² summary for text → compute-feature prediction."""

    model_config = _FORBID

    median_r2: float
    mean_r2: float
    n_features_r2_above_01: int
    n_features_r2_above_0: int
    best_r2: float
    worst_r2: float


class PerModeSurfaceVsCompute(BaseModel):
    """One mode's surface-vs-compute recall gap."""

    model_config = _FORBID

    surface_recall: float
    compute_recall: float
    gap_compute_minus_surface: float
    sub_semantic: bool
    surface_n: int
    compute_n: int


class PerModeSurfaceVsComputeResult(BaseModel):
    """Per-mode surface-vs-compute decomposition summary."""

    model_config = _FORBID

    per_mode: dict[str, PerModeSurfaceVsCompute]
    n_sub_semantic: int
    sub_semantic_modes: list[str]
    n_surface_dominant: int
    surface_dominant_modes: list[str]


class ShuffleControlsResult(BaseModel):
    """Shuffle-control classifications for one compute feature set."""

    model_config = _FORBID

    within_topic_shuffle: ClassificationScore
    global_shuffle: ClassificationScore


class PerTierSemanticResult(BaseModel):
    """Semantic orthogonality battery for a single tier."""

    model_config = _FORBID

    n_features: int | None = None
    classification: SemanticClassifierBundle | None = None
    mantel_tfidf_cosine: MantelResult | None = None
    mantel_sbert_cosine: MantelResult | None = None
    text_to_compute_r2: TextToComputeR2 | None = None
    per_mode_surface_vs_compute: PerModeSurfaceVsComputeResult | None = None
    shuffle_controls: ShuffleControlsResult | None = None
    error: str | None = None


class RetrievalFeatureSet(BaseModel):
    """kNN mode-match retrieval output for one feature set."""

    model_config = _FORBID

    mean_mode_match: float
    std_mode_match: float
    per_mode: dict[str, float]
    k: int


class JaccardStats(BaseModel):
    """Jaccard overlap statistics between two kNN neighbour sets."""

    model_config = _FORBID

    mean: float
    std: float


class RetrievalResult(BaseModel):
    """kNN retrieval diagnostics across feature sets."""

    model_config = _FORBID

    compute_t2t25: RetrievalFeatureSet | None = None
    tfidf: RetrievalFeatureSet | None = None
    sbert: RetrievalFeatureSet | None = None
    combined_compute_sbert: RetrievalFeatureSet | None = None
    jaccard_compute_tfidf: JaccardStats | None = None
    jaccard_compute_sbert: JaccardStats | None = None


class ContrastiveProjectionComparisonEntry(BaseModel):
    """One feature-set entry in the contrastive projection comparison."""

    model_config = _FORBID

    test_knn_accuracy: float
    train_knn_accuracy: float
    train_test_gap: float
    silhouette: float | None
    fold_accs: list[float]


class ContrastiveProjectionComparisonResult(BaseModel):
    """Contrastive projection comparison across feature sets.

    The section either returns an error stub or populates one or more
    feature-set entries (tfidf, compute_t2t25, sbert, combined_compute_sbert).
    """

    model_config = _FORBID

    compute_t2t25: ContrastiveProjectionComparisonEntry | None = None
    tfidf: ContrastiveProjectionComparisonEntry | None = None
    sbert: ContrastiveProjectionComparisonEntry | None = None
    combined_compute_sbert: ContrastiveProjectionComparisonEntry | None = None
    error: str | None = None


class PromptSwapTierResult(BaseModel):
    """Per-tier prompt-swap confound outcome."""

    model_config = _FORBID

    n_features: int
    follows_system_prompt: int
    follows_execution: int
    follows_neither: int
    pct_system: float
    pct_execution: float
    signal_type: str
    per_swap_type: dict[str, Any]


class PromptSwapConfoundResult(BaseModel):
    """Prompt-swap confound diagnostics.

    All fields Optional because the canonical runs hit the error path
    (``{"error": "No prompt-swap samples found"}``).
    """

    model_config = _FORBID

    n_swap_samples: int | None = None
    swap_types: list[str] | None = None
    per_tier: dict[str, PromptSwapTierResult] | None = None
    error: str | None = None


class SemanticResult(BaseModel):
    """Section 9 result.

    Top-level error ``{"error": "No generated text available"}`` round-trips
    via the Optional fields. ``per_tier_semantic`` is only populated by
    v2 runs; older baseline snapshots omit it.
    """

    model_config = _FORBID

    tfidf_classification: SemanticClassifierBundle | None = None
    sbert_classification: SemanticClassifierBundle | None = None
    per_tier_semantic: dict[str, PerTierSemanticResult] | None = None
    compute_classification: SemanticClassifierBundle | None = None
    combined_classification: SemanticClassifierBundle | None = None
    semantic_noise_classification: SemanticClassifierBundle | None = None
    mantel_tfidf: MantelResult | None = None
    mantel_sbert: MantelResult | None = None
    mantel_tfidf_cosine: MantelResult | None = None
    mantel_sbert_cosine: MantelResult | None = None
    mantel_tfidf_euclidean: MantelResult | None = None
    mantel_sbert_euclidean: MantelResult | None = None
    text_to_compute_r2: TextToComputeR2 | None = None
    per_mode_surface_vs_compute: PerModeSurfaceVsComputeResult | None = None
    shuffle_controls: ShuffleControlsResult | None = None
    contrastive_projection_comparison: ContrastiveProjectionComparisonResult | None = None
    retrieval: RetrievalResult | None = None
    prompt_swap_confound: PromptSwapConfoundResult | None = None
    error: str | None = None
