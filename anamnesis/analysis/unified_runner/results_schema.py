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

from typing import Any

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
