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
