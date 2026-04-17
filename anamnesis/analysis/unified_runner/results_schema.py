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

from pydantic import BaseModel, ConfigDict, Field

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
