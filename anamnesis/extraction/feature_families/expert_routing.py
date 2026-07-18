"""MoE expert-routing features (Source.expert_routing; prefix `xrt_`).

The fifth substrate in the source×method×depth map — read only on models with a
Mixture-of-Experts MLP (vmb arm A7, M6 = DeepSeek-V2-Lite class: 2 shared + 64
routed experts, top-6 greedy). Reads the per-generated-token router allocation
distribution (RawGenerationData.router_dist) and the shared/routed branch output
norms (RawGenerationData.router_branch_norms), both banked by the M6 capture
hooks. Dense models leave those fields None → this family returns empty (the
gate_features None-guard pattern), so it is safe to enable everywhere.

Per sampled MoE layer L (dense layers, e.g. DeepSeek layer 0, contribute NOTHING
— they have no router; the family skips any layer absent from router_dist, by
construction not by error), ~10 features over the generated-token span:

  Static (per-layer means / span summaries):
    xrt_L{L}_alloc_entropy_mean  — mean entropy of the per-token expert distribution
    xrt_L{L}_top1_margin_mean    — mean (top1 − top2) allocation weight   [magnitude]
    xrt_L{L}_shared_mass_mean    — mean ‖shared‖/(‖shared‖+‖routed‖)       [magnitude]
    xrt_L{L}_coverage            — unique experts selected over the span / n_experts
    xrt_L{L}_load_kl             — KL(selected-expert histogram ‖ uniform) over the span
  Dynamic (dispersion / change over generation time):
    xrt_L{L}_alloc_entropy_std   — std of per-token entropy
    xrt_L{L}_alloc_entropy_slope — linear slope of per-token entropy vs step
    xrt_L{L}_switch_rate         — P(top1 expert_t ≠ top1 expert_{t−1})
    xrt_L{L}_hist_drift          — JSD(first-half hist, second-half hist) of selections
    xrt_L{L}_shared_mass_std     — std of shared_mass

feature_map places every name (Source.expert_routing; margin/mass → magnitude,
rest → distributional; std/slope/switch/drift → dynamic). `FeatureMap.unclassified()`
== 0 on this family is the onboarding acceptance check (spec §3).

Selection semantics: under greedy top-k routing (DeepSeek-V2-Lite topk_method
"greedy") the selected experts are exactly argtop-k of the banked distribution, so
coverage/load/drift derive the selection in-module from router_dist. `top_k`
(num_experts_per_tok, 6 for DSV2-Lite) is a parameter; if a future MoE routes
non-greedily (group-limited) the banked distribution's argtop-k may diverge from
the model's actual selection — stamp and revisit (audit plan §R).

Pure numpy (no torch/model deps) — the state_extractor design constraint.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from anamnesis.extraction.feature_families import FeatureFamilyResult
from anamnesis.extraction.state_extractor import RawGenerationData

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]

# The ten feature suffixes emitted per MoE layer, in fixed order (so a short-gen
# layer emits zeros for the same names — the modal-vector guard expects fixed length).
_STATIC_SUFFIXES = (
    "alloc_entropy_mean",
    "top1_margin_mean",
    "shared_mass_mean",
    "coverage",
    "load_kl",
)
_DYNAMIC_SUFFIXES = (
    "alloc_entropy_std",
    "alloc_entropy_slope",
    "switch_rate",
    "hist_drift",
    "shared_mass_std",
)
N_FEATURES_PER_LAYER = len(_STATIC_SUFFIXES) + len(_DYNAMIC_SUFFIXES)


def _xrt_layer_names(layer_idx: int) -> list[str]:
    """Fixed-order feature names for one MoE layer."""
    prefix = f"xrt_L{layer_idx}"
    return [f"{prefix}_{s}" for s in (*_STATIC_SUFFIXES, *_DYNAMIC_SUFFIXES)]


def _entropy(p: F64) -> float:
    """Natural-log entropy of a distribution row (zeros safe)."""
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def _jsd(p: F64, q: F64) -> float:
    """Jensen–Shannon divergence (natural log; range [0, ln 2]). p, q are histograms."""
    ps, qs = p.sum(), q.sum()
    if ps <= 0 or qs <= 0:
        return 0.0
    p = p / ps
    q = q / qs
    m = 0.5 * (p + q)

    def _kl(a: F64, b: F64) -> float:
        mask = a > 0
        return float((a[mask] * np.log(a[mask] / b[mask])).sum())

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _slope(y: F64) -> float:
    """Linear-regression slope of y vs its index (0 if <2 points)."""
    if y.size < 2:
        return 0.0
    x = np.arange(y.size, dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def extract_expert_routing_features(
    data: RawGenerationData,
    sampled_layers: list[int] | None = None,
    top_k: int = 6,
) -> FeatureFamilyResult:
    """Extract MoE expert-routing statistics.

    Parameters
    ----------
    data : RawGenerationData
        Must have `router_dist` populated (per-token expert distribution) for MoE
        layers; `router_branch_norms` feeds the shared_mass features (its absence
        only zeros the two shared_mass features, not the rest).
    sampled_layers : list[int], optional
        Which layers to emit features for. Defaults to the M6 sampled set. Layers
        absent from `router_dist` (e.g. the dense layer 0) emit zeros for their
        names — fixed-length output for the modal-vector guard.
    top_k : int
        num_experts_per_tok — the selection size for coverage/load/drift under
        greedy routing (6 for DeepSeek-V2-Lite).
    """
    if sampled_layers is None:
        sampled_layers = [5, 11, 15, 18, 22, 26]

    # Dense model or capture not banked → empty family (safe-everywhere guard).
    if not data.router_dist:
        logger.info("No router_dist available — returning empty expert_routing features")
        return FeatureFamilyResult.empty("expert_routing")

    branch_norms = data.router_branch_norms or {}

    features: list[float] = []
    names: list[str] = []

    for l_idx in sampled_layers:
        layer_names = _xrt_layer_names(l_idx)
        dist_list = data.router_dist.get(l_idx)

        # Layer has no router (dense) or too short a span → fixed-length zeros.
        if not dist_list or len(dist_list) < 1:
            features.extend([0.0] * len(layer_names))
            names.extend(layer_names)
            continue

        # [T, n_experts] per-token distribution over routed experts.
        P = np.asarray(np.stack(dist_list), dtype=np.float64)
        if P.ndim != 2 or P.shape[0] == 0:
            features.extend([0.0] * len(layer_names))
            names.extend(layer_names)
            continue
        T, n_experts = P.shape
        k = int(min(max(top_k, 1), n_experts))

        # ── Per-token entropy time series ──
        entropy_ts = np.array([_entropy(P[t]) for t in range(T)], dtype=np.float64)

        # ── Per-token top1/top2 margin ──
        # partial sort: two largest per row (n_experts >= 2 always for a MoE)
        if n_experts >= 2:
            part = np.partition(P, -2, axis=1)
            top1 = part[:, -1]
            top2 = part[:, -2]
        else:
            top1 = P[:, -1]
            top2 = np.zeros(T, dtype=np.float64)
        margin_ts = top1 - top2

        # ── Selected experts per token (argtop-k) — greedy selection == argtop-k(dist) ──
        # argpartition gives the k largest indices per row (unordered, fine for sets/hist).
        sel = np.argpartition(P, -k, axis=1)[:, -k:]  # [T, k]
        top1_expert = np.argmax(P, axis=1)            # [T]

        # coverage: unique experts ever selected / n_experts
        coverage = float(np.unique(sel).size) / float(n_experts)

        # load_kl: KL(selection histogram ‖ uniform) over the whole span
        hist = np.bincount(sel.reshape(-1), minlength=n_experts).astype(np.float64)
        total = hist.sum()
        if total > 0:
            phist = hist / total
            uniform = np.full(n_experts, 1.0 / n_experts, dtype=np.float64)
            mask = phist > 0
            load_kl = float((phist[mask] * np.log(phist[mask] / uniform[mask])).sum())
        else:
            load_kl = 0.0

        # switch_rate: P(top1 changes step-to-step)
        if T >= 2:
            switch_rate = float(np.mean(top1_expert[1:] != top1_expert[:-1]))
        else:
            switch_rate = 0.0

        # hist_drift: JSD of first-half vs second-half selection histograms
        if T >= 2:
            half = T // 2
            if half >= 1:
                h1 = np.bincount(sel[:half].reshape(-1), minlength=n_experts).astype(np.float64)
                h2 = np.bincount(sel[half:].reshape(-1), minlength=n_experts).astype(np.float64)
                hist_drift = _jsd(h1, h2)
            else:
                hist_drift = 0.0
        else:
            hist_drift = 0.0

        # ── shared_mass time series (needs branch norms) ──
        bn = branch_norms.get(l_idx)
        if bn and len(bn) >= 1:
            B = np.asarray(np.stack(bn), dtype=np.float64)  # [T, 2] = (shared, routed)
            if B.ndim == 2 and B.shape[1] >= 2:
                denom = B[:, 0] + B[:, 1]
                shared_mass_ts = np.divide(
                    B[:, 0], denom,
                    out=np.zeros(B.shape[0], dtype=np.float64),
                    where=denom > 1e-12,
                )
            else:
                shared_mass_ts = np.zeros(T, dtype=np.float64)
        else:
            shared_mass_ts = np.zeros(T, dtype=np.float64)

        # ── Assemble in fixed suffix order ──
        vals = [
            float(entropy_ts.mean()),            # alloc_entropy_mean
            float(margin_ts.mean()),             # top1_margin_mean
            float(shared_mass_ts.mean()),        # shared_mass_mean
            coverage,                            # coverage
            load_kl,                             # load_kl
            float(entropy_ts.std()),             # alloc_entropy_std
            _slope(entropy_ts),                  # alloc_entropy_slope
            switch_rate,                         # switch_rate
            hist_drift,                          # hist_drift
            float(shared_mass_ts.std()),         # shared_mass_std
        ]
        features.extend(vals)
        names.extend(layer_names)

    return FeatureFamilyResult(
        features=np.array(features, dtype=np.float32),
        feature_names=names,
        family_name="expert_routing",
    )
