"""Stage-0 floor estimation + the noise law (prereg-vmb-v1 §5, addendum 2026-07-12a).

Two floor types (ratified §1):
  - STOCHASTIC: matched-history, different-seed pair deltas within a prompt class
    (topic × task stratum). 80 classes × 10 seeds → C(10,2)=45 pairs/class →
    3,600 pairs/model.
  - FAITHFULNESS: replay-vs-replay deltas of an identical continuation, STRATIFIED
    (addendum item 3) into within-device (pinned repeats — pure replay determinism)
    and cross-device (operational jitter) components.

Delta metric (the exp11 template generalized over feature_map cells):
  per-feature robust z (median/MAD over the floor corpus) → per-cell delta of a
  pair = mean |z_i − z_j| over the cell's features. Whole-vector = the all-feature
  cell. Cells at three granularities: whole_vector, legacy family, source × band.

The law (§5, finalized by addendum item 1): for each (cell × model), n_min =
smallest n at which a two-sample comparison of delta distributions detects a
k=2× floor-median shift with power 0.9 at the effective per-test α. Published as
a TABLE over α_test ∈ {0.05, 0.01, 1e-3, 1e-4}; the ruled row at arm-prereg time
is α = 0.05 / m with m = that arm's pre-registered confirmatory cell count
(Bonferroni planning bound — conservative for BH). Battery n = 2× n_min
(A2 cells 4×). Shift definition (conservative reading, documented): the arm
delta's location exceeds the floor median by (k−1)×median — i.e. "the arm sits
at k× floor", the floor-ruler criterion — so effect size d = (k−1)·median/σ_floor.
A rank-test variant inflates n by 1/0.955 (Mann-Whitney ARE vs t).

Every emitted number is stamped (n, M, law, floor_type). No tier vocabulary.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import norm

from anamnesis.analysis.feature_map import FeatureMap
from anamnesis.analysis.battery.manifest import FloorType

logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]

#: α grid the law table is published over (addendum 2026-07-12a item 1).
ALPHA_GRID: tuple[float, ...] = (0.05, 0.01, 1e-3, 1e-4)

#: Mann-Whitney asymptotic relative efficiency vs the t-test (normal shift).
RANK_ARE: float = 0.955


class LawParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    k: float = Field(default=2.0, description="Floor-ruler multiple: arm sits at k× floor median.")
    power: float = 0.9
    alpha_grid: tuple[float, ...] = ALPHA_GRID
    shift_reading: str = Field(
        default="(k-1)*floor_median location shift (conservative; floor-ruler criterion)",
        description="Documented interpretation of 'detects a k× floor-median shift'.",
    )


class FloorCell(BaseModel):
    """One cell's floor distribution + its n_min row of the law table."""

    model_config = ConfigDict(frozen=True)

    cell: str                      # e.g. "whole_vector", "family:attention_flow", "source_band:attention|mid"
    model: str                     # M: which model this floor characterizes
    floor_type: FloorType
    n_pairs: int                   # n: paired deltas in the distribution
    n_features: int                # features aggregated by this cell
    median: float
    mad: float
    std: float
    q10: float
    q90: float
    n_min_by_alpha: dict[str, int]         # α_test (str) → n_min per group (t-approx)
    n_min_by_alpha_rank: dict[str, int]    # rank-test variant (ARE-inflated)
    effect_d: float                # (k−1)·median / σ_floor
    law: LawParams


class FloorReport(BaseModel):
    """The Stage-0 deliverable for one (model × floor_type): floors + law table."""

    model: str
    floor_type: FloorType
    n_gens: int
    n_pairs_total: int
    corpus: str                    # provenance path
    law: LawParams
    cells: list[FloorCell]
    notes: list[str] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        logger.info(f"floor report → {path} ({len(self.cells)} cells)")


# ---------------------------------------------------------------------------- loading

def load_signature_matrix(sig_dir: Path) -> tuple[F32, list[str], list[int]]:
    """Load gen_*.npz signatures → (X [n, d], feature_names, gen_ids). Skips empty/failed."""
    paths = sorted(sig_dir.glob("gen_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if not paths:
        raise FileNotFoundError(f"no gen_*.npz under {sig_dir}")
    rows: list[F32] = []
    gen_ids: list[int] = []
    names: Optional[list[str]] = None
    for p in paths:
        z = np.load(p, allow_pickle=True)
        if "features" not in z:
            logger.warning(f"{p.name}: no features key — skipped")
            continue
        f = np.asarray(z["features"], dtype=np.float32)
        if names is None:
            names = [str(x) for x in z["feature_names"]]
        elif len(f) != len(names):
            raise ValueError(f"{p.name}: feature length {len(f)} != {len(names)}")
        rows.append(f)
        gen_ids.append(int(p.stem.split("_")[1]))
    if names is None:
        raise ValueError(f"no usable signatures under {sig_dir}")
    return np.stack(rows), names, gen_ids


def load_class_labels(metadata_path: Path) -> dict[int, tuple[int, str]]:
    """gen_id → (topic_idx, stratum) from a Stage-0 run's metadata.json."""
    md = json.loads(metadata_path.read_text())
    gens = md["generations"] if isinstance(md, dict) and "generations" in md else md
    return {int(g["generation_id"]): (int(g["topic_idx"]), str(g["mode"])) for g in gens}


# ---------------------------------------------------------------------------- deltas

def robust_scale(X: F32) -> tuple[F32, F32]:
    """Per-feature (median, MAD·1.4826) over the floor corpus; degenerate MAD → std → 1."""
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826
    std = X.std(axis=0)
    scale = np.where(mad > 1e-9, mad, np.where(std > 1e-9, std, 1.0))
    return med.astype(np.float32), scale.astype(np.float32)


def build_cells(names: list[str], n_layers: int) -> dict[str, NDArray]:
    """Cell name → boolean feature mask, at the three floor granularities."""
    fm = FeatureMap(names, n_layers)
    cells: dict[str, NDArray] = {"whole_vector": np.ones(len(names), dtype=bool)}
    for i, t in enumerate(fm.tags):
        fam_key = f"family:{t.family}"
        cells.setdefault(fam_key, np.zeros(len(names), dtype=bool))[i] = True
        if t.band is not None:
            sb_key = f"source_band:{t.source.value}|{t.band.value}"
            cells.setdefault(sb_key, np.zeros(len(names), dtype=bool))[i] = True
    return cells


def pair_deltas_by_class(
    Z: F32,
    gen_ids: list[int],
    labels: dict[int, tuple[int, str]],
    cells: dict[str, NDArray],
) -> dict[str, F32]:
    """All same-class pair deltas per cell: mean |Δz| over the cell's features."""
    by_class: dict[tuple[int, str], list[int]] = {}
    for row, gid in enumerate(gen_ids):
        if gid not in labels:
            logger.warning(f"gen {gid} missing from metadata — excluded from pairing")
            continue
        by_class.setdefault(labels[gid], []).append(row)

    out: dict[str, list[float]] = {c: [] for c in cells}
    for _cls, rows in sorted(by_class.items()):
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                dz = np.abs(Z[rows[i]] - Z[rows[j]])
                for cname, mask in cells.items():
                    out[cname].append(float(dz[mask].mean()))
    return {c: np.asarray(v, dtype=np.float32) for c, v in out.items()}


# ---------------------------------------------------------------------------- the law

def n_min_for(effect_d: float, alpha: float, power: float) -> int:
    """Two-sample normal-approximation n per group for a standardized shift d."""
    if effect_d <= 0:
        return 10**9  # degenerate: no detectable shift definition → effectively unpowerable
    za = norm.ppf(1.0 - alpha / 2.0)
    zb = norm.ppf(power)
    return max(2, math.ceil(2.0 * ((za + zb) / effect_d) ** 2))


def floor_cell_from_deltas(
    cell: str,
    deltas: F32,
    n_features: int,
    model: str,
    floor_type: FloorType,
    law: LawParams,
) -> FloorCell:
    med = float(np.median(deltas))
    mad = float(np.median(np.abs(deltas - med)) * 1.4826)
    std = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
    effect = ((law.k - 1.0) * med / std) if std > 1e-12 else 0.0
    n_by_a = {str(a): n_min_for(effect, a, law.power) for a in law.alpha_grid}
    n_by_a_rank = {a: min(10**9, math.ceil(n / RANK_ARE)) for a, n in n_by_a.items()}
    return FloorCell(
        cell=cell, model=model, floor_type=floor_type,
        n_pairs=len(deltas), n_features=n_features,
        median=med, mad=mad, std=std,
        q10=float(np.quantile(deltas, 0.10)), q90=float(np.quantile(deltas, 0.90)),
        n_min_by_alpha=n_by_a, n_min_by_alpha_rank=n_by_a_rank,
        effect_d=float(effect), law=law,
    )


# ---------------------------------------------------------------------------- entry points

def compute_stochastic_floors(
    sig_dir: Path,
    metadata_path: Path,
    model: str,
    n_layers: int,
    law: LawParams | None = None,
) -> FloorReport:
    """The Stage-0 stochastic-floor pipeline for one model (§5)."""
    law = law or LawParams()
    X, names, gen_ids = load_signature_matrix(sig_dir)
    labels = load_class_labels(metadata_path)
    med, scale = robust_scale(X)
    Z = (X - med) / scale
    cells = build_cells(names, n_layers)
    deltas = pair_deltas_by_class(Z, gen_ids, labels, cells)
    fcells = [
        floor_cell_from_deltas(c, d, int(cells[c].sum()), model, FloorType.stochastic, law)
        for c, d in deltas.items() if len(d) > 0
    ]
    return FloorReport(
        model=model, floor_type=FloorType.stochastic,
        n_gens=len(gen_ids), n_pairs_total=len(deltas["whole_vector"]),
        corpus=str(sig_dir), law=law, cells=fcells,
        notes=[
            "bare floors (no system prompt) — planning estimates; arm-time within-condition "
            "variance >~1.5x these floors triggers a flag + n top-up (addendum item 2)",
        ],
    )


def compute_faithfulness_floors(
    sig_dir: Path,
    replay_index_path: Path,
    model: str,
    n_layers: int,
    law: LawParams | None = None,
    scale_from: tuple[F32, F32] | None = None,
) -> list[FloorReport]:
    """Faithfulness floors, stratified within/cross-device (addendum item 3).

    replay_index_path: JSON list of {"sig": "gen_XXX", "continuation_id": int,
    "replay_idx": int, "device": str} — written by the Stage-0 replay driver.
    Standardization: pass scale_from=(median, scale) FROM THE STOCHASTIC FLOOR
    CORPUS so faithfulness deltas live on the same z scale as everything else.
    """
    law = law or LawParams()
    X, names, gen_ids = load_signature_matrix(sig_dir)
    index = {e["sig"]: e for e in json.loads(replay_index_path.read_text())}
    if scale_from is None:
        med, scale = robust_scale(X)
        logger.warning("faithfulness floors standardized on their OWN corpus — pass the "
                       "stochastic-floor scale for cross-floor comparability")
    else:
        med, scale = scale_from
    Z = (X - med) / scale
    cells = build_cells(names, n_layers)

    by_cont: dict[int, list[tuple[int, str]]] = {}   # continuation → [(row, device)]
    for row, gid in enumerate(gen_ids):
        e = index.get(f"gen_{gid:03d}")
        if e is None:
            logger.warning(f"replay sig gen_{gid:03d} missing from index — skipped")
            continue
        by_cont.setdefault(int(e["continuation_id"]), []).append((row, str(e["device"])))

    within: dict[str, list[float]] = {c: [] for c in cells}
    cross: dict[str, list[float]] = {c: [] for c in cells}
    for _cont, rows in sorted(by_cont.items()):
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                (ri, di), (rj, dj) = rows[i], rows[j]
                dz = np.abs(Z[ri] - Z[rj])
                bucket = within if di == dj else cross
                for cname, mask in cells.items():
                    bucket[cname].append(float(dz[mask].mean()))

    reports: list[FloorReport] = []
    for ftype, bucket in (
        (FloorType.faithfulness_within_device, within),
        (FloorType.faithfulness_cross_device, cross),
    ):
        fcells = [
            floor_cell_from_deltas(c, np.asarray(v, dtype=np.float32), int(cells[c].sum()),
                                   model, ftype, law)
            for c, v in bucket.items() if len(v) > 0
        ]
        n_pairs = len(bucket["whole_vector"]) if bucket["whole_vector"] else 0
        reports.append(FloorReport(
            model=model, floor_type=ftype,
            n_gens=len(gen_ids), n_pairs_total=n_pairs,
            corpus=str(sig_dir), law=law, cells=fcells,
            notes=["arm-time floor-ruling uses the component matching how that cell's "
                   "pairs were actually scheduled (addendum item 3)"],
        ))
    return reports
