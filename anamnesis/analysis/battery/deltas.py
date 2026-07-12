"""Paired-delta construction (§6b) — typed signatures; implementation lands with Wave 1.

The unit of analysis is the PAIRED DELTA, never raw signature position (§1).
Pairing rules are declared per floor type:
  - stochastic cells: matched-history pairs (same prompt class + condition, different seed)
  - matched-token (replay) cells: perturbed-replay vs native-replay of the SAME continuation
Per-feature standardization uses the Stage-0 floor scale (median/MAD of the model's
stochastic floor corpus) so arm deltas and floors share one z space.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from anamnesis.analysis.battery.manifest import BatteryCell

F32 = NDArray[np.float32]


class PairedDelta(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    cell_id: str
    pair: tuple[int, int]            # gen ids
    dz: object                       # per-feature |Δz| vector (np.ndarray)


def build_paired_deltas(
    cell: BatteryCell,
    sig_dir_a: Path,
    sig_dir_b: Path | None,
    floor_scale: tuple[F32, F32],
) -> list[PairedDelta]:
    """Construct the cell's paired deltas under its declared pairing rule.

    Wave-1 implementation. Stage 0 uses the specialized floor paths in
    floors.py (pair_deltas_by_class / the faithfulness stratification).
    """
    raise NotImplementedError("Wave-1: arm paired-delta construction (prereg §6b deltas.py)")
