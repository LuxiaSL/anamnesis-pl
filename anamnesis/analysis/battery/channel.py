"""Channel decomposition (§1, fourth readout) — Wave-1 implementation.

Every arm: compare the matched-token delta (replay channel; deformation at fixed
tokens = DIRECT component) against the free-generation delta; the remainder is
the TOKEN-MEDIATED component. Structural predictions live in §2c (A1 = 100%
token-mediated — matched-token delta AT faithfulness floor, parameter-free;
A4/A5 have direct components by construction). Output: the map's channel column.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from anamnesis.analysis.battery.stats import StampedValue


class ChannelSplit(BaseModel):
    model_config = ConfigDict(frozen=True)

    cell_id: str
    direct: StampedValue           # matched-token delta vs faithfulness floor
    token_mediated: StampedValue   # free-gen delta minus the direct component
    direct_at_floor: bool          # True = direct component indistinguishable from floor


def decompose_channel(matched_token_deltas: object, free_gen_deltas: object) -> ChannelSplit:
    raise NotImplementedError("Wave-1: channel decomposition (prereg §6b channel.py)")
