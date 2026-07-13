"""Shared A5 (activation-write) helpers: on-policy agreement + dosing norms.

Used by the incremental-decoding smoke, the matched-token pilot gate (C§3
>=0.85 top-1 agreement), and the cell chains. Kept script-side (not extraction/)
because everything here is a thin numpy/torch utility over banked artifacts.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


@torch.no_grad()
def teacher_forced_agreement(
    model: Any,
    input_ids: list[int] | torch.Tensor,
    prompt_length: int,
) -> float:
    """Top-1 agreement between the model's next-token argmax and a forced continuation.

    One full-sequence forward (any registered write hooks fire; injection specs
    should already carry start_pos=prompt_length). Token at position t is
    predicted by logits at t-1; the first generated token is predicted from the
    last prompt position (pre-injection — consistent with generation-side
    semantics where injection starts at the first generated position).

    This is the C§3 on-policy gate primitive: forced tokens = a banked
    continuation; agreement = how on-policy those tokens are for the (steered)
    model. Returns fraction in [0, 1].
    """
    device = next(model.parameters()).device
    ids = torch.as_tensor(input_ids, dtype=torch.long, device=device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    P = int(prompt_length)
    L = int(ids.shape[1])
    if not 0 < P < L:
        raise ValueError(f"prompt_length {P} out of range for sequence length {L}")
    out = model(ids, use_cache=False, return_dict=True)
    preds = out.logits[0, P - 1:L - 1].argmax(dim=-1)   # predicts tokens at P..L-1
    targets = ids[0, P:L]
    return float((preds == targets).float().mean().item())


@torch.no_grad()
def median_residual_norm(
    model: Any,
    input_ids: list[int] | torch.Tensor,
    prompt_length: int,
    layer_idx: int,
) -> float:
    """Median ||h|| entering decoder layer `layer_idx` over GENERATED positions.

    The C§3 alpha unit: hidden_states output index layer_idx equals the input
    to decoder layer layer_idx (hidden_states[0] = embeddings = input to layer 0).
    Computed on unsteered forwards of banked continuations; the vector bank
    stamps the value used per site so every absolute alpha is reconstructible.
    """
    device = next(model.parameters()).device
    ids = torch.as_tensor(input_ids, dtype=torch.long, device=device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    P = int(prompt_length)
    out = model(ids, use_cache=False, output_hidden_states=True, return_dict=True)
    h = out.hidden_states[layer_idx][0, P:]  # input to layer layer_idx, generated positions
    return float(h.float().norm(dim=-1).median().item())


def load_vector(npz_path: str, key: str) -> np.ndarray:
    """Load one unit vector from a banked vector npz (unit-normalized on save)."""
    bank = np.load(npz_path)
    if key not in bank:
        raise KeyError(f"vector key {key!r} not in {npz_path} (has {list(bank.keys())})")
    v = bank[key].astype(np.float32)
    n = float(np.linalg.norm(v))
    if not 0.99 < n < 1.01:
        logger.warning(f"vector {key} in {npz_path} has norm {n:.4f} (expected unit)")
    return v
