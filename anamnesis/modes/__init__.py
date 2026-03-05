"""Mode definitions for extraction experiments.

Provides configurable mode sets for different experimental conditions:
  - run4: Original 5 modes (linear, analogical, socratic, contrastive, dialectical)
  - extended: Run 4 + adapted Run 3 modes (structured, compressed, associative) with format control
  - prompt_swap: Prompt-swap pairs for confound testing
"""

from __future__ import annotations

from .run4_modes import RUN4_MODES, RUN4_MODE_INDEX
from .extended_modes import EXTENDED_MODES, EXTENDED_MODE_INDEX
from .prompt_swap import PROMPT_SWAP_PAIRS, PromptSwapPair

__all__ = [
    "RUN4_MODES",
    "RUN4_MODE_INDEX",
    "EXTENDED_MODES",
    "EXTENDED_MODE_INDEX",
    "PROMPT_SWAP_PAIRS",
    "PromptSwapPair",
]
