"""Extended data loader for the unified analysis runner.

Wraps the geometric trio data_loader and adds generated text loading.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..geometric_trio.data_loader import (
    Run4Data,
    SampleMeta,
    TIER_KEYS,
    TIER_GROUPS,
    load_run4,
    check_data_quality,
)


@dataclass
class AnalysisData:
    """Extended data container with optional text fields."""

    run4: Run4Data
    run_name: str
    generated_texts: list[str] | None = None
    system_prompts: list[str] | None = None
    user_prompts: list[str] | None = None
    generation_lengths: NDArray[np.int64] | None = None

    # Delegate common accessors
    @property
    def n_samples(self) -> int:
        return self.run4.n_samples

    @property
    def modes(self) -> NDArray:
        return self.run4.modes

    @property
    def topics(self) -> NDArray:
        return self.run4.topics

    @property
    def unique_modes(self) -> list[str]:
        return self.run4.unique_modes

    @property
    def unique_topics(self) -> list[str]:
        return self.run4.unique_topics

    def get_tier(self, name: str) -> NDArray[np.float32]:
        return self.run4.get_tier(name)

    def mode_mask(self, mode: str) -> NDArray[np.bool_]:
        return self.run4.mode_mask(mode)

    def topic_mask(self, topic: str) -> NDArray[np.bool_]:
        return self.run4.topic_mask(topic)


def load_analysis_data(
    signature_dir: Path | str,
    run_name: str,
    core_only: bool = True,
    load_text: bool = True,
    addon_dirs: list[Path | str] | None = None,
    mode_filter: list[str] | None = None,
) -> AnalysisData:
    """Load signature data with optional text fields for semantic analysis.

    Parameters
    ----------
    signature_dir : Path
        Directory containing gen_NNN.npz and gen_NNN.json files.
    run_name : str
        Label for this run (e.g. "8b_baseline").
    core_only : bool
        If True, load only one rep per topic-mode pair.
    load_text : bool
        If True, also load generated text from JSON metadata.
    addon_dirs : list[Path], optional
        Additional directories with features_* arrays to merge.
    mode_filter : list[str], optional
        If provided, only include samples whose mode is in this list.
    """
    sig_dir = Path(signature_dir)
    run4 = load_run4(
        signature_dir=sig_dir,
        core_only=core_only,
        addon_dirs=addon_dirs,
        mode_filter=mode_filter,
    )

    generated_texts: list[str] | None = None
    system_prompts: list[str] | None = None
    user_prompts: list[str] | None = None
    gen_lengths: NDArray[np.int64] | None = None

    if load_text:
        texts = []
        sys_prompts = []
        usr_prompts = []
        lengths = []

        for sample in run4.samples:
            json_path = sig_dir / f"{sample.file_stem}.json"
            if json_path.exists():
                with open(json_path) as f:
                    meta = json.load(f)
                texts.append(meta.get("generated_text", ""))
                sys_prompts.append(meta.get("system_prompt", ""))
                usr_prompts.append(meta.get("user_prompt", ""))
                lengths.append(meta.get("num_generated_tokens", 0))
            else:
                texts.append("")
                sys_prompts.append("")
                usr_prompts.append("")
                lengths.append(0)

        generated_texts = texts
        system_prompts = sys_prompts
        user_prompts = usr_prompts
        gen_lengths = np.array(lengths, dtype=np.int64)

    return AnalysisData(
        run4=run4,
        run_name=run_name,
        generated_texts=generated_texts,
        system_prompts=system_prompts,
        user_prompts=user_prompts,
        generation_lengths=gen_lengths,
    )
