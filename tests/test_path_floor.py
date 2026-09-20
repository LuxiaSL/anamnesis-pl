import runpy
from pathlib import Path

import numpy as np
import pytest

from anamnesis.config import ExtractionConfig
from anamnesis.extraction.path_floor import (
    first_position_coordinates,
    first_incremental_coordinates,
    coordinate_lower_bound,
)
from anamnesis.extraction.state_extractor import RawGenerationData, extract_tier1
from test_gpu_replay import tiny_loaded


@pytest.mark.parametrize(
    "steps,width", [(1, 128), (2, 3072), (127, 4096), (127, 8192), (512, 128)]
)
def test_first_coordinates_exactly_match_full_reference(steps, width):
    rng = np.random.default_rng(14)
    hidden = rng.normal(size=(steps, 4, width)).astype(np.float32)
    pm = rng.normal(0, 0.01, size=(4, steps + 20, width)).astype(np.float32)
    cfg = ExtractionConfig(sampled_layers=[0, 1, 2])
    raw = RawGenerationData(
        hidden_states=list(hidden),
        attentions=[],
        logits=[np.zeros(5, dtype=np.float32) for _ in range(steps)],
        chosen_token_ids=np.zeros(steps, dtype=np.float32),
        pre_rope_keys={},
        prompt_length=11,
        positional_means=pm,
    )
    full, names = extract_tier1(raw, cfg)
    values, selected = first_position_coordinates(hidden[0], pm, 11, cfg)
    expected = full[[names.index(n) for n in selected]]
    assert values.tobytes() == expected.tobytes()


def test_first_forward_matches_full_incremental_replay():
    helper = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/extraction_perf_phase0.py")
    )
    loaded = tiny_loaded()
    tokens = np.random.default_rng(92).integers(0, 64, size=29).tolist()
    pm = np.zeros((4, 29, 32), dtype=np.float32)
    cfg = ExtractionConfig(sampled_layers=[0, 1, 2])
    raw = helper["incremental_raw"](loaded, dict(tokens=tokens, start=11, end=29), pm)
    full, names = extract_tier1(raw, cfg)
    first, selected = first_incremental_coordinates(loaded, tokens, 11, 29, pm, cfg)
    assert first.tobytes() == full[[names.index(n) for n in selected]].tobytes()
    assert not loaded.hook_state.pre_rope_keys


def test_lower_bound_never_exceeds_full_distance():
    rng = np.random.default_rng(19)
    for _ in range(100):
        a = rng.normal(size=4086).astype(np.float32)
        b = rng.normal(size=4086).astype(np.float32)
        sigma = np.exp(rng.normal(size=4086))
        weights = np.ones(4086)
        indices = rng.choice(4086, size=80, replace=False)
        bound = coordinate_lower_bound(
            a[indices], b[indices], sigma[indices], weights[indices]
        )
        assert (
            0
            <= bound["floor_b_lower_bound"]
            <= np.linalg.norm((a.astype(float) - b) * weights / sigma)
        )
    bound = coordinate_lower_bound(a, a, np.ones(4086), weights)
    assert bound["floor_b_lower_bound"] == 0
