import numpy as np
import torch

from anamnesis.extraction.gpu_features import GpuFeatureLane


def test_window_hit_and_miss_preserve_values_and_count_transfers():
    lane = object.__new__(GpuFeatureLane)
    lane.pm = np.arange(4 * 30 * 8, dtype=np.float32).reshape(4, 30, 8)
    lane.device = torch.device("cpu")
    lane.positional_cache_range = (10, 20)
    lane.positional_cache = torch.as_tensor(np.ascontiguousarray(lane.pm[1:, 10:20]))
    for start, steps, hit in (
        (10, 10, True),
        (11, 5, True),
        (9, 5, False),
        (18, 4, False),
    ):
        for layer in range(3):
            result, transferred = lane._positional_slice(layer, start, steps)
            expected = lane.pm[layer + 1, start : start + steps]
            assert result.numpy().tobytes() == expected.tobytes()
            assert transferred == (0 if hit else expected.nbytes)
