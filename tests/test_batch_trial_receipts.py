import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location(
    "batch_trial", scripts / "extraction_perf_batch_trial.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_numpy_comparisons_are_json_safe_without_loosening_gate():
    for distance, bound, error, expected in (
        (0.01, 0.1, 0.01, True),
        (0.11, 0.1, 0.01, False),
        (0.01, 0.1, 0.11, False),
    ):
        value = module.feature_gate(
            np.float64(distance), np.float64(bound), np.float32(error)
        )
        assert type(value) is bool
        assert json.loads(json.dumps({"G1": value}))["G1"] is expected
