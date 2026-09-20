import importlib.util
from pathlib import Path
import sys

import pytest

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location(
    "native_trial", scripts / "extraction_perf_native_trial.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_length_extremes_and_median_are_fixed_before_measurement():
    entries = {
        str(i): dict(input_ids=[1] * n) for i, n in enumerate((20, 15, 30, 12, 16))
    }
    assert module.select_entries(entries) == ["3", "4", "2"]
    assert module.select_entries(dict(reversed(list(entries.items())))) == [
        "3",
        "4",
        "2",
    ]


def test_ties_are_ordered_by_generation_id():
    assert module.select_entries(
        {str(i): dict(input_ids=[1] * 10) for i in (10, 2, 1)}
    ) == ["1", "2", "10"]


def test_too_few_inputs_rejected():
    with pytest.raises(ValueError):
        module.select_entries({})
