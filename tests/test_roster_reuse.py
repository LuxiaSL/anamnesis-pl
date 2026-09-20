import importlib.util
from pathlib import Path
import sys

import pytest

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location(
    "roster_reuse", scripts / "extraction_perf_roster_reuse.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def evidence():
    row = dict(
        key=["doc", "B", 0, "h1"],
        bank_vector_sha256="bank",
        input_sha256="input",
        start=299,
        end=427,
    )
    record = dict(
        key=row["key"],
        telemetry=dict(input_tokens_sha256="input", span_start=299, span_end=427),
    )
    return row, record


def test_identical_inputs_and_reference_bytes_are_required():
    row, record = evidence()
    assert module.matches(row, record, "bank")
    assert not module.matches(row, record, "different bank")


@pytest.mark.parametrize(
    "field,value",
    [("input_tokens_sha256", "different"), ("span_start", 298), ("span_end", 426)],
)
def test_same_logical_id_is_not_sufficient(field, value):
    row, record = evidence()
    record["telemetry"][field] = value
    assert not module.matches(row, record, "bank")
