import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location(
    "supplement", scripts / "extraction_perf_roster_supplement.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def evidence():
    rows = [
        dict(
            key=[str(i)],
            bank=np.array([i], dtype=np.float32),
            tokens=[i, 1, 2],
            start=1,
            end=3,
        )
        for i in range(3)
    ]
    selected = [rows[2], rows[0]]
    plan = dict(
        measurement_rows=2,
        required_measurements=[
            dict(
                key=r["key"],
                start=r["start"],
                end=r["end"],
                bank_vector_sha256=hashlib.sha256(r["bank"].tobytes()).hexdigest(),
                input_sha256=hashlib.sha256(
                    np.asarray(r["tokens"], dtype="<i8").tobytes()
                ).hexdigest(),
            )
            for r in selected
        ],
    )
    return rows, plan


def test_supplement_preserves_owner_order_not_reuse_plan_order():
    rows, plan = evidence()
    assert [r["key"] for r in module.select_supplement(rows, plan)] == [["0"], ["2"]]


@pytest.mark.parametrize(
    "field,value",
    [("tokens", [9, 1, 2]), ("start", 2), ("bank", np.array([9], dtype=np.float32))],
)
def test_changed_input_or_reference_rejected(field, value):
    rows, plan = evidence()
    rows[0][field] = value
    with pytest.raises(ValueError, match="input/reference changed"):
        module.select_supplement(rows, plan)


def test_missing_measurement_refused():
    rows, plan = evidence()
    with pytest.raises(ValueError, match="population"):
        module.select_supplement(rows[1:], plan)
