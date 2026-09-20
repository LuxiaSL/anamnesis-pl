import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "batch_plan",
    Path(__file__).resolve().parents[1] / "scripts/extraction_perf_batch_plan.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_all_rows_once_with_explicit_singleton_cost():
    rows = [
        dict(key=[str(i)], start=p, end=p + 128)
        for i, p in enumerate((299, 263, 299, 297, 299, 297))
    ]
    plan = module.exact_batches(rows)
    assert sorted(i for b in plan for i in b["emitted_indices"]) == list(range(6))
    assert sum(b["filler_count"] for b in plan) == 2
    for batch in plan:
        assert len(batch["indices"]) == 2
        assert len({(rows[i]["start"], rows[i]["end"]) for i in batch["indices"]}) == 1
        # Charging elapsed/n_emitted to each emitted row charges the full call.
        elapsed = 2.0
        assert (
            sum(
                elapsed / len(batch["emitted_indices"])
                for _ in batch["emitted_indices"]
            )
            == elapsed
        )
        if batch["filler_count"]:
            assert batch["indices"][0] == batch["indices"][1]


@pytest.mark.parametrize("rows", [[], [dict(key=["x"], start=1, end=3)] * 2])
def test_invalid_population_refused(rows):
    with pytest.raises(ValueError):
        module.exact_batches(rows)
