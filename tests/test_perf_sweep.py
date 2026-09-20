"""The synthetic G3 population is predeclared, complete and reproducible."""

import importlib.util
from pathlib import Path
import sys

import pytest

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location(
    "perf_sweep", scripts / "extraction_perf_sweep.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_sweep_population():
    bank = [
        dict(
            key=[str(i), "own_cont", 0, "h1"],
            tokens=[i, i + 1, i + 2],
            end=3,
            excluded=False,
        )
        for i in range(5)
    ]
    rows = module.sweep_rows(bank)
    assert rows == module.sweep_rows(bank)
    assert len(rows) == 30
    assert len({tuple(r["key"]) for r in rows}) == 30
    for length in module.LENGTHS:
        group = [r for r in rows if r["start"] == length]
        assert len(group) == 5
        assert len({tuple(r["tokens"]) for r in group}) == 5
        assert all(len(r["tokens"]) == r["end"] == length + 128 for r in group)
    assert max(r["end"] - 2 for r in rows) == 2174


def test_missing_seeds_rejected():
    with pytest.raises(ValueError):
        module.sweep_rows([])
