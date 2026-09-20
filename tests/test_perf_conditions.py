"""Ruled comparators must not silently become cell-minus-anchor."""

import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "perf_conditions", SCRIPTS / "extraction_perf_conditions.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def key(cell, branch=0):
    return ("rubric-140", "doc", cell, branch, "h1")


def test_ruled_comparators():
    keys = [
        key(c)
        for c in (
            "own_cont",
            "paraphrase_own",
            "xfam_paraphrase",
            "rung_control",
            "paraphrase_own_rep2",
            "xfam_paraphrase_rep2",
        )
    ]
    keys += [key("null::own_cont", i) for i in range(3)]
    rows = {tuple(r["key"]): r for r in module.condition_entries(keys)}
    assert rows[key("own_cont")]["contrast_status"] == "null-by-rule"
    assert rows[key("own_cont")]["reason"]
    for suffix in ("", "_rep2"):
        for cell in ("paraphrase_own", "xfam_paraphrase"):
            row = rows[key(cell + suffix)]
            assert row["left"] == list(key("xfam_paraphrase" + suffix))
            assert row["right"] == list(key("paraphrase_own" + suffix))
    for i in range(3):
        row = rows[key("null::own_cont", i)]
        assert row["left"] == list(key("null::own_cont", i))
        assert row["right"] == list(key("own_cont"))


@pytest.mark.parametrize(
    "keys",
    [
        [key("unruled")],
        [key("paraphrase_own")],
        [key("own_cont"), key("own_cont")],
        [key("paraphrase_own_repbad")],
    ],
)
def test_invalid_population_rejected(keys):
    with pytest.raises(ValueError):
        module.condition_entries(keys)
