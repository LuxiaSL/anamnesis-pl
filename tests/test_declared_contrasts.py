import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "contrasts",
    Path(__file__).resolve().parents[1] / "scripts/extraction_perf_contrast_rules.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def k(cell):
    return ("bank", "doc", cell, 0, "h1")


def test_min_over_each_rows_declared_contrasts():
    keys = tuple(
        k(c)
        for c in (
            "own_cont",
            "paraphrase_own",
            "xfam_paraphrase",
            "rung_control",
            "null::own_cont",
        )
    )
    cpu = np.array([[0], [10], [12], [11], [0]], dtype=np.float32)
    gpu = cpu + np.float32(0.01)
    inputs = {key: str(i) for i, key in enumerate(keys)}
    inputs[keys[-1]] = inputs[keys[0]]
    pairs, distances, reasons, receipts = module.resolve_entries(
        module.declared_entries(keys),
        keys,
        cpu,
        gpu,
        gpu,
        np.ones(1),
        np.ones(1),
        inputs,
        dict.fromkeys(keys, (299, 427)),
    )
    assert distances[keys[0]] is None
    assert distances[keys[1]] == 1  # B-E beats F-B
    assert distances[keys[2]] == 1  # F-E beats F-B
    assert distances[keys[3]] == 1  # B-E beats F-E
    assert distances[keys[4]] is None and reasons[keys[4]]
    assert receipts[-1]["terms"][0]["omitted_exact_identical_input_zero"]


def test_missing_partners_are_recorded_without_dropping_row():
    keys = (k("rung_control"),)
    entries = module.declared_entries(keys)
    assert len(entries) == 1 and not entries[0]["contrasts"]
    assert len(entries[0]["unavailable_contrasts"]) == 2
    values = np.array([[1]], dtype=np.float32)
    _, distances, reasons, _ = module.resolve_entries(
        entries,
        keys,
        values,
        values,
        values,
        np.ones(1),
        np.ones(1),
        {keys[0]: "input"},
        {keys[0]: (1, 3)},
    )
    assert distances[keys[0]] is None and reasons[keys[0]]


def test_different_input_zero_gets_no_relief():
    keys = (k("own_cont"), k("null::own_cont"))
    values = np.zeros((2, 1), dtype=np.float32)
    _, distances, _, _ = module.resolve_entries(
        module.declared_entries(keys),
        keys,
        values,
        values + 0.01,
        values + 0.01,
        np.ones(1),
        np.ones(1),
        dict(zip(keys, ("one", "two"))),
        dict.fromkeys(keys, (1, 3)),
    )
    assert distances[keys[1]] == 0


def test_small_nonzero_is_not_an_epsilon_exemption():
    keys = (k("own_cont"), k("null::own_cont"))
    values = np.array([[0], [1e-12]], dtype=np.float32)
    _, distances, _, _ = module.resolve_entries(
        module.declared_entries(keys),
        keys,
        values,
        values,
        values,
        np.ones(1),
        np.ones(1),
        dict.fromkeys(keys, "same"),
        dict.fromkeys(keys, (1, 3)),
    )
    assert 0 < distances[keys[1]] < 1e-11
