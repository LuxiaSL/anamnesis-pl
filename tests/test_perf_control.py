import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from test_perf_verify_sweep import sweep as sweep

spec = importlib.util.spec_from_file_location(
    "perf_control",
    Path(__file__).resolve().parents[1] / "scripts/extraction_perf_compare_control.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def control(sweep):
    cpu, gpu, sigma = sweep
    folder = cpu.parent / "control"
    shutil.copytree(cpu, folder)
    for path in (cpu, folder):
        pin = json.loads((path / "deployment.json").read_text())
        pin["source"] = {"reference.py": "a" * 64}
        if path == folder:
            pin.update(
                mode="cpu-control",
                workspace=":16:8",
                deterministic=True,
                preferred_blas_library="Cublaslt",
            )
        module.write_json(path / "deployment.json", pin)
    return SimpleNamespace(cpu=cpu, control=folder, gpu=gpu, sigma=sigma)


def test_control_records_backend_without_certifying(control):
    result = module.evaluate(control)
    assert result["certification"] is False
    assert result["arithmetic"][1]["preferred_blas_library"] == "Cublaslt"
    assert all(
        r["comparisons"]["original_vs_control"]["l2"] == 0 for r in result["rows"]
    )


def test_control_rejects_nonfinite(control):
    values = np.array([np.nan, 1], dtype=np.float32)
    np.savez(control.control / "row-00000.npz", features=values, repeat=values)
    with pytest.raises(ValueError, match="invalid control"):
        module.evaluate(control)


def test_control_rejects_incomplete_population(control):
    module.write_json(control.control / "complete.json", dict(rows=29))
    with pytest.raises(ValueError, match="population"):
        module.evaluate(control)
