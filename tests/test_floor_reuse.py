import importlib.util
from pathlib import Path
import sys

import pytest

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location(
    "verify_bank", scripts / "extraction_perf_verify_bank.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def evidence():
    cpu = dict(anamnesis={"cpu.py": "a"}, caller={"caller.py": "b"})
    adaptive = dict(
        source=cpu["anamnesis"],
        caller=cpu["caller"],
        gpu_deployment_sha256="origin",
        helper_sha256="helper",
        standardizer_sha256="sigma",
    )
    origin = dict(
        calibration_sha256="cal",
        standardizer_sha256="sigma",
        reference_files={"bank": "c"},
        caller=cpu["caller"],
        source={"gpu.py": "old"},
    )
    candidate = dict(origin, source={"gpu.py": "new"})
    return adaptive, cpu, candidate, origin, "origin", "helper"


def test_changed_candidate_can_reuse_unchanged_cpu_path_bound():
    module.validate_floor_origin(*evidence())


@pytest.mark.parametrize(
    "field", ["calibration_sha256", "standardizer_sha256", "reference_files", "caller"]
)
def test_changed_reference_cannot_reuse_bound(field):
    args = list(evidence())
    args[2][field] = "changed"
    with pytest.raises(ValueError, match="reference mismatch"):
        module.validate_floor_origin(*args)


def test_wrong_original_deployment_refused():
    args = list(evidence())
    args[4] = "not-origin"
    with pytest.raises(ValueError, match="origin provenance"):
        module.validate_floor_origin(*args)
