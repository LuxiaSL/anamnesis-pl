import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location(
    "verify_batch", scripts / "extraction_perf_verify_batch_trial.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def pilot(tmp_path):
    folder = tmp_path / "pilot"
    folder.mkdir()
    sigma = tmp_path / "sigma.npz"
    np.savez(
        sigma,
        sigma_cal=np.ones(2),
        weights=np.ones(2),
        feature_names=np.array(["a", "b"]),
    )
    keys = [[f"d{i}", "F", 0, "h1"] for i in range(12)]
    rulers = tmp_path / "rulers.json"
    module.write_json(
        rulers,
        dict(vectors=dict(rows=[dict(key=["bank", *key], bound=0.05) for key in keys])),
    )
    module.write_json(
        folder / "deployment.json",
        dict(
            batch_size=2,
            standardizer_sha256=module.sha(sigma),
            rulers_sha256=module.sha(rulers),
            calibration_sha256="a" * 64,
            selected=[dict(key=k) for k in keys],
        ),
    )
    module.write_json(folder / "complete.json", dict(rows=12))
    module.write_json(
        folder / "process-vram.json",
        dict(
            errors=[],
            phases={
                p: dict(samples=1, peak_bytes=134 * 2**30)
                for p in ("warmup", "batch", "batch_repeat")
            },
        ),
    )
    schema = hashlib.sha256(json.dumps(["a", "b"]).encode()).hexdigest()
    for i, key in enumerate(keys):
        values = np.array([1, 2], dtype=np.float32)
        np.savez(
            folder / f"row-{i:05d}.npz",
            gpu=values,
            repeat=values,
            bank=values,
            single=values,
        )

        def meta(repeat):
            return dict(
                fresh_cache=True,
                batch_size=2,
                feature_schema_sha256=schema,
                calibration_sha256="a" * 64,
                replay_id=f"row{i}/{repeat}",
                lane_id="candidate",
                batch_invocation_id=f"batch{i // 2}/{repeat}",
                batch_row=i % 2,
                batch_composition_sha256=str(i // 2),
            )

        module.write_json(
            folder / f"row-{i:05d}.json",
            dict(
                key=key,
                telemetry=meta(0),
                repeat_telemetry=meta(1),
                batch_seconds=1.0,
                batch_repeat_seconds=1.0,
                seconds_per_row=0.5,
                single_seconds=1.0,
                G0=True,
                G1=True,
            ),
        )
    return folder, sigma, rulers


def test_pilot_is_not_full_certification(pilot):
    result = module.verify(*pilot)
    assert result["pilot_vector_memory_checks_pass"]
    assert result["paired_speedup"] == 2
    assert result["certification"] is False


def test_pass_flags_do_not_override_bad_arrays(pilot):
    folder = pilot[0]
    a = np.array([1, 2], dtype=np.float32)
    np.savez(folder / "row-00000.npz", gpu=a + 0.2, repeat=a + 0.2, bank=a, single=a)
    result = module.verify(*pilot)
    assert not result["pilot_vector_memory_checks_pass"]
    assert not result["rows"][0]["G1"]


def test_g0_failure_stops_g1(pilot):
    folder = pilot[0]
    a = np.array([1, 2], dtype=np.float32)
    np.savez(folder / "row-00000.npz", gpu=a, repeat=a + 0.01, bank=a, single=a)
    result = module.verify(*pilot)
    assert not result["rows"][0]["G0"]
    assert result["rows"][0]["G1"] is None


def test_incomplete_pilot_refused(pilot):
    module.write_json(pilot[0] / "complete.json", dict(rows=6))
    with pytest.raises(ValueError, match="incomplete"):
        module.verify(*pilot)


def test_prefill_policy_mismatch_rejected(pilot):
    path = pilot[0] / "deployment.json"
    pin = json.loads(path.read_text())
    pin["prefill_policy"] = "independent"
    module.write_json(path, pin)
    with pytest.raises(ValueError, match="metadata mismatch"):
        module.verify(*pilot)
