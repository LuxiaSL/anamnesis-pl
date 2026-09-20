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
    "verify_sweep", scripts / "extraction_perf_verify_sweep.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def sweep(tmp_path):
    cpu, gpu = tmp_path / "cpu", tmp_path / "gpu"
    sigma = tmp_path / "sigma.npz"
    np.savez(
        sigma,
        sigma_cal=np.ones(2),
        weights=np.ones(2),
        feature_names=np.array(["a", "b"]),
    )
    inputs = [
        dict(
            key=["synthetic", length, seed],
            start=length,
            end=length + 128,
            tokens=[seed] * (length + 128),
        )
        for length in module.LENGTHS
        for seed in range(5)
    ]
    for folder in (cpu, gpu):
        folder.mkdir()
        module.write_json(folder / "inputs.json", inputs)
        module.write_json(folder / "complete.json", dict(rows=30))
        pin = dict(
            inputs_sha256=module.sha(folder / "inputs.json"),
            bank_pin_sha256="a" * 64,
            calibration_sha256="b" * 64,
            standardizer_sha256=module.sha(sigma),
            caller={},
            torch="test",
            transformers="test",
            numpy="test",
            runner_sha256="c" * 64,
            mode=folder.name,
            deterministic=folder == gpu,
            tf32=False,
            workspace=":16:8" if folder == gpu else None,
        )
        module.write_json(folder / "deployment.json", pin)
        for i, row in enumerate(inputs):
            values = np.array([1, 2], dtype=np.float32)
            if folder == gpu:
                values += np.float32(1e-5)
            np.savez(folder / f"row-{i:05d}.npz", features=values, repeat=values)
            metadata = [
                dict(
                    fresh_cache=True,
                    input_tokens_sha256=hashlib.sha256(
                        np.asarray(row["tokens"], dtype="<i8").tobytes()
                    ).hexdigest(),
                    feature_schema_sha256=hashlib.sha256(
                        json.dumps(["a", "b"]).encode()
                    ).hexdigest(),
                    calibration_sha256="b" * 64,
                    replay_id=f"{i}/{repeat}",
                    lane_id="test",
                )
                for repeat in range(2)
            ]
            module.write_json(
                folder / f"row-{i:05d}.json",
                dict(
                    key=row["key"],
                    cache_length=row["start"],
                    independent_repeat_exact=True,
                    metadata=metadata,
                    inside_calibration=True,
                    all_feature_positions_count_gt5=True,
                    last_feature_position=row["start"] + 126,
                    seconds=[1, 1],
                ),
            )
    return cpu, gpu, sigma


def test_complete_sweep_is_evidence_not_automatic_certification(sweep):
    result = module.verify(*sweep)
    assert result["coverage_complete"] and result["per_feature_pass"]
    assert len(result["lengths"]) == 6
    assert result["certification"] is False


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("independent_repeat_exact", False, "replay mismatch"),
        ("cache_length", 999, "alignment"),
    ],
)
def test_bad_receipts_rejected(sweep, field, value, match):
    path = sweep[1] / "row-00000.json"
    record = json.loads(path.read_text())
    record[field] = value
    module.write_json(path, record)
    with pytest.raises(ValueError, match=match):
        module.verify(*sweep)


def test_reduction_reuse_is_not_independent_replay(sweep):
    path = sweep[1] / "row-00000.json"
    record = json.loads(path.read_text())
    record["metadata"][1]["replay_id"] = record["metadata"][0]["replay_id"]
    module.write_json(path, record)
    with pytest.raises(ValueError, match="reused replay"):
        module.verify(*sweep)


def test_changed_determinism_rejected(sweep):
    path = sweep[1] / "deployment.json"
    pin = json.loads(path.read_text())
    pin["deterministic"] = False
    module.write_json(path, pin)
    with pytest.raises(ValueError, match="deterministic"):
        module.verify(*sweep)


def test_agreeing_but_wrong_span_geometry_rejected(sweep):
    for folder in sweep[:2]:
        path = folder / "inputs.json"
        rows = json.loads(path.read_text())
        rows[0]["end"] -= 1
        module.write_json(path, rows)
        pin_path = folder / "deployment.json"
        pin = json.loads(pin_path.read_text())
        pin["inputs_sha256"] = module.sha(path)
        module.write_json(pin_path, pin)
    with pytest.raises(ValueError, match="span geometry"):
        module.verify(*sweep)
