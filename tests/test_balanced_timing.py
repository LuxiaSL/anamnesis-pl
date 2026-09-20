import importlib.util
from pathlib import Path
import sys

import pytest

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location(
    "balanced_report", scripts / "extraction_perf_balanced_report.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def records():
    return [
        dict(
            batch=i,
            shape=[299, 128],
            keys=[str(i)],
            all_samples_retained=True,
            feature_bytes_unchanged=True,
            samples=[
                dict(
                    round=r,
                    order=["single", "batch"] if r % 2 == 0 else ["batch", "single"],
                    single_seconds_per_row=2.0,
                    batch_seconds_per_row=1.0,
                )
                for r in range(6)
            ],
        )
        for i in range(6)
    ]


def test_all_samples_and_orders_reported():
    result = module.summarize(records(), 6)
    assert result["all_samples"] == dict(
        n=36, single_mean=2.0, batch_mean=1.0, speedup=2.0
    )
    assert all(v["n"] == 18 for v in result["by_first_mode"].values())


def test_dropping_slow_round_rejected():
    data = records()
    data[0]["samples"].pop()
    with pytest.raises(ValueError, match="missing timing"):
        module.summarize(data, 6)


def test_unbalanced_order_rejected():
    data = records()
    data[0]["samples"][1]["order"] = ["single", "batch"]
    with pytest.raises(ValueError, match="order differs"):
        module.summarize(data, 6)
