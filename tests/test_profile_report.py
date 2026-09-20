import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import extraction_perf_profile_report as report


def test_nested_host_scopes_are_not_double_counted(tmp_path, monkeypatch):
    events = [
        dict(cat="cpu_op", ph="X", tid=1, ts=0, dur=100, name="op"),
        dict(
            cat="user_annotation",
            ph="X",
            tid=1,
            ts=10,
            dur=80,
            name="extraction::forward",
        ),
        dict(
            cat="user_annotation",
            ph="X",
            tid=1,
            ts=30,
            dur=20,
            name="extraction::attention",
        ),
        dict(cat="kernel", ph="X", tid=7, ts=10, dur=90, name="kernel"),
    ]
    (tmp_path / "trace.json").write_text(json.dumps(dict(traceEvents=events)))
    (tmp_path / "row-00000.json").write_text(
        json.dumps(dict(seconds_per_row=1, G0=True))
    )
    out = tmp_path / "report.json"
    monkeypatch.setattr(
        sys, "argv", ["report", "--capture", str(tmp_path), "--out", str(out)]
    )
    report.main()
    result = json.loads(out.read_text())
    parts = {p["name"]: p["percent"] for p in result["host_partition"]}
    assert parts == {
        "extraction::forward": 60,
        "extraction::attention": 20,
        "other_host_work_or_wait": 20,
    }
    assert sum(parts.values()) == 100
    assert result["device_activity_ms"] == {"kernel": 0.09}
    with pytest.raises(FileExistsError):
        report.main()


def test_missing_scopes_rejected(tmp_path, monkeypatch):
    (tmp_path / "trace.json").write_text(json.dumps(dict(traceEvents=[])))
    monkeypatch.setattr(
        sys,
        "argv",
        ["report", "--capture", str(tmp_path), "--out", str(tmp_path / "report.json")],
    )
    with pytest.raises(ValueError, match="missing host"):
        report.main()
