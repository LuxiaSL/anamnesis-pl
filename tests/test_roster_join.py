import importlib.util
from pathlib import Path
import sys

import pytest

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location(
    "roster_join", scripts / "extraction_perf_join_roster.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_changed_logical_key_uses_supplement_not_old_rider():
    key = ("doc", "B_rep0", 0, "h1")
    assert module.choose_evidence(key, {}, {key: "old"}, {key: "new"}) == "new"


def test_missing_supplement_cannot_fall_back_to_old_rider():
    key = ("doc", "B_rep0", 0, "h1")
    with pytest.raises(ValueError, match="missing required"):
        module.choose_evidence(key, {}, {key: "old"}, {})


def test_matching_original_is_reused_only_when_declared():
    key = ("doc", "B", 0, "h1")
    assert (
        module.choose_evidence(key, {key: True}, {key: "original"}, {key: "other"})
        == "original"
    )
