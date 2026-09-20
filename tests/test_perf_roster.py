import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location(
    "perf_roster", scripts / "extraction_perf_roster.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def roster(tmp_path):
    mirror = tmp_path / "mirror"
    folder = mirror / "run"
    folder.mkdir(parents=True)
    shard = folder / "own_content_shard0.npz"
    np.savez(
        shard,
        feature_names=np.array(["a", "b"]),
        doc_id=np.array(["x", "y"]),
        cell=np.array(["B", "B"]),
        branch_id=np.array([0, 0]),
        excluded=np.array([False, True]),
        cluster=np.array(["x", "y"]),
        sig_ancestor_h1_raw=np.ones((2, 2), dtype=np.float32),
    )
    docs = [
        dict(
            doc_id=d,
            cell="B",
            branch_id=0,
            tokens=[1, 2, 3],
            spans=[dict(role="holder", index=1, start=1, end=3)],
        )
        for d in ("x", "y")
    ]
    (folder / "branch_documents_shard0.jsonl").write_text(
        "\n".join(json.dumps(d) for d in docs) + "\n"
    )
    path = tmp_path / "roster.json"
    manifest = dict(
        files=[
            dict(
                path="/bank/run/own_content_shard0.npz",
                sha256=module.sha(shard),
                n_roster_rows=1,
                rows={"B": [["x", "0"]]},
            )
        ]
    )
    module.write_json(path, manifest)
    return path, {"/bank": mirror}, manifest


def test_owner_selection_applied_before_logical_join(roster):
    rows, names, files = module.roster_rows(*roster[:2])
    assert len(rows) == 1 and rows[0]["key"] == ["x", "B", 0, "h1"]
    assert names == ["a", "b"] and files[0]["rows"] == 1


def test_excluded_row_cannot_be_silently_certified(roster):
    path, mirrors, manifest = roster
    manifest["files"][0]["rows"] = {"B": [["y", "0"]]}
    module.write_json(path, manifest)
    with pytest.raises(ValueError, match="excluded row"):
        module.roster_rows(path, mirrors)


def test_missing_row_refused(roster):
    path, mirrors, manifest = roster
    manifest["files"][0]["rows"] = {"B": [["missing", "0"]]}
    module.write_json(path, manifest)
    with pytest.raises(ValueError, match="tuple missing"):
        module.roster_rows(path, mirrors)


def test_changed_shard_refused(roster):
    path, mirrors, manifest = roster
    manifest["files"][0]["sha256"] = "0" * 64
    module.write_json(path, manifest)
    with pytest.raises(ValueError, match="hash-mismatched"):
        module.roster_rows(path, mirrors)


def test_duplicate_roster_tuple_refused(roster):
    path, mirrors, manifest = roster
    item = manifest["files"][0]
    item["rows"]["B"] *= 2
    item["n_roster_rows"] = 2
    module.write_json(path, manifest)
    with pytest.raises(ValueError, match="duplicate tuple"):
        module.roster_rows(path, mirrors)
