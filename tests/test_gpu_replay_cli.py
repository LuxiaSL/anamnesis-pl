import pytest

from anamnesis.scripts.run_gpu_replay import parser, read_generation_metadata
import json


BASE = [
    "--model",
    "3b",
    "--model-path",
    "/model",
    "--calib-dir",
    "/calibration",
    "--manifest",
    "/manifest.json",
    "--output",
    "/new-output",
]


def test_feature_only_requires_explicit_acknowledgement():
    with pytest.raises(SystemExit):
        parser().parse_args(BASE)
    args = parser().parse_args(BASE + ["--experimental-feature-only"])
    assert args.experimental_feature_only


@pytest.mark.parametrize(
    "extra",
    [
        ["--adapter-path", "/adapter"],
        ["--inject-npz", "/write.npz"],
        ["--batch-size", "2"],
    ],
)
def test_unqualified_execution_modes_are_not_accepted(extra):
    with pytest.raises(SystemExit):
        parser().parse_args(BASE + ["--experimental-feature-only", *extra])


def test_source_metadata_labels_are_preserved(tmp_path):
    path = tmp_path / "metadata.json"
    record = dict(
        generation_id=7, mode="linear", topic="topic", mode_idx=0, topic_idx=2
    )
    path.write_text(json.dumps(dict(generations=[record])))
    assert read_generation_metadata(path) == {7: record}
    path.write_text(json.dumps(dict(generations=[record, record])))
    with pytest.raises(ValueError, match="duplicate"):
        read_generation_metadata(path)
