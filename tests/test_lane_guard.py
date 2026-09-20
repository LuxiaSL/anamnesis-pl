import json

import numpy as np
import pytest

from anamnesis.analysis.lane_guard import MixedLaneError, require_single_lane
from anamnesis.analysis.geometric_trio.data_loader import load_run4


def test_legacy_inputs_are_not_invented_cpu_certification():
    assert require_single_lane([{}, {}]) is None
    assert (
        require_single_lane(
            [{"lane_id": "gpu"}, {"extraction_lane": {"lane_id": "gpu"}}]
        )
        == "gpu"
    )


@pytest.mark.parametrize(
    "metadata",
    [
        [{"lane_id": "gpu"}, {}],
        [{"lane_id": "gpu"}, {"lane_id": "cpu"}],
        [{"lane_id": "gpu", "extraction_lane": {"lane_id": "other"}}],
        [{"lane_id": ""}],
        [
            {"lane_id": "gpu", "extraction_lane": {"calibration_sha256": "a"}},
            {"lane_id": "gpu", "extraction_lane": {"calibration_sha256": "b"}},
        ],
    ],
)
def test_mixed_or_conflicting_identity_rejected(metadata):
    with pytest.raises(MixedLaneError):
        require_single_lane(metadata)


def write_row(folder, i, lane=None, feature_key="features_tier1"):
    folder.mkdir(exist_ok=True)
    np.savez(
        folder / f"gen_{i:03d}.npz",
        **{feature_key: np.array([1.0, 2.0], dtype=np.float32)},
        feature_names=np.array(["a", "b"]),
    )
    meta = dict(
        generation_id=i,
        topic="topic",
        topic_idx=0,
        mode="linear",
        mode_idx=0,
        num_generated_tokens=128,
    )
    if lane is not None:
        meta["lane_id"] = lane
    (folder / f"gen_{i:03d}.json").write_text(json.dumps(meta))


def test_loader_preserves_and_enforces_lane_identity(tmp_path):
    write_row(tmp_path, 0, "gpu")
    write_row(tmp_path, 1, "gpu")
    data = load_run4(tmp_path, core_only=False)
    assert data.lane_id == "gpu" and data.n_samples == 2
    write_row(tmp_path, 1, "other")
    with pytest.raises(MixedLaneError):
        load_run4(tmp_path, core_only=False)


def test_loader_refuses_untagged_addon_on_gpu_lane(tmp_path):
    base, addon = tmp_path / "base", tmp_path / "addon"
    write_row(base, 0, "gpu")
    write_row(addon, 0, feature_key="features_tier2")
    with pytest.raises(MixedLaneError):
        load_run4(base, core_only=False, addon_dirs=[addon])


def test_loader_legacy_behavior_remains_available(tmp_path):
    write_row(tmp_path, 0)
    assert load_run4(tmp_path, core_only=False).lane_id is None


def test_tagged_dataset_cannot_silently_skip_missing_metadata(tmp_path):
    write_row(tmp_path, 0, "gpu")
    write_row(tmp_path, 1, "gpu")
    (tmp_path / "gen_001.json").unlink()
    with pytest.raises(MixedLaneError, match="without lane metadata"):
        load_run4(tmp_path, core_only=False)
