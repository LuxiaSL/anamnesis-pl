import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "workspace_evidence",
    Path(__file__).resolve().parents[1]
    / "scripts/extraction_perf_workspace_evidence.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def evidence():
    cells = ("own_cont", "paraphrase_own", "xfam_paraphrase", "null::own_cont")
    audit = dict(
        row_manifest=[dict(key=["doc", cell, 0, "h1"]) for cell in cells],
        files=[dict(shard_sha256="bank", documents_sha256="inputs")],
        standardizer_sha256="sigma",
        feature_count=4086,
    )
    row = dict(
        n_features=4086,
        feature_names_match_banked=True,
        all_finite=True,
        byte_identical_vs_sig_ancestor_h1_raw=True,
    )
    base = dict(
        torch_cuda_deterministic_algorithms_enabled=False,
        standardization_npz_sha256="sigma",
        npz_in_sha256="bank",
        branch_jsonl_sha256="inputs",
        errors=[],
        n_rows_selected=4,
        n_rows_completed_ok=4,
        rows={cell + "/0": dict(row) for cell in cells},
        doc_id="doc",
        torch_version="2.9.1+cu128",
        model_path="model",
        preset="70b-base",
    )
    return {
        "unset": dict(
            copy.deepcopy(base), cublas_workspace_config_seen_by_this_process=None
        ),
        "ws16_8": dict(
            copy.deepcopy(base), cublas_workspace_config_seen_by_this_process=":16:8"
        ),
    }, audit


def test_matched_owner_positive_control_is_scoped():
    result = module.assess_workspace_probe(*evidence())
    assert result["owner_probe_pass"]
    assert result["deterministic_mode_in_probe"] is False


def test_nonreproduction_is_not_hidden_by_successful_process():
    probe, audit = evidence()
    probe["ws16_8"]["rows"]["own_cont/0"]["byte_identical_vs_sig_ancestor_h1_raw"] = (
        False
    )
    assert not module.assess_workspace_probe(probe, audit)["owner_probe_pass"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("torch_cuda_deterministic_algorithms_enabled", True),
        ("npz_in_sha256", "other"),
        ("standardization_npz_sha256", "other"),
    ],
)
def test_changed_probe_identity_refused(field, value):
    probe, audit = evidence()
    probe["ws16_8"][field] = value
    with pytest.raises(ValueError):
        module.assess_workspace_probe(probe, audit)
