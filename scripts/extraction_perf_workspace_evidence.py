"""Validate the owner's four-row CPU workspace reproduction receipt and scope."""

from __future__ import annotations


def assess_workspace_probe(probe, audit):
    required = {
        "own_cont/0",
        "paraphrase_own/0",
        "xfam_paraphrase/0",
        "null::own_cont/0",
    }
    roster_keys = {tuple(r["key"]) for r in audit["row_manifest"]}
    files = {(f["shard_sha256"], f["documents_sha256"]) for f in audit["files"]}
    for label, workspace in (("unset", None), ("ws16_8", ":16:8")):
        record = probe[label]
        if (
            record["cublas_workspace_config_seen_by_this_process"] != workspace
            or record["torch_cuda_deterministic_algorithms_enabled"] is not False
            or record["standardization_npz_sha256"] != audit["standardizer_sha256"]
            or (record["npz_in_sha256"], record["branch_jsonl_sha256"]) not in files
            or record["errors"]
            or record["n_rows_selected"] != 4
            or record["n_rows_completed_ok"] != 4
            or set(record["rows"]) != required
        ):
            raise ValueError("owner workspace probe identity/population mismatch")
        for name, row in record["rows"].items():
            cell, branch = name.rsplit("/", 1)
            if (record["doc_id"], cell, int(branch), "h1") not in roster_keys:
                raise ValueError("workspace probe row is outside owner roster")
            if (
                row["n_features"] != audit["feature_count"]
                or not row["feature_names_match_banked"]
                or not row["all_finite"]
            ):
                raise ValueError("workspace probe schema/nonfinite mismatch")
    for field in (
        "doc_id",
        "torch_version",
        "model_path",
        "preset",
        "npz_in_sha256",
        "branch_jsonl_sha256",
    ):
        if probe["unset"][field] != probe["ws16_8"][field]:
            raise ValueError("workspace cells changed more than the declared setting")
    positive = all(
        r["byte_identical_vs_sig_ancestor_h1_raw"] is True
        for r in probe["unset"]["rows"].values()
    )
    pinned = all(
        r["byte_identical_vs_sig_ancestor_h1_raw"] is True
        for r in probe["ws16_8"]["rows"].values()
    )
    return dict(
        owner_probe_pass=positive and pinned,
        literal_capture_reproduced=positive,
        explicit_workspace_reproduced=pinned,
        workspace=":16:8",
        deterministic_mode_in_probe=False,
        document=probe["unset"]["doc_id"],
        rows=4,
        torch=probe["unset"]["torch_version"],
        scope="owner-selected four-row CPU bank reproduction only; not GPU flag-OFF repeatability or all-shape equivalence",
    )
