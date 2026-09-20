"""Load only owner-selected rows, validating file hashes and explicit precedence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from extraction_perf_phase0 import sha, write_json


def roster_rows(roster_path: Path, mirrors: dict[str, Path]):
    roster = json.loads(roster_path.read_text())
    rows = []
    names = None
    seen = set()
    seen_files = set()
    provenance = []
    for item in roster["files"]:
        source = Path(item["path"])
        matches = [
            (Path(origin), local)
            for origin, local in mirrors.items()
            if source.is_relative_to(Path(origin))
        ]
        if len(matches) != 1:
            raise ValueError(f"ambiguous/unmapped roster file {source}")
        origin, local = matches[0]
        relative = source.relative_to(origin)
        if ".." in relative.parts:
            raise ValueError("roster path traversal")
        path = local / relative
        if str(source) in seen_files or sha(path) != item["sha256"]:
            raise ValueError("duplicate or hash-mismatched roster shard")
        seen_files.add(str(source))
        selected = [
            (doc, cell, int(branch), "h1")
            for cell, tuples in item["rows"].items()
            for doc, branch in tuples
        ]
        if len(selected) != item["n_roster_rows"] or len(set(selected)) != len(
            selected
        ):
            raise ValueError("roster count/duplicate tuple mismatch")
        document_path = path.with_name(
            path.name.replace("own_content_", "branch_documents_").replace(
                ".npz", ".jsonl"
            )
        )
        with document_path.open() as stream:
            documents = [json.loads(line) for line in stream if line.strip()]
        by_document = {
            (d["doc_id"], d["cell"], int(d["branch_id"]), "h1"): d for d in documents
        }
        if len(by_document) != len(documents):
            raise ValueError("duplicate input document tuple")
        with np.load(path, allow_pickle=False) as z:
            current = z["feature_names"].astype(str).tolist()
            if names is not None and names != current:
                raise ValueError("roster feature schema mismatch")
            names = current
            keys = list(
                zip(
                    z["doc_id"].astype(str),
                    z["cell"].astype(str),
                    z["branch_id"].astype(int),
                    ["h1"] * len(z["cell"]),
                    strict=True,
                )
            )
            if len(set(keys)) != len(keys):
                raise ValueError("duplicate shard row tuple")
            indices = {key: i for i, key in enumerate(keys)}
            values = z["sig_ancestor_h1_raw"]
            excluded = z["excluded"]
            clusters = z["cluster"].astype(str)
            for key in selected:
                if key in seen:
                    raise ValueError("roster precedence leaves duplicate logical row")
                if key not in indices or key not in by_document:
                    raise ValueError(
                        "roster tuple missing from shard or token documents"
                    )
                index = indices[key]
                if excluded[index]:
                    raise ValueError("owner roster selected an excluded row")
                document = by_document[key]
                spans = [
                    s
                    for s in document["spans"]
                    if s["role"] == "holder" and s["index"] == 1
                ]
                if len(spans) != 1:
                    raise ValueError("roster row has no unique holder span")
                span = spans[0]
                vector = values[index]
                if (
                    vector.dtype != np.float32
                    or vector.shape != (len(names),)
                    or not np.isfinite(vector).all()
                ):
                    raise ValueError("invalid bank feature vector")
                rows.append(
                    dict(
                        key=list(key),
                        bank=vector.copy(),
                        tokens=document["tokens"],
                        start=span["start"],
                        end=span["end"],
                        excluded=False,
                        cluster=clusters[index],
                        index=index,
                        shard=str(path),
                        origin_shard=str(source),
                    )
                )
                seen.add(key)
        provenance.append(
            dict(
                origin=str(source),
                shard_sha256=sha(path),
                documents_sha256=sha(document_path),
                rows=len(selected),
            )
        )
    if not rows or names is None:
        raise ValueError("empty owner roster")
    return rows, names, provenance


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ("roster", "reference", "extension", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows, names, files = roster_rows(
        args.roster,
        {
            "/mnt/beegfs/luxia/palimpsest-banks/rubric-140": args.reference,
            "/mnt/beegfs/luxia/palimpsest-banks/rubric-144": args.extension,
        },
    )
    standardization = json.loads(args.roster.read_text())["standardization"]
    sigma_path = args.reference / "standardization_floor_p05.npz"
    if sha(sigma_path) != standardization["sha256"]:
        raise ValueError("roster standardizer mismatch")
    with np.load(sigma_path) as z:
        if z["feature_names"].astype(str).tolist() != names:
            raise ValueError("roster/standardizer schema mismatch")
    result = dict(
        roster_sha256=sha(args.roster),
        rows=len(rows),
        feature_count=len(names),
        feature_schema_sha256=hashlib.sha256(json.dumps(names).encode()).hexdigest(),
        standardizer_sha256=standardization["sha256"],
        files=files,
        row_manifest=[
            dict(
                key=r["key"],
                origin_shard=r["origin_shard"],
                index=r["index"],
                start=r["start"],
                end=r["end"],
                bank_vector_sha256=hashlib.sha256(r["bank"].tobytes()).hexdigest(),
                input_sha256=hashlib.sha256(
                    np.asarray(r["tokens"][: r["end"]], dtype="<i8").tobytes()
                ).hexdigest(),
            )
            for r in rows
        ],
        certification=False,
    )
    write_json(args.out, result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("files", "row_manifest")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
