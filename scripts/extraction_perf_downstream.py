"""Evaluate F-B and each null fork separately against pinned CPU references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from extraction_perf_phase0 import rows_from_bank, sha, write_json
from anamnesis.extraction.fidelity import DownstreamAnchor, verify_downstream


def evaluate(args):
    """Recompute all four gates from arrays; never trust a prior pass flag."""
    import palimpsest.genpair as genpair

    if not (args.gpu / "complete.json").exists():
        raise ValueError("complete GPU bank required")
    rows, names = rows_from_bank(args.reference)
    reference = {tuple(r["key"]): r for r in rows}
    candidate = {}
    for path in sorted(args.gpu.glob("row-*.json")):
        record = json.loads(path.read_text())
        key = tuple(record["key"])
        if key in candidate or not record["G0"]:
            raise ValueError("duplicate row or failed G0")
        with np.load(path.with_suffix(".npz")) as data:
            if data["gpu"].tobytes() != data["repeat"].tobytes():
                raise ValueError("G0 does not reproduce")
            if data["bank"].tobytes() != reference[key]["bank"].tobytes():
                raise ValueError("CPU reference changed")
            candidate[key] = data["gpu"].copy()
    if set(reference) != set(candidate):
        raise ValueError("reference population mismatch")
    gpu_pin = json.loads((args.gpu / "deployment.json").read_text())
    if sha(Path(genpair.__file__)) != gpu_pin["caller"]["genpair.py"]:
        raise ValueError("statistic source differs from frozen caller")
    anchor_doc = json.loads(args.anchor.read_text())
    sigma_path = args.reference / "standardization_floor_p05.npz"
    if sha(sigma_path) != anchor_doc["standardization"]["sha256"]:
        raise ValueError("standardizer differs from reference")
    with np.load(sigma_path) as z:
        if z["feature_names"].astype(str).tolist() != names:
            raise ValueError("feature schema mismatch")
        sigma = z["sigma_cal"].astype(float)
        weights = z["weights"].astype(float)
    results = {}
    for name, left, branch, right in [
        ("F-B", "xfam_paraphrase", 0, "paraphrase_own"),
        *[(f"null::own_cont#{i}", "null::own_cont", i, "own_cont") for i in range(3)],
    ]:
        docs = []
        clusters = []
        cpu_delta = []
        gpu_delta = []
        for doc in sorted({k[0] for k in reference}):
            ka = (doc, "own_cont", 0, "h1")
            kl = (doc, left, branch, "h1")
            kr = (doc, right, 0, "h1")
            if not all(
                k in reference and not reference[k]["excluded"] for k in (ka, kl, kr)
            ):
                continue
            for source, output in (
                ({k: reference[k]["bank"] for k in (ka, kl, kr)}, cpu_delta),
                (candidate, gpu_delta),
            ):
                a, lhs, rhs = [
                    weights * (source[k].astype(float) / sigma) for k in (ka, kl, kr)
                ]
                output.append((lhs - a) - (rhs - a))
            docs.append(doc)
            clusters.append(reference[ka]["cluster"])
        ref = (
            anchor_doc["fb_contrast"] if name == "F-B" else anchor_doc["per_cell"][name]
        )
        if len(docs) != ref["n_documents"]:
            raise ValueError(
                f"{name}: matched-n reproduction failed ({len(docs)} vs {ref['n_documents']})"
            )
        anchor = DownstreamAnchor(
            ref["ratio"],
            ref["se_corrected"] if name == "F-B" else ref["jackknife_se"],
            tuple(docs),
            tuple(clusters),
            sha(args.anchor),
            sha(Path(genpair.__file__)),
        )
        result = verify_downstream(
            np.stack(cpu_delta),
            np.stack(gpu_delta),
            docs,
            clusters,
            anchor,
            genpair.jackknife_coherence_ratio,
            sha(Path(genpair.__file__)),
            flag_se_sampling_scale=name != "F-B",
        )
        result["documents"] = docs
        result["clusters"] = clusters
        results[name] = result
        print(
            name,
            json.dumps(
                {k: v for k, v in result.items() if k not in ("documents", "clusters")}
            ),
            flush=True,
        )
    return dict(
        consumer_acceptance="pending confirmation of current anchor/references",
        anchor_sha256=sha(args.anchor),
        separate_forks=True,
        results=results,
        all_numeric_bounds_pass=all(r["G2"] for r in results.values()),
        certification=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--gpu", type=Path, required=True)
    ap.add_argument("--anchor", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    write_json(args.out, evaluate(args))


if __name__ == "__main__":
    main()
