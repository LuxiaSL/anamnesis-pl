"""Localize short-cache drift without changing any qualification gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from extraction_perf_phase0 import sha, write_json


def evaluate(args):
    folders = (args.cpu, args.control, args.gpu)
    pins = [json.loads((f / "deployment.json").read_text()) for f in folders]
    manifests = [json.loads((f / "inputs.json").read_text()) for f in folders]
    for f, pin in zip(folders, pins, strict=True):
        if (
            not (f / "complete.json").exists()
            or sha(f / "inputs.json") != pin["inputs_sha256"]
        ):
            raise ValueError("missing completion or changed inputs")
        if pin["standardizer_sha256"] != sha(args.sigma):
            raise ValueError("standardizer mismatch")
        complete = json.loads((f / "complete.json").read_text())
        manifest = json.loads((f / "inputs.json").read_text())
        if (
            not manifest
            or complete["rows"] != len(manifest)
            or len({tuple(r["key"]) for r in manifest}) != len(manifest)
        ):
            raise ValueError("incomplete/duplicate control population")
        if pin["tf32"] is not False:
            raise ValueError("unexpected TF32 setting")
    if pins[0]["source"] != pins[1]["source"] or pins[1]["mode"] != "cpu-control":
        raise ValueError("control must retain original CPU extractor")
    for field in (
        "calibration_sha256",
        "caller",
        "bank_pin_sha256",
        "torch",
        "transformers",
        "numpy",
    ):
        if not pins[0][field] == pins[1][field] == pins[2][field]:
            raise ValueError(f"provenance mismatch: {field}")
    if (
        pins[0]["workspace"] is not None
        or pins[1]["workspace"] != pins[2]["workspace"]
        or pins[1]["workspace"] != ":16:8"
    ):
        raise ValueError("wrong arithmetic control")
    if pins[1]["deterministic"] is not True or pins[2]["deterministic"] is not True:
        raise ValueError("control deterministic flag mismatch")
    with np.load(args.sigma) as z:
        sigma, weights = z["sigma_cal"].astype(float), z["weights"].astype(float)
        names = z["feature_names"].astype(str)
    results, artifacts = [], {}
    for row in manifests[1]:
        values = []
        for folder, manifest in zip(folders, manifests, strict=True):
            indices = [
                i
                for i, candidate in enumerate(manifest)
                if candidate["key"] == row["key"]
            ]
            if len(indices) != 1 or manifest[indices[0]] != row:
                raise ValueError("control input differs")
            path = folder / f"row-{indices[0]:05d}.npz"
            with np.load(path) as z:
                if any(
                    z[name].dtype != np.float32
                    or z[name].shape != sigma.shape
                    or not np.isfinite(z[name]).all()
                    for name in ("features", "repeat")
                ):
                    raise ValueError("invalid control feature array")
                if z["features"].tobytes() != z["repeat"].tobytes():
                    raise ValueError("independent repeats differ")
                values.append(z["features"].astype(float))
            artifacts[str(path)] = sha(path)
        pairs = {}
        for name, left, right in (
            ("original_vs_control", 0, 1),
            ("control_vs_gpu", 1, 2),
            ("original_vs_gpu", 0, 2),
        ):
            delta = values[right] - values[left]
            errors = np.abs(delta) / sigma
            pairs[name] = dict(
                l2=float(np.linalg.norm(delta * weights / sigma)),
                max_feature_sigma_error=float(errors.max()),
                worst_feature=names[int(errors.argmax())],
            )
        results.append(dict(key=row["key"], comparisons=pairs))
    result = dict(
        rows=results,
        artifacts_sha256=artifacts,
        deployment_sha256=[sha(f / "deployment.json") for f in folders],
        certification=False,
        purpose="diagnostic localization only",
        arithmetic=[
            {
                k: pin.get(k, "not recorded in original receipt")
                for k in (
                    "mode",
                    "workspace",
                    "deterministic",
                    "tf32",
                    "preferred_blas_library",
                )
            }
            for pin in pins
        ],
    )
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ("cpu", "control", "gpu", "sigma", "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = evaluate(args)
    write_json(args.out, result)
    print(json.dumps(result["rows"], indent=2))


if __name__ == "__main__":
    main()
