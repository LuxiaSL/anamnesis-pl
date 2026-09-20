"""Decompose banked sweep error; squared-L2 shares add, L2 values do not."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anamnesis.analysis.feature_map import FeatureMap
from extraction_perf_phase0 import sha, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    sigma_path = args.root / "reference/standardization_floor_p05.npz"
    with np.load(sigma_path) as z:
        names = z["feature_names"].astype(str).tolist()
        sigma, weights = z["sigma_cal"].astype(float), z["weights"].astype(float)
    fm = FeatureMap(names, n_layers=80)
    groups = {
        "/".join(v.value for v in key): indices
        for key, indices in fm.cells("source", "method").items()
    }
    # Explicit unthresholded witness families, not a claim that all other
    # features are discrete. Overlapping witnesses are not additive partitions.
    witnesses = {
        prefix: [i for i, name in enumerate(names) if name.startswith(prefix)]
        for prefix in (
            "activation_norm_",
            "pca_",
            "attn_entropy_",
            "head_agreement_",
            "delta_",
        )
    }
    summaries, pins = [], {}
    for length_index, length in enumerate((64, 128, 299, 512, 1024, 2048)):
        errors = []
        for seed in range(5):
            index = 5 * length_index + seed
            arrays = []
            for folder in ("sweep-cpu-001", "sweep-gpu-001"):
                path = args.root / folder / f"row-{index:05d}.npz"
                receipt = json.loads(path.with_suffix(".json").read_text())
                if receipt["key"] != ["synthetic", length, seed]:
                    raise ValueError("row alignment mismatch")
                with np.load(path) as z:
                    arrays.append(z["features"].astype(float))
                pins[str(path)] = sha(path)
            errors.append((arrays[1] - arrays[0]) / sigma)
        e = np.stack(errors)
        weighted = e * weights
        total = float(np.square(weighted).sum())

        def describe(indices):
            sub = e[:, indices]
            squared = float(np.square(weighted[:, indices]).sum())
            return dict(
                n_features=len(indices),
                squared_l2_share=squared / total,
                row_l2=np.linalg.norm(weighted[:, indices], axis=1).tolist(),
                max_sigma_error=float(np.abs(sub).max()) if indices else 0,
                coordinates_above_1e_3_per_row=(np.abs(sub) > 1e-3)
                .sum(axis=1)
                .tolist(),
            )

        partition = {name: describe(indices) for name, indices in groups.items()}
        if not np.isclose(sum(g["squared_l2_share"] for g in partition.values()), 1):
            raise ValueError("partition does not conserve squared error")
        summaries.append(
            dict(
                cache_length=length,
                feature_steps=127,
                partition=partition,
                continuous_witnesses={
                    name: describe(idx) for name, idx in witnesses.items()
                },
                total_row_l2=np.linalg.norm(weighted, axis=1).tolist(),
            )
        )
    write_json(
        args.out,
        dict(
            standardizer_sha256=sha(sigma_path),
            arrays_sha256=pins,
            partition="canonical source/method; disjoint",
            witnesses="overlapping, not additive",
            lengths=summaries,
            certification=False,
        ),
    )
    for summary in summaries:
        print(
            "CACHE",
            summary["cache_length"],
            "activation_norm",
            summary["continuous_witnesses"]["activation_norm_"],
            flush=True,
        )
        if summary["cache_length"] == 64:
            print(json.dumps(summary["partition"], indent=2))


if __name__ == "__main__":
    main()
