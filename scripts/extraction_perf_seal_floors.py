"""Verify and seal completed rows after an intentionally stopped floor survey."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from extraction_perf_phase0 import sha, source_pin, stamp, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", type=Path, nargs="+")
    args = ap.parse_args()
    for folder in args.folders:
        pin = json.loads((folder / "deployment.json").read_text())
        if source_pin(Path(pin["anamnesis_root"])) != pin["anamnesis"]:
            raise ValueError("stopped source differs from launch pin")
        if source_pin(Path(pin["caller_root"])) != pin["caller"]:
            raise ValueError("stopped caller differs from launch pin")
        if (
            sha(Path(pin["arguments"]["sigma"]))
            != pin["files"][pin["arguments"]["sigma"]]
        ):
            raise ValueError("standardizer changed")
        with np.load(pin["arguments"]["sigma"]) as z:
            sigma = z["sigma_cal"].astype(float)
            weights = z["weights"].astype(float)
        files = {}
        for path in sorted(folder.glob("row-*.json")):
            record = json.loads(path.read_text())
            data_path = path.with_suffix(".npz")
            with np.load(data_path, allow_pickle=False) as data:
                if data["cpu"].tobytes() != data["bank"].tobytes():
                    raise ValueError(f"reference mismatch: {path}")
                value = float(
                    np.linalg.norm(
                        (data["incremental"].astype(float) - data["cpu"])
                        * weights
                        / sigma
                    )
                )
                if not np.isclose(value, record["floor_b"], rtol=1e-12, atol=1e-12):
                    raise ValueError(f"floor does not reproduce: {path}")
            files[path.name] = sha(path)
            files[data_path.name] = sha(data_path)
        receipt = dict(
            utc=stamp(),
            status="intentionally_stopped_partial",
            reason="user authorized adaptive lower-bound certificates to reduce GPU occupancy",
            n_complete_rows=len(files) // 2,
            source_verified=True,
            files=files,
            deployment_sha256=sha(folder / "deployment.json"),
            uncommitted_npz=[
                p.name for p in folder.glob("row-*.npz") if p.name not in files
            ],
            certification=False,
        )
        write_json(folder / "stopped.json", receipt)
        print(
            json.dumps({k: v for k, v in receipt.items() if k != "files"}), flush=True
        )


if __name__ == "__main__":
    main()
