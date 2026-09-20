"""Plot recomputed sweep evidence; do not imply a ratified flatness threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.receipt.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = data["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    lengths = sorted({r["cache_length"] for r in rows})
    for seed in range(5):
        group = sorted(
            (r for r in rows if r["key"][2] == seed), key=lambda r: r["cache_length"]
        )
        axes[0].plot(
            [r["cache_length"] for r in group],
            [r["floor_c"] for r in group],
            "o-",
            alpha=0.7,
            label=f"stream {seed}",
        )
    axes[0].set_ylabel("CPU–GPU standardized L2: floor(c)")
    axes[0].set_title("Five synthetic streams; no fitted acceptance cutoff")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=7)
    for field, label in (
        ("cpu_seconds", "CPU reference"),
        ("gpu_seconds", "GPU feature lane"),
    ):
        means = [
            float(np.mean([r[field][0] for r in rows if r["cache_length"] == length]))
            for length in lengths
        ]
        axes[1].plot(lengths, means, "o-", label=label)
    axes[1].set_ylabel("seconds/span, first measured replay")
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("Matched inputs; prefill + replay + features")
    axes[1].legend()
    axes[2].plot(
        lengths,
        [
            max(r["delta_self"] for r in rows if r["cache_length"] == length)
            for length in lengths
        ],
        "o-",
    )
    axes[2].set_ylabel("within-lane repeat distance")
    axes[2].set_ylim(-0.1, 0.1)
    axes[2].set_title("Independent complete GPU replays")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(lengths, [str(n) for n in lengths], rotation=30)
        ax.set_xlabel("cache length (tokens)")
    scope = (
        "All points position-calibrated"
        if all(r["calibrated_positions"] for r in rows)
        else "Includes out-of-calibration points"
    )
    fig.suptitle(
        f"70B instrument sweep · {scope} · consumer scope remains banked regime"
    )
    for ext in ("png", "svg"):
        fig.savefig(args.out / f"length-sweep.{ext}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
