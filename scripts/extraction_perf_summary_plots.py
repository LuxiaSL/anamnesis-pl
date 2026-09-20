"""Scoped performance panels: matched 70B, balanced batching, native pilots."""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from extraction_perf_phase0 import sha, write_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = []
    fig, axes = plt.subplots(3, 2, figsize=(13, 13), layout="constrained")
    paths = sorted((args.root / "gpu-trial-003").glob("row-*.json"))
    sources.extend(paths)
    pilot = [json.loads(p.read_text()) for p in paths]
    cpu = [r["cpu_matched_s"] for r in pilot]
    gpu = [r["seconds_per_row"] for r in pilot]

    def bars(ax, cpu, gpu, title, subtitle, cpu_repeat=None, gpu_repeat=None):
        means = [float(np.mean(cpu)), float(np.mean(gpu))]
        ax.bar([0, 1], means, width=0.55, color=["steelblue", "darkorange"], alpha=0.7)
        for x, values in enumerate((cpu, gpu)):
            ax.scatter(
                x + np.linspace(-0.1, 0.1, len(values)),
                values,
                color="black",
                s=24,
                zorder=3,
                label="first measured calls" if x == 0 else None,
            )
        if cpu_repeat is not None:
            for x, values in enumerate((cpu_repeat, gpu_repeat)):
                ax.scatter(
                    x + np.linspace(-0.1, 0.1, len(values)),
                    values,
                    facecolors="none",
                    edgecolors="black",
                    s=28,
                    zorder=3,
                    label="independent repeats" if x == 0 else None,
                )
        ax.set_xticks([0, 1], ["CPU features", "GPU features"])
        ax.set_ylabel("seconds / span (model forward stays on GPU)")
        ax.set_ylim(bottom=0)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.text(
            0.5,
            0.97,
            f"{means[0]:.3f} → {means[1]:.3f} s · {means[0] / means[1]:.2f}×",
            ha="center",
            va="top",
            transform=ax.transAxes,
        )
        ax.margins(y=0.2)

    bars(
        axes[0, 0],
        cpu,
        gpu,
        "70B cached replay · matched four-row pilot",
        "Frozen single-row lane; expanded-roster gate evidence is separate",
    )
    timing_paths = sorted((args.root / "batch-trial-005").glob("timing-batch-*.json"))
    sources.extend(timing_paths)
    timing = sorted(
        [json.loads(p.read_text()) for p in timing_paths], key=lambda r: r["shape"]
    )
    ax = axes[0, 1]
    for i, record in enumerate(timing):
        gains = [
            1000 * (r["single_seconds_per_row"] - r["batch_seconds_per_row"])
            for r in record["samples"]
        ]
        ax.scatter(
            i + np.linspace(-0.12, 0.12, len(gains)), gains, color="gray", alpha=0.7
        )
        ax.scatter(
            i, float(np.mean(gains)), marker="D", color="darkgreen", s=50, zorder=3
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(timing)), [str(r["shape"][0]) for r in timing])
    ax.set_xlabel("input pair (cache length); different content per pair")
    ax.set_ylabel("milliseconds saved / row (single − batch)")
    ax.set_title(
        "70B independent-prefill batch2 · all 36 warm samples\nDiamonds: pair means; fidelity qualified, speed benefit is sample-dependent",
        fontsize=10,
    )
    for model, ax in zip(("3b", "8b"), axes[1], strict=True):
        path = args.root / f"native-{model}-comparison-001.json"
        sources.append(path)
        data = json.loads(path.read_text())
        rows = data["rows"]
        bars(
            ax,
            [r["cpu_seconds"][0] for r in rows],
            [r["gpu_seconds"][0] for r in rows],
            f"{model.upper()} native full replay · {rows[0]['feature_count']} features",
            "Three-input characterization only; no transferred tolerance certificate",
            [r["cpu_seconds"][1] for r in rows],
            [r["gpu_seconds"][1] for r in rows],
        )
    handles, labels = axes[1, 0].get_legend_handles_labels()
    batch_path = args.root / "batch-timing-bank-002-v2.json"
    sources.append(batch_path)
    batch = json.loads(batch_path.read_text())
    ax = axes[2, 0]
    labels_batch = [
        "balanced pilot\n36 paired samples",
        "operational sample\n91 rows",
        "operational sample\n86 non-filler rows",
    ]
    balanced_path = args.root / "balanced-timing-005.json"
    sources.append(balanced_path)
    # Compute the balanced ratio directly from its source timing samples.
    samples = [s for record in timing for s in record["samples"]]
    balanced_ratio = sum(s["single_seconds_per_row"] for s in samples) / sum(
        s["batch_seconds_per_row"] for s in samples
    )
    ratios = [
        balanced_ratio,
        batch["paired_sample_speedup"],
        batch["paired_sample_by_filler"]["no_filler"]["speedup"],
    ]
    ax.bar(range(3), ratios, color=["darkgreen", "indianred", "indianred"], alpha=0.8)
    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(3), labels_batch, fontsize=8)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("single time / batch time (>1 favors batching)")
    ax.set_title(
        "Batch speed is not uniformly positive\nDifferent sampling designs; no pooled speedup claim",
        fontsize=10,
    )
    for i, value in enumerate(ratios):
        ax.text(i, value + 0.02, f"{value:.3f}×", ha="center")
    ax = axes[2, 1]
    values = [batch["first_call_seconds_per_row"], batch["repeat_call_seconds_per_row"]]
    ax.bar([0, 1], values, color=["darkorange", "goldenrod"], alpha=0.8)
    ax.set_xticks([0, 1], ["first complete replay", "independent repeat"])
    ax.set_ylim(0, max(values) * 1.3)
    ax.set_ylabel("seconds / emitted row")
    ax.set_title(
        "Original bank: 2,719 rows / 1,367 batch calls\n15 filler rows charged; setup excluded; not a matched speedup",
        fontsize=10,
    )
    for i, value in enumerate(values):
        ax.text(i, value + 0.03, f"{value:.3f} s", ha="center")
    fig.legend(
        handles,
        labels,
        fontsize=9,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.025),
        ncol=2,
    )
    fig.suptitle(
        "Extraction performance — measured scopes kept separate; outliers retained",
        fontsize=14,
    )
    for ext in ("png", "svg"):
        fig.savefig(
            args.out / f"performance-summary.{ext}", dpi=170, bbox_inches="tight"
        )
    plt.close(fig)
    write_json(
        args.out / "sources.json",
        dict(
            script_sha256=sha(Path(__file__)),
            sources={str(p): sha(p) for p in sources},
            note="Timing evidence, not a cross-scope certification or full-capture forecast",
        ),
    )


if __name__ == "__main__":
    main()
