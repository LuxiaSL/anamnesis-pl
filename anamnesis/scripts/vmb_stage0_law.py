"""vmb Stage-0 law computation: floors → n_min tables (prereg §5 + addendum 2026-07-12a).

Runs on CPU over extracted signatures. Per model:
  1. stochastic floors from the floor corpus (80 classes × 10 seeds → 3,600 pair deltas)
  2. faithfulness floors from the stratified replay set (within/cross-device components)
  3. the law table: n_min per (cell × α_test grid), battery n = 2× (A2 4×)
Outputs (raw-artifacts-next-to-claims): FloorReport JSONs + a human-readable law table
markdown, all under --out-dir. Faithfulness deltas standardized on the STOCHASTIC floor
scale so all floors share one z space.

Usage:
    PYTHONPATH=. python -m anamnesis.scripts.vmb_stage0_law \
        --model 3b --n-layers 28 \
        --floor-sig-dir outputs/battery/vmb_stage0_3b/signatures_v3 \
        --floor-metadata outputs/battery/vmb_stage0_3b/metadata.json \
        --faith-sig-dir outputs/battery/vmb_stage0_3b/faithfulness/signatures_v3 \
        --faith-index outputs/battery/vmb_stage0_3b/faithfulness/replay_index.json \
        --out-dir outputs/battery/vmb_stage0_3b/floors
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from anamnesis.analysis.battery.floors import (
    LawParams,
    compute_faithfulness_floors,
    compute_stochastic_floors,
    load_signature_matrix,
    robust_scale,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def law_table_md(reports, model: str) -> str:
    lines = [
        f"# Stage-0 law table — {model}",
        "",
        "Battery n per cell = 2 × n_min (A2 cells 4×; ratified). α_test at arm-prereg time",
        "= 0.05 / m with m = that arm's pre-registered confirmatory cell count",
        "(addendum 2026-07-12a item 1). Shift reading: arm sits at k=2× floor median",
        "(effect = (k−1)·median/σ_floor). Rank-test n = n/0.955.",
        "",
    ]
    for rep in reports:
        lines.append(f"## {rep.floor_type.value}  (n_gens={rep.n_gens}, pairs={rep.n_pairs_total}, "
                     f"M={rep.model})")
        lines.append("")
        lines.append("| cell | n_feat | floor median | σ | effect d | " +
                     " | ".join(f"n_min@α={a}" for a in rep.law.alpha_grid) + " |")
        lines.append("|" + "---|" * (5 + len(rep.law.alpha_grid)))
        for c in sorted(rep.cells, key=lambda c: c.cell):
            ns = " | ".join(str(c.n_min_by_alpha[str(a)]) for a in rep.law.alpha_grid)
            lines.append(f"| {c.cell} | {c.n_features} | {c.median:.4f} | {c.std:.4f} | "
                         f"{c.effect_d:.2f} | {ns} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-layers", type=int, required=True)
    ap.add_argument("--floor-sig-dir", type=Path, required=True)
    ap.add_argument("--floor-metadata", type=Path, required=True)
    ap.add_argument("--faith-sig-dir", type=Path, default=None)
    ap.add_argument("--faith-index", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--power", type=float, default=0.9)
    args = ap.parse_args()

    law = LawParams(k=args.k, power=args.power)
    reports = []

    stoch = compute_stochastic_floors(
        args.floor_sig_dir, args.floor_metadata, model=args.model,
        n_layers=args.n_layers, law=law,
    )
    stoch.save(args.out_dir / f"floors_stochastic_{args.model}.json")
    reports.append(stoch)

    if args.faith_sig_dir and args.faith_index:
        X, _names, _ids = load_signature_matrix(args.floor_sig_dir)
        scale = robust_scale(X)   # faithfulness on the stochastic-floor z scale
        for rep in compute_faithfulness_floors(
            args.faith_sig_dir, args.faith_index, model=args.model,
            n_layers=args.n_layers, law=law, scale_from=scale,
        ):
            rep.save(args.out_dir / f"floors_{rep.floor_type.value}_{args.model}.json")
            reports.append(rep)

    md = law_table_md(reports, args.model)
    (args.out_dir / f"law_table_{args.model}.md").write_text(md)
    logger.info(f"law table → {args.out_dir}/law_table_{args.model}.md")

    wv = [c for c in stoch.cells if c.cell == "whole_vector"][0]
    logger.info(f"[{args.model}] whole-vector stochastic floor: median={wv.median:.4f} "
                f"d={wv.effect_d:.2f} n_min@0.05={wv.n_min_by_alpha['0.05']} "
                f"@1e-4={wv.n_min_by_alpha['0.0001']} (n={wv.n_pairs} pairs, M={args.model}, "
                f"law k={law.k} power={law.power}, floor=stochastic)")


if __name__ == "__main__":
    main()
