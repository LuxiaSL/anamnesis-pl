"""Retro-read A1 + A3 magnitude columns through the 12e decomposed ruler.

Mandated by addendum 2026-07-12e item 2: already-emitted verdicts stand as
written (flagged pairwise-compressed); this emits the decomposed reading for
the campaign log. Free-gen confirmatory cells only.

Usage:
    PYTHONPATH=. python -m anamnesis.scripts.vmb_magnitude_retroread \
        --battery-root ../outputs/battery --out-dir ../outputs/battery/arms
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from anamnesis.analysis.battery.deltas import ConditionCorpus, build_cells, load_floor_scale
from anamnesis.analysis.battery.magnitude import decomposed_magnitude
from anamnesis.analysis.battery.manifest import MODEL_META

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

A1_CELLS = ["source:output", "source:residual", "source_band:attention|mid"]
A1_CONTRASTS = [("native", "t03"), ("native", "t09"), ("native", "t12"),
                ("native", "p07"), ("native", "p10")]
A3_CELLS = ["whole_vector", "source:attention", "source_band:attention|mid"]
MODES = ["linear", "analogical", "socratic", "contrastive", "dialectical"]
N_PERM = 1000


def rekey_topic(cc: ConditionCorpus) -> ConditionCorpus:
    rekeyed: dict[tuple[int, str], list[int]] = {}
    for (tidx, _m), rows in cc.rows_by_class.items():
        rekeyed.setdefault((tidx, "t"), []).extend(rows)
    cc.rows_by_class = rekeyed
    return cc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--models", default="3b,8b")
    ap.add_argument("--skip-a3", action="store_true",
                    help="models without pure-mode corpora (A1-only rosters)")
    args = ap.parse_args()

    results = {"prereg": "addendum 2026-07-12e item 2 retro-read; n_perm=%d" % N_PERM,
               "note": "original pairwise verdicts stand as written (flagged "
                       "pairwise-compressed); this is the decomposed reading",
               "A1": {}, "A3": {}}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        meta = MODEL_META[model]
        floor_dir = args.battery_root / meta.stage0_dir
        med, scale = load_floor_scale(floor_dir / "signatures_v3")

        # ── A1: native (Stage-0) vs each dose ──
        native = ConditionCorpus(floor_dir / "signatures_v3", floor_dir / "metadata.json",
                                 med, scale, f"{model}-native")
        names = native.feature_names
        cells = build_cells(names, meta.n_layers)
        a1_cells = {c: cells[c] for c in A1_CELLS}
        a1 = {}
        for (_, dose) in A1_CONTRASTS:
            d = args.battery_root / f"vmb_a1_{model}_{dose}"
            if not (d / "signatures_v3").exists():
                logger.warning(f"{model} {dose}: missing — skipped")
                continue
            cond = ConditionCorpus(d / "signatures_v3", d / "metadata.json",
                                   med, scale, f"{model}-{dose}")
            a1[f"native|{dose}"] = decomposed_magnitude(native, cond, a1_cells, N_PERM)
            logger.info(f"A1 {model} native|{dose} done")
        results["A1"][model] = a1

        if args.skip_a3:
            continue
        # ── A3: all 10 mode pairs ──
        conds = {}
        for mode in MODES:
            d = args.battery_root / f"vmb_a2_{model}_pure_{mode}"
            conds[mode] = rekey_topic(ConditionCorpus(
                d / "signatures_v3", d / "metadata.json", med, scale,
                f"{model}-pure_{mode}"))
        a3_cells = {c: cells[c] for c in A3_CELLS}
        a3 = {}
        for i, ma in enumerate(MODES):
            for mb in MODES[i + 1:]:
                a3[f"{ma}|{mb}"] = decomposed_magnitude(conds[ma], conds[mb], a3_cells, N_PERM)
                logger.info(f"A3 {model} {ma}|{mb} done")
        results["A3"][model] = a3

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "magnitude_retroread_12e.json"
    out.write_text(json.dumps(results, indent=1))

    lines = ["# 12e decomposed-ruler retro-read (A1 + A3)", "",
             f"Permutation nulls: n_perm={N_PERM}, labels shuffled within prompt "
             "class. shift = centroid displacement (floor-z); disp = within-cloud "
             "dispersion ratio (condition/reference).", ""]
    for arm, cellset in (("A1", A1_CELLS), ("A3", A3_CELLS)):
        for model, block in results[arm].items():
            lines.append(f"## {arm} {model}")
            lines.append("| contrast | cell | shift (z) | p_shift | disp ratio | p_wider | p_narrower |")
            lines.append("|---|---|---|---|---|---|---|")
            for contrast, cellvals in block.items():
                for cname, v in cellvals.items():
                    lines.append(
                        f"| {contrast} | {cname} | {v['centroid_shift']:.3f} | "
                        f"{v['p_shift']:.4g} | {v['dispersion_ratio']:.3f} | "
                        f"{v['p_disp_wider']:.4g} | {v['p_disp_narrower']:.4g} |")
            lines.append("")
    (args.out_dir / "magnitude_retroread_12e.md").write_text("\n".join(lines))
    logger.info(f"→ {out} + .md")


if __name__ == "__main__":
    main()
