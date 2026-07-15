"""ANNEX (non-quotable) — the SAME-FAMILY cell: score a formula vector against ITS OWN target.

V7 = unit(P[16:256]·grad S_entropy) is the ENTROPY gradient. Stage-2 (§B.7) scored it against
**dir0** and only dir0 (cross-family; cos(V7, V3_L14) = 0.116) and found it formula-visible /
formula-inert. Its SAME-family target is V_temp (cos(V7, Vtemp_L14) = 0.451) and that cell was
never scored. This runs the identical readout the main lane used for V_temp
(`vmb_c3_orphaned_lever.py`): project each steered cell's signature shift onto the banked C2
orphaned axis, reference to the SAME pooled main-grid alpha=0 riders, and ratio against
matched-SUPPORT nulls.

The ONLY generalization vs the main-lane script is `--null-prefix` (C3's nulls are `Rc*`; the
§B.7 band cells' matched-support nulls are `Rband*` — random units confined to the same
[16:256] band, which is the correct matched null for a band-confined vector per 13e).

Machinery is IMPORTED, not reimplemented, so the floor-z scaling, topic rekeying, corpus
loading and projection are bit-identical to the main-lane readout.

ANNEX RULE: nothing here is quotable until it graduates via a frozen prereg cell.
CPU-only.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from anamnesis.analysis.battery.deltas import ConditionCorpus
from anamnesis.scripts.vmb_arm_a5_analyze import parse_cell_dir, rekey_topic, text_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-run-dir", type=Path, required=True, help="e.g. vmb_b7_3b (V7 + Rband)")
    ap.add_argument("--main-run-dir", type=Path, required=True, help="vmb_a5_3b (alpha=0 riders)")
    ap.add_argument("--axis-npz", type=Path, required=True, help="c2_orphaned_axis_3b.npz")
    ap.add_argument("--null-prefix", default="RBAND",
                    help="uppercase prefix identifying matched-support null vectors")
    ap.add_argument("--model", default="3b")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    # C2 axis + its exact z-normalization (banked with the axis) — identical to the C3 readout
    A = np.load(args.axis_npz)
    axis = A["axis"].astype(np.float64)
    idx = A["feature_indices"].astype(int)
    med, scale = A["med"].astype(np.float64), A["scale"].astype(np.float64)
    logger.info(f"C2 axis: {len(idx)} non-trivial features (norm={np.linalg.norm(axis):.3f})")

    def corpus(d: Path, label: str) -> ConditionCorpus:
        return rekey_topic(ConditionCorpus(d / "signatures_v3", d / "metadata.json",
                                           med.astype(np.float32), scale.astype(np.float32), label))

    # reference (unsteered): pooled main-grid alpha=0 riders — the SAME convention C3 used, so the
    # V7 rows and the banked V_temp rows are referenced to an identical origin.
    riders = sorted(d for d in args.main_run_dir.iterdir()
                    if d.name.endswith("_a0.0") and (d / "signatures_v3").exists())
    if not riders:
        raise SystemExit(f"no alpha=0 rider cells under {args.main_run_dir}")
    ref_Z = np.vstack([corpus(d, f"rider-{d.name}").Z for d in riders])
    ref_centroid = ref_Z.mean(axis=0)
    logger.info(f"reference: {ref_Z.shape[0]} alpha=0 gens from {len(riders)} rider cells")

    rows = []
    for d in sorted(args.cell_run_dir.iterdir()):
        if not (d / "signatures_v3").exists():
            continue
        info = parse_cell_dir(d.name)
        if info is None or info["alpha_frac"] == 0.0:
            continue
        cc = corpus(d, d.name)
        shift = cc.Z.mean(axis=0) - ref_centroid
        targeting = float(abs(shift[idx] @ axis))       # projection onto the ORPHANED coordinate
        total = float(np.linalg.norm(shift))
        st = text_stats(d / "metadata.json")
        rows.append({"cell": d.name, "vector": info["vector"], "site": info["site"],
                     "alpha_frac": info["alpha_frac"], "n": int(cc.Z.shape[0]),
                     "orphaned_targeting": targeting, "total_deformation": total,
                     "efficiency": float(targeting / max(total, 1e-9)),
                     "is_null": info["vector"].upper().startswith(args.null_prefix.upper()),
                     "coherence": {k: st[k] for k in ("mean_len", "mean_ttr", "mean_trigram_rep")}})

    def null_mean(site, af, metric):
        v = [r[metric] for r in rows if r["is_null"] and r["site"] == site and r["alpha_frac"] == af]
        return float(np.mean(v)) if v else None

    for r in rows:
        if r["is_null"]:
            continue
        for metric in ("orphaned_targeting", "efficiency"):
            base = null_mean(r["site"], r["alpha_frac"], metric)
            r[f"{metric}_over_null"] = float(r[metric] / max(base, 1e-9)) if base else None
        r["lever_2x"] = bool((r.get("orphaned_targeting_over_null") or 0) >= 2.0)
        r["clears_1p5x"] = bool((r.get("orphaned_targeting_over_null") or 0) >= 1.5)

    out = {"model": args.model,
           "arm": "ANNEX same-family cell (V7 = grad S_entropy scored against V_temp's coordinate)",
           "STATUS": "ANNEX-GRADE — NOT QUOTABLE until graduated via a frozen prereg cell",
           "null_prefix": args.null_prefix,
           "law": "shift . C2-axis on 1282 non-trivial feats; ref = pooled main-grid alpha=0 riders "
                  "(identical to the C3 V_temp readout); ratio vs matched-SUPPORT nulls at the same "
                  "(site, alpha). §B.7 scored these same cells against dir0 ONLY.",
           "frozen_predictions": {
               "P1_raw_targeting_clears_1.5x_at_alpha_le_.1": 0.80,
               "P2_does_NOT_reach_Vtemp_efficiency": 0.70,
               "P3_B5_tail_law_is_the_explanation": 0.55,
           },
           "rows": rows}
    args.out_json.write_text(json.dumps(out, indent=1))
    logger.info(f"wrote {args.out_json}")
    for r in sorted((x for x in rows if not x["is_null"]), key=lambda x: (x["site"], x["alpha_frac"])):
        logger.info(f"  {r['vector']} L{r['site']} a{r['alpha_frac']:<5} "
                    f"tgt/null={r.get('orphaned_targeting_over_null')} "
                    f"eff/null={r.get('efficiency_over_null')} lever2x={r['lever_2x']} "
                    f"ttr={r['coherence']['mean_ttr']:.3f}")


if __name__ == "__main__":
    main()
