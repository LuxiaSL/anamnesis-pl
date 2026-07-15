"""C3 orphaned-lever readout (PREFLIGHT §4 leg (d); session-4 Part C). Does V_temp move the
banked C2 ORPHANED COORDINATE (the t03-vs-t09 discriminant on non-trivial surfaces) more than
the matched-norm Rc nulls? Projects each steered-cell signature shift onto the C2 axis
(`c2_orphaned_axis_3b.npz`) — NOT dir0. (d) = targeting(V_temp) ÷ targeting(mean Rc) at matched
(site, α); ≥ 2 → the coordinate is rank-1-WRITABLE. Masking is primary (the axis lives on the
1282 non-trivial features); length-resid is the C2 robustness column, not re-applied here.

⚠ LEVER leg only. The certifying consequences (b entropy / c A1-dissociation / f resample-
diversity) need the logit-retaining replay (overnight-staged). A lever WITHOUT certifying =
PREFLIGHT outcome 2 "constitutively token-mediated" territory (C2+C4 predicted it).
CPU-only. First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from anamnesis.analysis.battery.deltas import ConditionCorpus, load_floor_scale
from anamnesis.analysis.battery.manifest import MODEL_META
from anamnesis.scripts.vmb_arm_a5_analyze import parse_cell_dir, rekey_topic, text_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--c3-run-dir", type=Path, required=True, help="vmb_c3_3b")
    ap.add_argument("--main-run-dir", type=Path, required=True, help="vmb_a5_3b (α=0 riders)")
    ap.add_argument("--axis-npz", type=Path, required=True, help="c2_orphaned_axis_3b.npz")
    ap.add_argument("--model", default="3b")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    mm = MODEL_META[args.model]

    # C2 axis + its exact z-normalization (banked with the axis)
    A = np.load(args.axis_npz)
    axis = A["axis"].astype(np.float64)               # unit, on the non-trivial features
    idx = A["feature_indices"].astype(int)            # 1282 non-trivial feature columns
    med, scale = A["med"].astype(np.float64), A["scale"].astype(np.float64)
    logger.info(f"C2 axis: {len(idx)} non-trivial features (‖axis‖={np.linalg.norm(axis):.3f})")

    def corpus(d: Path, label: str) -> ConditionCorpus:
        return rekey_topic(ConditionCorpus(d / "signatures_v3", d / "metadata.json",
                                           med.astype(np.float32), scale.astype(np.float32), label))

    # reference (unsteered): pooled main-grid α=0 riders — same convention as the other analyses
    riders = sorted(d for d in args.main_run_dir.iterdir()
                    if d.name.endswith("_a0.0") and (d / "signatures_v3").exists())
    ref_Z = np.vstack([corpus(d, f"rider-{d.name}").Z for d in riders])
    ref_centroid = ref_Z.mean(axis=0)
    logger.info(f"reference: {ref_Z.shape[0]} α=0 gens from {len(riders)} rider cells")

    # discover C3 cells
    rows = []
    for d in sorted(args.c3_run_dir.iterdir()):
        if not (d / "signatures_v3").exists():
            continue
        info = parse_cell_dir(d.name)
        if info is None or info["alpha_frac"] == 0.0:
            continue
        cc = corpus(d, d.name)
        shift = cc.Z.mean(axis=0) - ref_centroid
        targeting = float(abs(shift[idx] @ axis))     # projection onto the orphaned coordinate
        total = float(np.linalg.norm(shift))
        st = text_stats(d / "metadata.json")
        rows.append({"cell": d.name, "vector": info["vector"], "site": info["site"],
                     "alpha_frac": info["alpha_frac"], "n": int(cc.Z.shape[0]),
                     "orphaned_targeting": targeting, "total_deformation": total,
                     "efficiency": float(targeting / max(total, 1e-9)),
                     "is_null": info["vector"].upper().startswith("RC"),
                     "coherence": {k: st[k] for k in ("mean_len", "mean_ttr", "mean_trigram_rep")}})

    # (d) matched: V_temp targeting ÷ mean(Rc targeting) at each (site, α)
    def rc_mean(site, af, metric):
        v = [r[metric] for r in rows if r["is_null"] and r["site"] == site and r["alpha_frac"] == af]
        return float(np.mean(v)) if v else None

    for r in rows:
        if r["is_null"]:
            continue
        for metric in ("orphaned_targeting", "efficiency"):
            base = rc_mean(r["site"], r["alpha_frac"], metric)
            r[f"{metric}_over_Rc"] = float(r[metric] / max(base, 1e-9)) if base else None
        r["lever_2x"] = bool((r.get("orphaned_targeting_over_Rc") or 0) >= 2.0)

    out = {"model": args.model, "arm": "C3-orphaned-lever (PREFLIGHT §4 (d))",
           "STATUS": "FIRST_READ_PENDING (C§8) — LEVER leg only; certifying (b)/(c)/(f) need logit replay",
           "law": "shift·C2-axis on 1282 non-trivial feats; ref=pooled main-grid α=0 riders; "
                  "(d) = Vtemp orphaned-targeting ÷ mean(Rc) at matched (site,α); ≥2 = rank-1-writable",
           "rows": rows}
    args.out_json.write_text(json.dumps(out, indent=1))
    logger.info(f"wrote {args.out_json}")
    for r in sorted((x for x in rows if not x["is_null"]), key=lambda x: (x["site"], x["alpha_frac"])):
        logger.info(f"  Vtemp L{r['site']} α{r['alpha_frac']:<5} tgt/Rc={r.get('orphaned_targeting_over_Rc')} "
                    f"eff/Rc={r.get('efficiency_over_Rc')} lever2x={r['lever_2x']} ttr={r['coherence']['mean_ttr']:.3f}")


if __name__ == "__main__":
    main()
