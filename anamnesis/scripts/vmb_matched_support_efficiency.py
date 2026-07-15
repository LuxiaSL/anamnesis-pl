"""Matched-support efficiency analyzer (session-4 Part A1; DESIGN-V4 §B.5 + §B.7).

targeting ÷ deformation with EVERY vector read against ITS OWN support-matched null —
R_top/R_tail for the §B.5 spectral split, band-[16:256] randoms for §B.7 stage-2. ⚠ THE
constraint (campaign-log): never pool nulls across supports. A tail-confined vector is
mechanically high-deformation; a pooled-R baseline that mixes top/tail/band silently
corrupts both readouts. Here the null SUPPORT is inferred from the vector NAME
(top/tail/band/full) and nulls are aggregated only within a support, at matched α.

Metrics (floor-z signature space, reference = pooled main-grid α=0 riders):
  shift          = mean(Z_cell) − ref_centroid
  targeting      = |shift · dir0_axis|            (dir0 = LDA analogical-vs-contrastive, unit)
  deformation    = ‖shift‖                        (total state movement)
  off_target     = sqrt(deformation² − targeting²)
  effect_per_off = targeting / off_target         (A5-inv metric of record; B2 lens)
  efficiency     = targeting / deformation         (§B.5 primary cross-subspace lens)
Construction-level check (vector-only): maha_v = vᵀΣ⁻¹v from banked Σ_L14 (confirms
top-cheap / tail-expensive; a CHECK, never the efficiency denominator).

CPU only. First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from anamnesis.analysis.battery.deltas import ConditionCorpus, load_floor_scale
from anamnesis.analysis.battery.manifest import MODEL_META
from anamnesis.scripts.vmb_arm_a5_analyze import parse_cell_dir, rekey_topic, text_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]
DIR0_PAIR = ("analogical", "contrastive")
PROMOTE = 1.5  # the standing 1.5×-matched-null promotion rule


def support_of(vec: str) -> str:
    """Infer null-matching support class from the vector NAME (never mix these)."""
    v = vec.lower()
    if "top" in v:
        return "top"
    if "tail" in v:
        return "tail"
    if "band" in v or vec == "V7":
        return "band"
    return "full"


def is_null(vec: str) -> bool:
    """R_top / R_tail / R_band / R1-3 are nulls; V-vectors are targets."""
    return vec.upper().startswith("R")


def maha_inv(v: F32, evals: NDArray, evecs: NDArray) -> float:
    """vᵀΣ⁻¹v from the banked eigendecomposition (evals may be ascending)."""
    c = evecs.T @ v.astype(np.float64)
    ev = np.clip(evals.astype(np.float64), 1e-12, None)
    return float(np.sum(c * c / ev))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--split-run-dir", type=Path, required=True, help="vmb_b5_3b or vmb_b7_3b")
    ap.add_argument("--main-run-dir", type=Path, required=True, help="vmb_a5_3b (V3 + R1-3 + α=0 riders)")
    ap.add_argument("--sigma", type=Path, required=True, help="a5_sigma_L14_3b.npz")
    ap.add_argument("--split-vectors", type=Path, required=True, help="split a5_vectors.npz (V3top/V7/… keys)")
    ap.add_argument("--main-vectors", type=Path, default=None, help="main a5_vectors.npz (V3_L14/R{1,2,3}_L14)")
    ap.add_argument("--arm", required=True, choices=["b5", "b7"])
    ap.add_argument("--model", default="3b")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    model = args.model
    mm = MODEL_META[model]

    # ── z-normalization + dir0 axis (shared with the main A5 analyzer) ──
    stage0 = args.battery_root / mm.stage0_dir
    med, scale = load_floor_scale(stage0 / "signatures_v3")

    def corpus(d: Path, label: str) -> ConditionCorpus:
        return rekey_topic(ConditionCorpus(d / "signatures_v3", d / "metadata.json", med, scale, label))

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    pures = {}
    for m_ in DIR0_PAIR:
        d = args.battery_root / f"vmb_a2_{model}_pure_{m_}"
        pures[m_] = ConditionCorpus(d / "signatures_v3", d / "metadata.json", med, scale, f"pure-{m_}")
    Xp = np.vstack([pures[DIR0_PAIR[0]].Z, pures[DIR0_PAIR[1]].Z])
    yp = np.r_[np.ones(pures[DIR0_PAIR[0]].Z.shape[0]), np.zeros(pures[DIR0_PAIR[1]].Z.shape[0])]
    axis = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xp, yp).coef_[0].astype(np.float64)
    axis /= np.linalg.norm(axis)
    logger.info(f"dir0 axis fit on {Xp.shape[0]} pure gens ({DIR0_PAIR})")

    # ── reference centroid: pooled α=0 riders from the MAIN grid ──
    rider_dirs = sorted(d for d in args.main_run_dir.iterdir()
                        if d.name.endswith("_a0.0") and (d / "signatures_v3").exists())
    if not rider_dirs:
        raise SystemExit(f"no α=0 rider cells under {args.main_run_dir}")
    ref_Z = np.vstack([corpus(d, f"rider-{d.name}").Z for d in rider_dirs])
    ref_centroid = ref_Z.mean(axis=0)
    logger.info(f"reference: {ref_Z.shape[0]} α=0 gens from {len(rider_dirs)} rider cells")

    # ── Σ eigendecomposition (construction-level maha check) ──
    S = np.load(args.sigma)
    evals, evecs = S["evals"], S["evecs"]
    split_vecs = dict(np.load(args.split_vectors))
    if args.main_vectors:
        for k, v in np.load(args.main_vectors).items():
            split_vecs.setdefault(k, v)

    # ── discover cells: split run dir + (V3 + R1-3 full-support) from the main grid ──
    def discover(run_dir: Path, keep=None, site=None) -> dict:
        out = {}
        for d in sorted(run_dir.iterdir()) if run_dir.exists() else []:
            if not (d / "signatures_v3").exists():
                continue
            info = parse_cell_dir(d.name)
            if info is None or info["alpha_frac"] == 0.0:
                continue
            if keep and info["vector"] not in keep:
                continue
            if site is not None and info["site"] != site:
                continue
            out[d.name] = info | {"dir": d}
        return out

    cells = discover(args.split_run_dir)
    # full-support V3/R reference: MAP SITE (L14) only — the split vectors are all L14-derived,
    # and the main grid injected V3 at all four sites (mixing sites corrupts the parity denominator).
    cells.update(discover(args.main_run_dir, keep={"V3", "R1", "R2", "R3"}, site=14))
    logger.info(f"{len(cells)} steered cells (split + full-support V3/R reference)")

    rows = []
    for cname, info in sorted(cells.items()):
        cc = corpus(info["dir"], cname)
        shift = cc.Z.mean(axis=0) - ref_centroid
        deform = float(np.linalg.norm(shift))
        tgt = float(abs(shift @ axis))
        off = float(np.sqrt(max(deform ** 2 - tgt ** 2, 0.0)))
        vec = info["vector"]
        vkey = f"{vec}_L{info['site']}"
        v_unit = split_vecs.get(vkey, split_vecs.get(f"{vec}_L14"))
        st = text_stats(info["dir"] / "metadata.json")
        rows.append({
            "cell_run": cname, "vector": vec, "support": support_of(vec),
            "is_null": is_null(vec), "site": info["site"], "alpha_frac": info["alpha_frac"],
            "n": int(cc.Z.shape[0]),
            "targeting": tgt, "deformation": deform, "off_target": off,
            "effect_per_offtarget": float(tgt / max(off, 1e-9)),
            "efficiency": float(tgt / max(deform, 1e-9)),
            "maha_inv_construction": maha_inv(np.asarray(v_unit), evals, evecs) if v_unit is not None else None,
            "coherence": {k: st[k] for k in ("mean_len", "mean_ttr", "mean_trigram_rep")},
        })

    # ── matched-support aggregation: target metric ÷ mean(null metric) within (support, α) ──
    def null_mean(metric: str, support: str, af: float) -> float | None:
        vals = [r[metric] for r in rows if r["is_null"] and r["support"] == support and r["alpha_frac"] == af]
        return float(np.mean(vals)) if vals else None

    for r in rows:
        if r["is_null"]:
            continue
        af, sup = r["alpha_frac"], r["support"]
        for metric in ("effect_per_offtarget", "efficiency", "targeting"):
            base = null_mean(metric, sup, af)
            r[f"{metric}_over_matched_null"] = float(r[metric] / max(base, 1e-9)) if base is not None else None
        # promotion flag on the primary lens (efficiency), matched-support
        eff_ratio = r.get("efficiency_over_matched_null")
        r["clears_1.5x_matched_null"] = bool(eff_ratio is not None and eff_ratio >= PROMOTE)
        # parity-vs-V3: target's effect ÷ full-support V3's effect at the same α
        v3 = next((x for x in rows if x["vector"] == "V3" and x["alpha_frac"] == af), None)
        if v3 is not None and r["vector"] != "V3":
            r["effect_vs_V3"] = float(r["effect_per_offtarget"] / max(v3["effect_per_offtarget"], 1e-9))
            r["efficiency_vs_V3"] = float(r["efficiency"] / max(v3["efficiency"], 1e-9))

    out = {
        "model": model, "arm": args.arm,
        "STATUS": "FIRST_READ_PENDING (C§8 — no stamps ship before outer-loop read)",
        "law": "matched-support efficiency; dir0=LDA(analogical,contrastive) unit z-space; "
               "targeting=|shift·dir0|, deformation=‖shift‖, ref=pooled main-grid α=0 riders; "
               "nulls grouped by NAME-inferred support (top/tail/band/full), matched α; 1.5×-null promotion rule",
        "supports_present": sorted({r["support"] for r in rows}),
        "n_reference_gens": int(ref_Z.shape[0]),
        "rows": rows,
    }
    args.out_json.write_text(json.dumps(out, indent=1))
    logger.info(f"wrote {args.out_json}")
    # console summary: targets only, primary lens
    for r in sorted((x for x in rows if not x["is_null"]), key=lambda x: (x["vector"], x["alpha_frac"])):
        logger.info(f"  {r['vector']:9s} α{r['alpha_frac']:<5} sup={r['support']:4s} "
                    f"eff={r['efficiency']:.3f} eff/null={r.get('efficiency_over_matched_null')} "
                    f"eff/V3={r.get('efficiency_vs_V3')} clears1.5x={r['clears_1.5x_matched_null']}")


if __name__ == "__main__":
    main()
