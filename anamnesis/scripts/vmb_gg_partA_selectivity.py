"""Graded-Goodhart Part A — uniform C§6 selectivity re-derivation + Σ-geometry + roster inventory
(DESIGN-graded-goodhart-selectivity-2026-07-15; session-7 baton Part-A item 2).

ONE pipeline pass over banked replays for EVERY roster vector (never harvested from heterogeneous
per-arm JSONs — different floors/conventions would manufacture the correlation). Uniform:
  - dir0 axis   = pure analogical−contrastive LDA, floor-z (unit)
  - reference   = pooled 480-gen α=0 riders (vmb_a5_3b), shared across all cells
  - SELECTIVITY = C§6 effect_per_offtarget = |shift·dir0| / sqrt(|shift|²−|shift·dir0|²)  [sig space]
  - EFFICACY(a) = raw target movement |shift·dir0|                                        [sig space]
  - EFFICACY(b) = frac_analogical = clf.predict(cell.Z).mean()  (behavioral; α≤.3 scope)
  - GEOMETRY    = vᵀΣ⁻¹v at L14 (banked Σ_L14, ridge as banked) + top/bottom-768 eigenmass [resid space]

⚠ P1/P2/P3 NOT scored this session (P3 sequencing waits on 14j; roster has holes → correlation is
PRELIMINARY, direction-only). Output = the uniform table + the INVENTORY of which roster members
lack a selectivity triple at α=.1 (→ the generation ask, which needs its own ratification addendum).
CPU-only, banked. First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from anamnesis.analysis.battery.deltas import ConditionCorpus, load_floor_scale
from anamnesis.analysis.battery.manifest import MODEL_META
from anamnesis.scripts.vmb_arm_a5_analyze import parse_cell_dir, rekey_topic

DIR0_PAIR = ("analogical", "contrastive")

# roster: label -> (run_dir, vec_npz, vec_key, site). site 14 unless noted.
ROSTER = {
    "V1": ("vmb_a5_3b", "a5_vectors_3b", "V1_L14", 14),
    "V2": ("vmb_a5_3b", "a5_vectors_3b", "V2_L13", 13),
    "V3": ("vmb_a5_3b", "a5_vectors_3b", "V3_L14", 14),
    "V4": ("vmb_a5_3b", "a5_vectors_3b", "V4_L14", 14),
    "V3selbare": ("vmb_a5_3b", "a5_vectors_3b_v3sel", "V3sel_bare_L14", 14),
    "R1": ("vmb_a5_3b", "a5_vectors_3b", "R1", 14),
    "R2": ("vmb_a5_3b", "a5_vectors_3b", "R2", 14),
    "R3": ("vmb_a5_3b", "a5_vectors_3b", "R3", 14),
    "V3top": ("vmb_b5_3b", "a5_vectors_3b_b5", "V3top_L14", 14),
    "V3tail": ("vmb_b5_3b", "a5_vectors_3b_b5", "V3tail_L14", 14),
    "Rtop1": ("vmb_b5_3b", "a5_vectors_3b_b5", "Rtop1_L14", 14),
    "Rtop2": ("vmb_b5_3b", "a5_vectors_3b_b5", "Rtop2_L14", 14),
    "Rtop3": ("vmb_b5_3b", "a5_vectors_3b_b5", "Rtop3_L14", 14),
    "Rtail1": ("vmb_b5_3b", "a5_vectors_3b_b5", "Rtail1_L14", 14),
    "Rtail2": ("vmb_b5_3b", "a5_vectors_3b_b5", "Rtail2_L14", 14),
    "Rtail3": ("vmb_b5_3b", "a5_vectors_3b_b5", "Rtail3_L14", 14),
    "V7": ("vmb_b7_3b", "a5_vectors_3b_b7", "V7_L14", 14),
    "Rband1": ("vmb_b7_3b", "a5_vectors_3b_b7", "Rband1_L14", 14),
    "Rband2": ("vmb_b7_3b", "a5_vectors_3b_b7", "Rband2_L14", 14),
    "Rband3": ("vmb_b7_3b", "a5_vectors_3b_b7", "Rband3_L14", 14),
    "Vtemp": ("vmb_c3_3b", "vector_banks/a5_vectors_3b_c3", "Vtemp_L14", 14),
    "Rc1": ("vmb_c3_3b", "vector_banks/a5_vectors_3b_c3", "Rc1_L14", 14),
    "Rc2": ("vmb_c3_3b", "vector_banks/a5_vectors_3b_c3", "Rc2_L14", 14),
    "Rc3": ("vmb_c3_3b", "vector_banks/a5_vectors_3b_c3", "Rc3_L14", 14),
    "V1b": ("vmb_a5_3b", "a5_vectors_3b_v1b", "V1b_L14", 14),   # vector built; NO gen cell → HOLE
    # 14r cell R-A (ratified 9064bea; annex runs it). Roster row only — every statistic for RA
    # comes from the SAME code paths as the banked roster (rider 2: no new readout code).
    "RA": ("annex/vmb_14r_3b", "annex/a5_vectors_3b_14r", "RA_L14", 14),
}
IS_NULL = lambda lbl: lbl.upper().startswith(("R",)) and lbl.upper() != "RA"  # RA = 14r cell, not a null
DATA_ROUTE = {"V1", "V2", "V3", "V3selbare", "Vtemp", "V1b", "V3top", "V3tail"}  # Δμ / contrast
FORMULA_ROUTE = {"V4", "V7", "RA"}                                               # gradient


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--model", default="3b")
    ap.add_argument("--sigma-npz", type=Path, required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.03, 0.1, 0.3])
    ap.add_argument("--primary-alpha", type=float, default=0.1)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    br, mm = args.battery_root, MODEL_META[args.model]
    med, scale = load_floor_scale(br / mm.stage0_dir / "signatures_v3")

    def cc(d: Path, label: str) -> ConditionCorpus:
        return rekey_topic(ConditionCorpus(d / "signatures_v3", d / "metadata.json", med, scale, label))

    # dir0 axis + behavioral classifier (identical to F-rung)
    pures = {m: ConditionCorpus(br / f"vmb_a2_{args.model}_pure_{m}" / "signatures_v3",
                                br / f"vmb_a2_{args.model}_pure_{m}" / "metadata.json", med, scale, m)
             for m in DIR0_PAIR}
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
        np.vstack([pures[DIR0_PAIR[0]].Z, pures[DIR0_PAIR[1]].Z]),
        np.r_[np.ones(len(pures[DIR0_PAIR[0]].Z)), np.zeros(len(pures[DIR0_PAIR[1]].Z))])
    dir0 = clf.coef_[0].astype(np.float64); dir0 /= np.linalg.norm(dir0)

    # pooled α=0 rider reference (shared)
    a5 = br / f"vmb_a5_{args.model}"
    ref_centroid = np.vstack([cc(d, d.name).Z for d in sorted(a5.iterdir())
                              if d.name.endswith("_a0.0") and (d / "signatures_v3").exists()]).mean(0)

    # Σ geometry
    S = np.load(args.sigma_npz)
    evals, evecs, ridge = S["evals"].astype(np.float64), S["evecs"].astype(np.float64), float(S["ridge"])
    inv = 1.0 / (evals + ridge)
    order = np.argsort(evals)[::-1]
    top, bot = order[:768], order[-768:]

    def geometry(vec_npz: str, key: str):
        p = br / vec_npz / "a5_vectors.npz"
        if not p.exists() or key not in np.load(p):
            return None
        v = np.load(p)[key].astype(np.float64); v /= np.linalg.norm(v)
        proj = evecs.T @ v
        e = proj ** 2
        return {"mahalanobis": float(np.sum(e * inv)),
                "top_768_eigenmass": float(e[top].sum() / e.sum()),
                "bottom_768_eigenmass": float(e[bot].sum() / e.sum())}

    def find_cell(run_dir: Path, label: str, site: int, af: float):
        for d in sorted(run_dir.iterdir()) if run_dir.exists() else []:
            info = parse_cell_dir(d.name)
            if info and info["vector"] == label and info["site"] == site \
                    and abs(info["alpha_frac"] - af) < 1e-9 and (d / "signatures_v3").exists():
                return d
        return None

    rows, inventory = [], {}
    for label, (run, vnpz, vkey, site) in ROSTER.items():
        run_dir = br / run
        geo = geometry(vnpz, vkey)
        present = {}
        for af in args.alphas:
            cell = find_cell(run_dir, label, site, af)
            present[af] = cell is not None
            if cell is None:
                continue
            C = cc(cell, f"{label}-a{af}")
            shift = C.Z.mean(0) - ref_centroid
            tgt = float(abs(shift @ dir0))
            total = float(np.linalg.norm(shift))
            off = float(np.sqrt(max(total ** 2 - tgt ** 2, 0)))
            rows.append({
                "vector": label, "route": ("null" if IS_NULL(label) else
                                           "data" if label in DATA_ROUTE else
                                           "formula" if label in FORMULA_ROUTE else "other"),
                "site": site, "alpha_frac": af, "n": int(C.Z.shape[0]),
                "selectivity_effect_per_offtarget": round(tgt / max(off, 1e-9), 5),
                "efficacy_raw_target_movement": round(tgt, 5),
                "efficacy_behavior_frac_analogical": round(float(clf.predict(C.Z).mean()), 5),
                "geometry_mahalanobis": round(geo["mahalanobis"], 3) if geo else None,
                "geometry_top768_eigenmass": round(geo["top_768_eigenmass"], 4) if geo else None,
                "geometry_bottom768_eigenmass": round(geo["bottom_768_eigenmass"], 4) if geo else None,
            })
        has_triple = all(present.get(a, False) for a in args.alphas)
        inventory[label] = {"site": site, "cells_present": {str(a): present.get(a, False) for a in args.alphas},
                            "has_selectivity_triple": has_triple,
                            "geometry_available": geo is not None}

    # PRELIMINARY correlation at primary α (direction-only; roster has holes, P3 deferred)
    prim = [r for r in rows if abs(r["alpha_frac"] - args.primary_alpha) < 1e-9
            and r["geometry_mahalanobis"] is not None]
    corr = None
    if len(prim) >= 4:
        m = np.array([r["geometry_mahalanobis"] for r in prim])
        s = np.array([r["selectivity_effect_per_offtarget"] for r in prim])
        rho, p = spearmanr(m, s)
        corr = {"n_roster": len(prim), "alpha": args.primary_alpha,
                "spearman_rho_mahal_vs_selectivity": round(float(rho), 4),
                "asymptotic_p": round(float(p), 4),
                "note": "PRELIMINARY / direction-only — roster incomplete + permutation-null and "
                        "P1/P2/P3 scoring deferred (P3 waits on 14j per sequencing rule)",
                "members": [r["vector"] for r in prim]}

    holes = [k for k, v in inventory.items() if not v["has_selectivity_triple"]]
    out = {"arm": "graded-Goodhart Part A — uniform selectivity + Σ-geometry + inventory",
           "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop; P1/P2/P3 NOT scored this session",
           "model": args.model,
           "law": "uniform: dir0=pure-LDA floor-z; ref=pooled 480 α=0 riders; selectivity=C§6 "
                  "effect_per_offtarget (sig space); geometry=vᵀΣ⁻¹v @L14 (resid space, banked Σ)",
           "roster_inventory": inventory,
           "holes_lacking_selectivity_triple": holes,
           "hole_note": "V1b = vector built (a5_vectors_3b_v1b) but NO free-gen cell exists → the "
                        "generation ask for a full triple; needs its own ratification addendum "
                        "(cross-site V3sel_bare_L{7,18,21}/V1_L7 also uninventoried here — L14-scoped pass)",
           "preliminary_correlation": corr,
           "rows": rows}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=1))
    # console
    print(f"roster members: {len(inventory)} | holes (no α-triple): {holes}")
    if corr:
        print(f"PRELIMINARY ρ(mahal, selectivity) @α={corr['alpha']}: {corr['spearman_rho_mahal_vs_selectivity']} "
              f"(n={corr['n_roster']}, p={corr['asymptotic_p']}) — direction-only, NOT scored")
    print(f"{'vector':11} {'route':8} {'sel':>8} {'eff_raw':>8} {'frac':>6} {'mahal':>8} {'top768':>7}")
    for r in sorted((x for x in rows if abs(x['alpha_frac']-args.primary_alpha) < 1e-9),
                    key=lambda x: x['geometry_mahalanobis'] or 0):
        print(f"  {r['vector']:11} {r['route']:8} {r['selectivity_effect_per_offtarget']:>8.4f} "
              f"{r['efficacy_raw_target_movement']:>8.4f} {r['efficacy_behavior_frac_analogical']:>6.3f} "
              f"{r['geometry_mahalanobis'] or 0:>8.2f} {r['geometry_top768_eigenmass'] or 0:>7.3f}")


if __name__ == "__main__":
    main()
