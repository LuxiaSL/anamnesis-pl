"""R9 — duality rider (14a §5): is the PROMPT-induced mode state the same as the
INJECTION-induced mode state, in PROFILE space? (2×2 standing rule: trait identity lives
in profile space = readout-vector × dose, never per-facet orthogonality.)

Two mode-induction routes, each as a DEFORMATION DIRECTION in floor-z signature space
(centroids cancel content):
  Δ_prompt  = centroid(pure_analogical) − centroid(pure_contrastive)   [prompt route]
  Δ_inject  = centroid(V3-steered @α)   − centroid(V3 @α=0 rider)       [injection route]
  Δ_random  = centroid(R1-steered @α)   − centroid(R1 @α=0 rider)       [generic-deformation null]

Metrics:
  cos(Δ_prompt, Δ_inject)  — profile alignment. ~1 => IDENTICAL routes; <1 => DISTINCT.
  cos(Δ_prompt, Δ_random)  — the generic-deformation null the alignment is judged against.
  dir0 projection of each  — the SHARED mode component (both should load positively on dir0).
  off-dir0 residual cos    — distinctness AFTER removing the shared mode axis.
Both outcomes named: DISTINCT (cos_pi meaningfully <1, and > null => specific-but-distinct)
vs IDENTICAL (cos_pi ≈ 1). Filed R9: DISTINCT, P=0.80.

First-read → outer loop; nothing stamped. CPU. Run from repo root.
Usage: python -m anamnesis.scripts.vmb_a5_duality --model 3b --alpha 0.3 --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from anamnesis.analysis.battery.deltas import ConditionCorpus, load_floor_scale
from anamnesis.analysis.battery.manifest import MODEL_META

BATTERY = Path("outputs/battery")


def _cc(rel: str, med, scale, label):
    d = BATTERY / rel
    return ConditionCorpus(d / "signatures_v3", d / "metadata.json", med, scale, label)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _cos(a, b):
    return float(np.dot(_unit(a), _unit(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=list(MODEL_META.keys()))
    ap.add_argument("--alpha", default="0.3", help="injection dose cell to compare (default 0.3)")
    ap.add_argument("--map-site", type=int, default=14)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    m, M, a = args.model, args.map_site, args.alpha
    meta = MODEL_META[m]
    med, scale = load_floor_scale(BATTERY / meta.stage0_dir / "signatures_v3")

    # dir0 = analogical↔contrastive LDA unit axis (floor-z) — the mode coordinate
    ana = _cc(f"vmb_a2_{m}_pure_analogical", med, scale, "ana")
    con = _cc(f"vmb_a2_{m}_pure_contrastive", med, scale, "con")
    X = np.vstack([ana.Z, con.Z]); y = np.r_[np.ones(len(ana.Z)), np.zeros(len(con.Z))]
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)
    dir0 = _unit(lda.coef_[0].astype(np.float64))

    # route deformation directions
    d_prompt = ana.Z.mean(0) - con.Z.mean(0)
    v3s = _cc(f"vmb_a5_{m}/V3_L{M}_L{M}_a{a}", med, scale, "v3s")
    v30 = _cc(f"vmb_a5_{m}/V3_L{M}_L{M}_a0.0", med, scale, "v30")
    d_inject = v3s.Z.mean(0) - v30.Z.mean(0)
    r1s = _cc(f"vmb_a5_{m}/R1_L{M}_a{a}", med, scale, "r1s")
    r10 = _cc(f"vmb_a5_{m}/R1_L{M}_a0.0", med, scale, "r10")
    d_random = r1s.Z.mean(0) - r10.Z.mean(0)

    cos_pi = _cos(d_prompt, d_inject)
    cos_pr = _cos(d_prompt, d_random)
    proj = {"prompt_on_dir0": float(np.dot(_unit(d_prompt), dir0)),
            "inject_on_dir0": float(np.dot(_unit(d_inject), dir0)),
            "random_on_dir0": float(np.dot(_unit(d_random), dir0))}
    # off-dir0 residual (remove the shared mode axis, compare what's left)
    def _resid(v):
        u = _unit(v); return u - np.dot(u, dir0) * dir0
    cos_pi_offdir0 = _cos(_resid(d_prompt), _resid(d_inject))

    # verdict: DISTINCT iff not identical (cos_pi < 0.9) — both outcomes named
    verdict = "DISTINCT" if cos_pi < 0.9 else "IDENTICAL"
    specificity = ("specific-but-distinct (inject aligns with prompt > random-null)"
                   if cos_pi > cos_pr + 0.05 else
                   "generic (inject ≈ random-null alignment)")

    res = {"model": m, "alpha": a, "map_site": M,
           "cos_prompt_inject": cos_pi, "cos_prompt_random_NULL": cos_pr,
           "cos_prompt_inject_off_dir0": cos_pi_offdir0,
           "dir0_projections": proj,
           "n": {"analogical": len(ana.Z), "contrastive": len(con.Z),
                 "v3_steered": len(v3s.Z), "v3_rider": len(v30.Z),
                 "r1_steered": len(r1s.Z), "r1_rider": len(r10.Z)},
           "VERDICT": verdict, "specificity": specificity,
           "filed": "R9 DISTINCT P=0.80",
           "law": "profile-space (deformation-direction cosine, floor-z); dir0 = ana↔con LDA unit"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
