"""Graded-Goodhart Part A — P1/P2 permutation-null scoring harness
(DESIGN-graded-goodhart-selectivity-2026-07-15 §1/§4; session-8 baton Part A4).

Reads the banked `gg_partA_selectivity` table and scores the frozen claims with a
PERMUTATION null over roster labels (§4: "Correlation significance by permutation over
roster labels, not asymptotic p"):

  P1 (positive, P=.75): spearman(mahalanobis, selectivity) NEGATIVE + significant at the
     primary dose (α=.1), surviving the label-permutation null.
  P2 (the must-hold negative, P=.85): mahalanobis does NOT predict EFFICACY — neither raw
     target movement (a) nor behavioral consummation (b); |ρ| n.s. against the same null.
  P3 (construction split): data-route Δμ vs formula-route on efficacy CLASS. ⚠ NOT SCORED
     HERE — P3 wording is outer-loop-owned (14j sequencing) and V7's efficacy entry uses
     its SAME-family temperature result; this harness reports the raw data/formula split
     numbers for the outer loop but files no P3 verdict.

Scoring waits on the V1b row (Part B) to complete the roster to n≈25; run now on the
banked n=24 table to VERIFY the harness, re-run once V1b lands. Robustness columns:
targets-only (excl. nulls) beside all-roster; a positive P1 must not be a nulls artifact.

CPU-only, reads the banked JSON. First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def _perm_p(x: np.ndarray, y: np.ndarray, observed: float, n_perm: int, seed: int,
            two_sided: bool) -> float:
    """Permutation p: shuffle y labels against x, rebuild spearman ρ, count as-extreme."""
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = spearmanr(x, rng.permutation(y)).statistic
    if two_sided:
        return float((np.abs(null) >= abs(observed) - 1e-12).mean())
    # one-sided NEGATIVE (P1): fraction of null ρ ≤ observed
    return float((null <= observed + 1e-12).mean())


def _corr(rows: list[dict], xkey: str, ykey: str, n_perm: int, seed: int,
          two_sided: bool) -> dict | None:
    pts = [(r[xkey], r[ykey]) for r in rows
           if r.get(xkey) is not None and r.get(ykey) is not None]
    if len(pts) < 4:
        return None
    x = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    rho = float(spearmanr(x, y).statistic)
    return {"n": len(pts), "spearman_rho": round(rho, 4),
            "perm_p": round(_perm_p(x, y, rho, n_perm, seed, two_sided), 4),
            "test": "two-sided" if two_sided else "one-sided-negative",
            "n_perm": n_perm, "seed": seed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gg-json", type=Path, required=True, help="gg_partA_selectivity_3b.json")
    ap.add_argument("--primary-alpha", type=float, default=0.1)
    ap.add_argument("--n-perm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260715)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    gg = json.loads(args.gg_json.read_text())
    rows = [r for r in gg["rows"] if abs(r["alpha_frac"] - args.primary_alpha) < 1e-9
            and r.get("geometry_mahalanobis") is not None]
    targets = [r for r in rows if r["route"] != "null"]

    MAHAL, SEL = "geometry_mahalanobis", "selectivity_effect_per_offtarget"
    EFF_A, EFF_B = "efficacy_raw_target_movement", "efficacy_behavior_frac_analogical"

    def block(rs: list[dict]) -> dict:
        return {
            "P1_mahal_vs_selectivity": _corr(rs, MAHAL, SEL, args.n_perm, args.seed, two_sided=False),
            "P2a_mahal_vs_efficacy_raw": _corr(rs, MAHAL, EFF_A, args.n_perm, args.seed + 1, two_sided=True),
            "P2b_mahal_vs_efficacy_behavior": _corr(rs, MAHAL, EFF_B, args.n_perm, args.seed + 2, two_sided=True),
        }

    all_roster = block(rows)
    targets_only = block(targets)

    # data-vs-formula efficacy split (RAW numbers for the outer loop; P3 verdict NOT filed here)
    def route_stats(route: str, key: str) -> dict:
        vals = [r[key] for r in rows if r["route"] == route and r.get(key) is not None]
        return {"n": len(vals), "mean": round(float(np.mean(vals)), 5) if vals else None,
                "members": sorted({r["vector"] for r in rows if r["route"] == route})}
    p3_raw = {rt: {k: route_stats(rt, k) for k in (SEL, EFF_A, EFF_B, MAHAL)}
              for rt in ("data", "formula", "null", "other")}

    # verdict scoring (P1/P2 only; the V1b hole gates the FINAL call)
    p1 = all_roster["P1_mahal_vs_selectivity"]
    p2a = all_roster["P2a_mahal_vs_efficacy_raw"]
    p2b = all_roster["P2b_mahal_vs_efficacy_behavior"]
    v1b_present = any(r["vector"] == "V1b" for r in rows)

    def verdict_p1(c: dict | None) -> str:
        if c is None:
            return "insufficient-n"
        return ("INSIDE (neg + sig, perm p<.05)" if c["spearman_rho"] < 0 and c["perm_p"] < 0.05
                else "neg-not-sig" if c["spearman_rho"] < 0 else "wrong-sign")

    def verdict_p2(c: dict | None) -> str:
        if c is None:
            return "insufficient-n"
        return "INSIDE (n.s. as required)" if c["perm_p"] >= 0.05 else "OUTSIDE (mahal predicts efficacy)"

    out = {
        "arm": "graded-Goodhart Part A — P1/P2 permutation scoring",
        "STATUS": ("FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop. "
                   + ("HARNESS-VERIFY on n=24 (V1b hole open; NOT the final score)" if not v1b_present
                      else "V1b present — roster complete")),
        "model": gg.get("model", "3b"), "primary_alpha": args.primary_alpha,
        "n_perm": args.n_perm, "seed": args.seed,
        "seed_scheme": "P1 = seed, P2a = seed+1, P2b = seed+2 (same triplet both blocks)",
        "filed_P": {"P1_neg_sig": 0.75, "P2_efficacy_ns": 0.85,
                    "P3_construction_split": "outer-loop-owned (14j sequencing) — NOT scored here"},
        "n_roster_at_primary_alpha": len(rows), "n_targets_only": len(targets),
        "v1b_present": v1b_present,
        "all_roster": all_roster, "targets_only": targets_only,
        "provisional_verdicts": {
            "P1": verdict_p1(p1), "P2a_raw": verdict_p2(p2a), "P2b_behavior": verdict_p2(p2b),
            "gate": ("FINAL requires V1b (roster complete) + outer-loop P3 re-word"
                     if not v1b_present else "roster complete; P1/P2 scorable, P3 still outer-loop"),
        },
        "p3_construction_split_RAW": p3_raw,
        "law": "spearman over roster @primary-α; P1 one-sided-negative label-permutation null; "
               "P2 two-sided (n.s. required); targets-only column excludes R-nulls (P1 must not be "
               "a nulls artifact). P3 NOT filed — outer-loop wording, V7 uses same-family efficacy.",
    }
    args.out_json.write_text(json.dumps(out, indent=1))
    print(f"n_roster={len(rows)} n_targets={len(targets)} v1b_present={v1b_present}")
    print(f"P1 mahal×selectivity: {p1}")
    print(f"  → {out['provisional_verdicts']['P1']}")
    print(f"P2a mahal×eff_raw: {p2a} → {out['provisional_verdicts']['P2a_raw']}")
    print(f"P2b mahal×eff_behavior: {p2b} → {out['provisional_verdicts']['P2b_behavior']}")
    print(f"targets-only P1: {targets_only['P1_mahal_vs_selectivity']}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
