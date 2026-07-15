"""ANNEX — THE SHAPE AUDIT: the third leg that location+scale still misses.

⚠ ANNEX LANE: nothing here is quotable until it graduates through a frozen prereg cell.
Ledger: outputs/battery/annex/ANNEX-LEDGER.md

WHY THIS EXISTS — the lesson generalised TWICE, the hard way, on two different corpora.

  Session 1: cell-PC1 (25% var, gate enrichment 10.83, split-half .988, nuisance audit spotless)
             was the 512-token CAP. Caught only by `corr(|score|, cap) = .829` while
             `corr(score, cap) = .012`. Lesson written: "audit LOCATION AND SCALE."

  Session 2: A4-power PC1/PC2 (58% of variance, attn_res enrichment 15.64, structure coefficients
             ±1.000) are **8–17 outlier rows out of 23,758** — kurtosis **11,861**, minority
             cluster **1/23,758**, and the middle 98% of the data spanning **0.02 sd**.
             **BOTH nuisance channels certify them CLEAN** (corr(score,glen) −.0005,
             corr(|score|,glen) −.0019).

⇒ **Location + scale is STILL not enough.** Session 1's lesson would have passed session 2's
artifact. The mechanism was neither location nor scale in a covariate — it was a handful of
degenerate rows, which no covariate audit can see. The only thing that catches it is the SHAPE of
the score distribution itself:

    kurtosis          — 3.0 is Gaussian. Session 1's artifact: 14.1. Session 2's: 11,861.
    minority cluster  — a 1-D 2-means split. A real axis splits ~balanced; an artifact peels off
                        a shard (28/800, then 1/23,758).
    percentile span   — p1..p99. If the middle 98% spans a fraction of an sd, the "variance" is
                        a few rows and the axis is not an axis.

**A variance-maximizing method will hand you the outliers as PC1 every time, dressed in whatever
family those rows happen to load.** Run this on EVERY candidate axis, always, before naming it.

    PYTHONPATH=pipeline python -m anamnesis.scripts.annex_shape_audit --corpus power --variant cell
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from anamnesis.scripts.annex_corpus import REPO, load_power, load_venue, prepare
from anamnesis.scripts.annex_spectrum import pca

logger = logging.getLogger(__name__)
F32 = NDArray[np.float32]
OUT = REPO / "outputs/battery/annex"

# Thresholds are TRIPWIRES, not proofs — they flag, you look.
KURT_FLAG = 10.0        # session 1's artifact was 14.1; a healthy axis sits near 3
MINORITY_FLAG = 0.02    # a cluster under 2% of the corpus is a shard, not a mode
SPAN_FLAG = 0.5         # p1..p99 spanning < 0.5 sd ⇒ the "variance" is a few rows


def minority_cluster(z: F32) -> int:
    """1-D 2-means; returns the smaller cluster's size. A real split is ~balanced; an artifact
    peels off a shard."""
    ctr = np.array([z.min() / 2, z.max() / 2], dtype=np.float64)
    lab = np.zeros(len(z), dtype=np.int64)
    for _ in range(50):
        lab = np.abs(z[:, None] - ctr[None, :]).argmin(1)
        for k in (0, 1):
            if (lab == k).any():
                ctr[k] = z[lab == k].mean()
    return int(min((lab == 0).sum(), (lab == 1).sum()))


def audit_axis(score: F32, covariates: dict[str, NDArray]) -> dict:
    """Location + scale + SHAPE. The first two are session 1's lesson; the third is session 2's."""
    z = ((score - score.mean()) / max(score.std(), 1e-12)).astype(np.float64)
    kurt = float((z ** 4).mean())
    mino = minority_cluster(z)
    p1, p25, p50, p75, p99 = (float(x) for x in np.percentile(z, [1, 25, 50, 75, 99]))
    span = p99 - p1
    out = {
        "kurtosis": round(kurt, 2),
        "minority_cluster": mino,
        "minority_frac": round(mino / len(z), 5),
        "percentiles": {"p1": round(p1, 3), "p25": round(p25, 3), "p50": round(p50, 3),
                        "p75": round(p75, 3), "p99": round(p99, 3)},
        "p1_p99_span_sd": round(span, 3),
        "n_abs_z_gt5": int((np.abs(z) > 5).sum()),
        "n_abs_z_gt10": int((np.abs(z) > 10).sum()),
        "covariates": {},
    }
    for name, v in covariates.items():
        v = np.asarray(v, dtype=np.float64)
        if v.std() < 1e-12:
            out["covariates"][name] = {"location": None, "scale": None,
                                       "note": "no variance (controlled by construction)"}
            continue
        out["covariates"][name] = {
            "location": round(float(np.corrcoef(z, v)[0, 1]), 4),        # session 1: necessary
            "scale": round(float(np.corrcoef(np.abs(z), v)[0, 1]), 4),   # session 1: and this
        }
    flags = []
    if kurt > KURT_FLAG:
        flags.append(f"KURTOSIS {kurt:.1f} > {KURT_FLAG}")
    if mino / len(z) < MINORITY_FLAG:
        flags.append(f"MINORITY {mino}/{len(z)} < {MINORITY_FLAG:.0%}")
    if span < SPAN_FLAG:
        flags.append(f"p1..p99 SPAN {span:.3f} sd < {SPAN_FLAG}")
    for name, cv in out["covariates"].items():
        if cv.get("scale") is not None and abs(cv["scale"]) > 0.3:
            flags.append(f"SCALE vs {name} = {cv['scale']:+.3f}")
    out["flags"] = flags
    out["verdict"] = "ARTIFACT-SHAPED — do not name this axis" if flags else "shape OK"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["venue", "venuecap", "power"], default="power")
    ap.add_argument("--variant", default="cell")
    ap.add_argument("--partner", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    c = (load_power(partner=args.partner) if args.corpus == "power"
         else load_venue(capped_only=(args.corpus == "venuecap")))
    X, df = prepare(c, args.variant)
    print(f"{args.corpus}/{args.variant}: X {X.shape} df={df}")

    covs: dict[str, NDArray] = dict(c.covariates)
    for k, v in c.factors.items():
        if k not in ("conv",):
            covs[k] = v

    sp = pca(X, df, k=args.k)
    rows = {}
    for j in range(args.k):
        r = audit_axis((X @ sp.components[j]).astype(np.float32), covs)
        r["var_ratio"] = round(float(sp.var_ratio[j]), 4)
        rows[f"PC{j+1}"] = r
        pc = r["percentiles"]
        print(f"\n=== PC{j+1}  var {r['var_ratio']:.4f} — {r['verdict']} ===")
        print(f"  kurtosis {r['kurtosis']:>10.2f}  (3.0 = gaussian)")
        print(f"  minority {r['minority_cluster']}/{len(X)} = {100*r['minority_frac']:.2f}%")
        print(f"  p1/p25/p50/p75/p99 = {pc['p1']}/{pc['p25']}/{pc['p50']}/{pc['p75']}/{pc['p99']}"
              f"   span {r['p1_p99_span_sd']} sd")
        print(f"  |z|>5: {r['n_abs_z_gt5']}   |z|>10: {r['n_abs_z_gt10']}")
        for name, cv in r["covariates"].items():
            if cv.get("location") is not None:
                print(f"    {name:12s} location {cv['location']:+.4f}   scale {cv['scale']:+.4f}")
        for f in r["flags"]:
            print(f"  ⚠ {f}")

    p = args.out or OUT / f"annex_shape_audit_{args.corpus}_{args.variant}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "ANNEX_RULE": "NOT QUOTABLE until graduated via a frozen prereg cell",
        "rung": "shape audit — the third leg (location + scale is NOT enough)",
        "why": "A4-power PC1/PC2 held 58% of the variance with BOTH nuisance channels clean and "
               "were 8-17 outlier rows of 23,758 (kurtosis 11,861). Session 1's location+scale "
               "lesson would have passed it. Only SHAPE catches a degenerate-row artifact.",
        "thresholds": {"kurtosis": KURT_FLAG, "minority_frac": MINORITY_FLAG,
                       "p1_p99_span_sd": SPAN_FLAG,
                       "note": "TRIPWIRES, not proofs — they flag, you look."},
        "corpus": args.corpus, "variant": args.variant, "partner": args.partner,
        "n": int(len(X)), "df": int(df), "axes": rows,
    }, indent=1))
    print(f"\n  → {p}")


if __name__ == "__main__":
    main()
