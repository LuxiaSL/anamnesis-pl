"""ANNEX 14r — rider-1 shape-assay completion for R-A: the Σ-screen (13d §2.1 statistic,
banked-convention-exact) extended to the band-era vectors (V7, Rband1-3, R-A), CPU-only.

The prereg 14r block orders "shape assay first (C§1 screen + Σ-screen); then ONE steering
block." The rider-1 filing carried the Σ-basis mass profile; this adds the Mahalanobis /
eigenmass screen with the SAME convention as the banked `a5_covariance_screen_3b.json`
(ridge = ridge_rel × mean eigenvalue; coeff²/(λ+ridge) summed in the full eigenbasis), so
R-A's row is directly comparable to the banked R1-3/V1/V3/V4 rows. Banked rows are echoed
into the output and NOT recomputed (the bank is the record; echo + convention match is the
check). Analytic band-null reference: E[maha] for a random unit vector confined to the
[16:256] eigenband = mean_{i in band} 1/(λ_i + ridge).

Construction/shape-side only — rider 2 (no author steering-readout code) untouched.

Usage (from pipeline/):
    python -m anamnesis.scripts.annex_14r_ra_shape \
        --sigma ../outputs/battery/arms/A5/a5_sigma_L14_3b.npz \
        --screen ../outputs/battery/arms/A5/a5_covariance_screen_3b.json \
        --b7-vectors ../outputs/battery/a5_vectors_3b_b7/a5_vectors.npz \
        --ra-vectors ../outputs/battery/annex/a5_vectors_3b_14r/a5_vectors.npz \
        --out ../outputs/battery/annex/annex_14r_ra_shape.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BAND = (16, 256)  # vmb_b7_stage2_vectors convention (descending eigenvalue order)
BOTTOM_K = 768    # screen convention (bottom/top 768 of 3072)


def screen_row(v: np.ndarray, evals: np.ndarray, evecs: np.ndarray, ridge: float,
               order: np.ndarray) -> dict:
    """Identical statistics to vmb_a5_covariance_screen._screen + the band mass profile."""
    u = v.astype(np.float64) / np.linalg.norm(v)
    coeff = evecs.T @ u
    maha = float((coeff ** 2 / (evals + ridge)).sum())
    asc = np.argsort(evals)                    # ascending, the screen's bottom-k convention
    m = coeff ** 2
    bottom = float(m[asc[:BOTTOM_K]].sum())
    top = float(m[asc[-BOTTOM_K:]].sum())
    md = (evecs[:, order].T @ u) ** 2          # descending order for the b7 band profile
    return {
        "mahalanobis": maha,
        f"bottom_{BOTTOM_K}_eigenmass": bottom,
        f"top_{BOTTOM_K}_eigenmass": top,
        "tail_over_top": bottom / max(top, 1e-12),
        "mass_top16": float(md[:BAND[0]].sum()),
        "mass_band16_256": float(md[BAND[0]:BAND[1]].sum()),
        "mass_tail256plus": float(md[BAND[1]:].sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=Path, required=True)
    ap.add_argument("--screen", type=Path, required=True)
    ap.add_argument("--b7-vectors", type=Path, required=True)
    ap.add_argument("--ra-vectors", type=Path, required=True)
    ap.add_argument("--ridge-rel", type=float, default=1e-3,
                    help="must match the banked screen (ridge = ridge_rel x mean eigenvalue)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    S = np.load(args.sigma)
    evals = S["evals"].astype(np.float64)
    evecs = S["evecs"].astype(np.float64)
    ridge = args.ridge_rel * float(evals.mean())
    order = np.argsort(evals)[::-1]

    banked = json.loads(args.screen.read_text())["sites"]["14"]
    if not np.isclose(ridge, banked["ridge"], rtol=1e-6):
        raise SystemExit(f"ridge mismatch vs banked screen: {ridge} vs {banked['ridge']}")

    b7 = np.load(args.b7_vectors)
    ra = np.load(args.ra_vectors)
    rows = {name: screen_row(bank[name], evals, evecs, ridge, order)
            for bank in (b7, ra) for name in bank.files}

    lam_band = evals[order[BAND[0]:BAND[1]]]
    band_null_maha = float((1.0 / (lam_band + ridge)).mean())

    out = {
        "provenance": "annex 14r rider-1 shape-assay completion (Sigma-screen, 13d conventions "
                      "verbatim; ridge matched to banked a5_covariance_screen_3b.json)",
        "site": 14, "ridge": ridge, "band": list(BAND),
        "new_rows": rows,
        "banked_rows_echo": banked["vectors"],
        "band_null_maha_analytic": band_null_maha,
        "note": "band-confined unit vectors (V7/Rband/RA) have E[maha] = mean 1/(lambda+ridge) "
                "over the band if isotropic in-band; compare RA/V7 to Rband and to this analytic "
                "reference, not to the full-space R rows.",
    }
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
