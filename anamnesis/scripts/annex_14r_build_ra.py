"""ANNEX 14r cell R-A construction (ratified `9064bea`; rider 1: build + anatomy + predictions
BEFORE the steering block; rider 2 note: this is CONSTRUCTION — the steering readout stays the
banked 14k analyzer, no new readout code here).

R-A = unit(P[16:256] · V4_L14): the missing cell of the {output, internal} x {raw, band-passed}
gradient 2x2. Conventions are vmb_b7_stage2_vectors.py's VERBATIM (descending-eigenvalue order,
Uband = evecs[:, order[16:256]], renormalize) so R-A differs from V7 in exactly one factor (the
functional) and from V4 in exactly one factor (the band-pass).

Emits the vector bank + a construction-anatomy JSON. The load-bearing anatomy number is
band_mass_of_V4 = ||P.V4|| / ||V4||: if V4 barely lives in the band, R-A is amplified residue,
not a controlled completion — that number must be quoted next to any R-A result.

Usage (from pipeline/):
    python -m anamnesis.scripts.annex_14r_build_ra \
        --vectors ../outputs/battery/a5_vectors_3b/a5_vectors.npz \
        --sigma ../outputs/battery/arms/A5/a5_sigma_L14_3b.npz \
        --stamps ../outputs/battery/a5_vectors_3b/a5_vectors_stamps.json \
        --b7-vectors ../outputs/battery/a5_vectors_3b_b7/a5_vectors.npz \
        --out-dir ../outputs/battery/annex/a5_vectors_3b_14r
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BAND = (16, 256)   # vmb_b7_stage2_vectors.BAND — do not drift
SITE = 14


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", type=Path, required=True)
    ap.add_argument("--sigma", type=Path, required=True)
    ap.add_argument("--stamps", type=Path, required=True)
    ap.add_argument("--b7-vectors", type=Path, required=True, help="banked V7 + Rband nulls")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bank = np.load(args.vectors)
    v4 = bank["V4_L14"].astype(np.float64)
    v3 = bank["V3_L14"].astype(np.float64)
    b7 = np.load(args.b7_vectors)
    v7 = b7["V7_L14"].astype(np.float64)

    S = np.load(args.sigma)
    evals = S["evals"].astype(np.float64)
    evecs = S["evecs"].astype(np.float64)
    order = np.argsort(evals)[::-1]                     # verbatim b7 convention
    Uband = evecs[:, order[BAND[0]:BAND[1]]]            # (d, 240)

    proj = Uband @ (Uband.T @ v4)
    band_mass = float(np.linalg.norm(proj) / np.linalg.norm(v4))
    if np.linalg.norm(proj) <= 1e-12:
        raise SystemExit("V4 has no band component — R-A cannot be constructed")
    ra = (proj / np.linalg.norm(proj)).astype(np.float32)

    # band decomposition of the comparators, same basis (descending order)
    def mass_profile(v: np.ndarray) -> dict:
        u = v / np.linalg.norm(v)
        coef = evecs[:, order].T @ u
        m = coef ** 2
        return {"top16": float(m[:16].sum()), "band16_256": float(m[16:256].sum()),
                "tail256plus": float(m[256:].sum())}

    cosine = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    anatomy = {
        "construction": "unit(P[16:256] . V4_L14); conventions = vmb_b7_stage2_vectors verbatim",
        "band": list(BAND), "site": SITE,
        "band_mass_of_V4": band_mass,
        "mass_profiles": {"V4": mass_profile(v4), "RA": mass_profile(ra.astype(np.float64)),
                          "V7": mass_profile(v7), "V3": mass_profile(v3)},
        "cos": {"RA_V4": cosine(ra, v4), "RA_V7": cosine(ra, v7), "RA_V3": cosine(ra, v3),
                "V7_V4": cosine(v7, v4), "V7_V3": cosine(v7, v3)},
        "tail_mass_RA_by_construction": 0.0,
        "note": "R-A is band-confined => zero tail mass => inherits V7's predicted alpha=.3 "
                "fragility class (S7 pricing / SB.5 law) independent of the functional question.",
    }

    l14 = json.loads(args.stamps.read_text())["median_resid_norms"]["L14"]
    np.savez(args.out_dir / "a5_vectors.npz", RA_L14=ra)
    (args.out_dir / "a5_vectors_stamps.json").write_text(json.dumps(
        {"median_resid_norms": {"L14": l14}, "band": list(BAND), "site": SITE,
         "provenance": "14r cell R-A: band-passed V4 (the missing 2x2 cell)"}, indent=1))
    (args.out_dir / "ra_construction_anatomy.json").write_text(json.dumps(anatomy, indent=1))
    print(json.dumps(anatomy, indent=1))


if __name__ == "__main__":
    main()
