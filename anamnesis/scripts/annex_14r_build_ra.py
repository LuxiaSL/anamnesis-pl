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

BAND = (16, 256)   # vmb_b7_stage2_vectors.BAND — do not drift (default; overridable per-model)
SITE = 14          # default (3B); M6 passes --site 9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", type=Path, required=True)
    ap.add_argument("--sigma", type=Path, required=True)
    ap.add_argument("--stamps", type=Path, required=True)
    ap.add_argument("--b7-vectors", type=Path, required=True, help="banked V7 + Rband nulls")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--site", type=int, default=SITE,
                    help="injection layer (default 14 = 3B; M6 = 9). Selects the V4_L{site}/"
                         "V3_L{site}/V7_L{site} keys + the median_resid_norms[L{site}] scale.")
    ap.add_argument("--band", default=f"{BAND[0]},{BAND[1]}",
                    help="eigenvector index band-pass (frozen b7 default 16,256 — do not drift "
                         "without a ruling; a CLI knob only so per-model spectra can be revisited).")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    site = int(args.site)
    band = tuple(int(x) for x in args.band.split(","))
    if len(band) != 2:
        raise SystemExit(f"--band must be 'lo,hi', got {args.band!r}")
    lk = f"L{site}"

    bank = np.load(args.vectors)
    v4 = bank[f"V4_{lk}"].astype(np.float64)
    v3 = bank[f"V3_{lk}"].astype(np.float64)
    b7 = np.load(args.b7_vectors)
    v7 = b7[f"V7_{lk}"].astype(np.float64)

    S = np.load(args.sigma)
    evals = S["evals"].astype(np.float64)
    evecs = S["evecs"].astype(np.float64)
    order = np.argsort(evals)[::-1]                     # verbatim b7 convention
    Uband = evecs[:, order[band[0]:band[1]]]            # (d, hi-lo)

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
        return {f"top{band[0]}": float(m[:band[0]].sum()),
                f"band{band[0]}_{band[1]}": float(m[band[0]:band[1]].sum()),
                f"tail{band[1]}plus": float(m[band[1]:].sum())}

    cosine = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    anatomy = {
        "construction": f"unit(P[{band[0]}:{band[1]}] . V4_{lk}); conventions = "
                        "vmb_b7_stage2_vectors verbatim",
        "band": list(band), "site": site,
        "band_mass_of_V4": band_mass,
        "mass_profiles": {"V4": mass_profile(v4), "RA": mass_profile(ra.astype(np.float64)),
                          "V7": mass_profile(v7), "V3": mass_profile(v3)},
        "cos": {"RA_V4": cosine(ra, v4), "RA_V7": cosine(ra, v7), "RA_V3": cosine(ra, v3),
                "V7_V4": cosine(v7, v4), "V7_V3": cosine(v7, v3)},
        "tail_mass_RA_by_construction": 0.0,
        "note": "R-A is band-confined => zero tail mass => inherits V7's predicted alpha=.3 "
                "fragility class (S7 pricing / SB.5 law) independent of the functional question.",
    }

    lnorm = json.loads(args.stamps.read_text())["median_resid_norms"][lk]
    np.savez(args.out_dir / "a5_vectors.npz", **{f"RA_{lk}": ra})
    (args.out_dir / "a5_vectors_stamps.json").write_text(json.dumps(
        {"median_resid_norms": {lk: lnorm}, "band": list(band), "site": site,
         "provenance": "14r cell R-A: band-passed V4 (the missing 2x2 cell)"}, indent=1))
    (args.out_dir / "ra_construction_anatomy.json").write_text(json.dumps(anatomy, indent=1))
    print(json.dumps(anatomy, indent=1))


if __name__ == "__main__":
    main()
