"""Standalone band_mass_of_V4 readout — the R-A "amplified-residue" check as a reusable,
model-agnostic, no-rebuild readout.

band_mass_of_V4 = ||P[lo:hi] . V4|| / ||V4||, where P[lo:hi] projects onto the covariance
eigenvectors ranked lo..hi in DESCENDING eigenvalue order. This is the VERBATIM b7/14r
convention (`vmb_b7_stage2_vectors.BAND`, `annex_14r_build_ra`): R-A = unit(P[16:256]·V4),
so if V4 barely lives in the band, R-A is an amplified residue rather than a genuine
{internal, band-passed} 2x2 cell.

`annex_14r_build_ra` already emits this as a side-effect of building R-A (into
`ra_construction_anatomy.json`). This script recomputes it directly from a banked sigma +
vector, so the figure of record has a standalone tracked producer that needs no b7 stack —
the PM6-b L2 completion leg, and a steering-matrix / M7 pre-flight primitive (band-health at
any candidate site, any model).

Pure numpy on banked npz (CPU). First-read -> outer loop; nothing stamped.

    python -m anamnesis.scripts.vmb_a5_band_mass_readout \
        --sigma $BANK/arms/A5_dsv2/a5_sigma_L22_dsv2-lite.npz \
        --vectors $BANK/a5_vectors_dsv2_lite_v4_L22/a5_vectors.npz --vec-key V4_L22 \
        --site 22 --band 16,256 --out-json $BANK/arms/A5_dsv2/pm6b_L22_band_mass.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def band_mass(sigma_npz: Path, vec: np.ndarray, band: tuple[int, int]) -> dict:
    """Return band_mass + top/band/tail mass fractions of `vec` in the descending-eigenvalue
    eigenbasis of the covariance stored in `sigma_npz` (keys: evals, evecs)."""
    S = np.load(sigma_npz)
    for k in ("evals", "evecs"):
        if k not in S:
            raise SystemExit(f"{sigma_npz} missing key {k!r} (has {list(S.keys())})")
    evals = S["evals"].astype(np.float64)
    evecs = S["evecs"].astype(np.float64)
    if evecs.shape[0] != vec.shape[0]:
        raise SystemExit(f"dim mismatch: evecs {evecs.shape} vs vec {vec.shape}")
    lo, hi = band
    order = np.argsort(evals)[::-1]                       # verbatim b7 convention
    Uband = evecs[:, order[lo:hi]]
    proj = Uband @ (Uband.T @ vec)
    vnorm = float(np.linalg.norm(vec))
    if vnorm == 0.0:
        raise SystemExit("vector has zero norm")
    bm = float(np.linalg.norm(proj) / vnorm)
    coef = evecs[:, order].T @ vec                        # coords in descending eigenbasis
    m = coef ** 2 / (coef ** 2).sum()
    return {
        "band_mass": round(bm, 6),
        "mass_fractions": {f"top{lo}": round(float(m[:lo].sum()), 6),
                           f"band{lo}_{hi}": round(float(m[lo:hi].sum()), 6),
                           f"tail{hi}plus": round(float(m[hi:].sum()), 6)},
        "sqrt_band_massfrac_check": round(float(np.sqrt(m[lo:hi].sum())), 6),
        "vec_norm": round(vnorm, 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=Path, required=True, help="covariance npz (keys evals, evecs)")
    ap.add_argument("--vectors", type=Path, required=True, help="vector bank npz")
    ap.add_argument("--vec-key", required=True, help="key in --vectors (e.g. V4_L22)")
    ap.add_argument("--site", type=int, required=True, help="layer index (labeling only)")
    ap.add_argument("--band", default="16,256", help="descending-eigenvalue index band 'lo,hi'")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    band = tuple(int(x) for x in args.band.split(","))
    if len(band) != 2 or band[0] >= band[1]:
        raise SystemExit(f"--band must be 'lo,hi' with lo<hi, got {args.band!r}")

    V = np.load(args.vectors)
    if args.vec_key not in V:
        raise SystemExit(f"{args.vec_key!r} not in {args.vectors} (has {list(V.keys())})")
    vec = V[args.vec_key].astype(np.float64)

    res = band_mass(args.sigma, vec, band)
    out = {
        "arm": "A5 band_mass_of_V4 readout (R-A amplified-residue check)",
        "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED -> outer loop",
        "site": args.site, "band": list(band), "vec_key": args.vec_key,
        "sigma": str(args.sigma), "vectors": str(args.vectors),
        "law": "band_mass = ||P[lo:hi].V|| / ||V||, descending-eigenvalue basis (b7/14r verbatim); "
               "healthy band_mass => R-A is a genuine band-passed cell, not amplified residue.",
        **res,
    }
    args.out_json.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
