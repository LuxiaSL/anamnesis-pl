"""ANNEX — generic band-pass member builder: unit(P[16:256] . G) for each requested raw
gradient, b7 conventions VERBATIM (descending eigenorder, renormalize) — the
annex_14r_build_ra.py recipe generalized over keys (construction-side; rider 2 untouched).

Emits the member bank (V-keys) + per-member anatomy (band mass of the raw gradient, mass
profiles, cos table vs the banked comparators V7/V3/V4/RA) — the inputs each member's
rider-1-style prediction filing must quote BEFORE its steering block.

Usage (from pipeline/, CPU, after the pulses are pulled back):
    python -m anamnesis.scripts.annex_band_pass \
        --gradients ../outputs/battery/annex/roster_vectors_3b/roster_gradients.npz \
        --keys Gmargin_L14:Vconf_L14 Geos_L14:Veos_L14 Grep_L14:Vrep_L14 \
        --sigma ../outputs/battery/arms/A5/a5_sigma_L14_3b.npz \
        --stamps ../outputs/battery/a5_vectors_3b/a5_vectors_stamps.json \
        --compare ../outputs/battery/a5_vectors_3b/a5_vectors.npz:V3_L14,V4_L14 \
                  ../outputs/battery/a5_vectors_3b_b7/a5_vectors.npz:V7_L14 \
                  ../outputs/battery/annex/a5_vectors_3b_14r/a5_vectors.npz:RA_L14 \
        --out-dir ../outputs/battery/annex/roster_vectors_3b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BAND = (16, 256)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gradients", type=Path, required=True)
    ap.add_argument("--keys", nargs="+", required=True, help="RAWKEY:MEMBERKEY pairs")
    ap.add_argument("--sigma", type=Path, required=True)
    ap.add_argument("--stamps", type=Path, required=True,
                    help="banked a5 stamps (median_resid_norms source)")
    ap.add_argument("--compare", nargs="+", default=[], help="npz_path:key1,key2 comparators")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    S = np.load(args.sigma)
    evals = S["evals"].astype(np.float64)
    evecs = S["evecs"].astype(np.float64)
    order = np.argsort(evals)[::-1]
    Uband = evecs[:, order[BAND[0]:BAND[1]]]

    comparators: dict[str, np.ndarray] = {}
    for spec in args.compare:
        p, keys = spec.split(":")
        bank = np.load(p)
        for k in keys.split(","):
            comparators[k] = bank[k].astype(np.float64)

    grads = np.load(args.gradients)
    cosine = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    def mass_profile(v: np.ndarray) -> dict:
        u = v / np.linalg.norm(v)
        m = (evecs[:, order].T @ u) ** 2
        return {"top16": float(m[:BAND[0]].sum()), "band16_256": float(m[BAND[0]:BAND[1]].sum()),
                "tail256plus": float(m[BAND[1]:].sum())}

    members, anatomy = {}, {}
    for pair in args.keys:
        raw_key, member_key = pair.split(":")
        g = grads[raw_key].astype(np.float64)
        proj = Uband @ (Uband.T @ g)
        if np.linalg.norm(proj) <= 1e-12:
            raise SystemExit(f"{raw_key} has no band component — member cannot be constructed")
        members[member_key] = (proj / np.linalg.norm(proj)).astype(np.float32)
        anatomy[member_key] = {
            "raw_key": raw_key,
            "band_mass_of_raw": float(np.linalg.norm(proj) / np.linalg.norm(g)),
            "mass_profiles": {"raw": mass_profile(g),
                              "member": mass_profile(members[member_key].astype(np.float64))},
            "cos": {ck: cosine(members[member_key].astype(np.float64), cv)
                    for ck, cv in comparators.items()},
        }

    l14 = json.loads(args.stamps.read_text())["median_resid_norms"]
    np.savez(args.out_dir / "a5_vectors.npz", **members)
    (args.out_dir / "a5_vectors_stamps.json").write_text(json.dumps(
        {"median_resid_norms": l14, "band": list(BAND),
         "provenance": "in-shadow roster members: band-passed natural output-functional "
                       "gradients (PRICING-inshadow-roster-2026-07-16; b7 conventions verbatim)"},
        indent=1))
    (args.out_dir / "roster_member_anatomy.json").write_text(json.dumps(anatomy, indent=1))
    print(json.dumps(anatomy, indent=1))


if __name__ == "__main__":
    main()
