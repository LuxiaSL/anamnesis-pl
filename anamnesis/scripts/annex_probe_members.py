"""ANNEX probe program — member builder (CPU; runs node-side inside the overnight chain).

From banked raw probe gradients (annex_probe_pulses.py, 12 keys G*_L{7,14,21}) build:
  - 12 raw band members  P{func}_L{s}      = unit(P_band · G)  through the SITE's Σ
    (band [16:256] of the DESCENDING eigenorder — b7 conventions, per-site Σ)
  - 4 ⊥ members at L14   P{func}_perp_L14  = Gram-Schmidt vs V7_L14 (standing ⊥ rule;
    L7/L21 carry no banked V7 — scope named in the stamps)
  - 2 site nulls         Rband1_L7 / Rband1_L21 = first matched-support random from each
    site's band (NULL_SEED convention; L14's nulls stay the banked b7 Rband cells)
One merged a5_vectors.npz + a5_vectors_stamps.json (median_resid_norms for all 3 sites,
from the A5 stamps of record) + probe_anatomy.json (band norms, cos-to-V7 at L14,
⊥ residual fractions — the PP-5/PP-6 scoring inputs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BAND = (16, 256)
NULL_SEED = 20260714
FUNCS = ("attnent", "anchor", "vnorm", "gatemass")
PREFIX = {"attnent": "Gattnent", "anchor": "Ganchor",
          "vnorm": "Gvnorm", "gatemass": "Ggatemass"}


def band_basis(sigma_npz: Path) -> np.ndarray:
    S = np.load(sigma_npz)
    evals = S["evals"].astype(np.float64)
    evecs = S["evecs"].astype(np.float64)
    order = np.argsort(evals)[::-1]
    return evecs[:, order[BAND[0]:BAND[1]]]          # (d, 240)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gradients", type=Path, required=True, help="probe_gradients.npz")
    ap.add_argument("--sigma-l7", type=Path, required=True)
    ap.add_argument("--sigma-l14", type=Path, required=True)
    ap.add_argument("--sigma-l21", type=Path, required=True)
    ap.add_argument("--v7-npz", type=Path, required=True, help="b7 bank holding V7_L14")
    ap.add_argument("--norms-json", type=Path, required=True,
                    help="a5_vectors_stamps.json of record (L7/L14/L21 median norms)")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    G = dict(np.load(args.gradients))
    sigmas = {7: args.sigma_l7, 14: args.sigma_l14, 21: args.sigma_l21}
    v7 = np.load(args.v7_npz)["V7_L14"].astype(np.float64)
    v7 = v7 / np.linalg.norm(v7)
    norms_src = json.loads(args.norms_json.read_text())["median_resid_norms"]

    vectors: dict[str, np.ndarray] = {}
    anatomy: dict[str, dict] = {}
    for site, spath in sigmas.items():
        Ub = band_basis(spath)
        for f in FUNCS:
            gkey = f"{PREFIX[f]}_L{site}"
            if gkey not in G:
                raise SystemExit(f"missing raw gradient {gkey} in {args.gradients}")
            g = G[gkey].astype(np.float64)
            band = Ub @ (Ub.T @ g)
            band_norm = float(np.linalg.norm(band))
            if band_norm < 1e-12:
                raise SystemExit(f"{gkey}: zero band projection — pulse degenerate")
            m = band / band_norm
            mkey = f"P{f}_L{site}"
            vectors[mkey] = m.astype(np.float32)
            row = {"raw_key": gkey, "band_norm_fraction": band_norm /
                   max(float(np.linalg.norm(g)), 1e-30)}
            if site == 14:
                c = float(m @ v7)
                row["cos_to_V7"] = c
                perp = m - c * v7
                pn = float(np.linalg.norm(perp))
                row["perp_residual_fraction"] = pn
                if pn < 1e-6:
                    raise SystemExit(f"{mkey}: fully V7-collapsed — no ⊥ member exists")
                vectors[f"P{f}_perp_L14"] = (perp / pn).astype(np.float32)
            anatomy[mkey] = row
        # site null (L14's nulls are the banked b7 Rband cells — skip)
        if site != 14:
            rng = np.random.default_rng(NULL_SEED + site)
            r = Ub @ rng.standard_normal(Ub.shape[1])
            vectors[f"Rband1_L{site}"] = (r / np.linalg.norm(r)).astype(np.float32)

    stamps = {
        "median_resid_norms": {f"L{s}": float(norms_src[f"L{s}"]) for s in (7, 14, 21)},
        "band": list(BAND),
        "provenance": "probe program members: per-site Σ band-pass (b7 conventions); "
                      "⊥ vs V7_L14 at L14 only (L7/L21 have no banked V7 — scope named); "
                      "site nulls seeded NULL_SEED+site; norms = a5_vectors_3b stamps of "
                      "record",
    }
    np.savez(args.out_dir / "a5_vectors.npz", **vectors)
    (args.out_dir / "a5_vectors_stamps.json").write_text(json.dumps(stamps, indent=1))
    (args.out_dir / "probe_anatomy.json").write_text(json.dumps(anatomy, indent=1))
    print(json.dumps({"built": sorted(vectors.keys()),
                      "anatomy_keys": sorted(anatomy.keys())}, indent=1))


if __name__ == "__main__":
    main()
