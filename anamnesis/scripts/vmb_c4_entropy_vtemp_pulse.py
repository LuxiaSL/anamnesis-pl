"""C4 — free formula rung for the synthetic-temperature target (session-4 Part C; the mirror
of the §B.6 dir0 panel). ∇S_entropy (the output-facing entropy gradient, banked from the panel)
is SAME-FAMILY for V_temp (temperature IS the entropy facet). Question: where in Σ's eigenspectrum
does ∇S_entropy see V_temp — and how does that band location compare to where it saw dir0
(mid-band [16:256], z=3.7, §B.6)? Tests the same-family-tail / cross-family-mid-band regularity
for free (no new gen). CPU-only, seeded nulls. First-read → outer loop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NULL_SEED = 20260715
BANDS = [(0, 16), (16, 256), (256, 1024), (1024, 3072)]


def _cos(a, b):
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gradient-G", type=Path, required=True, help="v4panel_G_S_entropy_3b.npz")
    ap.add_argument("--sigma", type=Path, required=True, help="a5_sigma_L14_3b.npz")
    ap.add_argument("--vtemp", type=Path, required=True, help="ctemp a5_vectors.npz (Vtemp_L14)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    S = np.load(args.sigma)
    evals, evecs = S["evals"].astype(np.float64), S["evecs"].astype(np.float64)
    order = np.argsort(evals)[::-1]
    g = np.load(args.gradient_G)
    G = g["G"].astype(np.float64)
    mean_grad = g["mean_grad"].astype(np.float64) if "mean_grad" in g else G.mean(0)
    vtemp = np.load(args.vtemp)["Vtemp_L14"].astype(np.float64)
    rng = np.random.default_rng(NULL_SEED)

    # raw alignment + per-gen sign consistency + isotropic-null z
    cos_raw = _cos(mean_grad, vtemp)
    per_gen = np.array([_cos(G[i], vtemp) for i in range(len(G))])
    iso = np.array([_cos(rng.standard_normal(len(vtemp)), vtemp) for _ in range(2000)])
    z_iso = (cos_raw - iso.mean()) / max(iso.std(), 1e-12)

    # spectral band table vs band-matched nulls (the §B.6 mirror)
    bands = {}
    for lo, hi in BANDS:
        U = evecs[:, order[lo:hi]]                       # (d, hi-lo)
        cg, cv = U.T @ mean_grad, U.T @ vtemp
        cos_b = _cos(cg, cv)
        nb = rng.standard_normal((1000, U.shape[1]))
        nb /= np.linalg.norm(nb, axis=1, keepdims=True)
        null = nb @ (cv / max(np.linalg.norm(cv), 1e-30))
        z = (cos_b - null.mean()) / max(null.std(), 1e-12)
        bands[f"P[{lo}:{hi}]"] = {"cos": round(cos_b, 4), "band_matched_z": round(float(z), 2),
                                  "vtemp_norm_frac_in_band": round(float(np.linalg.norm(cv)), 3)}

    out = {"arm": "C4-entropy-Vtemp-pulse (mirror of §B.6 dir0 panel)",
           "STATUS": "FIRST_READ_PENDING (C§8) — free formula rung; no steered gen",
           "cos_gradSentropy_Vtemp": round(cos_raw, 4),
           "z_vs_isotropic": round(float(z_iso), 2),
           "per_gen_positive": f"{int((per_gen > 0).sum())}/{len(G)}",
           "per_gen_mean": round(float(per_gen.mean()), 4),
           "spectral_bands": bands,
           "reference_dir0_from_B6": {"band": "[16:256]", "z": 3.7,
                                      "note": "where ∇S_entropy saw dir0 (mid-band, cross-family)"},
           "law": "∇S_entropy vs V_temp; Σ_L14 descending eigenbands; band-matched unit-vector null (1000 draws); "
                  "same-family(entropy↔temperature) prediction = alignment shifts location vs the dir0 mirror"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({"cos_raw": out["cos_gradSentropy_Vtemp"], "z_iso": out["z_vs_isotropic"],
                      "per_gen_positive": out["per_gen_positive"],
                      "bands": {k: v["band_matched_z"] for k, v in bands.items()}}, indent=1))


if __name__ == "__main__":
    main()
