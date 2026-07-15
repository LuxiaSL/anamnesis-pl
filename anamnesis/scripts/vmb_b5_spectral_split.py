"""§B.5 — V3 spectral-split vector build (DESIGN-V4 §B.5; Luxia-ratified window add 14d item 3).

Interprets §B.3's finding (V3's steering efficacy vs the Σ eigenspectrum): split the data-route
mode vector V3_L14 at the rank-BOUNDARY of the banked residual-Σ eigenspectrum into
  V3_top  = V3 projected onto the top-`boundary` eigendirections (high-variance / on-manifold)
  V3_tail = V3 projected onto the remaining directions (low-variance / anti-manifold)
each unit-normalized, plus MATCHED-SUPPORT nulls R_top ×3 / R_tail ×3 (random unit vectors
confined to the SAME subspaces — so "top-concentration alone" cannot read as efficacy).

Which spectral half carries V3's lever? Free-gen readouts (downstream): dir0 targeting (A5-inv)
+ deformation-maha (§2.1) + efficiency = targeting ÷ deformation (primary cross-subspace) +
coherence. Banks an a5-vectors-format npz + stamps (reusing the L14 median residual norm).

CPU-only (eigenbasis projections). Deterministic null seed. Run:
    python -m anamnesis.scripts.vmb_b5_spectral_split \
      --sigma outputs/battery/arms/A5/a5_sigma_L14_3b.npz \
      --vectors <a5_vectors_3b/a5_vectors.npz> --stamps <a5_vectors_3b/a5_vectors_stamps.json> \
      --out-dir <a5_vectors_3b_b5> --boundary 256
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NULL_SEED = 20260714
SITE = 14


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=Path, required=True)
    ap.add_argument("--vectors", type=Path, required=True)
    ap.add_argument("--stamps", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--boundary", type=int, default=256, help="top/tail rank split of Σ eigenspectrum")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    S = np.load(args.sigma)
    evals = S["evals"].astype(np.float64)          # ascending
    evecs = S["evecs"].astype(np.float64)          # columns = eigenvectors
    order = np.argsort(evals)[::-1]                # descending rank
    top_idx = order[: args.boundary]
    tail_idx = order[args.boundary:]
    Utop = evecs[:, top_idx]                        # (d, boundary)
    Utail = evecs[:, tail_idx]                      # (d, d-boundary)

    v3 = np.load(args.vectors)["V3_L14"].astype(np.float64)

    def _proj(U, x):
        c = U.T @ x
        y = U @ c
        n = np.linalg.norm(y)
        return (y / n).astype(np.float32), float(n)

    vectors, meta = {}, {}
    v3top, ntop = _proj(Utop, v3)
    v3tail, ntail = _proj(Utail, v3)
    vectors["V3top_L14"] = v3top
    vectors["V3tail_L14"] = v3tail
    meta["V3top_L14"] = {"route": "V3 projected onto top-%d Σ-eigendirections" % args.boundary,
                         "captured_norm_fraction": ntop}
    meta["V3tail_L14"] = {"route": "V3 projected onto tail Σ-eigendirections",
                          "captured_norm_fraction": ntail}
    # matched-support random nulls (unit vectors confined to each subspace)
    rng = np.random.default_rng(NULL_SEED)
    for i in range(1, 4):
        ct = rng.standard_normal(Utop.shape[1]); rt = Utop @ ct; rt /= np.linalg.norm(rt)
        cl = rng.standard_normal(Utail.shape[1]); rl = Utail @ cl; rl /= np.linalg.norm(rl)
        vectors[f"Rtop{i}_L14"] = rt.astype(np.float32)
        vectors[f"Rtail{i}_L14"] = rl.astype(np.float32)
        meta[f"Rtop{i}_L14"] = {"route": "matched-support null (top subspace)"}
        meta[f"Rtail{i}_L14"] = {"route": "matched-support null (tail subspace)"}

    # sanity: V3top + V3tail (unnormalized reconstruction) recovers V3; report overlap
    cos_top_tail = float(v3top @ v3tail / max(np.linalg.norm(v3top) * np.linalg.norm(v3tail), 1e-30))

    # reuse the L14 median residual norm from the existing stamps (alpha = frac × this)
    src_stamps = json.loads(args.stamps.read_text())
    l14_norm = src_stamps["median_resid_norms"]["L14"]
    stamps_out = {"median_resid_norms": {"L14": l14_norm},
                  "boundary": args.boundary, "site": SITE,
                  "v3_top_norm_fraction": ntop, "v3_tail_norm_fraction": ntail,
                  "cos_v3top_v3tail": cos_top_tail, "vectors": meta}

    np.savez(args.out_dir / "a5_vectors.npz", **vectors)
    (args.out_dir / "a5_vectors_stamps.json").write_text(json.dumps(stamps_out, indent=1))
    print(json.dumps({"built": list(vectors.keys()), "v3_top_norm_frac": round(ntop, 4),
                      "v3_tail_norm_frac": round(ntail, 4), "cos_top_tail": round(cos_top_tail, 4),
                      "L14_median_norm": round(l14_norm, 3)}, indent=1))


if __name__ == "__main__":
    main()
