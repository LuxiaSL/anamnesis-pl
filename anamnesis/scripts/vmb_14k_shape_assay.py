"""14k — write-anatomy SHAPE assay (session-8 Part B6; ASSAY ONLY, no steering cells).

Per the 14k rider: each candidate coordinate emits a FROZEN shape-based writability prediction
(needle ⇒ data-only / field ⇒ formula-viable) BEFORE any steering cell. The shape is the
candidate's data-route vector geometry on the banked Σ_L14 — the Σ-screen + spectral split:

  mahalanobis  = vᵀΣ⁻¹v         (high = expensive/needle-like; low = cheap/field-like)
  top768/tail768/band eigenmass = share of v's energy in the top / bottom eigenmodes and in
                                  the [16:256] band (V7's field home)
  C§1 readability = energy OUTSIDE the trivial feature channels (a trivial-channel candidate
                    exits here — scoping result, not a prediction)

Anchored on banked references: V3 (data-route LEVER, the needle) and V7 (band/field, formula-
INERT) — the candidate is classed relative to them. Frozen rule (filed BEFORE any steering):
  FIELD (formula-viable) if band-concentrated + low-mahal (V7-like);
  NEEDLE (data-only)     if tail-concentrated + high-mahal (V3tail-like).
The ∇-panel (differentiability) is the GPU confirm; this CPU screen fixes the prediction.
First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def geom(v, evals, evecs, inv, order, band):
    v = v / np.linalg.norm(v)
    proj = evecs.T @ v.astype(np.float64)
    e = proj ** 2
    return {"mahalanobis": float(np.sum(e * inv)),
            "top768_eigenmass": float(e[order[:768]].sum() / e.sum()),
            "tail768_eigenmass": float(e[order[-768:]].sum() / e.sum()),
            "band_16_256_eigenmass": float(e[order[band[0]:band[1]]].sum() / e.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-npz", type=Path, required=True, help="a5_sigma_L14_3b.npz")
    ap.add_argument("--candidate-npz", type=Path, required=True)
    ap.add_argument("--candidate-keys", nargs="+", required=True, help="e.g. Ksoclin_L14")
    ap.add_argument("--ref-npz", type=Path, required=True, help="a5_vectors.npz (V3/R refs)")
    ap.add_argument("--b7-npz", type=Path, default=None, help="a5_vectors_3b_b7 (V7 ref)")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    S = np.load(args.sigma_npz)
    evals, evecs, ridge = S["evals"].astype(np.float64), S["evecs"].astype(np.float64), float(S["ridge"])
    inv = 1.0 / (evals + ridge)
    order = np.argsort(evals)[::-1]
    band = (16, 256)

    def load_key(npz, key):
        d = np.load(npz)
        return d[key] if key in d else None

    # references
    refs = {}
    ref = np.load(args.ref_npz)
    for k in ("V3_L14",):
        if k in ref:
            refs[k] = geom(ref[k], evals, evecs, inv, order, band)
    if args.b7_npz:
        b7 = np.load(args.b7_npz)
        if "V7_L14" in b7:
            refs["V7_L14"] = geom(b7["V7_L14"], evals, evecs, inv, order, band)

    v3, v7 = refs.get("V3_L14"), refs.get("V7_L14")
    rows = []
    for key in args.candidate_keys:
        v = load_key(args.candidate_npz, key)
        if v is None:
            rows.append({"key": key, "present": False}); continue
        g = geom(v, evals, evecs, inv, order, band)
        # frozen classification: FIELD if V7-like (band-conc, low-mahal), NEEDLE if V3-like
        field_like = (v7 is not None and g["band_16_256_eigenmass"] >= 0.8 * v7["band_16_256_eigenmass"]
                      and g["mahalanobis"] <= 1.5 * v7["mahalanobis"])
        needle_like = (v3 is not None and g["tail768_eigenmass"] >= v3["tail768_eigenmass"]
                       and g["mahalanobis"] >= v3["mahalanobis"])
        pred = ("FIELD (formula-viable)" if field_like and not needle_like
                else "NEEDLE (data-only)" if needle_like and not field_like
                else "INTERMEDIATE (ambiguous shape — ∇-panel decides)")
        rows.append({"key": key, "present": True, **g,
                     "frozen_prediction": pred,
                     "vs_V7_band_ratio": round(g["band_16_256_eigenmass"] / v7["band_16_256_eigenmass"], 3) if v7 else None,
                     "vs_V3_mahal_ratio": round(g["mahalanobis"] / v3["mahalanobis"], 3) if v3 else None})
        print(f"  {key}: mahal={g['mahalanobis']:.1f} band={g['band_16_256_eigenmass']:.3f} "
              f"tail={g['tail768_eigenmass']:.3f} → {pred}")

    out = {"arm": "14k write-anatomy SHAPE assay (Σ-screen + spectral; frozen prediction)",
           "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop; ASSAY ONLY (no steering cells)",
           "law": "mahalanobis vᵀΣ⁻¹v + top/tail/band-[16:256] eigenmass on banked Σ_L14; classed vs "
                  "V3 (data-route needle) / V7 (band field). FROZEN before any steering cell: FIELD "
                  "(band-conc+low-mahal)⇒formula-viable; NEEDLE (tail-conc+high-mahal)⇒data-only.",
           "references": refs, "rows": rows}
    args.out_json.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
