"""ANNEX — V7-orthogonalized roster members (the EOS/rep follow-up block, Luxia 2026-07-16).

v_perp = unit(v − (v·V7)V7), computed INSIDE the band (all inputs are band-confined unit
vectors, so v_perp stays band-confined). Purpose: separate each member's coordinate content
from its temperature projection — +/−Vrep_perp = the clean (anti-)repetition dial candidate,
Veos_perp = the "does EOS-hazard have ANY V7-independent direction" probe. Negative doses are
sign flips at injection (no separate vectors needed).

Emits the perp bank + anatomy (cos to V7 must be ~0 — hard-checked; residual-norm fraction =
how much of the member survives orthogonalization; cos table vs comparators).

Usage (from pipeline/):
    python -m anamnesis.scripts.annex_perp_vectors \
        --members ../outputs/battery/annex/roster_vectors_3b/a5_vectors.npz \
        --keys Veos_L14:Veos_perp_L14 Vrep_L14:Vrep_perp_L14 \
        --v7-npz ../outputs/battery/a5_vectors_3b_b7/a5_vectors.npz \
        --stamps ../outputs/battery/annex/roster_vectors_3b/a5_vectors_stamps.json \
        --out-dir ../outputs/battery/annex/eosrep_vectors_3b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=Path, required=True)
    ap.add_argument("--keys", nargs="+", required=True, help="MEMBERKEY:PERPKEY pairs")
    ap.add_argument("--v7-npz", type=Path, required=True)
    ap.add_argument("--v7-key", default="V7_L14",
                    help="V7 key in --v7-npz (site-specific; 3B=V7_L14, 8B=V7_L16, …)")
    ap.add_argument("--stamps", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    members = np.load(args.members)
    v7 = np.load(args.v7_npz)[args.v7_key].astype(np.float64)
    v7 = v7 / np.linalg.norm(v7)
    cosine = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    bank, anatomy = {}, {}
    for pair in args.keys:
        mk, pk = pair.split(":")
        v = members[mk].astype(np.float64)
        v = v / np.linalg.norm(v)
        proj = (v @ v7) * v7
        perp = v - proj
        resid_frac = float(np.linalg.norm(perp))
        if resid_frac <= 1e-6:
            raise SystemExit(f"{mk} is parallel to V7 — no perp component")
        perp = perp / np.linalg.norm(perp)
        c7 = cosine(perp, v7)
        if abs(c7) > 1e-8:
            raise SystemExit(f"{pk}: cos to V7 = {c7} — orthogonalization failed")
        bank[pk] = perp.astype(np.float32)
        anatomy[pk] = {
            "source": mk, "cos_source_V7": cosine(v, v7),
            "residual_norm_fraction": resid_frac,
            "cos_perp_V7": c7,
            "cos_perp_source": cosine(perp, v),
        }

    stamps = json.loads(args.stamps.read_text())
    np.savez(args.out_dir / "a5_vectors.npz", **bank)
    (args.out_dir / "a5_vectors_stamps.json").write_text(json.dumps(
        {"median_resid_norms": stamps["median_resid_norms"], "band": stamps.get("band"),
         "provenance": "V7-orthogonalized roster members (Gram-Schmidt in-band; EOS/rep "
                       "follow-up block 2026-07-16)"}, indent=1))
    (args.out_dir / "perp_anatomy.json").write_text(json.dumps(anatomy, indent=1))
    print(json.dumps(anatomy, indent=1))


if __name__ == "__main__":
    main()
