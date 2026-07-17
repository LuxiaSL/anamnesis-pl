"""§B.7 stage-2 vector build (DESIGN-V4 §B.7; Luxia-ratified 14d). Assembles the free-gen
vector bank: V7 (the entropy-band candidate, already banked in the stage-1 npz, key V7) +
3 MATCHED-SUPPORT random nulls (random unit vectors confined to the FROZEN [16:256] eigenspace
of banked Σ_L14, renormalized — 13e matched-support null). Injected at L14, generated positions
only. CPU-only; deterministic null seed. Downstream free-gen = submit_b7_stage2.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NULL_SEED = 20260714
BAND = (16, 256)
SITE = 14


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b7-npz", type=Path, required=True, help="v4_b7_entropy_fresh_G_3b.npz (key V7)")
    ap.add_argument("--sigma", type=Path, required=True)
    ap.add_argument("--stamps", type=Path, required=True, help="a5_vectors_stamps.json for the site median norm")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--site", type=int, default=SITE,
                    help="map site (3B=14, 8B=16 — the 8B 2×2 swaps site per its pricing doc; "
                         "band [16:256] is the same FROZEN slice applied to that site's Σ)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    site = args.site

    v7 = np.load(args.b7_npz)["V7"].astype(np.float32)
    S = np.load(args.sigma)
    evals = S["evals"].astype(np.float64)
    evecs = S["evecs"].astype(np.float64)
    order = np.argsort(evals)[::-1]
    Uband = evecs[:, order[BAND[0]:BAND[1]]]        # (d, 240)

    vectors = {f"V7_L{site}": v7}
    rng = np.random.default_rng(NULL_SEED)
    for i in range(1, 4):
        c = rng.standard_normal(Uband.shape[1])
        r = Uband @ c
        vectors[f"Rband{i}_L{site}"] = (r / np.linalg.norm(r)).astype(np.float32)

    lsite = json.loads(args.stamps.read_text())["median_resid_norms"][f"L{site}"]
    stamps = {"median_resid_norms": {f"L{site}": lsite}, "band": list(BAND), "site": site,
              "provenance": "§B.7 stage-2: V7 (entropy-band) + 3 [16:256]-matched-support randoms"}
    np.savez(args.out_dir / "a5_vectors.npz", **vectors)
    (args.out_dir / "a5_vectors_stamps.json").write_text(json.dumps(stamps, indent=1))
    print(json.dumps({"built": list(vectors.keys()),
                      f"L{site}_median_norm": round(lsite, 3)}, indent=1))


if __name__ == "__main__":
    main()
