"""C3 injection-bank assembly (PREFLIGHT §3; session-4 Part C). Adds the matched-norm
random nulls (13e) to the V_temp bank so the orphaned-target ladder has R baselines at
every (site, α). 3 seeded random UNIT vectors PER SITE (independent per site — a null must
not be a single global direction) → Rc{1,2,3}_L{site}, alongside Vtemp_L{site}. Injection
norm (median_resid_norms) is unchanged; α = frac × that, per site. CPU-only, deterministic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SITES = [7, 14, 18, 21]
NULL_SEED = 20260714


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctemp-dir", type=Path, required=True, help="a5_vectors_3b_ctemp (V_temp bank)")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    src = dict(np.load(args.ctemp_dir / "a5_vectors.npz"))
    stamps = json.loads((args.ctemp_dir / "a5_vectors_stamps.json").read_text())
    d = next(iter(src.values())).shape[0]

    vectors = dict(src)                        # keep Vtemp_L{site}
    rng = np.random.default_rng(NULL_SEED)
    for s in SITES:
        for i in range(1, 4):
            r = rng.standard_normal(d)
            vectors[f"Rc{i}_L{s}"] = (r / np.linalg.norm(r)).astype(np.float32)
            stamps["vectors"][f"Rc{i}_L{s}"] = {"route": f"matched-norm random null (site L{s})",
                                                "trait": "13e R-baseline for V_temp ladder"}
    np.savez(args.out_dir / "a5_vectors.npz", **vectors)
    (args.out_dir / "a5_vectors_stamps.json").write_text(json.dumps(stamps, indent=1))
    print(json.dumps({"built": sorted(vectors.keys()),
                      "median_resid_norms": stamps["median_resid_norms"]}, indent=1))


if __name__ == "__main__":
    main()
