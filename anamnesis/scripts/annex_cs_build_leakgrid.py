"""Build the {grid, anatomy} input JSONs that `annex_cs_leakscore` consumes, from a V7-stack
entropy readout + a member-vector .npz. Data-prep helper (the leak LAW lives in the scorer);
promoted from a scratch one-off per the scripts-live-in-project rule.

grid   = {"cells": [{"name": <cell dir>, "key": "<VECTOR>_L<site>", "frac": <signed dose>,
                     "null": <True for Rband*>}]}
anatomy = {"<MEMBER>_L<site>": {"cos_to_V7": cos(member, V7)}}  (V7 key excluded; ⊥ n/a)

Usage:
    python -m anamnesis.scripts.annex_cs_build_leakgrid \
        --entropy-json <entropy_qwen.json> --vectors-npz <qwen_v7gen.npz> \
        --v7-key V7_L21 --site 21 --out-grid <grid.json> --out-anatomy <anatomy.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entropy-json", type=Path, required=True)
    ap.add_argument("--vectors-npz", type=Path, required=True)
    ap.add_argument("--v7-key", required=True)
    ap.add_argument("--site", type=int, required=True)
    ap.add_argument("--out-grid", type=Path, required=True)
    ap.add_argument("--out-anatomy", type=Path, required=True)
    args = ap.parse_args()

    ent = json.loads(args.entropy_json.read_text())
    rows = ent["rows"] if isinstance(ent, dict) else ent
    vecs = np.load(args.vectors_npz)
    v7 = vecs[args.v7_key]

    cells, anatomy = [], {}
    for r in rows:
        vector = r["vector"]                       # V7 / RA / Rband1..3 / V3
        key = f"{vector}_L{args.site}"
        is_null = vector.upper().startswith("RBAND")
        cells.append({"name": r["cell"], "key": key,
                      "frac": float(r["alpha_frac"]), "null": is_null})
        if not is_null and vector != args.v7_key.split("_")[0] and key in vecs:
            if key not in anatomy:
                anatomy[key] = {"cos_to_V7": round(cos(vecs[key], v7), 4)}

    args.out_grid.write_text(json.dumps({"cells": cells}, indent=1))
    args.out_anatomy.write_text(json.dumps(anatomy, indent=1))
    print(f"grid: {len(cells)} cells -> {args.out_grid}")
    print(f"anatomy: {anatomy} -> {args.out_anatomy}")


if __name__ == "__main__":
    main()
