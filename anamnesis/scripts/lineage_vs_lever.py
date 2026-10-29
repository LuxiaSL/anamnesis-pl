"""Did a fine-tune learn a lever's direction? Phase 2's lever directions set against the lineage reader shifts.

Phase 2 (``msteer_p2_read.py``) signed each steered child worn and unworn on the same tokens. Each lever arm's
mean worn - unworn delta, with the generic "steered at all" component (the unit mean of the R arms) projected
out, is that lever's own signature direction. The lineage read (``lineage_read.py``) signed one set of texts
under several readers' weights, so each reader contrast's mean delta is that fine-tune's direction on fixed text.

Both are z-scored by one scale, the phase-2 unworn rows' across-row SD, and set on the features the two share.
The script reports the cosine of every reader contrast with every lever direction, with the generic component
removed from both and kept in both. If DPO learned P's direction, DPO - SFT points along P, more than along R,
and b01/b03 point the same way.

Usage: python lineage_vs_lever.py --p2 <phase-2 signed dir> --signed release=<dir> --signed sft=<dir> ...
       --contrast dpo10x-sft ... --out <json>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p2 = _sibling("msteer_p2_read")


def matrix(rows: dict, ids: list[str], pos: dict[str, int], n: int) -> np.ndarray:
    X = np.full((len(ids), n), np.nan)
    for i, rid in enumerate(ids):
        r = rows[rid]
        for name, v in zip(r["names"], r["vec"]):
            j = pos.get(name)
            if j is not None:
                X[i, j] = v
    return X


def cos(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    return round(float(x[ok] @ y[ok] / (np.linalg.norm(x[ok]) * np.linalg.norm(y[ok]) + 1e-30)), 4)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p2", required=True, type=Path)
    ap.add_argument("--signed", required=True, action="append")
    ap.add_argument("--contrast", required=True, action="append")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    P2 = p2.load_signed(args.p2)
    readers = {s.partition("=")[0]: p2.load_signed(Path(s.partition("=")[2])) for s in args.signed}
    names_p2 = set().union(*(set(r["names"]) for r in P2.values()))
    first = next(iter(readers.values()))
    names_lin = set().union(*(set(r["names"]) for r in first.values()))
    shared = sorted(names_p2 & names_lin)
    pos = {n: i for i, n in enumerate(shared)}

    ids2 = sorted(P2)
    X2 = matrix(P2, ids2, pos, len(shared))
    unworn = np.array([rid.endswith(".unworn") for rid in ids2])
    sd = np.nanstd(X2[unworn], axis=0)
    live = np.isfinite(sd) & (sd > 1e-12)
    Z2 = X2[:, live] / sd[live]
    row = {rid: i for i, rid in enumerate(ids2)}
    deltas: dict[str, list[np.ndarray]] = defaultdict(list)
    for rid in ids2:
        if rid.endswith(".worn"):
            twin = rid[:-5] + ".unworn"
            if twin in row:
                arm = rid.rsplit(".", 1)[0].split(":")[2]
                deltas[arm].append(Z2[row[rid]] - Z2[row[twin]])
    lever = {a: np.nanmean(np.stack(v), axis=0) for a, v in deltas.items()}
    r_arms = [a for a in lever if a.startswith("R_")]
    g = np.nan_to_num(np.nanmean(np.stack([lever[a] for a in r_arms]), axis=0))
    g /= np.linalg.norm(g) + 1e-30

    def resid(v: np.ndarray) -> np.ndarray:
        v = np.nan_to_num(v)
        return v - (v @ g) * g

    common = sorted(set.intersection(*(set(r) for r in readers.values())))
    Zr = {k: matrix(v, common, pos, len(shared))[:, live] / sd[live] for k, v in readers.items()}
    out: dict[str, object] = dict(features_shared=int(live.sum()), p2_rows=len(ids2), lineage_rows=len(common),
                                  lever_arms=sorted(lever), cosines={})
    for c in args.contrast:
        a, _, b = c.partition("-")
        m = np.nanmean(Zr[a] - Zr[b], axis=0)
        out["cosines"][c] = dict(
            generic=cos(m, g),
            raw={arm: cos(m, lever[arm]) for arm in sorted(lever)},
            generic_removed={arm: cos(resid(m), resid(lever[arm])) for arm in sorted(lever)})
        print(c, json.dumps(out["cosines"][c]["generic_removed"]), "generic", out["cosines"][c]["generic"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
