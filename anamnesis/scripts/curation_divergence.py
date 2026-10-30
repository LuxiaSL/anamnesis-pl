"""Reader divergence as a coherence measure (posttraining's candidate 4): does a fine-tune process a turn more
differently from the base where the turn is coherent (or incoherent)?

Per turn, e = z(writer's reading) - z(base's reading) of the same tokens, on the base reading's pooled within-scene
SD. The mean of e over a set is the fine-tune's lineage shift, nearly constant across texts. The residual
r = e - mean(e) is the turn's own divergence, and |r| is its size. The read is within scene, coherent >= --hi vs
<= --lo, using the curation pairs:

* pair accuracy of |r| as a one-number ranker (> .5: coherent turns diverge more; < .5: less);
* the same for log length alone;
* the same for |r| with scene-centred log length regressed out (the part length doesn't explain);
* |r| on length-matched pairs only (|dlog length| < 0.1 and < 0.2), where length cannot rank.

Usage: python curation_divergence.py --set label=<writer bank dir>:<base bank dir>[:<writer>] ... --out <json>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ctr = _sibling("curation_turn_read")
msr = sys.modules["msteer_replay_read"]


def pair_acc(s: np.ndarray, pairs: np.ndarray) -> float:
    d = s[pairs[:, 0]] - s[pairs[:, 1]]
    return float(np.mean(np.where(d > 0, 1.0, np.where(d < 0, 0.0, 0.5))))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, action="append")
    ap.add_argument("--hi", type=int, default=4)
    ap.add_argument("--lo", type=int, default=2)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    out = {}
    for spec in args.set:
        label, _, rest = spec.partition("=")
        parts = rest.split(":")
        wdir, bdir, writer = Path(parts[0]), Path(parts[1]), (parts[2] if len(parts) > 2 else "")
        _, rows = msr.load_rows(wdir / "rows.jsonl")
        Xw, idw, names, _, _ = msr.load_bank(wdir / "sigs", "engine-stats-b1")
        Xb, idb, names_b, _, _ = msr.load_bank(bdir / "sigs", "engine-stats-b1")
        if names != names_b:
            raise ValueError(f"{label}: feature names differ between the readers")
        bpos = {c: i for i, c in enumerate(idb)}
        keep = [i for i, c in enumerate(idw) if c in bpos and (not writer or rows[c].get("writer") == writer)]
        ids = [idw[i] for i in keep]
        A = Xw[keep].astype(np.float64)
        B = Xb[[bpos[c] for c in ids]].astype(np.float64)
        meta = [rows[c] for c in ids]
        scene = np.array([m["scene_id"] for m in meta])
        finite = np.isfinite(A).all(axis=0) & np.isfinite(B).all(axis=0)
        Bc = B[:, finite].copy()
        for s in np.unique(scene):
            u = scene == s
            Bc[u] -= Bc[u].mean(axis=0)
        sd = np.sqrt((Bc ** 2).sum(axis=0) / max(len(meta) - len(np.unique(scene)), 1))
        live = sd > 1e-9
        E = (A[:, finite][:, live] - B[:, finite][:, live]) / sd[live]
        shift = E.mean(axis=0)
        R = E - shift
        size = np.linalg.norm(R, axis=1)
        loglen = np.log(np.array([m["n_tokens"] for m in meta], dtype=float))
        sc_size, sc_len = size.copy(), loglen.copy()
        for s in np.unique(scene):
            u = scene == s
            sc_size[u] -= sc_size[u].mean()
            sc_len[u] -= sc_len[u].mean()
        beta = (sc_len @ sc_size) / (sc_len @ sc_len)
        resid = sc_size - beta * sc_len
        pairs, pscene = ctr.make_pairs(meta, args.hi, args.lo, 20, 0)
        out[label] = dict(turns=len(meta), pairs=int(len(pairs)), scenes=int(len(np.unique(pscene))),
                          shift_norm=round(float(np.linalg.norm(shift)), 2),
                          median_residual_norm=round(float(np.median(size)), 2),
                          corr_size_loglen=round(float(np.corrcoef(sc_size, sc_len)[0, 1]), 3),
                          acc_size=round(pair_acc(size, pairs), 4), acc_loglen=round(pair_acc(loglen, pairs), 4),
                          acc_size_len_resid=round(pair_acc(resid, pairs), 4))
        dl = np.abs(loglen[pairs[:, 0]] - loglen[pairs[:, 1]]) if len(pairs) else np.array([])
        for tol in (0.1, 0.2):
            mp = pairs[dl < tol]
            out[label][f"length_matched_{tol}"] = dict(pairs=int(len(mp)),
                                                       acc_size=round(pair_acc(size, mp), 4) if len(mp) else None,
                                                       acc_loglen=round(pair_acc(loglen, mp), 4) if len(mp) else None)
        print(label, json.dumps(out[label]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
