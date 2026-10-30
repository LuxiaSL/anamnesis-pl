"""Per-turn reader divergence for posttraining's round-3 gate (signature curation v2).

The writer's reading of each joined turn is its signature on the canonical drain lane (``curation_r2_join.py``
output). The base's reading is a drain of the same exact ids through the release (``curation_grid_rows.py`` rows,
drained by ``curation-grid-drain-node2.sh``, possibly in shards). On the base reading's pooled within-scene SD:

* E = z(writer reading) - z(base reading);
* the writer's mean E (its lineage shift) is removed per writer, and div_raw = |E - shift|;
* div_z: div_raw z-scored within its (scene, writer) group. Writers differ in scale (DPO-10x diverges more), and in
  round-2 h4 one scene has two writers.
* div_lenres_z: div_raw with log length regressed out (one slope per writer, on group-centred values), then
  z-scored within its group. Raw divergence falls with length, so cutting its low tail raw would cut long turns. This
  is the column to gate on.

For a check, the turns that carry a coherence label give the pair accuracy, coherent >= 4 vs <= 2 within a group, of
div_raw, of div_lenres_z, and of log length, overall and per set.

Usage: python curation_div_export.py --join <dir> --base-signed <dir> [--base-signed ...] --out <jsonl>
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


ctr = _sibling("curation_turn_read")
msr = sys.modules["msteer_replay_read"]
p2 = _sibling("msteer_p2_read")


def zgroup(x: np.ndarray, group: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    for g in np.unique(group):
        u = group == g
        sd = x[u].std()
        out[u] = (x[u] - x[u].mean()) / sd if sd > 0 else 0.0
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--join", required=True, type=Path)
    ap.add_argument("--base-signed", required=True, type=Path, action="append")
    ap.add_argument("--labels-from", type=Path, action="append", default=[],
                    help="labelled join dirs: their coherent labels, by (scene_id, turn), for the check")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    _, rows = msr.load_rows(args.join / "rows.jsonl")
    Xw, ids, names, _, _ = msr.load_bank(args.join / "sigs", "engine-stats-b1")
    base: dict = {}
    for d in args.base_signed:
        base.update(p2.load_signed(d))
    keep = [i for i, c in enumerate(ids) if c in base and base[c]["names"] == names]
    ids = [ids[i] for i in keep]
    W = Xw[keep].astype(np.float64)
    B = np.stack([base[c]["vec"] for c in ids]).astype(np.float64)
    labels: dict[tuple[str, int], int] = {}
    for d in args.labels_from:
        _, lrows = msr.load_rows(d / "rows.jsonl")
        for r in lrows.values():
            if r.get("coherent") is not None:
                labels[(r["scene_id"], int(r["turn"]))] = int(r["coherent"])
    meta = [dict(rows[c], coherent=labels.get((rows[c]["scene_id"], int(rows[c]["turn"])))) for c in ids]
    print(f"[div] turns {len(ids)} (writer readings {len(Xw)}, base readings {len(base)})", flush=True)
    scene = np.array([m["scene_id"] for m in meta])
    writer = np.array([str(m.get("writer")) for m in meta])
    group = np.array([f"{s}|{w}" for s, w in zip(scene, writer)])
    finite = np.isfinite(W).all(axis=0) & np.isfinite(B).all(axis=0)
    Bc = B[:, finite].copy()
    for s in np.unique(scene):
        u = scene == s
        Bc[u] -= Bc[u].mean(axis=0)
    sd = np.sqrt((Bc ** 2).sum(axis=0) / max(len(ids) - len(np.unique(scene)), 1))
    live = sd > 1e-9
    E = (W[:, finite][:, live] - B[:, finite][:, live]) / sd[live]
    div = np.zeros(len(ids))
    for w in np.unique(writer):
        u = writer == w
        div[u] = np.linalg.norm(E[u] - E[u].mean(axis=0), axis=1)
    loglen = np.log(np.array([m["n_tokens"] for m in meta], dtype=float))
    lenres = np.zeros(len(ids))
    for w in np.unique(writer):
        u = writer == w
        dc, lc = div[u].copy(), loglen[u].copy()
        for g in np.unique(group[u]):
            v = group[u] == g
            dc[v] -= dc[v].mean()
            lc[v] -= lc[v].mean()
        beta = float(lc @ dc / (lc @ lc)) if (lc @ lc) > 0 else 0.0
        lenres[u] = div[u] - beta * loglen[u]
    div_z, lenres_z = zgroup(div, group), zgroup(lenres, group)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for c, m, a, b, d in zip(ids, meta, div, div_z, lenres_z):
            fh.write(json.dumps(dict(scene_id=m["scene_id"], turn=m["turn"], writer=m.get("writer"), set=m.get("set"),
                                     sign_id=m.get("sign_id"), n_tokens=m["n_tokens"], div_raw=round(float(a), 4),
                                     div_z=round(float(b), 4), div_lenres_z=round(float(d), 4))) + "\n")
    # check on labelled turns
    check = {}
    for setname in ["all"] + sorted({str(m.get("set")) for m in meta}):
        idx = [i for i, m in enumerate(meta) if m.get("coherent") is not None and (setname == "all"
                                                                                   or str(m.get("set")) == setname)]
        if not idx:
            continue
        sub = [dict(meta[i], scene_id=group[i]) for i in idx]
        pairs, _ = ctr.make_pairs(sub, 4, 2, 20, 0)
        if not len(pairs):
            continue
        acc = lambda s: round(float(np.mean(np.where(s[pairs[:, 0]] > s[pairs[:, 1]], 1.0,  # noqa: E731
                                                     np.where(s[pairs[:, 0]] < s[pairs[:, 1]], 0.0, 0.5)))), 4)
        check[setname] = dict(pairs=int(len(pairs)), div_raw=acc(div[idx]), div_lenres_z=acc(lenres_z[idx]),
                              loglen=acc(loglen[idx]))
    summary = dict(turns=len(ids), by_set={s: int((np.array([str(m.get('set')) for m in meta]) == s).sum())
                                           for s in sorted({str(m.get('set')) for m in meta})},
                   check_coherent_over_incoherent=check, out=str(args.out))
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
