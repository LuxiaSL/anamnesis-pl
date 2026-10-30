"""Where the coherence effect lives in tsumugi's signature: a decomposition of the curation read.

Rows and pairs are the curation read's (``curation_turn_read.py`` on ``curation_r2_join.py`` output): the signature
z-scored by the pooled within-scene SD, and pairs (coherent >= --hi vs <= --lo) within one scene and writer. Each
pair's difference D = z(coherent) - z(incoherent) is the unit.

Three views:

1. **Per feature: paired effects.**
   - d: mean(D) / sd(D) over pairs.
   - t: across scenes, the mean of scene-mean D over its standard error (scenes are the independent units).
   - Length-controlled: the intercept of D ~ a + b * dlog(length), over the residual SD, with the same scene t.
     Length alone ranks coherence at ~.69, so the raw effects partly restate length.
   - Aggregated by source, method, static/dynamic, depth band and layer (``feature_map.classify``, 72 layers):
     each group's share of the total squared effect, its median |d|, and how many features have |t| > 5.
2. **Rankers by group** (cross-fit as in the read: GroupKFold by scene, C by inner GroupKFold).
   - solo: the group's features alone;
   - drop: every feature but the group's.
   - Groups are the sources, the depth bands and static/dynamic.
3. **Structure coefficients.** The ranker fitted on every pair scores each turn. Each feature's correlation with
   that score, over turns centred within their scene, says what the ranker's direction is made of (raw weights
   do not, under correlated features).

Usage: python curation_decomp.py --join <dir> [--join ...] [--writer sft] --feature-map <feature_map.py>
       --out <json>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

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
_G: dict[str, Any] = {}


def scene_t(D: np.ndarray, pscene: np.ndarray) -> np.ndarray:
    """Per column: the mean of scene-mean D over its standard error across scenes."""
    scenes = np.unique(pscene)
    M = np.stack([D[pscene == s].mean(axis=0) for s in scenes])
    se = M.std(axis=0, ddof=1) / np.sqrt(len(scenes))
    return M.mean(axis=0) / np.where(se > 0, se, np.inf)


def _ranker(task: tuple[str, list[int]]) -> tuple[str, float, int]:
    from sklearn.model_selection import GroupKFold

    name, cols = task
    Z, pairs, pscene = _G["Z"][:, cols], _G["pairs"], _G["pscene"]
    correct = np.empty(len(pairs))
    for tr, te in GroupKFold(n_splits=5).split(pairs, groups=pscene):
        C = msr.pick_c(pairs[tr], Z, pscene[tr])
        x, y = msr.design(pairs[tr], Z)
        w = msr.fit(x, y, C).coef_.ravel()
        s = (Z[pairs[te, 0]] - Z[pairs[te, 1]]) @ w
        correct[te] = np.where(s > 0, 1.0, np.where(s < 0, 0.0, 0.5))
    return name, round(float(correct.mean()), 4), len(cols)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--join", required=True, type=Path, action="append")
    ap.add_argument("--writer", default=None)
    ap.add_argument("--feature-map", required=True, type=Path)
    ap.add_argument("--hi", type=int, default=4)
    ap.add_argument("--lo", type=int, default=2)
    ap.add_argument("--pairs-per-scene", type=int, default=20)
    ap.add_argument("--layers", type=int, default=72)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    classify = msr.load_classify(args.feature_map)
    if classify is None:
        raise SystemExit("feature_map.classify is needed for the decomposition")

    meta: list[dict] = []
    blocks: list[np.ndarray] = []
    names: list[str] | None = None
    for d in args.join:
        _, rows = msr.load_rows(d / "rows.jsonl")
        X, ids, n, _, _ = msr.load_bank(d / "sigs", "engine-stats-b1")
        if names is not None and n != names:
            raise ValueError(f"{d}: feature names differ")
        names = n
        keep = [i for i, c in enumerate(ids) if args.writer is None or rows[c].get("writer") == args.writer]
        meta += [rows[ids[i]] for i in keep]
        blocks.append(X[keep].astype(np.float64))
    X = np.concatenate(blocks)
    scene = np.array([m["scene_id"] for m in meta])
    finite = np.isfinite(X).all(axis=0)
    resid, dof = np.zeros(X.shape[1]), 0
    for s in np.unique(scene):
        u = np.flatnonzero(scene == s)
        if len(u) >= 2:
            resid += ((X[u] - X[u].mean(axis=0)) ** 2).sum(axis=0)
            dof += len(u) - 1
    sd = np.sqrt(resid / max(dof, 1))
    live = finite & (sd > 1e-9 * np.maximum(1.0, np.nan_to_num(np.abs(X)).max(axis=0)))
    Z = np.nan_to_num(X[:, live] / sd[live])
    fnames = [n for n, k in zip(names, live) if k]
    tags = [classify(n, args.layers) for n in fnames]
    pairs, pscene = ctr.make_pairs(meta, args.hi, args.lo, args.pairs_per_scene, args.seed)
    print(f"[decomp] rows {len(meta)}, pairs {len(pairs)}, scenes {len(np.unique(pscene))}, features {len(fnames)}",
          flush=True)

    # 1. paired effects, raw and length-controlled
    D = Z[pairs[:, 0]] - Z[pairs[:, 1]]
    dlen = np.log([meta[a]["n_tokens"] for a, _ in pairs]) - np.log([meta[b]["n_tokens"] for _, b in pairs])
    A = np.stack([np.ones(len(dlen)), dlen], axis=1)
    coef, *_ = np.linalg.lstsq(A, D, rcond=None)
    R = D - A @ coef
    eff = {"raw": dict(d=D.mean(axis=0) / np.where(D.std(axis=0) > 0, D.std(axis=0), np.inf), t=scene_t(D, pscene)),
           "length_controlled": dict(d=coef[0] / np.where(R.std(axis=0) > 0, R.std(axis=0), np.inf),
                                     t=scene_t(R + coef[0], pscene))}

    def group_of(tag: Any, by: str) -> str:
        v = getattr(tag, by)
        return str(getattr(v, "value", v))

    groups_out: dict[str, Any] = {}
    for kind, e in eff.items():
        total = float((e["d"] ** 2).sum())
        groups_out[kind] = {}
        for by in ("source", "method", "dynamic", "band", "layer"):
            agg: dict[str, list[int]] = defaultdict(list)
            for j, tag in enumerate(tags):
                agg[group_of(tag, by)].append(j)
            groups_out[kind][by] = {
                g: dict(n=len(js), share=round(float((e["d"][js] ** 2).sum()) / total, 4),
                        median_abs_d=round(float(np.median(np.abs(e["d"][js]))), 4),
                        n_t_gt_5=int((np.abs(e["t"][js]) > 5).sum()))
                for g, js in sorted(agg.items(), key=lambda kv: -float((e["d"][kv[1]] ** 2).sum()))}
        order = np.argsort(-np.abs(e["t"]))[:30]
        groups_out[kind]["top"] = [dict(name=fnames[j], d=round(float(e["d"][j]), 4), t=round(float(e["t"][j]), 2),
                                        source=group_of(tags[j], "source"), method=group_of(tags[j], "method"),
                                        layer=tags[j].layer) for j in order]
    lens_slope = dict(median_abs_b=round(float(np.median(np.abs(coef[1]))), 4))

    # 2. rankers by group
    _G.update(Z=Z, pairs=pairs, pscene=pscene)
    tasks: list[tuple[str, list[int]]] = [("all", list(range(len(fnames))))]
    for by in ("source", "band", "dynamic"):
        agg = defaultdict(list)
        for j, tag in enumerate(tags):
            agg[group_of(tag, by)].append(j)
        for g, js in agg.items():
            if len(js) >= 3:
                tasks.append((f"solo:{by}={g}", js))
                rest = sorted(set(range(len(fnames))) - set(js))
                tasks.append((f"drop:{by}={g}", rest))
    from concurrent.futures import ProcessPoolExecutor

    rankers: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, acc, n in pool.map(_ranker, tasks):
            rankers[name] = dict(acc=acc, features=n)
            print(f"[decomp] {name}: {acc} ({n})", flush=True)

    # 3. structure coefficients
    C = msr.pick_c(pairs, Z, pscene)
    x, y = msr.design(pairs, Z)
    w = msr.fit(x, y, C).coef_.ravel()
    score = Z @ w
    Zc, sc = Z.copy(), score.copy()
    for s in np.unique(scene):
        u = scene == s
        Zc[u] -= Zc[u].mean(axis=0)
        sc[u] -= sc[u].mean()
    r = (Zc * sc[:, None]).sum(axis=0) / (np.linalg.norm(Zc, axis=0) * np.linalg.norm(sc) + 1e-30)
    structure: dict[str, Any] = {}
    for by in ("source", "band", "dynamic"):
        agg = defaultdict(list)
        for j, tag in enumerate(tags):
            agg[group_of(tag, by)].append(j)
        structure[by] = {g: dict(n=len(js), mean_r2=round(float((r[js] ** 2).mean()), 4),
                                 max_abs_r=round(float(np.abs(r[js]).max()), 4)) for g, js in agg.items()}
    order = np.argsort(-np.abs(r))[:30]
    structure["top"] = [dict(name=fnames[j], r=round(float(r[j]), 4), source=group_of(tags[j], "source"),
                             layer=tags[j].layer) for j in order]

    out = dict(rows=len(meta), pairs=len(pairs), scenes=int(len(np.unique(pscene))), features=len(fnames),
               writer=args.writer, effects=groups_out, length_slope=lens_slope, rankers=rankers,
               structure=structure, C=C)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
