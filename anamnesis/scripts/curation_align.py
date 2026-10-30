"""Do coherent turns written by different models line up in tsumugi's signature? (curation grid, one reader)

The writer x reader matrix (``curation_grid_bank.py``) showed the reader barely matters, so every writer's turns are
read here by ONE reader (default: the base) and z-scored on one scale: the pooled within-group SD over every set.
A group is (set, scene, writer). Every turn is centred on its group mean, so scene and writer offsets drop out and
what's left is how a turn deviates from the rest of its scene.

Per writer set (coherent >= --hi, incoherent <= --lo):

* g, b: the mean centred signature of its coherent turns and of its incoherent turns. m = g - b is its coherence
  direction.
* Alignment: the cosine of m, g and b between every two sets. The ceiling is each set's split-half cosine (scenes
  split at random into halves, the same vector from each). "Good" alignment is the cosine of g; "bad" is b's.
* Spread: the norms |g| and |b|, and the mean pairwise cosine of single centred turns, coherent with coherent and
  incoherent with incoherent, within a set and across sets. Do good turns agree with each other more than bad ones?
* Transfer: a pairwise ranker (the read's: antisymmetric logistic on within-group pair differences, C by inner
  GroupKFold) trained on set A's pairs and tested on set B's. The diagonal is cross-fit by scene.
* Absolute: a logistic classifier of centred turns (coherent vs incoherent, middling turns left out), trained on A,
  AUC on B. That is a gate that needs no scene-mates to compare with. The diagonal is cross-fit by scene.

Every quantity is reported raw, and with log(length) regressed out of the centred features (--length-control).

Usage: python curation_align.py --set base=<bank dir> --set esft=<bank dir> --set h4sft=<bank dir>:sft
       --set h4b01x=<bank dir>:b01x --set hyb=<bank dir> --out <json>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def offsets(Zc: np.ndarray, coh: np.ndarray, hi: int, lo: int) -> tuple[np.ndarray, np.ndarray]:
    return Zc[coh >= hi].mean(axis=0), Zc[coh <= lo].mean(axis=0)


def mean_pair_cos(A: np.ndarray, B: np.ndarray, n: int, rng: np.random.Generator, same: bool) -> float:
    if len(A) < 2 or len(B) < 2:
        return float("nan")
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-30)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-30)
    i, j = rng.integers(0, len(An), n), rng.integers(0, len(Bn), n)
    if same:
        keep = i != j
        i, j = i[keep], j[keep]
    return float((An[i] * Bn[j]).sum(axis=1).mean())


def auc(score: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, action="append", help="label=<bank dir>[:<writer>]")
    ap.add_argument("--hi", type=int, default=4)
    ap.add_argument("--lo", type=int, default=2)
    ap.add_argument("--pairs-per-scene", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    rng = np.random.default_rng(args.seed)
    labels, metas, blocks, names = [], [], [], None
    for spec in args.set:
        label, _, rest = spec.partition("=")
        path, _, writer = rest.partition(":")
        _, rows = msr.load_rows(Path(path) / "rows.jsonl")
        X, ids, n, _, _ = msr.load_bank(Path(path) / "sigs", "engine-stats-b1")
        if names is not None and n != names:
            raise ValueError(f"{label}: feature names differ")
        names = n
        keep = [i for i, c in enumerate(ids) if not writer or rows[c].get("writer") == writer]
        m = [dict(rows[ids[i]], set=label, group=f"{label}|{rows[ids[i]]['scene_id']}|{rows[ids[i]].get('writer')}")
             for i in keep]
        labels.append(label)
        metas.append(m)
        blocks.append(X[keep].astype(np.float64))
    X = np.concatenate(blocks)
    meta = [m for ms in metas for m in ms]
    group = np.array([m["group"] for m in meta])
    setl = np.array([m["set"] for m in meta])
    coh = np.array([m["coherent"] if m.get("coherent") is not None else -1 for m in meta], dtype=float)
    loglen = np.log(np.array([m["n_tokens"] for m in meta], dtype=float))
    finite = np.isfinite(X).all(axis=0)
    X = X[:, finite]
    Xc = X.copy()
    Lc = loglen.copy()
    for g in np.unique(group):
        u = group == g
        Xc[u] -= Xc[u].mean(axis=0)
        Lc[u] -= Lc[u].mean()
    dof = max(len(meta) - len(np.unique(group)), 1)
    sd = np.sqrt((Xc ** 2).sum(axis=0) / dof)
    live = sd > 1e-9
    Zc_raw = Xc[:, live] / sd[live]
    beta = (Lc @ Zc_raw) / (Lc @ Lc)
    variants = {"raw": Zc_raw, "length_controlled": Zc_raw - np.outer(Lc, beta)}
    print(f"[align] turns {len(meta)}, sets {labels}, features {int(live.sum())}", flush=True)

    out: dict[str, Any] = dict(sets={l: int((setl == l).sum()) for l in labels}, features=int(live.sum()),
                               hi=args.hi, lo=args.lo, variants={})
    for vname, Zc in variants.items():
        v: dict[str, Any] = {}
        vec = {}
        for l in labels:
            u = setl == l
            g, b = offsets(Zc[u], coh[u], args.hi, args.lo)
            vec[l] = dict(g=g, b=b, m=g - b)
            # split-half ceiling: scenes split at random
            groups = np.unique(group[u])
            half = set(rng.choice(groups, size=len(groups) // 2, replace=False))
            h1 = u & np.isin(group, list(half))
            h2 = u & ~np.isin(group, list(half))
            g1, b1 = offsets(Zc[h1], coh[h1], args.hi, args.lo)
            g2, b2 = offsets(Zc[h2], coh[h2], args.hi, args.lo)
            v.setdefault("split_half", {})[l] = dict(m=round(cos(g1 - b1, g2 - b2), 3), g=round(cos(g1, g2), 3),
                                                      b=round(cos(b1, b2), 3))
            v.setdefault("norms", {})[l] = dict(g=round(float(np.linalg.norm(g)), 3),
                                                b=round(float(np.linalg.norm(b)), 3),
                                                n_good=int((coh[u] >= args.hi).sum()),
                                                n_bad=int((coh[u] <= args.lo).sum()))
        for key in ("m", "g", "b"):
            v[f"cos_{key}"] = {a: {c: round(cos(vec[a][key], vec[c][key]), 3) for c in labels} for a in labels}
        # single-turn agreement: good with good, bad with bad, within and across sets
        spread: dict[str, Any] = {}
        for a in labels:
            for c in labels:
                ua, uc = setl == a, setl == c
                spread[f"{a}~{c}"] = dict(
                    good=round(mean_pair_cos(Zc[ua & (coh >= args.hi)], Zc[uc & (coh >= args.hi)], 20000, rng,
                                             a == c), 4),
                    bad=round(mean_pair_cos(Zc[ua & (coh <= args.lo) & (coh >= 0)],
                                            Zc[uc & (coh <= args.lo) & (coh >= 0)], 20000, rng, a == c), 4))
        v["turn_cos"] = spread
        # transfer: pairwise ranker and absolute classifier
        pairs_by, pscene_by = {}, {}
        for l in labels:
            idx = np.flatnonzero(setl == l)
            sub = [dict(meta[i], scene_id=meta[i]["group"]) for i in idx]
            p, ps = ctr.make_pairs(sub, args.hi, args.lo, args.pairs_per_scene, args.seed)
            pairs_by[l], pscene_by[l] = idx[p] if len(p) else p, ps
        rank_t: dict[str, dict[str, float]] = {}
        abs_t: dict[str, dict[str, float]] = {}
        for a in labels:
            pa, sa = pairs_by[a], pscene_by[a]
            C = msr.pick_c(pa, Zc, sa)
            x, y = msr.design(pa, Zc)
            w = msr.fit(x, y, C).coef_.ravel()
            ua = (setl == a) & ((coh >= args.hi) | ((coh <= args.lo) & (coh >= 0)))
            ya = (coh[ua] >= args.hi).astype(int)
            clf = LogisticRegression(C=C, max_iter=3000).fit(Zc[ua], ya)
            rank_t[a], abs_t[a] = {}, {}
            for c in labels:
                pc, sc = pairs_by[c], pscene_by[c]
                uc = (setl == c) & ((coh >= args.hi) | ((coh <= args.lo) & (coh >= 0)))
                yc = (coh[uc] >= args.hi).astype(int)
                if a == c:  # cross-fit by scene
                    correct = np.empty(len(pc))
                    for tr, te in GroupKFold(n_splits=5).split(pc, groups=sc):
                        xx, yy = msr.design(pc[tr], Zc)
                        ww = msr.fit(xx, yy, C).coef_.ravel()
                        correct[te] = ((Zc[pc[te, 0]] - Zc[pc[te, 1]]) @ ww) > 0
                    rank_t[a][c] = round(float(correct.mean()), 4)
                    idx = np.flatnonzero(uc)
                    score = np.empty(len(idx))
                    for tr, te in GroupKFold(n_splits=5).split(idx, groups=group[idx]):
                        m_ = LogisticRegression(C=C, max_iter=3000).fit(Zc[idx[tr]], yc[tr])
                        score[te] = m_.decision_function(Zc[idx[te]])
                    abs_t[a][c] = round(auc(score, yc), 4)
                else:
                    rank_t[a][c] = round(float((((Zc[pc[:, 0]] - Zc[pc[:, 1]]) @ w) > 0).mean()), 4)
                    abs_t[a][c] = round(auc(clf.decision_function(Zc[uc]), yc), 4)
            print(f"[align] {vname} train {a}: rank {rank_t[a]} abs {abs_t[a]}", flush=True)
        v["transfer_rank"] = rank_t
        v["transfer_abs_auc"] = abs_t
        out["variants"][vname] = v
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
