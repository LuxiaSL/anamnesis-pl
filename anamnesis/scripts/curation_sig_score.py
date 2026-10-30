"""A per-turn signature coherence score: the curation read's signature ranker, exported for every signed turn.

The ranker is the read's (``curation_turn_read.py``): an antisymmetric no-intercept logistic regression on
within-scene, within-writer pair differences (coherent >= --hi vs <= --lo), the signature z-scored by the pooled
within-scene SD of the labelled turns, C picked by an inner GroupKFold. A turn's score is w . z(v).

Cross-fit: labelled scenes (scenes with at least one pair) are split into --folds GroupKFold folds by scene. Every turn of a fold's scenes
(labelled or not) is scored by the model fitted on the other folds, so no scene is scored by a model that saw its
labels. Turns of scenes with no labels are scored by the model fitted on every labelled scene (fold "all").

The score ranks turns WITHIN a scene; its level across scenes is not calibrated (scene effects cancel in the
pairs). score_z is the score z-scored within its scene, for gates that compare across scenes.

Inputs are ``curation_r2_join.py`` output dirs: --train (labelled; one or more) and --score (turns to score; a
labelled dir may be given as both).

Usage: python curation_sig_score.py --train <join dir> [--train ...] --score <join dir> [--score ...] --out <jsonl>
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


def load_dir(d: Path) -> tuple[list[dict], np.ndarray, list[str]]:
    _, rows = msr.load_rows(d / "rows.jsonl")
    X, ids, names, _, _ = msr.load_bank(d / "sigs", "engine-stats-b1")
    return [rows[c] for c in ids], X.astype(np.float64), names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, type=Path, action="append")
    ap.add_argument("--score", required=True, type=Path, action="append")
    ap.add_argument("--hi", type=int, default=4)
    ap.add_argument("--lo", type=int, default=2)
    ap.add_argument("--pairs-per-scene", type=int, default=20)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    from sklearn.model_selection import GroupKFold

    meta: list[dict] = []
    blocks: list[np.ndarray] = []
    names: list[str] | None = None
    for d in args.train:
        m, X, n = load_dir(d)
        if names is not None and n != names:
            raise ValueError(f"{d}: feature names differ from the first train dir")
        names = n
        meta += m
        blocks.append(X)
    Xt = np.concatenate(blocks)
    scene = np.array([m["scene_id"] for m in meta])
    finite = np.isfinite(Xt).all(axis=0)
    resid, dof = np.zeros(Xt.shape[1]), 0
    for s in np.unique(scene):
        u = np.flatnonzero(scene == s)
        if len(u) >= 2:
            resid += ((Xt[u] - Xt[u].mean(axis=0)) ** 2).sum(axis=0)
            dof += len(u) - 1
    sd = np.sqrt(resid / max(dof, 1))
    live = finite & (sd > 1e-9 * np.maximum(1.0, np.nan_to_num(np.abs(Xt)).max(axis=0)))

    def z(X: np.ndarray) -> np.ndarray:
        return np.nan_to_num(X[:, live] / sd[live])

    Zt = z(Xt)
    pairs, pscene = ctr.make_pairs(meta, args.hi, args.lo, args.pairs_per_scene, args.seed)
    labelled = np.unique(pscene)
    print(f"[score] train rows {len(meta)}, pairs {len(pairs)}, labelled scenes {len(labelled)}, "
          f"features {int(live.sum())}", flush=True)

    fold_of: dict[str, int] = {}
    models: dict[str, np.ndarray] = {}
    gkf = GroupKFold(n_splits=args.folds)
    for k, (tr, te) in enumerate(gkf.split(pairs, groups=pscene)):
        C = msr.pick_c(pairs[tr], Zt, pscene[tr])
        x, y = msr.design(pairs[tr], Zt)
        models[str(k)] = msr.fit(x, y, C).coef_.ravel()
        for s in np.unique(pscene[te]):
            fold_of[str(s)] = k
        acc = np.mean(((Zt[pairs[te, 0]] - Zt[pairs[te, 1]]) @ models[str(k)]) > 0)
        print(f"[score] fold {k}: C {C}, held-out pair accuracy {acc:.4f}", flush=True)
    C = msr.pick_c(pairs, Zt, pscene)
    x, y = msr.design(pairs, Zt)
    models["all"] = msr.fit(x, y, C).coef_.ravel()

    out_rows: list[dict[str, Any]] = []
    for d in args.score:
        m, X, n = load_dir(d)
        if n != names:
            raise ValueError(f"{d}: feature names differ from the train dirs")
        Z = z(X.astype(np.float64))
        for i, r in enumerate(m):
            fold = fold_of.get(r["scene_id"])
            key = "all" if fold is None else str(fold)
            out_rows.append(dict(scene_id=r["scene_id"], turn=r["turn"], writer=r.get("writer"),
                                 sign_id=r.get("sign_id"), score=float(Z[i] @ models[key]), fold=key,
                                 coherent=r.get("coherent"), set=r.get("set"), source=str(d)))
    by_scene: dict[str, list[dict]] = {}
    for r in out_rows:
        by_scene.setdefault(r["scene_id"], []).append(r)
    for rs in by_scene.values():
        s = np.array([r["score"] for r in rs])
        sdv = s.std()
        for r, v in zip(rs, s):
            r["score_z"] = float((v - s.mean()) / sdv) if sdv > 0 else 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")
    folds = {}
    for r in out_rows:
        folds[r["fold"]] = folds.get(r["fold"], 0) + 1
    print(json.dumps(dict(scored=len(out_rows), scenes=len(by_scene), by_fold=folds, out=str(args.out))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
