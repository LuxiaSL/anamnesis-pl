"""Curation turns: does the signature separate coherent from incoherent turns beyond the text?

Rows: ``curation_turn_rows.py`` (posttraining's per-turn labels, base-weights subset), signed on the
tsumugi replay lane by ``sign_koto_replay_rows.py``. Each row is one turn replayed under its own
lead-in, as the loom rendered it.

The contrast is held within a scene, so context is matched. A pair is (a turn rated coherent >= --hi, a
turn rated <= --lo) from the same scene, at most --pairs-per-scene pairs per scene, drawn at random.
Each pair's difference d = v(hi) - v(lo) enters with label 1 and -d with label 0, so a no-intercept
logistic regression is an antisymmetric ranker (as in ``msteer_replay_read.py``). Folds are an outer
GroupKFold by scene; C is picked by an inner GroupKFold. The read reports out-of-fold accuracy over pairs.

Rankers:
  length       log span tokens
  logprob      the signer's mean log-prob of the span under its lead-in
  position     the turn index and log context tokens (late turns drift; their contexts are longer)
  echo         content-word overlap with the previous turn
  surface      all four of the above together
  content      a sentence embedding of the turn text (all-MiniLM-L6-v2), plus surface
  signature    the signature alone (z-scored by pooled within-scene SD)
  sig+content  signature, content embedding and surface together
Each accuracy carries a scene-bootstrap 95% interval. The signature and sig+content rankers also carry a
sign-flip permutation p, and the read reports a paired scene-bootstrap of sig+content minus content: the
signature's contribution beyond the text.

Usage: python curation_turn_read.py --rows <replay-rows.jsonl> --shards <sigs dir> --labels
<turn-labels.jsonl> --sample <sample.json> --support <support.tsv> --out <json>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"{name}.py not found beside {__file__}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


msr = _sibling("msteer_replay_read")

STOP = set("a an the and or but if of to in on at for with is are was were be been it its this that i you he she we "
           "they me him her us them my your his our their not no so do does did have has had what who why how".split())


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w not in STOP and len(w) > 2}


def echo(prev: str | None, text: str) -> float:
    if not prev:
        return 0.0
    a, b = content_words(text), content_words(prev)
    return len(a & b) / len(a) if a else 0.0


def make_pairs(meta: list[dict], hi: int, lo: int, per_scene: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    by_scene: dict[str, tuple[list[int], list[int]]] = {}
    for i, m in enumerate(meta):
        c = m.get("coherent")
        if c is None:
            continue
        h, l_ = by_scene.setdefault(m["scene_id"], ([], []))
        if c >= hi:
            h.append(i)
        elif c <= lo:
            l_.append(i)
    pairs, scene = [], []
    for s, (h, l_) in sorted(by_scene.items()):
        if not h or not l_:
            continue
        allp = [(a, b) for a in h for b in l_]
        take = rng.choice(len(allp), size=min(per_scene, len(allp)), replace=False)
        for k in take:
            pairs.append(allp[k])
            scene.append(s)
    return np.array(pairs, dtype=int), np.array(scene)


def scene_bootstrap(correct: np.ndarray, scene: np.ndarray, n: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    us = np.unique(scene)
    idx = {s: np.flatnonzero(scene == s) for s in us}
    stats = []
    for _ in range(n):
        pick = rng.choice(us, size=len(us), replace=True)
        rows = np.concatenate([idx[s] for s in pick])
        stats.append(correct[rows].mean())
    return [round(float(np.percentile(stats, 2.5)), 4), round(float(np.percentile(stats, 97.5)), 4)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, type=Path)
    ap.add_argument("--shards", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path, help="turn-labels.jsonl (turn text)")
    ap.add_argument("--sample", required=True, type=Path, help="sample.json (previous turns)")
    ap.add_argument("--support", required=True, type=Path)
    ap.add_argument("--lane", default="659aea48")
    ap.add_argument("--admit-steps", type=int, default=4)
    ap.add_argument("--hi", type=int, default=4)
    ap.add_argument("--lo", type=int, default=2)
    ap.add_argument("--pairs-per-scene", type=int, default=20)
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor

    from sklearn.model_selection import GroupKFold

    _, rows = msr.load_rows(args.rows)
    X, ids, names, mean_lp, lanes = msr.load_bank(args.shards, args.lane)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate cand_ids across shards")
    meta = [rows[c] for c in ids]
    steps = np.array([m["n_tokens"] - 1 for m in meta])
    support = msr.load_support(args.support)
    cols = [i for i, n in enumerate(names) if support.get(n, (10**9, ""))[0] <= args.admit_steps]
    keep = steps >= args.admit_steps
    Xa = X[keep][:, cols].astype(np.float64)
    meta = [m for m, k in zip(meta, keep) if k]
    lp = mean_lp[keep]
    finite = np.isfinite(Xa).all(axis=0)
    Xa = Xa[:, finite]
    fnames = [names[c] for c, f in zip(cols, finite) if f]

    # Pooled within-scene SD, so the scale is "how much turns of one scene differ".
    scene = np.array([m["scene_id"] for m in meta])
    resid, dof = np.zeros(Xa.shape[1]), 0
    for s in np.unique(scene):
        u = np.flatnonzero(scene == s)
        if len(u) >= 2:
            resid += ((Xa[u] - Xa[u].mean(axis=0)) ** 2).sum(axis=0)
            dof += len(u) - 1
    sd = np.sqrt(resid / max(dof, 1))
    live = sd > 1e-9 * np.maximum(1.0, np.abs(Xa).max(axis=0))
    Z = Xa[:, live] / sd[live]
    print(f"[read] lanes {lanes}; rows {len(meta)} (admit steps {args.admit_steps}); features {int(live.sum())}",
          flush=True)

    # Text side.
    texts: dict[tuple[str, int], str] = {}
    for line in args.labels.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            texts[(r["id"], int(r["turn"]))] = r["text"]
    scenes = {s["id"]: s for s in json.loads(args.sample.read_text())["scenes"]}
    tx = [texts[(m["scene_id"], int(m["turn"]))] for m in meta]
    prev = []
    for m in meta:
        turns = scenes[m["scene_id"]]["turns"]
        t = int(m["turn"])
        prev.append(turns[t - 1]["text"] if t >= 1 else None)
    surface = np.stack([
        np.log(np.array([m["n_tokens"] for m in meta], dtype=float)),
        lp,
        np.array([float(m["turn"]) for m in meta]),
        np.log(np.array([m["prompt_length"] for m in meta], dtype=float)),
        np.array([echo(p, t) for p, t in zip(prev, tx)]),
    ], axis=1)
    S = (surface - surface.mean(axis=0)) / np.where(surface.std(axis=0) > 0, surface.std(axis=0), 1.0)
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(args.embedder, device="cpu").encode(tx, batch_size=64, normalize_embeddings=True)
    E = (emb - emb.mean(axis=0)) / np.where(emb.std(axis=0) > 0, emb.std(axis=0), 1.0)

    V = {"length": S[:, [0]], "logprob": S[:, [1]], "position": S[:, [2, 3]], "echo": S[:, [4]],
         "surface": S, "content": np.concatenate([E, S], axis=1), "signature": Z,
         "sig+content": np.concatenate([Z, E, S], axis=1)}

    pairs, pscene = make_pairs(meta, args.hi, args.lo, args.pairs_per_scene, args.seed)
    n_scenes = len(np.unique(pscene))
    print(f"[read] pairs {len(pairs)} over {n_scenes} scenes (hi >= {args.hi}, lo <= {args.lo})", flush=True)
    folds = list(GroupKFold(n_splits=min(5, n_scenes)).split(pairs, groups=pscene))

    msr._G["_V"] = V
    msr._G["curation"] = dict(pairs=pairs, pctx=pscene, folds=folds, Cs={})
    out: dict[str, Any] = dict(rows=len(meta), features=int(live.sum()), lanes=lanes, admit_steps=args.admit_steps,
                               hi=args.hi, lo=args.lo, pairs=len(pairs), scenes=n_scenes, rankers={})
    correct: dict[str, np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for _, name, c, chosen in pool.map(msr._ranker_worker, [("curation", n) for n in V]):
            correct[name] = c
            msr._G["curation"]["Cs"][name] = chosen
            out["rankers"][name] = dict(acc=round(float(c.mean()), 4), ci95=scene_bootstrap(c, pscene), C=chosen)
            print(f"[read] {name}: {out['rankers'][name]}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [("curation", n, args.seed + 1000 + k) for n in ("signature", "sig+content")
                for k in range(args.permutations)]
        null: dict[str, list[float]] = {}
        for _, name, acc in pool.map(msr._perm_worker, jobs):
            null.setdefault(name, []).append(acc)
    for name, accs in null.items():
        a = np.array(accs)
        out["rankers"][name]["perm_p"] = round(float((1 + (a >= correct[name].mean()).sum()) / (1 + len(a))), 4)
        out["rankers"][name]["perm_null_mean"] = round(float(a.mean()), 4)
    out["beyond_text"] = msr.paired_diff_bootstrap(correct["sig+content"], correct["content"], pscene)
    out["wall_seconds"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("rows", "pairs", "scenes", "beyond_text")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
