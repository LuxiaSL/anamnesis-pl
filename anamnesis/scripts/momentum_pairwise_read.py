"""Within-node pairwise read of the momentum contrast on tsumugi signatures.

The test of record from posttraining's momentum design: for sibling candidates at
one loom node (same context, speaker and position) whose panel grades differ by at
least ``--min-gap``, does a signature difference rank the higher-graded sibling
higher, beyond what the text-side covariates already rank?

Inputs: the momentum v2 signature shards (``signatures-*.npz`` from
``sign_koto_replay_rows.py``), the v2 labels JSONL, and the feature support table.
Features are the banked readout (``full_readout``) restricted to one lane id,
masked by the support table at the shortest admitted span, then standardised on
the training rows only.

Pairs: both siblings graded, survivors and not scene-breaking, at least
``--min-tokens`` tokens, grade gap at least ``--min-gap``. Each pair enters as
d = x_hi - x_lo with label 1 and its mirror with label 0, so a no-intercept
logistic model is an antisymmetric ranker. Rankers, each fit on the train split
(by source seed) and scored on the test split:

  length       log n_tokens
  surprisal    mean log-prob of the span
  form         question, imperative, second person, closure, ended at EOS
  meters       end-of-text pressure, length-normalised novelty
  text_all     all of the above
  signature    the masked signature
  sig+text     signature plus text_all

Reported per ranker: held-out pairwise accuracy (fraction of test pairs ordered
correctly), with a node-bootstrap 95% interval, and for signature and sig+text a
sign-flip permutation p-value. The same is reported on the length-matched subset
(n_chars ratio within ``--len-ratio``). Regularisation is chosen by inner
GroupKFold over source seeds on the training split.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

FORM = ("question", "imperative", "second_person", "closure", "ended_eos")
METERS = ("eot_pressure", "novelty_lennorm")


def load_bank(shard_dir: Path, lane_prefix: str):
    feats, ids, names = [], [], None
    for path in sorted(shard_dir.glob("signatures-*of*.npz")):
        z = np.load(path, allow_pickle=False)
        keep = np.array([str(l).startswith(lane_prefix) for l in z["lane_ids"]])
        feats.append(z["full_readout"][keep])
        ids += [str(c) for c in z["cand_ids"][keep]]
        names = [str(n) for n in z["full_names"]]
    return np.concatenate(feats), ids, names


def load_support(path: Path) -> dict[str, int]:
    with path.open() as fh:
        # "never": constant at every measured length (e.g. a count that cannot fire below
        # the sweep's range), so it is never admitted.
        return {row["feature_name"]: (10**9 if row["min_steps_computed"] == "never"
                                      else int(row["min_steps_computed"]))
                for row in csv.DictReader(fh, delimiter="\t")}


def load_labels(path: Path):
    cands, nodes = {}, []
    with path.open() as fh:
        for i, line in enumerate(fh):
            rec = json.loads(line)
            if rec.get("_header"):
                continue
            nodes.append(rec)
            for c in rec["candidates"]:
                cands[c["cand_id"]] = dict(c, node_id=rec["node_id"], seed=rec["source_seed"],
                                           split=rec["split"], tree=rec["tree_id"])
    return cands, nodes


def build_pairs(nodes, cands, have, min_tokens, min_gap):
    pairs = []
    for node in nodes:
        ok = [c for c in node["candidates"]
              if c.get("grade") is not None and c.get("survivor") and not c.get("breaks_scene")
              and c["n_tokens"] >= min_tokens and c["cand_id"] in have]
        for a in ok:
            for b in ok:
                if a["grade"] - b["grade"] >= min_gap:
                    pairs.append((a["cand_id"], b["cand_id"]))
    return pairs


def text_cov(c: dict, which: str) -> list[float]:
    out = []
    if which in ("length", "text_all"):
        out.append(np.log(max(c["n_tokens"], 1)))
    if which in ("surprisal", "text_all"):
        out.append(float(c["mean_logprob"]))
    if which in ("form", "text_all"):
        out += [float(bool(c.get(k))) for k in FORM]
    if which in ("meters", "text_all"):
        out += [float(c.get(k) or 0.0) for k in METERS]
    return out


def design(pairs, vec):
    d = np.stack([vec(hi) - vec(lo) for hi, lo in pairs])
    x = np.concatenate([d, -d])
    y = np.concatenate([np.ones(len(d)), np.zeros(len(d))])
    return x, y


def fit_ranker(x, y, groups, grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    n_groups = len(set(groups))
    best_c, best = grid[0], -1.0
    if n_groups >= 3:
        folds = GroupKFold(n_splits=min(5, n_groups))
        for c in grid:
            scores = []
            for tr, va in folds.split(x, y, groups):
                m = LogisticRegression(C=c, fit_intercept=False, max_iter=5000).fit(x[tr], y[tr])
                scores.append((m.predict(x[va]) == y[va]).mean())
            if np.mean(scores) > best:
                best, best_c = float(np.mean(scores)), c
    model = LogisticRegression(C=best_c, fit_intercept=False, max_iter=5000).fit(x, y)
    return model, best_c


def pair_correct(model, pairs, vec):
    d = np.stack([vec(hi) - vec(lo) for hi, lo in pairs])
    score = d @ model.coef_.ravel()
    # A tie (identical covariates, e.g. two questions) orders nothing: half credit.
    return np.where(score > 0, 1.0, np.where(score < 0, 0.0, 0.5))


def node_bootstrap(correct, pair_nodes, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    nodes = sorted(set(pair_nodes))
    by = {k: np.flatnonzero(np.array(pair_nodes) == k) for k in nodes}
    stats = []
    for _ in range(n):
        pick = rng.choice(len(nodes), len(nodes))
        idx = np.concatenate([by[nodes[i]] for i in pick])
        stats.append(correct[idx].mean())
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def sign_flip_p(model_fn, train_pairs, test_pairs, train_groups, vec, observed, n=200, seed=0):
    """Refit on train pairs with each pair's order flipped at random; p = share >= observed."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        flip = rng.random(len(train_pairs)) < 0.5
        shuffled = [(lo, hi) if f else (hi, lo) for (hi, lo), f in zip(train_pairs, flip)]
        x, y = design(shuffled, vec)
        model = LogisticRegression(C=model_fn, fit_intercept=False, max_iter=5000).fit(x, y)
        if pair_correct(model, test_pairs, vec).mean() >= observed:
            hits += 1
    return (hits + 1) / (n + 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--support", required=True, type=Path)
    ap.add_argument("--lane", default="659aea48")
    ap.add_argument("--min-tokens", type=int, default=16)
    ap.add_argument("--min-gap", type=float, default=2.0)
    ap.add_argument("--len-ratio", type=float, default=1.5)
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    x_all, ids, names = load_bank(args.shards, args.lane)
    support = load_support(args.support)
    steps_floor = args.min_tokens - 1
    cols = [i for i, n in enumerate(names) if support.get(n, 10**9) <= steps_floor]
    x_all = x_all[:, cols]
    finite = np.isfinite(x_all).all(axis=0)
    x_all, kept = x_all[:, finite], [names[c] for c, f in zip(cols, finite) if f]
    row = {c: i for i, c in enumerate(ids)}
    cands, nodes = load_labels(args.labels)

    pairs = build_pairs(nodes, cands, set(ids), args.min_tokens, args.min_gap)
    train = [p for p in pairs if cands[p[0]]["split"] == "train"]
    test = [p for p in pairs if cands[p[0]]["split"] == "test"]
    train_rows = sorted({row[c] for p in train for c in p})
    mu = x_all[train_rows].mean(axis=0)
    sd = x_all[train_rows].std(axis=0)
    live = sd > 1e-12
    z_all = (x_all[:, live] - mu[live]) / sd[live]

    def sig(c):
        return z_all[row[c]]

    def make_text(which):
        cov = {c: np.array(text_cov(cands[c], which)) for c in cands}
        return lambda c: cov[c]

    def both(c):
        return np.concatenate([sig(c), np.array(text_cov(cands[c], "text_all"))])

    rankers = {w: make_text(w) for w in ("length", "surprisal", "form", "meters", "text_all")}
    rankers["signature"] = sig
    rankers["sig+text"] = both

    def chars_ok(p):
        a, b = cands[p[0]]["n_chars"], cands[p[1]]["n_chars"]
        return max(a, b) / max(min(a, b), 1) <= args.len_ratio

    report = dict(lane=args.lane, min_tokens=args.min_tokens, min_gap=args.min_gap,
                  features_kept=int(live.sum()), features_banked=len(names),
                  pairs=dict(train=len(train), test=len(test),
                             test_len_matched=sum(chars_ok(p) for p in test),
                             test_nodes=len({cands[p[0]]["node_id"] for p in test})),
                  rankers={})
    groups = [cands[p[0]]["seed"] for p in train]
    groups2 = groups + groups
    for name, vec in rankers.items():
        x, y = design(train, vec)
        model, c = fit_ranker(x, y, groups2)
        entry = {"C": c}
        for subset, sel in (("all", test), ("len_matched", [p for p in test if chars_ok(p)])):
            if not sel:
                continue
            correct = pair_correct(model, sel, vec)
            lo, hi = node_bootstrap(correct, [cands[p[0]]["node_id"] for p in sel])
            entry[subset] = dict(n=len(sel), acc=round(float(correct.mean()), 4), ci95=[round(lo, 4), round(hi, 4)])
            if name in ("signature", "sig+text") and args.permutations:
                entry[subset]["p_signflip"] = round(sign_flip_p(c, train, sel, groups, vec, correct.mean(),
                                                                n=args.permutations), 4)
        report["rankers"][name] = entry
        print(name, json.dumps(entry), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
