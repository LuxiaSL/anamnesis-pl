"""Column C freeze read: the A<->C bridge at adequate n, the decode-shaped last position, and per-family tests.

Data: the momentum-steer phase-2 children no lever touched (arms unworn, T1.1, min_p.05; 120 contexts x 8 slots
each), graded by posttraining's panel (``grades.jsonl``, depth 1). Lane A is their engine-lane signature
(``msteer-replay/p2/signed``, the ``.unworn`` signing); Column C is ``column_c_extract_koto.py`` on the same exact
ids (``msteer_colc_rows.py``).

Pairs are siblings of one context and one arm with a grade gap >= --min-gap. **Long** pairs have both members
>= --min-tokens (A's protocol of record); **short** pairs have a member under it. Rows either judge marks as
breaking the scene are left out.

Every number is out-of-fold: GroupKFold over contexts (--folds). In each fold the standardisers (and the
length residualisation) are fit on training rows, and the ranker (antisymmetric no-intercept logistic, C by inner
GroupKFold over contexts) on training pairs. Ties earn half credit. Reported:

* **A** on long pairs; **C** variants on long pairs (fit on long train pairs: the bridge) and on short pairs (fit on
  all train pairs).
* **Bridge agreement** on long pairs: the share of pairs where A's and C's out-of-fold orderings agree, against two
  independent rankers at those accuracies (acc_A acc_C + (1-acc_A)(1-acc_C)), with a context bootstrap of the
  difference and a permutation p (C's scores shuffled over pairs), and the score correlation.
* **Positions:** ``alast`` (the whole-answer chunk), ``alast_step`` (a 1-token decode step: the engine's shape),
  ``a0`` (first-token identity within a fan, a control) and ``plast`` (byte-shared within a fan: must read 0.5).
* **Families:** all, depth, read, route; all minus the engineered ``depth.band*`` and minus ``depth.area.*``.
* **Text baselines:** log length, the span's mean log-prob (from the A signing), the last token's identity
  (alast's text twin), and all three together; and C stacked with them.
* **The nonlinear rung** (--mlp): an MLP on pair differences, antisymmetrised (score = p(d) - p(-d)), on the
  first --mlp-pcs principal components of the training rows.

Usage: python column_c_freeze_read.py --colc <column_c_floor.npz> --a-signed <dir> --grades <grades.jsonl>
       --rows <colc rows.jsonl> --out <json>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from momentum_pairwise_read import design, fit_ranker  # noqa: E402


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


Pair = tuple[str, str]  # (higher grade, lower grade)


def tie_credit(s: np.ndarray) -> np.ndarray:
    return np.where(s > 0, 1.0, np.where(s < 0, 0.0, 0.5))


def ctx_bootstrap(values: np.ndarray, ctx: np.ndarray, n: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    keys = np.unique(ctx)
    by = {k: np.flatnonzero(ctx == k) for k in keys}
    stats = [values[np.concatenate([by[k] for k in rng.choice(keys, len(keys))])].mean() for _ in range(n)]
    return [round(float(np.percentile(stats, 2.5)), 4), round(float(np.percentile(stats, 97.5)), 4)]


def standardise(x: np.ndarray, train_rows: np.ndarray, resid: np.ndarray | None) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if resid is not None:
        d = np.column_stack([np.ones(len(x)), resid])
        beta, *_ = np.linalg.lstsq(d[train_rows], x[train_rows], rcond=None)
        x = x - d @ beta
    mu, sd = x[train_rows].mean(axis=0), x[train_rows].std(axis=0)
    live = np.isfinite(sd) & (sd > 1e-9)
    return (x[:, live] - mu[live]) / sd[live]


def oof_scores(X: np.ndarray, row: dict[str, int], train_sets: list[list[Pair]], test_sets: list[list[Pair]],
               ctx_of: Callable[[str], str], resid: np.ndarray | None,
               mlp: tuple[int, int] | None = None) -> tuple[list[np.ndarray], list[float]]:
    """Per fold: standardise on training rows, fit on training pairs, score the test pairs."""
    out, cs = [], []
    for tr, te in zip(train_sets, test_sets):
        if not tr or not te:
            out.append(np.zeros(len(te)))
            continue
        rows_tr = np.array(sorted({row[c] for p in tr for c in p}))
        Z = standardise(X, rows_tr, resid)
        vec = lambda c, _Z=Z: _Z[row[c]]  # noqa: E731
        if mlp is None:
            x, y = design(tr, vec)
            m, c = fit_ranker(x, y, [ctx_of(p[0]) for p in tr] * 2)
            d = np.stack([vec(a) - vec(b) for a, b in te])
            out.append(d @ m.coef_.ravel())
            cs.append(c)
        else:
            from sklearn.decomposition import PCA
            from sklearn.neural_network import MLPClassifier
            pcs, hidden = mlp
            pca = PCA(n_components=min(pcs, len(rows_tr) - 1), random_state=0).fit(Z[rows_tr])
            P = pca.transform(Z) / np.sqrt(pca.explained_variance_[0])
            v2 = lambda c, _P=P: _P[row[c]]  # noqa: E731
            x, y = design(tr, v2)
            m = MLPClassifier(hidden_layer_sizes=(hidden,), alpha=1e-1, max_iter=400, early_stopping=True,
                              random_state=0).fit(x, y)
            d = np.stack([v2(a) - v2(b) for a, b in te])
            out.append(m.predict_proba(d)[:, 1] - m.predict_proba(-d)[:, 1])
    return out, cs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--colc", required=True, type=Path, action="append", help="column_c_floor.npz (repeat for shards)")
    ap.add_argument("--a-signed", required=True, type=Path)
    ap.add_argument("--grades", required=True, type=Path)
    ap.add_argument("--rows", required=True, type=Path)
    ap.add_argument("--min-tokens", type=int, default=16)
    ap.add_argument("--min-gap", type=float, default=2.0)
    ap.add_argument("--len-ratio", type=float, default=1.5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-last", type=int, default=40)
    ap.add_argument("--mlp", action="store_true")
    ap.add_argument("--mlp-pcs", type=int, default=128)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    p2 = _sibling("msteer_p2_read")

    # ── rows, labels ──
    meta: dict[str, dict] = {}
    last_id: dict[str, int] = {}
    for line in args.rows.read_text().splitlines():
        r = json.loads(line)
        if r.get("_header"):
            continue
        meta[r["cand_id"]] = dict(r["meta"], n_tokens=int(r["n_tokens"]))
        last_id[r["cand_id"]] = int(r["input_ids"][-1])
    key = {(m["context_index"], m["arm"], m["slot"]): c for c, m in meta.items()}
    grade: dict[str, float] = {}
    dropped = Counter()
    for line in args.grades.read_text().splitlines():
        g = json.loads(line)
        c = key.get((g["context_index"], g["arm"], g["slot"]))
        if g.get("depth") != 1 or c is None:
            continue
        if g.get("grade") is None:
            dropped["no_grade"] += 1
        elif g.get("breaks_opus") or g.get("breaks_fable"):
            dropped["breaks_scene"] += 1
        else:
            grade[c] = float(g["grade"])

    # ── A ──
    signed = p2.load_signed(args.a_signed)
    a_ids = [c for c in meta if c in signed]
    names_a = signed[a_ids[0]]["names"]
    a_ids = [c for c in a_ids if signed[c]["names"] == names_a]
    XA = np.stack([signed[c]["vec"] for c in a_ids]).astype(np.float64)
    XA = XA[:, np.isfinite(XA).all(axis=0)]
    row_a = {c: i for i, c in enumerate(a_ids)}
    mean_lp = {c: float((signed[c]["rec"].get("reducer_meta") or {}).get("mean_logprob", np.nan)) for c in a_ids}

    # ── C ──
    parts = [np.load(f, allow_pickle=False) for f in args.colc]
    if len({str(q["digest"]) for q in parts}) != 1:
        raise ValueError("Column C shards carry different schema digests")
    if len({tuple(q["names"]) for q in parts}) != 1:
        raise ValueError("Column C shards carry different names")
    shared = ("names", "digest", "version")
    z = {k: (parts[0][k] if k in shared else np.concatenate([q[k] for q in parts])) for k in parts[0].files}
    names_c = [str(n) for n in z["names"]]
    ids_c = [str(c) for c in z["cand_ids"]]
    row_c = {c: i for i, c in enumerate(ids_c)}
    ntok = np.array([meta[c]["n_tokens"] for c in ids_c], dtype=float)
    log_n = np.log(np.maximum(ntok, 1.0))

    usable = [c for c in ids_c if c in grade and c in row_a]
    by_group: dict[tuple, list[str]] = defaultdict(list)
    for c in usable:
        by_group[(meta[c]["context_index"], meta[c]["arm"])].append(c)
    long_pairs: list[Pair] = []
    short_pairs: list[Pair] = []
    for members in by_group.values():
        for a in members:
            for b in members:
                if grade[a] - grade[b] >= args.min_gap:
                    if min(meta[a]["n_tokens"], meta[b]["n_tokens"]) >= args.min_tokens:
                        long_pairs.append((a, b))
                    else:
                        short_pairs.append((a, b))
    ctx_of = lambda c: f"c{meta[c]['context_index']}"  # noqa: E731
    contexts = sorted({ctx_of(c) for c in usable})
    gkf = GroupKFold(n_splits=args.folds)
    fold_of = {}
    for k, (_, te) in enumerate(gkf.split(contexts, groups=contexts)):
        for i in te:
            fold_of[contexts[i]] = k
    split = lambda ps, k, test: [p for p in ps if (fold_of[ctx_of(p[0])] == k) == test]  # noqa: E731
    long_tr = [split(long_pairs, k, False) for k in range(args.folds)]
    long_te = [split(long_pairs, k, True) for k in range(args.folds)]
    all_pairs = long_pairs + short_pairs
    all_tr = [split(all_pairs, k, False) for k in range(args.folds)]
    short_te = [split(short_pairs, k, True) for k in range(args.folds)]
    flat = lambda sets: [p for s in sets for p in s]  # noqa: E731
    LT, ST = flat(long_te), flat(short_te)

    def len_ok(p: Pair) -> bool:
        a, b = meta[p[0]]["n_tokens"], meta[p[1]]["n_tokens"]
        return max(a, b) / max(min(a, b), 1) <= args.len_ratio

    def summarise(scores: np.ndarray, pairs: list[Pair]) -> dict[str, Any]:
        corr = tie_credit(scores)
        ctx = np.array([ctx_of(p[0]) for p in pairs])
        lm = np.array([len_ok(p) for p in pairs])
        return dict(n=len(pairs), contexts=int(len(np.unique(ctx))), acc=round(float(corr.mean()), 4),
                    ci95=ctx_bootstrap(corr, ctx), len_matched=dict(
                        n=int(lm.sum()), acc=round(float(corr[lm].mean()), 4) if lm.any() else None))

    report: dict[str, Any] = dict(
        colc_digest=str(z["digest"]), colc_version=str(z["version"]), colc_names=len(names_c),
        a_lane=signed[a_ids[0]]["rec"].get("lane_id"), a_features=int(XA.shape[1]), min_tokens=args.min_tokens,
        min_gap=args.min_gap, folds=args.folds, rows=len(ids_c), graded=len(grade), dropped=dict(dropped),
        pairs=dict(long=len(long_pairs), short=len(short_pairs),
                   long_contexts=len({ctx_of(p[0]) for p in long_pairs}),
                   long_len_matched=sum(map(len_ok, long_pairs)), short_len_matched=sum(map(len_ok, short_pairs)),
                   long_by_arm=dict(Counter(meta[p[0]]["arm"] for p in long_pairs))),
        long={}, short={}, bridge={}, families={}, baselines={})
    print(json.dumps(report["pairs"]), flush=True)

    # A on long pairs
    def by_test(sc: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(sc) if sc else np.zeros(0)

    sa, cs = oof_scores(XA, row_a, long_tr, long_te, ctx_of, None)
    sA = by_test(sa)
    report["long"]["A"] = dict(summarise(sA, LT), C=cs)
    accA = float(tie_credit(sA).mean())
    print("A", json.dumps(report["long"]["A"]), flush=True)

    # text baselines (on rows of C's order)
    vocab = [t for t, _ in Counter(last_id[c] for c in ids_c).most_common(args.n_last)]
    vindex = {t: i for i, t in enumerate(vocab)}
    T_len = log_n[:, None]
    T_lp = np.array([[mean_lp.get(c, np.nan)] for c in ids_c])
    T_last = np.zeros((len(ids_c), len(vocab) + 1))
    for i, c in enumerate(ids_c):
        T_last[i, vindex.get(last_id[c], len(vocab))] = 1.0
    T_all = np.hstack([T_len, T_lp, T_last])
    texts = dict(length=T_len, mean_logprob=T_lp, last_token=T_last, text_all=T_all)
    for nm, T in texts.items():
        sl, _ = oof_scores(T, row_c, long_tr, long_te, ctx_of, None)
        ss, _ = oof_scores(T, row_c, all_tr, short_te, ctx_of, None)
        report["baselines"][nm] = dict(long=summarise(by_test(sl), LT), short=summarise(by_test(ss), ST))
        print("text", nm, json.dumps(report["baselines"][nm]), flush=True)

    fam_idx = {
        "all": list(range(len(names_c))),
        "depth": [i for i, n in enumerate(names_c) if n.startswith("depth.")],
        "read": [i for i, n in enumerate(names_c) if n.startswith("read.")],
        "route": [i for i, n in enumerate(names_c) if n.startswith("route.")],
        "all-band": [i for i, n in enumerate(names_c) if not n.startswith("depth.band")],
        "all-area": [i for i, n in enumerate(names_c) if not n.startswith("depth.area.")],
        "all-band-area": [i for i, n in enumerate(names_c) if not n.startswith(("depth.band", "depth.area."))],
    }
    report["family_sizes"] = {k: len(v) for k, v in fam_idx.items()}

    def run_c(position: str, fam: str, resid: bool, bridge: bool = True, short: bool = True,
              mlp: tuple[int, int] | None = None) -> dict[str, Any]:
        X = z[f"floor_{position}"][:, fam_idx[fam]]
        r = log_n if resid else None
        out: dict[str, Any] = {}
        if bridge:
            sc, cs_ = oof_scores(X, row_c, long_tr, long_te, ctx_of, r, mlp)
            sC = by_test(sc)
            out["long"] = dict(summarise(sC, LT), C=cs_)
            accC = float(tie_credit(sC).mean())
            agree_v = (np.sign(sA) == np.sign(sC)).astype(float)
            indep = accA * accC + (1 - accA) * (1 - accC)
            rng = np.random.default_rng(1)
            null = np.array([np.mean(np.sign(sA) == np.sign(rng.permutation(sC))) for _ in range(2000)])
            ctx = np.array([ctx_of(p[0]) for p in LT])
            out["agreement"] = dict(agree=round(float(agree_v.mean()), 4), if_independent=round(indep, 4),
                                    excess=round(float(agree_v.mean() - indep), 4),
                                    excess_ci95=[round(v - indep, 4) for v in ctx_bootstrap(agree_v, ctx)],
                                    p_perm=round(float((np.sum(null >= agree_v.mean()) + 1) / 2001), 4),
                                    score_corr=round(float(np.corrcoef(sA, sC)[0, 1]), 4))
        if short:
            sc, _ = oof_scores(X, row_c, all_tr, short_te, ctx_of, r, mlp)
            out["short"] = summarise(by_test(sc), ST)
        return out

    variants = [(pos, fam, res) for pos in ("alast", "alast_step") for fam in ("all", "depth", "read", "route")
                for res in (False, True)]
    variants += [("a0", "all", True), ("plast", "all", False)]
    for pos, fam, res in variants:
        tag = f"{pos}.{fam}.{'resid' if res else 'raw'}"
        out = run_c(pos, fam, res)
        report["long"][tag] = out["long"]
        report["short"][tag] = out["short"]
        report["bridge"][tag] = out["agreement"]
        print("C", tag, json.dumps(out), flush=True)
    for fam in ("all-band", "all-area", "all-band-area"):
        tag = f"alast_step.{fam}.resid"
        out = run_c("alast_step", fam, True)
        report["families"][tag] = out
        print("C", tag, json.dumps(out), flush=True)

    # C stacked with the text baselines (alast_step, resid)
    Xs = np.hstack([np.nan_to_num(z["floor_alast_step"].astype(np.float64)), T_all])
    sc, _ = oof_scores(Xs, row_c, all_tr, short_te, ctx_of, log_n)
    report["short"]["alast_step.all.resid+text_all"] = summarise(by_test(sc), ST)
    sc, _ = oof_scores(Xs, row_c, long_tr, long_te, ctx_of, log_n)
    report["long"]["alast_step.all.resid+text_all"] = summarise(by_test(sc), LT)
    print("C+text", json.dumps(report["short"]["alast_step.all.resid+text_all"]),
          json.dumps(report["long"]["alast_step.all.resid+text_all"]), flush=True)

    if args.mlp:
        out = run_c("alast_step", "all", True, mlp=(args.mlp_pcs, 64))
        report["mlp"] = dict(pcs=args.mlp_pcs, hidden=64, **out)
        print("MLP", json.dumps(report["mlp"]), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
