"""Gate 1 for Column C: the A<->C bridge on momentum v2, and C on the short siblings A can't read.

The A side is the test of record, unchanged (``momentum_pairwise_read``: lane 659aea48, siblings
>= ``--min-tokens``, grade gap >= ``--min-gap``, support-table mask at the shortest admitted span,
train/test by the labels' source-seed split, inner GroupKFold over seeds for C). The C side reads
the FLOOR vector at the first answer position (``floor_a0`` from ``column_c_extract_koto``), for
every row, through the same antisymmetric pairwise ranker.

Reported:

* **bridge** (long pairs, both siblings in A): A and C held-out accuracy; per-pair agreement of
  the two rankers' orderings, against the agreement two independent rankers at those accuracies
  would show (acc_A acc_C + (1 - acc_A)(1 - acc_C)); the score correlation.
* **short** (test pairs with a sibling under ``--min-tokens``, which A cannot read): C trained on
  long pairs only (the transfer the bridge licenses) and on all train pairs; beside them the
  text-side baselines on the same pairs (length, whole-span surprisal, form, meters, all text, and
  the first-token log-probs, which are a0's text-side twin; with ``--rows``, the last token's
  identity, alast's twin). ``C+text`` stacks C with all text covariates (incl. last token).
* C variants: per family (depth, read, route), length-residualised (each feature regressed on
  log n_tokens over training rows) or not, and the shared last-prompt column ``plast`` as a
  sanity null (byte-shared within a fan, so its within-pair difference must be ~0 and the
  ranker must read chance).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from momentum_pairwise_read import (design, fit_ranker, load_bank, load_labels,  # noqa: E402
                                    load_support, node_bootstrap, pair_correct, text_cov)


def score(model, pairs, vec):
    d = np.stack([vec(hi) - vec(lo) for hi, lo in pairs])
    return d @ model.coef_.ravel()


def acc_entry(model, pairs, vec, cands, perm_fn=None):
    correct = pair_correct(model, pairs, vec)
    lo, hi = node_bootstrap(correct, [cands[p[0]]["node_id"] for p in pairs])
    e = dict(n=len(pairs), acc=round(float(correct.mean()), 4), ci95=[round(lo, 4), round(hi, 4)])
    if perm_fn is not None:
        e["p_signflip"] = round(perm_fn(correct.mean()), 4)
    return e


def sign_flip(c, train, test, vec, observed, n, seed=0):
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        flip = rng.random(len(train)) < 0.5
        shuffled = [(lo, hi) if f else (hi, lo) for (hi, lo), f in zip(train, flip)]
        x, y = design(shuffled, vec)
        m = LogisticRegression(C=c, fit_intercept=False, max_iter=5000).fit(x, y)
        hits += pair_correct(m, test, vec).mean() >= observed
    return (hits + 1) / (n + 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--support", required=True, type=Path)
    ap.add_argument("--colc", required=True, type=Path, help="column_c_floor.npz")
    ap.add_argument("--lane", default="659aea48")
    ap.add_argument("--min-tokens", type=int, default=16)
    ap.add_argument("--min-gap", type=float, default=2.0)
    ap.add_argument("--len-ratio", type=float, default=1.5)
    ap.add_argument("--permutations", type=int, default=100)
    ap.add_argument("--rows", type=Path, default=None,
                    help="replay-rows JSONL; adds the last-token baseline (alast's text-side twin)")
    ap.add_argument("--n-last", type=int, default=40, help="last-token vocabulary size (rest pooled)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    cands, nodes = load_labels(args.labels)

    # ── A, exactly as the test of record ──
    xa, ids_a, names_a = load_bank(args.shards, args.lane)
    support = load_support(args.support)
    cols = [i for i, n in enumerate(names_a) if support.get(n, 10**9) <= args.min_tokens - 1]
    xa = xa[:, cols]
    xa = xa[:, np.isfinite(xa).all(axis=0)]
    row_a = {c: i for i, c in enumerate(ids_a)}

    # ── C ──
    z = np.load(args.colc, allow_pickle=False)
    names_c = [str(n) for n in z["names"]]
    ids_c = [str(c) for c in z["cand_ids"]]
    row_c = {c: i for i, c in enumerate(ids_c)}
    ntok_c = z["n_tokens"].astype(float)
    first = {c: [float(z["logprob_a0"][i]),
                 float(np.nan_to_num(z["logprob_a1"][i], nan=0.0)),
                 float(z["n_tokens"][i] < 2)] for c, i in row_c.items()}

    def ok(c):
        return c.get("grade") is not None and c.get("survivor") and not c.get("breaks_scene")

    long_pairs, all_pairs = [], []
    for node in nodes:
        good = [c for c in node["candidates"] if ok(c) and c["cand_id"] in row_c]
        for a in good:
            for b in good:
                if a["grade"] - b["grade"] >= args.min_gap:
                    all_pairs.append((a["cand_id"], b["cand_id"]))
                    if (a["n_tokens"] >= args.min_tokens and b["n_tokens"] >= args.min_tokens
                            and a["cand_id"] in row_a and b["cand_id"] in row_a):
                        long_pairs.append((a["cand_id"], b["cand_id"]))
    is_short = lambda p: min(cands[p[0]]["n_tokens"], cands[p[1]]["n_tokens"]) < args.min_tokens  # noqa: E731
    split = lambda ps, s: [p for p in ps if cands[p[0]]["split"] == s]  # noqa: E731
    long_tr, long_te = split(long_pairs, "train"), split(long_pairs, "test")
    all_tr = split(all_pairs, "train")
    short_te = [p for p in split(all_pairs, "test") if is_short(p)]
    short_tr = [p for p in all_tr if is_short(p)]

    def chars_ok(p):
        a, b = cands[p[0]]["n_chars"], cands[p[1]]["n_chars"]
        return max(a, b) / max(min(a, b), 1) <= args.len_ratio

    groups = lambda ps: [cands[p[0]]["seed"] for p in ps] * 2  # noqa: E731

    # standardisers fit on the rows of a pair set's training pairs
    def standardiser(x, rows_of, train_pairs, resid=None):
        tr = sorted({rows_of[c] for p in train_pairs for c in p})
        x = x.astype(np.float64)
        if resid is not None:
            design_ = np.column_stack([np.ones(len(x)), resid])
            beta, *_ = np.linalg.lstsq(design_[tr], x[tr], rcond=None)
            x = x - design_ @ beta
        mu, sd = x[tr].mean(axis=0), x[tr].std(axis=0)
        live = (sd > 1e-9) & np.isfinite(sd)
        return (x[:, live] - mu[live]) / sd[live], int(live.sum())

    za, n_a = standardiser(xa, row_a, long_tr)
    vec_a = lambda c: za[row_a[c]]  # noqa: E731

    fam_cols = {
        "all": list(range(len(names_c))),
        "depth": [i for i, n in enumerate(names_c) if n.startswith("depth.")],
        "read": [i for i, n in enumerate(names_c) if n.startswith("read.")],
        "route": [i for i, n in enumerate(names_c) if n.startswith("route.")],
    }
    log_n = np.log(np.maximum(ntok_c, 1.0))

    def c_vec(position, fam, resid, train_pairs):
        x = z[f"floor_{position}"][:, fam_cols[fam]]
        zc, n_live = standardiser(np.nan_to_num(x, nan=0.0), row_c, train_pairs,
                                  resid=log_n if resid else None)
        return (lambda c: zc[row_c[c]]), n_live

    report = dict(colc_digest=str(z["digest"]), colc_names=len(names_c), lane=args.lane,
                  min_tokens=args.min_tokens, min_gap=args.min_gap, a_features=n_a,
                  pairs=dict(long_train=len(long_tr), long_test=len(long_te), all_train=len(all_tr),
                             short_train=len(short_tr), short_test=len(short_te),
                             short_test_len_matched=sum(map(chars_ok, short_te)),
                             short_test_nodes=len({cands[p[0]]["node_id"] for p in short_te})),
                  bridge={}, short={}, baselines_long={}, baselines_short={})
    print(json.dumps(report["pairs"]), flush=True)

    # ── A on long pairs (the test of record, refit here) ──
    xA, yA = design(long_tr, vec_a)
    mA, cA = fit_ranker(xA, yA, groups(long_tr))
    sA = score(mA, long_te, vec_a)
    accA = float(pair_correct(mA, long_te, vec_a).mean())
    report["A_long"] = dict(C=cA, all=acc_entry(mA, long_te, vec_a, cands),
                            len_matched=acc_entry(mA, [p for p in long_te if chars_ok(p)], vec_a, cands))
    print("A long", json.dumps(report["A_long"]), flush=True)

    # ── text baselines ──
    last_tok: dict[str, np.ndarray] = {}
    if args.rows is not None:
        import json as _json
        from collections import Counter
        ends = {}
        with args.rows.open() as fh:
            for line in fh:
                rec = _json.loads(line)
                if rec.get("_header"):
                    continue
                ends[rec["cand_id"]] = int(rec["input_ids"][-1])
        vocab = [t for t, _ in Counter(ends.values()).most_common(args.n_last)]
        index = {t: i for i, t in enumerate(vocab)}
        for c, t in ends.items():
            v = np.zeros(len(vocab) + 1)
            v[index.get(t, len(vocab))] = 1.0
            last_tok[c] = v
        report["last_token_vocab"] = len(vocab)

    def text_vec(which):
        if which == "last":
            return lambda c: last_tok[c]
        if which == "text+last":
            return lambda c: np.concatenate([text_cov(cands[c], "text_all"), first[c], last_tok[c]])
        if which == "first":
            return lambda c: np.array(first[c])
        if which == "text+first":
            return lambda c: np.concatenate([text_cov(cands[c], "text_all"), first[c]])
        return lambda c: np.array(text_cov(cands[c], which))

    for which in ("length", "surprisal", "form", "meters", "text_all", "first", "text+first",
                  *(("last", "text+last") if last_tok else ())):
        v = text_vec(which)
        x, y = design(long_tr, v)
        m, c = fit_ranker(x, y, groups(long_tr))
        report["baselines_long"][which] = acc_entry(m, long_te, v, cands)
        x, y = design(all_tr, v)
        m, c = fit_ranker(x, y, groups(all_tr))
        report["baselines_short"][which] = dict(all=acc_entry(m, short_te, v, cands),
                                                len_matched=acc_entry(m, [p for p in short_te if chars_ok(p)], v, cands))
        print("baseline", which, json.dumps(report["baselines_long"][which]),
              json.dumps(report["baselines_short"][which]), flush=True)

    # ── C variants ──
    variants = [(pos_, fam, resid) for pos_ in ("a0", "alast") for fam in ("all", "depth", "read", "route")
                for resid in (False, True)]
    variants += [("plast", "all", False), ("a3", "all", True)]
    for position, fam, resid in variants:
        tag = f"{position}.{fam}.{'resid' if resid else 'raw'}"
        if position == "a3":
            # exploratory: the 4th answer position, rows with >= 4 tokens only
            keep = lambda p: min(cands[p[0]]["n_tokens"], cands[p[1]]["n_tokens"]) >= 4  # noqa: E731
        else:
            keep = lambda p: True  # noqa: E731
        # bridge: C fit on long train pairs, scored on long test pairs, compared with A
        vec_c, n_live = c_vec(position, fam, resid, long_tr)
        x, y = design(long_tr, vec_c)
        mC, cC = fit_ranker(x, y, groups(long_tr))
        sC = score(mC, long_te, vec_c)
        accC = float(pair_correct(mC, long_te, vec_c).mean())
        agree = float(np.mean(np.sign(sA) == np.sign(sC)))
        chance_agree = accA * accC + (1 - accA) * (1 - accC)
        pf = (lambda obs, _c=cC, _v=vec_c: sign_flip(_c, long_tr, long_te, _v, obs, args.permutations)) \
            if fam == "all" and position in ("a0", "alast") else None
        b = dict(features=n_live, C=cC, C_long=acc_entry(mC, long_te, vec_c, cands, pf),
                 agreement=round(agree, 4), agreement_if_independent=round(chance_agree, 4),
                 score_corr=round(float(np.corrcoef(sA, sC)[0, 1]), 4))
        # agreement p: permute pairing of C scores against A scores within the test set
        rng = np.random.default_rng(1)
        null = [np.mean(np.sign(sA) == np.sign(rng.permutation(sC))) for _ in range(2000)]
        b["agreement_p_perm"] = round(float((np.sum(np.array(null) >= agree) + 1) / 2001), 4)
        # short pairs: transfer (long-trained) and all-trained
        st = [p for p in short_te if keep(p)]
        b_short = dict(transfer_from_long=dict(all=acc_entry(mC, st, vec_c, cands),
                                               len_matched=acc_entry(mC, [p for p in st if chars_ok(p)], vec_c, cands)))
        tr_all = [p for p in all_tr if keep(p)]
        vec_c2, _ = c_vec(position, fam, resid, tr_all)
        x, y = design(tr_all, vec_c2)
        m2, c2 = fit_ranker(x, y, groups(tr_all))
        pf2 = (lambda obs, _c=c2, _v=vec_c2: sign_flip(_c, tr_all, st, _v, obs, args.permutations)) \
            if fam == "all" and position in ("a0", "alast") else None
        b_short["trained_all"] = dict(C=c2, all=acc_entry(m2, st, vec_c2, cands, pf2),
                                      len_matched=acc_entry(m2, [p for p in st if chars_ok(p)], vec_c2, cands))
        if position in ("a0", "alast") and fam == "all":
            # C beside the text: does C add to text+first?
            tf = text_vec("text+last" if last_tok else "text+first")
            both = lambda c, _v=vec_c2: np.concatenate([_v(c), tf(c)])  # noqa: E731
            x, y = design(tr_all, both)
            m3, c3 = fit_ranker(x, y, groups(tr_all))
            b_short["C+text+first"] = acc_entry(m3, st, both, cands)
        report["bridge"][tag] = b
        report["short"][tag] = b_short
        print("C", tag, json.dumps(b), json.dumps(b_short), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
