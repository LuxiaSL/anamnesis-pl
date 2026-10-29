"""Does steered text, replayed with no lever, sign like natural text from the same context?

Inputs: the replay rows from ``msteer_replay_rows.py`` (one per depth-1 child, with
arm, context_index, slot, n_tokens, grade), the signature shards that
``sign_koto_replay_rows.py`` wrote for them, and the tsumugi feature support table.

Design rules:

* Support mask. A feature is admitted at ``--admit-steps`` s (steps = tokens - 1)
  when its ``min_steps_computed`` <= s ("never" is never admitted). Rows with
  fewer than s steps are left out, so every row carries the same admitted set.
  The default s is the 10th percentile of the signed rows' steps.
* Matched nodes only. Every read compares a steered child with the unworn
  siblings at its own context (and, for Read B, with the unworn child of the same
  slot, which shares its seed). Rows are never pooled across contexts as samples
  of one distribution.
* Floor. Each feature is divided by the unworn siblings' pooled within-context SD
  (residuals from each context's unworn mean, pooled over contexts).

Read A (distance). For a steered child at context c, d = || z - mean(unworn at c) ||.
The floor at c is the median over unworn siblings of their leave-one-out distance
to the mean of the other unworn siblings. Ratio = d / floor_c. Per arm: the median
ratio over children, with a 95% context-bootstrap interval. The unworn arm's own
LOO ratios are the reference (median 1 by construction). Under a null where a
steered child is one more unworn draw, E[d^2] / E[floor^2] = (1 + 1/n) / (1 + 1/(n-1)),
about 0.99 at n = 8, so the null ratio sits a hair under 1. The same read runs per
feature family (support-table family) and per source (feature_map.classify).

Read B (separability beyond text). For each arm, pairs (steered child, unworn child)
at the same (context, slot). d = v(steered) - v(unworn) enters with label 1 and
-d with label 0; a no-intercept logistic regression is an antisymmetric ranker.
Outer GroupKFold over context_index (out-of-fold accuracy over all pairs), C by an
inner GroupKFold. Rankers: log length, mean log-prob (the signer's natural-forward
mean_logprob of the span), ended_by, all text, signature, signature + text.
Accuracy carries a context-bootstrap 95% interval; the signature and sig+text
rankers carry a sign-flip permutation p (training pairs flipped at random, C held
at each fold's chosen value).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np

GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
UNWORN = "unworn"

# Fork-inherited globals for the permutation workers.
_G: dict[str, Any] = {}


# ------------------------------------------------------------------------ loading

def load_rows(path: Path) -> tuple[dict, dict[str, dict]]:
    header: dict = {}
    rows: dict[str, dict] = {}
    with path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("_header"):
                header = rec
                continue
            rec.pop("input_ids", None)
            rows[rec["cand_id"]] = rec
    return header, rows


def load_bank(shard_dir: Path, lane_prefix: str) -> tuple[np.ndarray, list[str], list[str], np.ndarray, dict]:
    feats, ids, lps, names = [], [], [], None
    lanes: dict[str, int] = {}
    for path in sorted(shard_dir.glob("signatures-*of*.npz")):
        z = np.load(path, allow_pickle=False)
        lane = np.array([str(l) for l in z["lane_ids"]])
        for l in lane:
            lanes[l] = lanes.get(l, 0) + 1
        keep = np.array([l.startswith(lane_prefix) for l in lane], dtype=bool)
        feats.append(z["full_readout"][keep])
        ids += [str(c) for c in z["cand_ids"][keep]]
        lps.append(z["mean_logprob"][keep])
        shard_names = [str(n) for n in z["full_names"]]
        if names is not None and shard_names != names:
            raise ValueError(f"{path}: full_names differ between shards")
        names = shard_names
    if names is None:
        raise FileNotFoundError(f"no signatures-*of*.npz under {shard_dir}")
    return np.concatenate(feats), ids, names, np.concatenate(lps), lanes


def load_receipts(shard_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(shard_dir.glob("receipts-*.jsonl")):
        for line in path.open():
            if line.strip():
                rec = json.loads(line)
                out[rec["cand_id"]] = rec
    return out


def load_support(path: Path) -> dict[str, tuple[int, str]]:
    with path.open() as fh:
        return {r["feature_name"]: (10**9 if r["min_steps_computed"] == "never" else int(r["min_steps_computed"]),
                                    r["family"])
                for r in csv.DictReader(fh, delimiter="\t")}


def load_classify(feature_map_path: Path | None) -> Callable[[str, int], Any] | None:
    try:
        if feature_map_path is not None and feature_map_path.exists():
            spec = importlib.util.spec_from_file_location("feature_map_staged", feature_map_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["feature_map_staged"] = mod
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return mod.classify
        from anamnesis.analysis.feature_map import classify  # type: ignore
        return classify
    except Exception as exc:  # the family column still carries the decomposition
        print(f"[read] feature_map.classify unavailable ({type(exc).__name__}: {exc}); source read skipped", flush=True)
        return None


# ------------------------------------------------------------------------ helpers

def ctx_bootstrap(values: np.ndarray, ctx: np.ndarray, stat: Callable[[np.ndarray], float],
                  n: int = 2000, seed: int = 0) -> list[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    uniq = np.unique(ctx)
    by = [np.flatnonzero(ctx == c) for c in uniq]
    stats = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        stats[b] = stat(values[np.concatenate([by[i] for i in pick])])
    return [round(float(np.percentile(stats, 2.5)), 4), round(float(np.percentile(stats, 97.5)), 4)]


def r4(x: float) -> float:
    return round(float(x), 4)


# ------------------------------------------------------------------------ Read A

def read_a(Z: np.ndarray, meta: list[dict], groups: dict[str, np.ndarray], arms: list[str],
           grade_cut: float) -> dict[str, Any]:
    """groups: name -> boolean feature mask (the 'all' group plus families and sources)."""
    gnames = list(groups)
    M = np.stack([groups[g] for g in gnames], axis=1).astype(np.float64)  # F x G
    ctx = np.array([m["context_index"] for m in meta])
    arm = np.array([m["arm"] for m in meta])
    ntok = np.array([m["n_tokens"] for m in meta])
    grade = np.array([np.nan if m.get("grade") is None else float(m["grade"]) for m in meta])

    child_ratio: list[np.ndarray] = []   # per steered child: ratios per group
    child_dist: list[float] = []
    child_ctx, child_arm, child_lenok, child_grade = [], [], [], []
    loo_ratio: list[np.ndarray] = []
    loo_ctx: list[int] = []
    floor_all: list[float] = []
    n_ctx_used = 0
    for c in np.unique(ctx):
        at = ctx == c
        u = np.flatnonzero(at & (arm == UNWORN))
        if len(u) < 3:
            continue
        n_ctx_used += 1
        U = Z[u]
        tot = U.sum(axis=0)
        loo_sq = np.stack([((U[i] - (tot - U[i]) / (len(u) - 1)) ** 2) @ M for i in range(len(u))])  # n_u x G
        loo_d = np.sqrt(loo_sq)
        floor = np.median(loo_d, axis=0)  # G
        floor = np.where(floor > 0, floor, np.nan)
        floor_all.append(float(floor[0]))
        loo_ratio.append(loo_d / floor)
        loo_ctx += [int(c)] * len(u)
        cen = tot / len(u)
        lo, hi = ntok[u].min(), ntok[u].max()
        for s in np.flatnonzero(at & (arm != UNWORN)):
            sq = ((Z[s] - cen) ** 2) @ M
            child_ratio.append(np.sqrt(sq) / floor)
            child_dist.append(float(np.sqrt(sq[0])))
            child_ctx.append(int(c))
            child_arm.append(arm[s])
            child_lenok.append(bool(lo <= ntok[s] <= hi))
            child_grade.append(grade[s])
    R = np.stack(child_ratio)
    cctx, carm = np.array(child_ctx), np.array(child_arm)
    lenok, cgrade = np.array(child_lenok), np.array(child_grade)
    LOO = np.concatenate(loo_ratio)
    lctx = np.array(loo_ctx)

    def summary(vals: np.ndarray, cx: np.ndarray, seed: int) -> dict[str, Any]:
        ok = np.isfinite(vals)
        vals, cx = vals[ok], cx[ok]
        if len(vals) == 0:
            return dict(n=0)
        return dict(n=int(len(vals)), n_contexts=int(len(np.unique(cx))), median_ratio=r4(np.median(vals)),
                    ci95=ctx_bootstrap(vals, cx, np.median, seed=seed),
                    share_above_floor_p90=r4(np.mean(vals > np.nanpercentile(LOO[:, 0], 90))))

    out: dict[str, Any] = dict(
        n_contexts_used=n_ctx_used, floor_median_zL2=r4(np.median(floor_all)),
        floor_p10_p90=[r4(np.percentile(floor_all, 10)), r4(np.percentile(floor_all, 90))],
        n_features=int(groups["all"].sum()),
        unworn_loo=dict(n=int(len(LOO)), median_ratio=r4(np.median(LOO[:, 0])),
                        p90_ratio=r4(np.percentile(LOO[:, 0], 90)),
                        ci95=ctx_bootstrap(LOO[:, 0], lctx, np.median, seed=1)),
        null_expected_ratio_at_n8=r4(np.sqrt((1 + 1 / 8) / (1 + 1 / 7))),
        arms={}, families={}, sources={},
    )
    for k, a in enumerate(arms):
        sel = carm == a
        entry = summary(R[sel, 0], cctx[sel], seed=10 + k)
        entry["median_zL2"] = r4(np.median(np.array(child_dist)[sel]))
        entry["len_matched"] = summary(R[sel & lenok, 0], cctx[sel & lenok], seed=40 + k)
        hi_g = sel & (cgrade >= grade_cut)
        lo_g = sel & (cgrade < grade_cut)
        entry["grade_ge_cut"] = summary(R[hi_g, 0], cctx[hi_g], seed=70 + k)
        entry["grade_lt_cut"] = summary(R[lo_g, 0], cctx[lo_g], seed=100 + k)
        out["arms"][a] = entry
    # per family / source: median ratio per arm, and the unworn LOO reference per group
    steered = carm != UNWORN
    for gi, g in enumerate(gnames):
        if g == "all":
            continue
        kind, _, label = g.partition(":")
        bucket = out["families" if kind == "family" else "sources"]
        rec: dict[str, Any] = dict(n_features=int(groups[g].sum()),
                                   unworn_loo_median=r4(np.nanmedian(LOO[:, gi])),
                                   all_steered=summary(R[steered, gi], cctx[steered], seed=200 + gi))
        rec["arms"] = {a: r4(np.nanmedian(R[carm == a, gi])) for a in arms if a != UNWORN}
        bucket[label] = rec
    out["_per_child"] = dict(ratio_all=R[:, 0], ctx=cctx, arm=carm, len_ok=lenok, grade=cgrade,
                             group_names=np.array(gnames), ratio_groups=R)
    return out


# ------------------------------------------------------------------------ Read B

def design(pairs: np.ndarray, V: np.ndarray, flip: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    d = V[pairs[:, 0]] - V[pairs[:, 1]]
    y1 = np.ones(len(d))
    if flip is not None:
        d = np.where(flip[:, None], -d, d)
    x = np.concatenate([d, -d])
    y = np.concatenate([y1, 1 - y1])
    return x, y


def fit(x: np.ndarray, y: np.ndarray, C: float):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, fit_intercept=False, max_iter=3000).fit(x, y)


def pick_c(pairs: np.ndarray, V: np.ndarray, pctx: np.ndarray) -> float:
    from sklearn.model_selection import GroupKFold
    if len(np.unique(pctx)) < 3:
        return 1.0
    best_c, best = GRID[0], -1.0
    folds = list(GroupKFold(n_splits=3).split(pairs, groups=pctx))
    for C in GRID:
        sc = []
        for tr, va in folds:
            x, y = design(pairs[tr], V)
            m = fit(x, y, C)
            s = (V[pairs[va, 0]] - V[pairs[va, 1]]) @ m.coef_.ravel()
            sc.append(np.mean(np.where(s > 0, 1.0, np.where(s < 0, 0.0, 0.5))))
        if np.mean(sc) > best:
            best, best_c = float(np.mean(sc)), C
    return best_c


def oof_correct(pairs: np.ndarray, V: np.ndarray, pctx: np.ndarray, folds: list, Cs: list[float] | None,
                flip_seed: int | None = None) -> tuple[np.ndarray, list[float]]:
    correct = np.empty(len(pairs))
    chosen = []
    rng = np.random.default_rng(flip_seed) if flip_seed is not None else None
    for k, (tr, te) in enumerate(folds):
        C = Cs[k] if Cs is not None else pick_c(pairs[tr], V, pctx[tr])
        chosen.append(C)
        flip = rng.random(len(tr)) < 0.5 if rng is not None else None
        x, y = design(pairs[tr], V, flip)
        m = fit(x, y, C)
        s = (V[pairs[te, 0]] - V[pairs[te, 1]]) @ m.coef_.ravel()
        correct[te] = np.where(s > 0, 1.0, np.where(s < 0, 0.0, 0.5))
    return correct, chosen


def _perm_worker(args: tuple[str, str, int]) -> float:
    arm, ranker, seed = args
    g = _G[arm]
    c, _ = oof_correct(g["pairs"], g["V"][ranker], g["pctx"], g["folds"], g["Cs"][ranker], flip_seed=seed)
    return float(c.mean())


def read_b(Z: np.ndarray, meta: list[dict], text: np.ndarray, arms: list[str], n_perm: int,
           workers: int, len_ratio: float) -> dict[str, Any]:
    from sklearn.model_selection import GroupKFold
    key = {(m["context_index"], m["arm"], m["slot"]): i for i, m in enumerate(meta)}
    ntok = np.array([m["n_tokens"] for m in meta], dtype=float)
    # text columns: 0 log length, 1 mean log-prob, 2 ended_by length
    T = (text - text.mean(axis=0)) / np.where(text.std(axis=0) > 0, text.std(axis=0), 1.0)
    rankers = {"length": T[:, [0]], "mean_logprob": T[:, [1]], "ended_by": T[:, [2]],
               "text_all": T, "signature": Z, "sig+text": np.concatenate([Z, T], axis=1)}
    out: dict[str, Any] = {}
    for arm in arms:
        if arm == UNWORN:
            continue
        pairs, pctx = [], []
        for (c, a, s), i in key.items():
            if a == arm and (c, UNWORN, s) in key:
                pairs.append((i, key[(c, UNWORN, s)]))
                pctx.append(c)
        pairs_a, pctx_a = np.array(pairs), np.array(pctx)
        if len(np.unique(pctx_a)) < 5:
            out[arm] = dict(n_pairs=len(pairs), note="too few contexts")
            continue
        folds = list(GroupKFold(n_splits=5).split(pairs_a, groups=pctx_a))
        lr = np.abs(np.log(ntok[pairs_a[:, 0]] / ntok[pairs_a[:, 1]]))
        lm = lr <= np.log(len_ratio)
        entry: dict[str, Any] = dict(n_pairs=int(len(pairs_a)), n_contexts=int(len(np.unique(pctx_a))),
                                     n_len_matched=int(lm.sum()), rankers={})
        Cs: dict[str, list[float]] = {}
        for name, V in rankers.items():
            correct, chosen = oof_correct(pairs_a, V, pctx_a, folds, None)
            Cs[name] = chosen
            e = dict(acc=r4(correct.mean()), ci95=ctx_bootstrap(correct, pctx_a, np.mean, seed=len(name)),
                     C_per_fold=chosen)
            if lm.sum() >= 20:
                e["len_matched"] = dict(n=int(lm.sum()), acc=r4(correct[lm].mean()),
                                        ci95=ctx_bootstrap(correct[lm], pctx_a[lm], np.mean, seed=len(name) + 7))
            entry["rankers"][name] = e
            print(f"[readB] {arm} {name}: {e['acc']} {e['ci95']}", flush=True)
        if n_perm:
            _G[arm] = dict(pairs=pairs_a, V=rankers, pctx=pctx_a, folds=folds, Cs=Cs)
            for name in ("signature", "sig+text"):
                t0 = time.time()
                jobs = [(arm, name, 1000 + k) for k in range(n_perm)]
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    null = np.array(list(ex.map(_perm_worker, jobs, chunksize=1)))
                obs = entry["rankers"][name]["acc"]
                entry["rankers"][name]["p_signflip"] = r4((np.sum(null >= obs) + 1) / (n_perm + 1))
                entry["rankers"][name]["null_mean_sd"] = [r4(null.mean()), r4(null.std())]
                entry["rankers"][name]["n_perm"] = n_perm
                print(f"[readB] {arm} {name} perm p={entry['rankers'][name]['p_signflip']} "
                      f"null={null.mean():.3f}±{null.std():.3f} ({time.time() - t0:.0f}s)", flush=True)
            _G.pop(arm, None)
        out[arm] = entry
    return out


# ------------------------------------------------------------------------ write-up

def markdown(rep: dict[str, Any], arms: list[str]) -> str:
    A, B = rep["read_a"], rep["read_b"]
    lines = [f"# msteer replay read: tables ({rep['date']})", "",
             f"Rows signed: {rep['counts']['signed']} of {rep['counts']['rows']}; in the read: "
             f"{rep['counts']['in_read']} (steps >= {rep['admit_steps']}). Admitted features: {rep['n_features']}.",
             f"Floor: unworn LOO z-L2 median {A['floor_median_zL2']} (p10-p90 {A['floor_p10_p90']}) over "
             f"{A['n_contexts_used']} contexts; unworn LOO ratio median {A['unworn_loo']['median_ratio']} "
             f"(n {A['unworn_loo']['n']}, p90 {A['unworn_loo']['p90_ratio']}); null expectation ~{A['null_expected_ratio_at_n8']}.",
             "", "## Read A: distance to the unworn centroid / floor", "",
             "| arm | n | median ratio | CI95 | share > unworn p90 | len-matched n | len-matched ratio | CI95 |",
             "|---|---|---|---|---|---|---|---|"]
    for a in arms:
        if a == UNWORN:
            continue
        e = A["arms"][a]
        lm = e["len_matched"]
        lines.append(f"| {a} | {e['n']} | {e['median_ratio']} | {e['ci95']} | {e['share_above_floor_p90']} | "
                     f"{lm.get('n')} | {lm.get('median_ratio')} | {lm.get('ci95')} |")
    lines += ["", f"### Read A by panel grade (cut {rep['grade_cut']})", "",
              "| arm | n (grade >= cut) | ratio | CI95 | n (grade < cut) | ratio | CI95 |", "|---|---|---|---|---|---|---|"]
    for a in arms:
        if a == UNWORN:
            continue
        h, l = A["arms"][a]["grade_ge_cut"], A["arms"][a]["grade_lt_cut"]
        lines.append(f"| {a} | {h.get('n')} | {h.get('median_ratio')} | {h.get('ci95')} | {l.get('n')} | "
                     f"{l.get('median_ratio')} | {l.get('ci95')} |")
    for kind in ("families", "sources"):
        if not A[kind]:
            continue
        lines += ["", f"### Read A per {kind[:-1] if kind == 'sources' else 'family'} (median ratio; all steered arms pooled, then per arm)", "",
                  "| group | n feats | unworn LOO | all steered | CI95 | " + " | ".join(a for a in arms if a != UNWORN) + " |",
                  "|---|---|---|---|---|" + "---|" * (len(arms) - 1)]
        order = sorted(A[kind].items(), key=lambda kv: -(kv[1]["all_steered"].get("median_ratio") or 0))
        for g, e in order:
            lines.append(f"| {g} | {e['n_features']} | {e['unworn_loo_median']} | {e['all_steered'].get('median_ratio')} | "
                         f"{e['all_steered'].get('ci95')} | " + " | ".join(str(e["arms"][a]) for a in arms if a != UNWORN) + " |")
    lines += ["", "## Read B: held-out pairwise accuracy (steered vs unworn, same context and slot)", "",
              "| arm | pairs | ctx | length | mean_logprob | ended_by | text_all | signature | p | sig+text | p |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        if a == UNWORN or a not in B or "rankers" not in B[a]:
            continue
        r = B[a]["rankers"]
        cell = lambda k: f"{r[k]['acc']} {r[k]['ci95']}"
        lines.append(f"| {a} | {B[a]['n_pairs']} | {B[a]['n_contexts']} | {cell('length')} | {cell('mean_logprob')} | "
                     f"{cell('ended_by')} | {cell('text_all')} | {cell('signature')} | {r['signature'].get('p_signflip')} | "
                     f"{cell('sig+text')} | {r['sig+text'].get('p_signflip')} |")
    lines += ["", f"### Read B, length-matched pairs (|log length ratio| <= log {rep['len_ratio']}; models trained on all pairs)", "",
              "| arm | n | text_all | signature | sig+text |", "|---|---|---|---|---|"]
    for a in arms:
        if a == UNWORN or a not in B or "rankers" not in B[a]:
            continue
        r = B[a]["rankers"]
        if "len_matched" not in r["signature"]:
            continue
        cell = lambda k: f"{r[k]['len_matched']['acc']} {r[k]['len_matched']['ci95']}"
        lines.append(f"| {a} | {r['signature']['len_matched']['n']} | {cell('text_all')} | {cell('signature')} | {cell('sig+text')} |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------ main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, type=Path)
    ap.add_argument("--shards", required=True, type=Path)
    ap.add_argument("--support", required=True, type=Path)
    ap.add_argument("--feature-map", type=Path, default=None, help="staged feature_map.py (else import)")
    ap.add_argument("--lane", default="659aea48")
    ap.add_argument("--admit-steps", default="p10", help="int, or pNN = that percentile of signed rows' steps")
    ap.add_argument("--grade-cut", type=float, default=3.0)
    ap.add_argument("--len-ratio", type=float, default=1.25)
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--skip-read-b", action="store_true")
    ap.add_argument("--out", required=True, type=Path, help="output JSON (an .md and .per_child.npz beside it)")
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    args = ap.parse_args(argv)
    t0 = time.time()

    header, rows = load_rows(args.rows)
    X, ids, names, mean_lp, lanes = load_bank(args.shards, args.lane)
    receipts = load_receipts(args.shards)
    status_by_arm: dict[str, dict[str, int]] = {}
    for cid, r in rows.items():
        st = receipts.get(cid, {}).get("status", "missing")
        st = st if not st.startswith("failed") else "failed"
        status_by_arm.setdefault(r["arm"], {}).setdefault(st, 0)
        status_by_arm[r["arm"]][st] += 1
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate cand_ids across shards")
    meta = [rows[c] for c in ids]
    steps = np.array([m["n_tokens"] - 1 for m in meta])
    if isinstance(args.admit_steps, str) and args.admit_steps.startswith("p"):
        admit = int(np.percentile(steps, float(args.admit_steps[1:])))
    else:
        admit = int(args.admit_steps)

    support = load_support(args.support)
    cols = [i for i, n in enumerate(names) if support.get(n, (10**9, ""))[0] <= admit]
    keep_rows = steps >= admit
    Xa = X[keep_rows][:, cols].astype(np.float64)
    meta = [m for m, k in zip(meta, keep_rows) if k]
    lp = mean_lp[keep_rows]
    finite = np.isfinite(Xa).all(axis=0)
    n_nonfinite = int((~finite).sum())
    Xa, fnames = Xa[:, finite], [names[c] for c, f in zip(cols, finite) if f]

    # pooled within-context SD of the unworn siblings
    ctx = np.array([m["context_index"] for m in meta])
    arm = np.array([m["arm"] for m in meta])
    resid, dof = np.zeros(Xa.shape[1]), 0
    for c in np.unique(ctx):
        u = np.flatnonzero((ctx == c) & (arm == UNWORN))
        if len(u) >= 2:
            resid += ((Xa[u] - Xa[u].mean(axis=0)) ** 2).sum(axis=0)
            dof += len(u) - 1
    sd = np.sqrt(resid / max(dof, 1))
    live = sd > 1e-9 * np.maximum(1.0, np.abs(Xa).max(axis=0))
    Z = Xa[:, live] / sd[live]
    fnames = [n for n, l in zip(fnames, live) if l]
    print(f"[read] rows {len(meta)}, admit steps {admit}, features admitted {len(cols)}, non-finite {n_nonfinite}, "
          f"live {len(fnames)}", flush=True)

    # feature groups: all, support-table family, feature_map source
    groups: dict[str, np.ndarray] = {"all": np.ones(len(fnames), dtype=bool)}
    fam = np.array([support[n][1] for n in fnames])
    for f in sorted(set(fam)):
        groups[f"family:{f}"] = fam == f
    classify = load_classify(args.feature_map)
    sources_ok = False
    if classify is not None:
        try:
            layers = [int(m.group(1)) for n in fnames for m in [re.search(r"_L(\d+)", n)] if m]
            n_layers = max(layers) + 1 if layers else 1
            tags = [classify(n, n_layers) for n in fnames]
            src = np.array([str(getattr(t.source, "value", t.source)) for t in tags])
            for s in sorted(set(src)):
                groups[f"source:{s}"] = src == s
            sources_ok = True
        except Exception as exc:
            print(f"[read] source classification failed: {type(exc).__name__}: {exc}", flush=True)

    arms = [UNWORN] + sorted({m["arm"] for m in meta} - {UNWORN})
    A = read_a(Z, meta, groups, arms, args.grade_cut)
    per_child = A.pop("_per_child")

    B: dict[str, Any] = {}
    if not args.skip_read_b:
        text = np.stack([np.log(np.array([m["n_tokens"] for m in meta], dtype=float)), lp,
                         np.array([float(m.get("ended_by") == "length") for m in meta])], axis=1)
        B = read_b(Z, meta, text, arms, args.permutations, args.workers, args.len_ratio)

    rep: dict[str, Any] = dict(
        date=args.date, rows=str(args.rows), rows_header={k: header.get(k) for k in ("version", "source", "source_sha256")},
        shards=str(args.shards), support=str(args.support), lane=args.lane, lanes_seen=lanes,
        admit_steps=admit, admit_rule=f"features with min_steps_computed <= {admit}; rows with steps < {admit} left out",
        grade_cut=args.grade_cut, len_ratio=args.len_ratio,
        counts=dict(rows=len(rows), signed=len(ids), in_read=len(meta),
                    left_out_short=int((~keep_rows).sum()), status_by_arm=status_by_arm,
                    per_arm_in_read={a: int((arm == a).sum()) for a in arms}),
        n_features_banked=len(names), n_features_admitted=len(cols), n_nonfinite_dropped=n_nonfinite,
        n_features=len(fnames), sources_classified=sources_ok,
        read_a=A, read_b=B, wall_seconds=round(time.time() - t0, 1),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2))
    args.out.with_suffix(".md").write_text(markdown(rep, arms))
    np.savez(args.out.with_suffix(".per_child.npz"), **per_child)
    print(f"[read] wrote {args.out} ({rep['wall_seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
