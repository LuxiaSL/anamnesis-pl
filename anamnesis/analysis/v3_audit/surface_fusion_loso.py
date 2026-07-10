"""Decompose UNIQUE vs REDUNDANT vs SYNERGISTIC mode signal across surfaces.

Per-surface floors (`surface_encoder_floor.py`) give each surface's standalone ceiling — but two
surfaces can both score high while reading the SAME signal (redundant) or DIFFERENT signal (unique).
This decomposes that, FOR MODE (redundancy is task-specific → re-run for taste on kotodama).

Smart fusion (avoids the width trap from the redundancy discussion): per fold, residualize each
surface on [prompt_len, gen_len], PCA-reduce each to K dims (train-fit), standardize, concat → small
fused rep. Then:

  TASK-LEVEL (what each surface uniquely contributes to predicting mode):
    - A_all                : fused logreg on all blocks
    - LOSO unique_s        : A_all − A_{−s}  (drop surface s's block). ≈0 ⇒ redundant; large ⇒ unique.
    - standalone floor_s   : logreg on block s alone (reduced-rep sanity vs the full encoder floor)
    - greedy incremental   : add surfaces in floor order; gain on entry = conditional value given the
                             rest already in. Exposes SYNERGY (pair > sum).

  DIRECTION-LEVEL (do surfaces encode mode the SAME way — basis-free, comparable across diff spaces):
    - pairwise prediction agreement / complementarity (exactly-one-correct = unique coverage)
    - pairwise CCA of LDA discriminant scores (max canonical corr of the two surfaces' mode discriminants)

Leak-proof GroupKFold-by-topic. PARALLEL: the (seed,fold) CV iterations are independent → a fork-based
process pool runs them concurrently (fork shares the big cached blocks copy-on-write — no IPC). Pin BLAS
to 1/worker. Pure sklearn/numpy, CPU, node1:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python surface_fusion_loso.py --model 3b --workers 16
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

CACHE = os.environ.get("SURFACE_CACHE_DIR", "/dev/shm/anamnesis_surface_caches")
OUTDIR = os.environ.get("SURFACE_FLOOR_DIR", ".")
SURFACES = ["residual", "keys", "values", "queries", "gate", "attention"]
K = 100
SEEDS = 3
SEED_PAIR = 0
_G: dict = {}   # shared with pool workers via fork (read-only; big blocks not pickled)


def load_all(model: str):
    blocks, ref_uid = {}, None
    mode = topic = C = None
    for s in SURFACES:
        path = os.path.join(CACHE, f"surface_cache_{model}_{s}.npz")
        if not os.path.exists(path):
            print(f"  [skip] {s}: cache missing", flush=True)
            continue
        z = np.load(path, allow_pickle=True)
        blocks[s] = np.nan_to_num(z["X"].astype(np.float32))
        uid = z["gen_uid"].astype(str)
        if ref_uid is None:
            ref_uid = uid
            mode = z["mode"].astype(str); topic = z["topic"].astype(int)
            C = np.column_stack([z["plen"].astype(float), z["glen"].astype(float)])
        elif not np.array_equal(uid, ref_uid):
            raise SystemExit(f"alignment error: {s} gen_uid != reference")
    return blocks, mode, topic, C


def _residualize(Ftr, Fte, Ctr, Cte):
    A = np.hstack([Ctr, np.ones((len(Ctr), 1))]); B = np.hstack([Cte, np.ones((len(Cte), 1))])
    coef, *_ = np.linalg.lstsq(A, Ftr, rcond=None)
    return Ftr - A @ coef, Fte - B @ coef


def _reduce_blocks(blocks, tr, te, C, seed):
    out = {}
    for s, X in blocks.items():
        Xtr, Xte = _residualize(X[tr], X[te], C[tr], C[te])
        k = min(K, Xtr.shape[0] - 1, Xtr.shape[1])
        pca = PCA(n_components=k, svd_solver="randomized", random_state=seed).fit(Xtr)
        Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
        sc = StandardScaler().fit(Ztr)
        out[s] = (sc.transform(Ztr), sc.transform(Zte))
    return out


def _fit_acc(red, surfs, ytr, yte):
    Xtr = np.hstack([red[s][0] for s in surfs]); Xte = np.hstack([red[s][1] for s in surfs])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    return float((clf.predict(Xte) == yte).mean())


def _task_level(args):
    """One (seed, fold): reduce → A_all, per-surface floor, LOSO drop, greedy curve."""
    seed, tr, te = args
    blocks, yi, C, surfs = _G["blocks"], _G["yi"], _G["C"], _G["surfs"]
    red = _reduce_blocks(blocks, tr, te, C, seed)
    ytr, yte = yi[tr], yi[te]
    acc_all = _fit_acc(red, surfs, ytr, yte)
    floors = {s: _fit_acc(red, [s], ytr, yte) for s in surfs}
    loso = {s: acc_all - _fit_acc(red, [x for x in surfs if x != s], ytr, yte) for s in surfs}
    order = sorted(surfs, key=lambda s: floors[s], reverse=True)
    cur, curve = [], []
    for s in order:
        cur = cur + [s]
        curve.append(_fit_acc(red, cur, ytr, yte))
    return {"acc_all": acc_all, "floors": floors, "loso": loso, "order": order, "curve": curve}


def _oof_fold(args):
    """One fold: per-surface LDA → held-out predictions + discriminant scores (for pairwise metrics)."""
    tr, te = args
    blocks, yi, C, surfs = _G["blocks"], _G["yi"], _G["C"], _G["surfs"]
    red = _reduce_blocks(blocks, tr, te, C, SEED_PAIR)
    out = {"te": te}
    for s in surfs:
        Ztr, Zte = red[s]
        lda = LDA().fit(Ztr, yi[tr])
        out[s] = (lda.predict(Zte), lda.transform(Zte))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["3b", "8b"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--K", type=int, default=100,
                    help="PCA dims/surface; raise to de-understate floors & confirm unique/redundant is K-robust")
    args = ap.parse_args()
    global K
    K = args.K
    blocks, mode, topic, C = load_all(args.model)
    surfs = [s for s in SURFACES if s in blocks]
    labels = sorted(set(mode.tolist()))
    yi = np.array([labels.index(m) for m in mode])
    n = len(yi)
    folds = list(GroupKFold(5).split(np.zeros(n), yi, topic))
    print(f"\n===== {args.model}  n={n}  topics={len(set(topic.tolist()))}  surfaces={surfs}  K={K}  "
          f"workers={args.workers}  (chance {1/len(labels):.0%}) =====", flush=True)

    _G.update(blocks=blocks, yi=yi, C=C, surfs=surfs)   # populated BEFORE pool → inherited via fork
    ctx = mp.get_context("fork")

    # ── TASK-LEVEL (parallel over seed×fold) ──
    tasks = [(seed, tr, te) for seed in range(SEEDS) for (tr, te) in folds]
    with ProcessPoolExecutor(max_workers=min(len(tasks), args.workers), mp_context=ctx) as ex:
        res = list(ex.map(_task_level, tasks))
    a_all = [r["acc_all"] for r in res]
    out = {
        "n": int(n), "labels": labels, "K": K,
        "A_all": {"mean": round(float(np.mean(a_all)), 4), "std": round(float(np.std(a_all)), 4)},
        "per_surface": {
            s: {"floor": round(float(np.mean([r["floors"][s] for r in res])), 4),
                "unique_loso_drop": round(float(np.mean([r["loso"][s] for r in res])), 4),
                "unique_std": round(float(np.std([r["loso"][s] for r in res])), 4)}
            for s in surfs
        },
        "greedy_example": {"order": res[0]["order"], "curve": [round(c, 4) for c in res[0]["curve"]]},
    }
    print(f"  A_all = {out['A_all']['mean']:.1%} ± {out['A_all']['std']:.1%}", flush=True)
    print("  surface         floor   unique(LOSO drop)", flush=True)
    for s in surfs:
        ps = out["per_surface"][s]
        print(f"    {s:12s}  {ps['floor']:.1%}    {ps['unique_loso_drop']:+.1%} ± {ps['unique_std']:.1%}", flush=True)

    # ── DIRECTION-LEVEL: pooled OOF (parallel over folds) → pairwise agreement / complementarity / CCA ──
    with ProcessPoolExecutor(max_workers=min(len(folds), args.workers), mp_context=ctx) as ex:
        oofs = list(ex.map(_oof_fold, folds))
    oof_pred = {s: np.full(n, -1) for s in surfs}
    oof_disc = {s: np.zeros((n, len(labels) - 1)) for s in surfs}
    for of in oofs:
        te = of["te"]
        for s in surfs:
            pred, disc = of[s]
            oof_pred[s][te] = pred
            oof_disc[s][te] = disc
    correct = {s: (oof_pred[s] == yi) for s in surfs}
    pairwise = {}
    for a, b in combinations(surfs, 2):
        agree = float(np.mean(oof_pred[a] == oof_pred[b]))
        complement = float(np.mean(correct[a] ^ correct[b]))
        try:
            cca = CCA(n_components=1).fit(oof_disc[a], oof_disc[b])
            ua, vb = cca.transform(oof_disc[a], oof_disc[b])
            canon = float(abs(np.corrcoef(ua[:, 0], vb[:, 0])[0, 1]))
        except Exception:
            canon = float("nan")
        pairwise[f"{a}|{b}"] = {"pred_agreement": round(agree, 3),
                                "complementarity": round(complement, 3),
                                "discriminant_cca": round(canon, 3)}
    out["pairwise"] = pairwise
    print("\n  pairwise (agreement / complementarity / discriminant-CCA):", flush=True)
    for pr, d in sorted(pairwise.items(), key=lambda kv: kv[1]["discriminant_cca"]):
        print(f"    {pr:24s} agree={d['pred_agreement']:.0%}  compl={d['complementarity']:.0%}  "
              f"CCA={d['discriminant_cca']:.2f}", flush=True)

    path = os.path.join(OUTDIR, f"surface_fusion_{args.model}_K{K}.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"\n  wrote {path}", flush=True)


if __name__ == "__main__":
    main()
