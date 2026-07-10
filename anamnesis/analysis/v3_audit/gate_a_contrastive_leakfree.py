"""Leak-free contrastive eval — NESTED topic-held-out CV (the honest number).

The Gate-A contrastive readout (97-99% residualized, 5-way) is a LEAK: the 800-dim
`cp_*` features in signatures_v2_contrastive were produced by ONE projection trained
(train_contrastive_projection.py) on the eval run's own raw, ALL topics. Topic-heldout
CV then holds topics out of the kNN only — the projection already saw every topic.

This script removes the leak the only way possible: it **retrains the projection inside
each topic fold, on train-topics' samples only**, then projects the held-out topic fresh.
Everything else matches gate_a_contrastive.py (5-way hard modes, gen-level 800-dim,
kNN k=7 + RF x300, GroupKFold by topic, raw + length-residualized) so the delta from
97-99% is EXACTLY the projection-level leak.

Faithful replication knobs (match the leaked pipeline, change only the leak):
  - projection trained on ALL non-swap modes (as the Mar-2 projection was), eval 5-way hard;
  - per-gen 800-vec = project each of the 5 layers x 5 temporal corrected hidden states
    (identical sampling to feature_families/contrastive_projection.py) -> 32-dim -> concat;
  - by-group (by-generation) internal val split in train_projection (the fixed bug).

PARALLELISM (128-core node): the per-gen raw read (zlib-decompression bound) is fanned out
across `--load-workers` processes; the 15 independent (repeat x fold) projection-trainings
run across `--eval-workers` processes. BLAS is pinned to 1 thread/worker (OMP set before
numpy import) — the documented oversubscription gotcha (many workers x multi-threaded numpy
on 128 cores -> thrash). CPU-only; no GPU needed.

Runs on node1. From ~/luxi-files/anamnesis-pl with PYTHONPATH=pipeline:
    PYTHONPATH=pipeline python gate_a_contrastive_leakfree.py --root /models/anamnesis-extract
First pass reads the raw (slow I/O); it caches extracted hidden states to
gate_a_lf_cache_{run}.npz so reruns are instant.
"""
from __future__ import annotations

import os

# MUST precede numpy/torch import: forked pool workers inherit this -> 1 BLAS thread each.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from anamnesis.extraction.feature_families.contrastive_projection import (
    ContrastiveProjectionInference,
    _correct_hidden_state,
)
from anamnesis.extraction.raw_saver import load_raw_tensors
from anamnesis.scripts.train_contrastive_projection import train_projection

warnings.filterwarnings("ignore")

try:  # direct script run (sys.path[0] = this dir) — node1 self-contained convention
    from _common import HARD, gen_metadata_by_id
except ImportError:  # imported as a package module
    from anamnesis.analysis.v3_audit._common import HARD, gen_metadata_by_id

PCA_LAYERS = {"3b": [7, 14, 18, 21, 24], "8b": [8, 16, 20, 24, 28]}
T_SAMPLES = 5
REF = {"3b": {"rf_raw_floor": 81, "length_only": 65},
       "8b": {"rf_raw_floor": 81, "length_only": 69}}


# ---------------------------------------------------------------------------
# Parallel loading: per-gen raw read + positional correction (workers).
# ---------------------------------------------------------------------------
_PM: np.ndarray | None = None  # positional means, loaded once per worker process


def _load_init(pm_path: str) -> None:
    global _PM
    _PM = np.load(pm_path)["positional_means"].astype(np.float32)


def _load_one(task: tuple) -> dict | None:
    gid, raw_dir, layers, mode, topic, plen, glen = task
    try:
        # Lean load: this script reads only the sampled-hidden residual slice —
        # skip attention/gate/logits reconstruction entirely (~6x faster on v3 banks).
        data = load_raw_tensors(gid, Path(raw_dir), surfaces=("hidden",))
    except Exception as e:  # noqa: BLE001
        return {"_err": f"gen {gid}: {e}"}
    T = len(data.hidden_states)
    if T < 2:
        return None
    t_idx = [int(round(i * (T - 1) / (T_SAMPLES - 1))) for i in range(T_SAMPLES)]
    rows: list[np.ndarray] = []
    for l in layers:
        ai = l + 1
        for t in t_idx:
            if t >= T:
                return None
            h = data.hidden_states[t][ai].astype(np.float32).copy()
            h = _correct_hidden_state(h, ai, data.prompt_length + t, _PM)
            rows.append(h)
    return dict(gen=gid, mode=mode, topic=topic, plen=plen, glen=glen,
                H=np.stack(rows, 0).astype(np.float32))


def load_recs(run: str, model: str, root: Path, workers: int, rebuild: bool) -> list[dict]:
    layers = PCA_LAYERS[model]
    cache = Path(f"gate_a_lf_cache_{run}.npz")
    if cache.exists() and not rebuild:
        z = np.load(cache, allow_pickle=True)
        H = z["H"]
        recs = [dict(gen=int(g), mode=str(m), topic=int(t), plen=int(p), glen=int(gl), H=H[i])
                for i, (g, m, t, p, gl) in enumerate(
                    zip(z["gen"], z["mode"], z["topic"], z["plen"], z["glen"]))]
        print(f"  [{run}] {len(recs)} gens from cache {cache}", flush=True)
        return recs

    rd = root / "runs" / run
    raw_dir = str(rd / "raw_tensors_v3")
    md = gen_metadata_by_id(rd / "metadata.json")
    pm_path = str(root / "calibration" / model / "positional_means.npz")

    tasks = []
    for gid in sorted(md):
        g = md[gid]
        mode, cond = g.get("mode", ""), g.get("condition", "standard")
        if cond.startswith("prompt_swap") or mode.startswith("swap_"):
            continue
        tasks.append((gid, raw_dir, layers, mode, int(g["topic_idx"]),
                      int(g["prompt_length"]), int(g["num_generated_tokens"])))

    print(f"  [{run}] loading {len(tasks)} gens with {workers} workers ...", flush=True)
    recs, errs, done = [], 0, 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_load_init,
                             initargs=(pm_path,)) as ex:
        for r in ex.map(_load_one, tasks, chunksize=1):
            done += 1
            if r is None:
                continue
            if "_err" in r:
                errs += 1
                print(f"    skip {r['_err']}", flush=True)
                continue
            recs.append(r)
            if done % 25 == 0:
                print(f"    [{run}] {done}/{len(tasks)} scanned, {len(recs)} kept", flush=True)

    print(f"  [{run}] {len(recs)} gens kept, {errs} errors", flush=True)
    np.savez_compressed(
        cache,
        H=np.stack([r["H"] for r in recs], 0),
        gen=np.array([r["gen"] for r in recs]),
        mode=np.array([r["mode"] for r in recs]),
        topic=np.array([r["topic"] for r in recs]),
        plen=np.array([r["plen"] for r in recs]),
        glen=np.array([r["glen"] for r in recs]),
    )
    return recs


# ---------------------------------------------------------------------------
# Parallel nested CV: one (repeat, fold) task = retrain projection + score.
# ---------------------------------------------------------------------------
def _residualize(Ftr, Fte, Ctr, Cte):
    A = np.hstack([Ctr, np.ones((len(Ctr), 1))])
    B = np.hstack([Cte, np.ones((len(Cte), 1))])
    coef, *_ = np.linalg.lstsq(A, Ftr, rcond=None)
    return Ftr - A @ coef, Fte - B @ coef


def _score(Vtr, ytr, Vte, yte, Ctr, Cte, resid, clf):
    if resid:
        Vtr, Vte = _residualize(Vtr, Vte, Ctr, Cte)
    if clf == "knn":  # match gate_a_contrastive.py exactly (k=7, scaled, euclidean)
        est = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))
    else:
        est = RandomForestClassifier(300, random_state=0, n_jobs=1)
    est.fit(Vtr, ytr)
    return float(est.score(Vte, yte))


def _load_cache(cache_path: str):
    z = np.load(cache_path, allow_pickle=True)
    return z["H"], z["mode"].astype(str), z["topic"].astype(int), z["plen"].astype(int), z["glen"].astype(int)


def _eval_fold(task: tuple) -> tuple:
    cache_path, rep, fold_idx, base_seed, n_epochs = task
    H, mode, topic, plen, glen = _load_cache(cache_path)
    is_hard = np.array([m in HARD for m in mode])
    hi = np.flatnonzero(is_hard)
    yi = np.unique(mode[hi], return_inverse=True)[1]
    th = topic[hi]
    Ch = np.column_stack([plen[hi], glen[hi]]).astype(float)

    # projection training pool = ALL non-swap samples (flatten [Ngen,25,hid] -> [N,hid])
    Ng, S, D = H.shape
    Xall = H.reshape(Ng * S, D)
    yall = np.repeat(mode, S)
    gall = np.repeat(np.arange(Ng), S)
    tall = np.repeat(topic, S)

    splits = list(GroupKFold(5).split(np.zeros(len(hi)), yi, th))
    tr, te = splits[fold_idx]
    tr_topics = set(th[tr].tolist())
    m = np.isin(tall, list(tr_topics))
    w = train_projection(Xall[m], yall[m], groups=gall[m], seed=base_seed + rep, n_epochs=n_epochs)
    proj = ContrastiveProjectionInference(**w)

    # per-gen 800-vec for hard gens: project [25,hid] -> [25,32] -> flatten (canonical order)
    V = np.stack([proj.project(H[g]).reshape(-1).astype(np.float64) for g in hi], 0)
    V = np.nan_to_num(V)
    V = V[:, V.std(0) > 1e-10]

    out = {}
    for resid in (False, True):
        for clf in ("knn", "rf"):
            out[f"{'resid' if resid else 'raw'}_{clf}"] = _score(
                V[tr], yi[tr], V[te], yi[te], Ch[tr], Ch[te], resid, clf)
    return rep, fold_idx, out


def length_only(cache_path: str, seeds=3):
    H, mode, topic, plen, glen = _load_cache(cache_path)
    hi = np.flatnonzero(np.array([m in HARD for m in mode]))
    yi = np.unique(mode[hi], return_inverse=True)[1]
    C = np.column_stack([plen[hi], glen[hi]]).astype(float)
    th = topic[hi]
    accs = []
    for s in range(seeds):
        fold = [RandomForestClassifier(300, random_state=s, n_jobs=1).fit(C[tr], yi[tr]).score(C[te], yi[te])
                for tr, te in GroupKFold(5).split(C, yi, th)]
        accs.append(np.mean(fold))
    return float(np.mean(accs))


def run_eval(run: str, cache_path: str, repeats: int, n_epochs: int, base_seed: int, workers: int):
    combos = ["raw_knn", "raw_rf", "resid_knn", "resid_rf"]
    per_rep = {c: [[] for _ in range(repeats)] for c in combos}
    tasks = [(cache_path, rep, fi, base_seed, n_epochs) for rep in range(repeats) for fi in range(5)]
    print(f"  [{run}] {len(tasks)} (repeat x fold) tasks across {workers} workers ...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for rep, fi, out in ex.map(_eval_fold, tasks):
            for c in combos:
                per_rep[c][rep].append(out[c])
            print(f"    [{run}] rep {rep} fold {fi} done", flush=True)
    res = {}
    for c in combos:
        rep_means = [float(np.mean(f)) for f in per_rep[c]]
        res[c] = (float(np.mean(rep_means)), float(np.std(rep_means)))
    return res


def build_merged_cache(runs, model, root, load_workers, rebuild):
    """Load each run (cached per-run), concat into one merged cache. gen ids are offset
    per run so the by-generation internal val split never groups across runs."""
    all_recs = []
    for ri, run in enumerate(runs):
        if not (root / "runs" / run / "raw_tensors_v3").exists():
            print(f"  [{run}] no raw_tensors_v3 — skipping", flush=True)
            continue
        recs = load_recs(run, model, root, load_workers, rebuild)
        for r in recs:
            all_recs.append({**r, "gen": r["gen"] + ri * 100000})
    if not all_recs:
        return None, 0
    merged = Path(f"gate_a_lf_cache_{model}_merged.npz")
    np.savez_compressed(
        merged,
        H=np.stack([r["H"] for r in all_recs], 0),
        gen=np.array([r["gen"] for r in all_recs]),
        mode=np.array([r["mode"] for r in all_recs]),
        topic=np.array([r["topic"] for r in all_recs]),
        plen=np.array([r["plen"] for r in all_recs]),
        glen=np.array([r["glen"] for r in all_recs]),
    )
    return str(merged.resolve()), len(all_recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/models/anamnesis-extract"))
    ap.add_argument("--groups", nargs="+", default=["3b_fat_01,3b_fat_ext", "8b_fat_01,8b_fat_ext"],
                    help="Comma-joined run groups to merge (fat_01,fat_ext per model)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--load-workers", type=int, default=24)
    ap.add_argument("--eval-workers", type=int, default=15)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("gate_a_contrastive_leakfree_results.json"))
    args = ap.parse_args()

    results = {}
    for grp in args.groups:
        runs = [r for r in grp.split(",") if r]
        model = "3b" if runs[0].startswith("3b") else "8b"
        name = "+".join(runs)
        print(f"\n===== {name} — leak-free contrastive (nested topic-held-out CV) =====", flush=True)
        cache, n_total = build_merged_cache(runs, model, args.root, args.load_workers, args.rebuild_cache)
        if cache is None:
            print(f"  no data for {name}", flush=True)
            continue
        lo = length_only(cache)
        res = run_eval(name, cache, args.repeats, args.epochs, 42, args.eval_workers)
        z = np.load(cache, allow_pickle=True)
        n_hard = int(np.sum([m in HARD for m in z["mode"].astype(str)]))
        n_topics = len(set(z["topic"].tolist()))
        results[name] = {"n_hard": n_hard, "n_nonswap": int(len(z["mode"])),
                         "n_topics": n_topics, "length_only": lo, **res}
        ref = REF[model]
        print(f"\n  {name}  n_hard={n_hard}  n_nonswap={len(z['mode'])}  topics={n_topics}  (chance 20%)", flush=True)
        print(f"  length-only RF:        {lo:.1%}", flush=True)
        print(f"  leak-free raw   kNN/RF: {res['raw_knn'][0]:.1%}+/-{res['raw_knn'][1]:.1%}  /  {res['raw_rf'][0]:.1%}+/-{res['raw_rf'][1]:.1%}", flush=True)
        print(f"  leak-free resid kNN/RF: {res['resid_knn'][0]:.1%}+/-{res['resid_knn'][1]:.1%}  /  {res['resid_rf'][0]:.1%}+/-{res['resid_rf'][1]:.1%}", flush=True)
        print(f"  [ref @n=100] LEAKED resid ~97-99%   RF-raw floor ~{ref['rf_raw_floor']}%   length-only ~{ref['length_only']}%", flush=True)

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
