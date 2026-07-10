"""Geometry of the space — SIGNATURES vs RAW hidden, on the merged corpus.

Answers, side by side for the engineered-signature space and the raw-hidden space:
  - "what does the space look like / are our sigs efficient?"  -> intrinsic dim (sig vs raw)
  - "best way to extract?"                                     -> δ-hyperbolicity (distances
        generalize? low δ_rel = curved/tree-like -> kNN fails, the collapse we saw)
  - "floor / ceiling, mode-wise?"                              -> CCGP cross-topic (linear SVC
        = linear mode-ceiling that generalizes; kNN = distance-based) on sig vs raw

Raw-space = the merged contrastive cache (corrected hidden states at pca_layers × temporal),
so run gate_a_contrastive_leakfree.py first (it builds gate_a_lf_cache_{model}_merged.npz).
Sig-space = merged signatures_v3 (hard modes). Both 5-way hard, GroupKFold by topic.

Runs on node (CPU). OMP_NUM_THREADS=8 PYTHONPATH=pipeline python geometry_merged.py
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
warnings.filterwarnings("ignore")

from itertools import combinations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:  # direct script run (sys.path[0] = this dir) — node1 self-contained convention
    from _common import HARD, load_signature_matrix
except ImportError:  # imported as a package module
    from anamnesis.analysis.v3_audit._common import HARD, load_signature_matrix

ROOT = Path("/models/anamnesis-extract")
GROUPS = [("3b", ["3b_fat_01", "3b_fat_ext"]), ("8b", ["8b_fat_01", "8b_fat_ext"])]


def load_sig_space(runs):
    """Merged hard-mode signatures -> (X[N,Dfeat], y, topic)."""
    m = load_signature_matrix(runs, ROOT / "runs")
    return m.X, m.y, m.topic


def load_raw_space(model):
    """Merged raw hidden from the contrastive cache -> (X[N, 25*hidden], y, topic), hard only."""
    cache = Path(f"gate_a_lf_cache_{model}_merged.npz")
    if not cache.exists():
        return None, None, None
    z = np.load(cache, allow_pickle=True)
    H, mode, topic = z["H"], z["mode"].astype(str), z["topic"].astype(int)
    hard = np.array([m in HARD for m in mode])
    X = H[hard].reshape(int(hard.sum()), -1)  # flatten 25 sampled hidden states
    return np.nan_to_num(X.astype(np.float64)), mode[hard], topic[hard]


def intrinsic_dim(X):
    Xs = StandardScaler().fit_transform(X)
    Xs = Xs[:, Xs.std(0) > 1e-10]
    out = {"ambient": int(Xs.shape[1]), "n": int(Xs.shape[0])}
    try:
        from skdim.id import TwoNN
        out["twonn"] = float(TwoNN().fit(Xs).dimension_)
    except Exception as e:  # noqa: BLE001
        out["twonn"] = f"err:{e}"
    try:
        from dadapy import Data
        d = Data(Xs); d.compute_id_2NN()
        out["dada2nn"] = float(d.intrinsic_dim)
    except Exception as e:  # noqa: BLE001
        out["dada2nn"] = f"err:{e}"
    return out


def delta_hyperbolicity(X, n_quads=200_000, seed=42):
    """Gromov 4-point δ, scale-free (δ_rel = δ/diameter). ~0 = tree/hyperbolic
    (distances don't generalize globally → kNN brittle); ~0.5 = flat/Euclidean."""
    Xs = StandardScaler().fit_transform(X).astype(np.float64)
    D = squareform(pdist(Xs))
    diam = float(D.max())
    n = D.shape[0]
    rng = np.random.default_rng(seed)
    q = rng.integers(0, n, size=(n_quads, 4))
    qs = np.sort(q, axis=1)  # all-4-distinct (matches core's choice(replace=False) semantics)
    ok = (qs[:, 0] != qs[:, 1]) & (qs[:, 1] != qs[:, 2]) & (qs[:, 2] != qs[:, 3])
    q = q[ok]
    i, j, k, l = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    s = np.sort(np.stack([D[i, j] + D[k, l], D[i, k] + D[j, l], D[i, l] + D[j, k]], 1), 1)
    deltas = (s[:, 2] - s[:, 1]) / 2.0
    return {"delta_rel_max": float(deltas.max()) / diam if diam else 0.0,
            "delta_rel_mean": float(deltas.mean()) / diam if diam else 0.0,
            "diameter": diam}


# ── CCGP: faithful copy of unified_runner.geometry._ccgp_variant (core calculation) ──
def _generate_topic_folds(topics, n_folds, rng):
    topics_arr = np.array(topics); n = len(topics_arr); fold_size = n // n_folds
    perm = rng.permutation(n); folds = []
    for i in range(n_folds):
        test_idx = perm[i * fold_size:(i + 1) * fold_size]
        train_idx = np.setdiff1d(perm, test_idx)
        folds.append((topics_arr[train_idx].tolist(), topics_arr[test_idx].tolist()))
    return folds


def _make_clf(clf_name, n_train, n_modes, seed, binary=False):
    if clf_name == "knn3":
        k = min(3, max(1, n_train // 2)) if binary else min(3, n_train // n_modes)
        return KNeighborsClassifier(n_neighbors=max(1, k))
    if clf_name == "knn5":
        k = min(5, max(1, n_train // 2)) if binary else min(5, n_train // n_modes)
        return KNeighborsClassifier(n_neighbors=max(1, k))
    if clf_name == "linearsvc":
        return LinearSVC(max_iter=5000, dual="auto")
    if clf_name == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=seed)
    raise ValueError(clf_name)


def ccgp_core(X, modes, topics, clf_name, n_folds=5, seed=42):
    """ccgp_score (fraction of binary dichotomies decodable cross-topic >0.65) + multiclass mean.
    Line-for-line the core _ccgp_variant calculation, returned as a dict."""
    rng = np.random.default_rng(seed)
    unique_modes = sorted(set(modes.tolist()))
    unique_topics = sorted(set(topics.tolist()))
    topic_folds = _generate_topic_folds(unique_topics, n_folds, rng)

    accs = []
    for train_topics, test_topics in topic_folds:
        trm, tem = np.isin(topics, train_topics), np.isin(topics, test_topics)
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(X[trm]); Xte_s = sc.transform(X[tem])
        clf = _make_clf(clf_name, int(trm.sum()), len(unique_modes), seed)
        clf.fit(Xtr_s, modes[trm])
        accs.append(float(np.mean(clf.predict(Xte_s) == modes[tem])))
    multiclass_mean = float(np.mean(accs)) if accs else 0.0

    specs = []
    n = len(unique_modes)
    for k in range(1, n // 2 + 1):
        for ga in combinations(unique_modes, k):
            gb = tuple(m for m in unique_modes if m not in ga)
            if k == n // 2 and ga > gb:
                continue
            specs.append((ga, gb))
    n_dec = 0
    for ga, gb in specs:
        fa = []
        for train_topics, test_topics in topic_folds:
            trm, tem = np.isin(topics, train_topics), np.isin(topics, test_topics)
            mm = np.isin(modes, ga + gb)
            tf, tg = trm & mm, tem & mm
            if tf.sum() < 4 or tg.sum() < 2:
                continue
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(X[tf]); Xte_s = sc.transform(X[tg])
            ytr = np.isin(modes[tf], ga).astype(int); yte = np.isin(modes[tg], ga).astype(int)
            clf = _make_clf(clf_name, int(tf.sum()), 2, seed, binary=True)
            clf.fit(Xtr_s, ytr)
            fa.append(float(np.mean(clf.predict(Xte_s) == yte)))
        if fa and np.mean(fa) > 0.65:
            n_dec += 1
    return {"ccgp_score": n_dec / max(len(specs), 1), "multiclass_mean": multiclass_mean,
            "n_decodable": n_dec, "n_dichotomies": len(specs)}


def main():
    results = {}
    for model, runs in GROUPS:
        Xs, ys, ts = load_sig_space(runs)
        Xr, yr, tr = load_raw_space(model)
        print(f"\n===== {model}  ({'+'.join(runs)}) =====", flush=True)
        if len(ys) == 0:
            print("  no sig data", flush=True); continue
        spaces = [("SIG", Xs, ys, ts)]
        if Xr is not None:
            spaces.append(("RAW", Xr, yr, tr))
        results[model] = {}
        for name, X, y, t in spaces:
            idd = intrinsic_dim(X)
            dh = delta_hyperbolicity(X)
            csvc = ccgp_core(X, y, t, "linearsvc")
            cknn = ccgp_core(X, y, t, "knn3")
            results[model][name] = {"id": idd, "delta": dh,
                                    "ccgp_linearsvc": csvc, "ccgp_knn3": cknn}
            tw = idd["twonn"] if isinstance(idd["twonn"], float) else -1
            dd = idd["dada2nn"] if isinstance(idd["dada2nn"], float) else -1
            print(f"  {name:3s} n={idd['n']} ambient={idd['ambient']:6d} | "
                  f"ID twonn={tw:5.1f} dada={dd:5.1f} | "
                  f"δ_rel(max/mean)={dh['delta_rel_max']:.3f}/{dh['delta_rel_mean']:.3f} | "
                  f"CCGP svc(score/mc)={csvc['ccgp_score']:.2f}/{csvc['multiclass_mean']:.0%} "
                  f"knn(score/mc)={cknn['ccgp_score']:.2f}/{cknn['multiclass_mean']:.0%}", flush=True)
    Path("geometry_merged_results.json").write_text(json.dumps(results, indent=2))
    print("\nWrote geometry_merged_results.json", flush=True)


if __name__ == "__main__":
    main()
