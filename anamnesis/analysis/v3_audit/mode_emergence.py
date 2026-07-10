"""Do modes EMERGE unsupervised once content is removed — and which modes conflate?

Companion to unsupervised_structure.py. That showed naive unsupervised clustering recovers TOPIC (content
dominates variance). imago's deeper ask / Luxia's question: can the data surface the MODE structure
unsupervised, so we see which modes conflate — without telling it the modes?

The move: cancel content using the paired design (every topic generated under all modes), NOT mode labels.
  - NAIVE      : PCA -> KMeans on the raw signature              (baseline: should recover TOPIC)
  - TOPIC-CTR  : subtract per-topic mean, then PCA -> KMeans     (content removed via the prompt only)
  - cPCA       : contrastive PCA — top eigvecs of (Cov[within-topic residuals] - α·Cov[topic means]);
                 explicitly keeps mode-residual variance, drops content variance (ladder #2, "motivated")
If mode NMI jumps from NAIVE -> TOPIC-CTR/cPCA, modes are organic structure (not arbitrary impositions);
the cluster×mode confusion + the pairwise mode-separability map show WHICH modes blur into which.

Conflation map: pairwise 2-mode separability (LDA, length-resid, GroupKFold-by-topic). Low = conflated.
(Uses mode labels — but as MEASUREMENT of how known modes relate, not to discover them.)

Self-checks: NMI(topic,topic)=1; NMI(clusters, shuffled)≈0; naive recovers topic (matches unsup test).
CPU, node1:  OMP_NUM_THREADS=8 PYTHONPATH=pipeline python -m anamnesis.analysis.v3_audit.mode_emergence
"""
from __future__ import annotations

import json
import os
import warnings
from itertools import combinations

os.environ.setdefault("OMP_NUM_THREADS", "8")
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:  # direct script run (sys.path[0] = this dir) — node1 self-contained convention
    from _common import load_signature_matrix
except ImportError:  # imported as a package module
    from anamnesis.analysis.v3_audit._common import load_signature_matrix

RUNS = Path(os.environ.get("ANAMNESIS_RUNS", "/models/anamnesis-extract/runs"))
ROOT = os.environ.get("ANAMNESIS_PL", ".")
GROUPS = [("3b", ["3b_fat_01", "3b_fat_ext"]), ("8b", ["8b_fat_01", "8b_fat_ext"])]
SEED = 0


def load_sig(runs):
    m = load_signature_matrix(runs, RUNS)
    return m.X, m.y, m.topic, m.C


def topic_center(X, topic):
    out = X.copy()
    for t in np.unique(topic):
        m = topic == t
        out[m] -= X[m].mean(0)
    return out


def km_nmi(Z, yi, topic, k):
    km = KMeans(k, n_init=10, random_state=SEED).fit_predict(Z)
    return km, round(NMI(km, yi), 3), round(ARI(km, yi), 3), round(NMI(km, topic), 3)


def cpca(Xs, topic, k=10, alpha=1.5):
    """Contrastive PCA: top eigvecs of Cov[within-topic residuals] - α·Cov[topic means].
    Keeps mode-residual variance, subtracts content variance. Returns the projection of the residuals."""
    resid = topic_center(Xs, topic)
    tmeans = np.stack([Xs[topic == t].mean(0) for t in np.unique(topic)])   # pure-content vectors
    Cfg = np.cov(resid, rowvar=False)
    Cbg = np.cov(tmeans, rowvar=False)
    w, V = np.linalg.eigh(Cfg - alpha * Cbg)
    dirs = V[:, np.argsort(-w)[:k]]
    return resid @ dirs


def conflation(X, yi, topic, C, labels):
    """Pairwise 2-mode separability (LDA, length-resid, GroupKFold-by-topic). Low = conflated."""
    pair = {}
    for a, b in combinations(range(len(labels)), 2):
        m = (yi == a) | (yi == b)
        Xa, ya, ta, Ca = X[m], (yi[m] == b).astype(int), topic[m], C[m]
        accs = []
        for tr, te in GroupKFold(5).split(Xa, ya, ta):
            A = np.hstack([Ca[tr], np.ones((len(tr), 1))]); B = np.hstack([Ca[te], np.ones((len(te), 1))])
            cf, *_ = np.linalg.lstsq(A, Xa[tr], rcond=None)
            Ftr, Fte = Xa[tr] - A @ cf, Xa[te] - B @ cf
            sc = StandardScaler().fit(Ftr); Ftr, Fte = sc.transform(Ftr), sc.transform(Fte)
            lda = LDA(solver="eigen", shrinkage="auto"); lda.fit(Ftr, ya[tr]); accs.append(lda.score(Fte, ya[te]))
        pair[f"{labels[a]}|{labels[b]}"] = round(float(np.mean(accs)), 3)
    return dict(sorted(pair.items(), key=lambda kv: kv[1]))   # ascending = most-conflated first


def main():
    out = {}
    for model, runs in GROUPS:
        X, mode, topic, C = load_sig(runs)
        labels = sorted(set(mode.tolist()))
        yi = np.array([labels.index(m) for m in mode])
        nt = len(set(topic.tolist()))
        Xs = StandardScaler().fit_transform(X)
        print(f"\n===== {model}  n={len(yi)}  modes={len(labels)} topics={nt} =====", flush=True)
        rng = np.random.default_rng(SEED); shuf = rng.permutation(yi)
        res = {}
        # --- emergence: does mode cluster unsupervised as content is removed? ---
        reps = {
            "NAIVE": PCA(30, random_state=SEED).fit_transform(Xs),
            "TOPIC_CTR": PCA(30, random_state=SEED).fit_transform(topic_center(Xs, topic)),
            "cPCA": cpca(Xs, topic, k=10, alpha=1.5),
        }
        print(f"  {'method':10s} NMI(mode) ARI(mode) NMI(topic)  control  [chance NMI≈0]", flush=True)
        for name, Z in reps.items():
            km, nm_mode, ari_mode, nm_topic = km_nmi(Z, yi, topic, len(labels))
            ctrl = round(NMI(km, shuf), 3)
            res[name] = {"NMI_mode": nm_mode, "ARI_mode": ari_mode, "NMI_topic": nm_topic, "control": ctrl}
            print(f"  {name:10s}  {nm_mode:6.2f}    {ari_mode:6.2f}    {nm_topic:6.2f}    {ctrl:5.2f}", flush=True)
            if name == "TOPIC_CTR":
                km5 = km
        # confusion of the TOPIC_CTR KMeans clusters vs true modes
        conf = np.zeros((len(labels), len(labels)), int)
        for c in range(len(labels)):
            for mi in range(len(labels)):
                conf[mi, c] = int(np.sum((km5 == c) & (yi == mi)))
        print("  TOPIC_CTR cluster×mode confusion (rows=mode, cols=cluster):", flush=True)
        for mi, lab in enumerate(labels):
            print(f"    {lab:12s} " + " ".join(f"{conf[mi,c]:3d}" for c in range(len(labels))), flush=True)
        # --- conflation map (supervised measurement) ---
        conf_pairs = conflation(X, yi, topic, C, labels)
        print("  pairwise mode separability (LDA-resid, topic-CV) — ascending = MOST conflated first:", flush=True)
        for pr, acc in conf_pairs.items():
            print(f"    {pr:26s} {acc:.0%}", flush=True)
        res["conflation_pairs"] = conf_pairs
        res["_check_NMI_topic_self"] = round(NMI(topic, topic), 3)
        out[model] = res
    Path("mode_emergence_results.json").write_text(json.dumps(out, indent=2))
    print("\nWrote mode_emergence_results.json", flush=True)


if __name__ == "__main__":
    main()
