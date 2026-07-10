"""imago's test — if you "let the data do its work" (partition by variance, no labels), do you get
MODE (processing) or TOPIC (content)?

imago's words: "partitioning this pool not by any preset categories but instead ... by how much variance
of something it explains." That's unsupervised (PCA / KMeans). This measures whether the unsupervised
structure of the signature/raw aligns with the mode axis (processing) or the topic axis (content).

Prediction (from Phase-0 "PCA dies under format control" + Mantel r=0.40 topic>mode): TOPIC >> MODE
unsupervised; MODE only appears with supervision. If so, "let the data do its work" surfaces *what it's
about*, not *how it was processed* — so the taste/processing axis needs a supervisory signal (ratings) or
a nonlinear learner, which is the answer to imago.

Measures (per model x surface x prep), n≈900 hard, 5 modes / 60 topics:
  - eta^2 attribution on the top PCs: how much of each top-variance direction is explained by topic vs mode.
  - KMeans alignment: NMI/ARI of unsupervised clusters with mode (k=5) and topic (k=60).
  - SUPERVISED anchor: LDA-resid mode accuracy (the ceiling — mode IS recoverable with supervision).
  - SELF-CHECKS: NMI(topic,topic)=1 (metric sanity); NMI(clusters, shuffled) ≈ 0 (random control).

CPU, node1:  OMP_NUM_THREADS=8 PYTHONPATH=pipeline python -m anamnesis.analysis.v3_audit.unsupervised_structure
"""
from __future__ import annotations

import json
import os
import warnings

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
    from _common import HARD, load_signature_matrix, residualize_all
except ImportError:  # imported as a package module
    from anamnesis.analysis.v3_audit._common import HARD, load_signature_matrix, residualize_all

RUNS = Path(os.environ.get("ANAMNESIS_RUNS", "/models/anamnesis-extract/runs"))
ROOT = os.environ.get("ANAMNESIS_PL", ".")
GROUPS = [("3b", ["3b_fat_01", "3b_fat_ext"]), ("8b", ["8b_fat_01", "8b_fat_ext"])]
SEED = 0


def load_sig(runs):
    m = load_signature_matrix(runs, RUNS)
    return m.X, m.y, m.topic, m.C


def load_raw(model):
    z = np.load(os.path.join(ROOT, f"gate_a_lf_cache_{model}_merged.npz"), allow_pickle=True)
    H, mode, topic = z["H"], z["mode"].astype(str), z["topic"].astype(int)
    plen, glen = z["plen"].astype(float), z["glen"].astype(float)
    hard = np.array([m in HARD for m in mode])
    X = np.nan_to_num(H[hard].reshape(int(hard.sum()), -1).astype(np.float64))
    return X, mode[hard], topic[hard], np.column_stack([plen[hard], glen[hard]])


def eta2(v, groups):
    """Fraction of a 1-D variable's variance explained by a grouping (between/total)."""
    grand = v.mean(); ssb = 0.0
    for g in np.unique(groups):
        m = groups == g
        ssb += m.sum() * (v[m].mean() - grand) ** 2
    sst = ((v - grand) ** 2).sum() + 1e-12
    return ssb / sst


def lda_mode_acc(F, yi, topic, C):
    accs = []
    for tr, te in GroupKFold(5).split(F, yi, topic):
        A = np.hstack([C[tr], np.ones((len(tr), 1))]); B = np.hstack([C[te], np.ones((len(te), 1))])
        cf, *_ = np.linalg.lstsq(A, F[tr], rcond=None)
        Ftr, Fte = F[tr] - A @ cf, F[te] - B @ cf
        sc = StandardScaler().fit(Ftr); Ftr, Fte = sc.transform(Ftr), sc.transform(Fte)
        lda = LDA(solver="eigen", shrinkage="auto"); lda.fit(Ftr, yi[tr]); accs.append(lda.score(Fte, yi[te]))
    return float(np.mean(accs))


def analyze(X, mode, topic, C, prep):
    yi = np.unique(mode, return_inverse=True)[1]
    nmode, ntopic = len(set(yi.tolist())), len(set(topic.tolist()))
    F = residualize_all(X, C) if prep == "resid" else X.copy()
    Z = PCA(30, random_state=SEED).fit_transform(StandardScaler().fit_transform(F))
    # eta^2 attribution on top PCs
    e_topic = [eta2(Z[:, j], topic) for j in range(10)]
    e_mode = [eta2(Z[:, j], yi) for j in range(10)]
    # KMeans alignment (each label gets its fair k)
    km5 = KMeans(nmode, n_init=10, random_state=SEED).fit_predict(Z)
    km_t = KMeans(ntopic, n_init=10, random_state=SEED).fit_predict(Z)
    rng = np.random.default_rng(SEED); shuf = rng.permutation(yi)
    return {
        "eta2_topic_pc1": round(e_topic[0], 3), "eta2_mode_pc1": round(e_mode[0], 3),
        "eta2_topic_top10": round(float(np.mean(e_topic)), 3), "eta2_mode_top10": round(float(np.mean(e_mode)), 3),
        "NMI_km5_mode": round(NMI(km5, yi), 3), "ARI_km5_mode": round(ARI(km5, yi), 3),
        "NMI_kmT_topic": round(NMI(km_t, topic), 3), "NMI_kmT_mode": round(NMI(km_t, yi), 3),
        "NMI_km5_topic": round(NMI(km5, topic), 3),
        "_control_NMI_km5_shuffledmode": round(NMI(km5, shuf), 3),
        "_check_NMI_topic_self": round(NMI(topic, topic), 3),
    }


def main():
    out = {}
    for model, runs in GROUPS:
        out[model] = {}
        sig = load_sig(runs)
        raw = load_raw(model)
        print(f"\n===== {model}  (5 modes / {len(set(sig[2].tolist()))} topics, n={len(sig[1])}) =====", flush=True)
        for sname, (X, mode, topic, C) in [("SIG", sig), ("RAW", raw)]:
            yi = np.unique(mode, return_inverse=True)[1]
            # supervised ceiling: LDA (covariance-aware) on the sig; on the ~1e5-dim raw the eigen-LDA's
            # p x p covariance is intractable, so cite the known logistic-resid raw (3B 0.89 / 8B 0.92).
            if X.shape[1] <= 3000:
                sup = lda_mode_acc(X, yi, topic, C)
                print(f"  --- {sname}  (supervised LDA-resid mode acc = {sup:.1%}; chance {1/len(set(yi)):.0%}) ---", flush=True)
                out[model][sname] = {"supervised_lda_mode_acc": round(sup, 4)}
            else:
                print(f"  --- {sname}  (supervised: known logistic-resid raw ≈ 0.89/0.92; chance {1/len(set(yi)):.0%}) ---", flush=True)
                out[model][sname] = {"supervised_lda_mode_acc": None, "supervised_note": "logistic-resid raw ~0.89(3b)/0.92(8b)"}
            for prep in ["raw", "resid"]:
                r = analyze(X, mode, topic, C, prep)
                out[model][sname][prep] = r
                print(f"    [{prep:5s}] eta2 top10: topic={r['eta2_topic_top10']:.2f} mode={r['eta2_mode_top10']:.2f}"
                      f" | KMeans NMI: topic(k={len(set(topic))})={r['NMI_kmT_topic']:.2f}"
                      f" mode(k=5)={r['NMI_km5_mode']:.2f} | control(shuf)={r['_control_NMI_km5_shuffledmode']:.2f}"
                      f" self(topic)={r['_check_NMI_topic_self']:.2f}", flush=True)
    Path("unsupervised_structure_results.json").write_text(json.dumps(out, indent=2))
    print("\nWrote unsupervised_structure_results.json", flush=True)


if __name__ == "__main__":
    main()
