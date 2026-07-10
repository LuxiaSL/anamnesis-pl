"""Gate-A in CONTRASTIVE space — the project's validated nonlinear readout.

The 800-dim contrastive_projection features are a learned nonlinear projection of the
residual stream; v3 didn't touch the residual stream (faithfulness r=1.0), so the banked
signatures_v2_contrastive ARE the v3 contrastive features. Question: does the mode signal
survive length control in this rich space (kNN + RF, raw vs length-residualized)?

5-way hard modes, GroupKFold by topic. Run from anamnesis_exps:
    OMP_NUM_THREADS=1 python3 research/notes/gate_a_contrastive.py
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
HARD = {"linear", "socratic", "contrastive", "dialectical", "analogical"}


def load(run: str):
    rd = Path("outputs/runs") / run
    meta = json.load(open(rd / "metadata.json"))
    gens = meta["generations"] if isinstance(meta, dict) and "generations" in meta else meta
    md = {int(g["generation_id"]): g for g in gens}
    sd = rd / "signatures_v2_contrastive"
    ids = sorted(int(p.stem.split("_")[1]) for p in sd.glob("gen_*.npz"))
    ids = [g for g in ids if g in md and md[g]["mode"] in HARD]
    names = None
    rows, y, topic, C = [], [], [], []
    for g in ids:
        z = np.load(sd / f"gen_{g:03d}.npz", allow_pickle=True)
        if names is None:
            names = [str(x) for x in z["feature_names"]]
        rows.append([float(v) for v in z["features"]])
        y.append(md[g]["mode"]); topic.append(md[g]["topic_idx"])
        C.append([md[g]["prompt_length"], md[g]["num_generated_tokens"]])
    return np.nan_to_num(np.array(rows, float)), np.array(y), np.array(topic), np.array(C, float), np.array(names)


def cv(F, y, topic, C=None, residualize=False, clf="knn", seeds=3):
    F = F[:, F.std(0) > 1e-10]
    yi = np.unique(y, return_inverse=True)[1]
    accs = []
    for s in range(seeds):
        fold = []
        for tr, te in GroupKFold(5).split(F, yi, topic):
            Ftr, Fte = F[tr].copy(), F[te].copy()
            if residualize:
                A = np.hstack([C[tr], np.ones((len(tr), 1))])
                B = np.hstack([C[te], np.ones((len(te), 1))])
                coef, *_ = np.linalg.lstsq(A, Ftr, rcond=None)
                Ftr, Fte = Ftr - A @ coef, Fte - B @ coef
            if clf == "knn":
                est = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))
            else:
                est = RandomForestClassifier(300, random_state=s, n_jobs=1)
            est.fit(Ftr, yi[tr])
            fold.append(est.score(Fte, yi[te]))
        accs.append(np.mean(fold))
    return float(np.mean(accs))


def main():
    import re
    for run in ["3b_fat_01", "8b_fat_01"]:
        X, y, topic, C, names = load(run)
        print(f"\n===== {run} (contrastive, 800-dim) =====  n={len(y)}  (chance 20%)", flush=True)
        print(f"length-only:           kNN {cv(C, y, topic, clf='knn'):.1%}  RF {cv(C, y, topic, clf='rf'):.1%}", flush=True)
        print(f"contrastive raw:       kNN {cv(X, y, topic, clf='knn'):.1%}  RF {cv(X, y, topic, clf='rf'):.1%}", flush=True)
        print(f"contrastive residual:  kNN {cv(X, y, topic, C, True, 'knn'):.1%}  RF {cv(X, y, topic, C, True, 'rf'):.1%}", flush=True)
        # Decompose: where does the (residualized) signal concentrate? — like the tier battery.
        layers = sorted({re.search(r"_L(\d+)_", n).group(1) for n in names}, key=int)
        print("  by layer    (resid kNN):  " + "  ".join(
            f"L{L} {cv(X[:, np.array([f'_L{L}_' in n for n in names])], y, topic, C, True, 'knn'):.0%}" for L in layers), flush=True)
        temps = sorted({re.search(r"_t(\d+)_", n).group(1) for n in names}, key=int)
        print("  by temporal (resid kNN):  " + "  ".join(
            f"t{t} {cv(X[:, np.array([f'_t{t}_' in n for n in names])], y, topic, C, True, 'knn'):.0%}" for t in temps), flush=True)


if __name__ == "__main__":
    main()
