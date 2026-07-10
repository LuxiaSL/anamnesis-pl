"""Linear probe + LDA-rank on the merged corpus — closes the picture before the encoder.

Is the mode signal linearly decodable (→ a nameable, portable direction) and low-rank (→ how many
directions carry the how-axis)? Mirrors the Gate-A protocol exactly (5-way hard, GroupKFold-by-topic,
per-feature length residualization on train folds) so **logistic-resid is directly comparable to the
RF-resid v3-ALL numbers** — the gap RF − logistic = the nonlinear premium.

  - SIG  = merged signatures_v3 (the nameable hand-feature space)
  - RAW  = merged contrastive cache (sampled corrected hidden states, flattened)
  - LDA-rank (SIG only; raw is too high-dim for LDA): cumulative cross-topic accuracy using the
    top-k LDA discriminant directions (k=1..4, nearest class-centroid) → is mode ~2-4 directions?

Runs on node (CPU). ANAMNESIS_RUNS defaults to /models/anamnesis-extract/runs.
    OMP_NUM_THREADS=8 PYTHONPATH=pipeline python linear_probe_merged.py
"""
from __future__ import annotations

import json
import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "8")
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HARD = {"linear", "socratic", "contrastive", "dialectical", "analogical"}
RUNS = Path(os.environ.get("ANAMNESIS_RUNS", "/models/anamnesis-extract/runs"))
GROUPS = [("3b", ["3b_fat_01", "3b_fat_ext"]), ("8b", ["8b_fat_01", "8b_fat_ext"])]


def load_sig(runs):
    names = None
    rows, y, topic, C = [], [], [], []
    for run in runs:
        rd = RUNS / run; sd = rd / "signatures_v3"
        if not (rd / "metadata.json").exists() or not sd.exists():
            continue
        meta = json.load(open(rd / "metadata.json"))
        gens = meta["generations"] if isinstance(meta, dict) and "generations" in meta else meta
        md = {int(g["generation_id"]): g for g in gens}
        for p in sorted(sd.glob("gen_*.npz"), key=lambda x: int(x.stem.split("_")[1])):
            g = int(p.stem.split("_")[1])
            if g not in md or md[g]["mode"] not in HARD:
                continue
            z = np.load(p, allow_pickle=True)
            nm = [str(x) for x in z["feature_names"]]
            if names is None:
                names = nm
            d = {n: float(v) for n, v in zip(nm, z["features"])}
            rows.append([d.get(n, 0.0) for n in names]); y.append(md[g]["mode"])
            topic.append(md[g]["topic_idx"])
            C.append([md[g]["prompt_length"], md[g]["num_generated_tokens"]])
    return np.nan_to_num(np.array(rows, float)), np.array(y), np.array(topic), np.array(C, float)


def load_raw(model):
    c = Path(f"gate_a_lf_cache_{model}_merged.npz")
    if not c.exists():
        return None, None, None, None
    z = np.load(c, allow_pickle=True)
    H, mode, topic = z["H"], z["mode"].astype(str), z["topic"].astype(int)
    plen, glen = z["plen"].astype(float), z["glen"].astype(float)
    hard = np.array([m in HARD for m in mode])
    X = np.nan_to_num(H[hard].reshape(int(hard.sum()), -1).astype(np.float64))
    return X, mode[hard], topic[hard], np.column_stack([plen[hard], glen[hard]])


def residualize(Ftr, Fte, Ctr, Cte):
    A = np.hstack([Ctr, np.ones((len(Ctr), 1))]); B = np.hstack([Cte, np.ones((len(Cte), 1))])
    coef, *_ = np.linalg.lstsq(A, Ftr, rcond=None)
    return Ftr - A @ coef, Fte - B @ coef


def cv(X, y, topic, C, resid, make, seeds=3):
    yi = np.unique(y, return_inverse=True)[1]
    accs = []
    for s in range(seeds):
        fold = []
        for tr, te in GroupKFold(5).split(X, yi, topic):
            Ftr, Fte = X[tr].copy(), X[te].copy()
            if resid:
                Ftr, Fte = residualize(Ftr, Fte, C[tr], C[te])
            est = make()
            est.fit(Ftr, yi[tr])
            fold.append(float(est.score(Fte, yi[te])))
        accs.append(float(np.mean(fold)))
    return float(np.mean(accs)), float(np.std(accs))


def lda_rank(X, y, topic, C, resid=True):
    """Cumulative cross-topic accuracy using top-k LDA directions (nearest class-centroid)."""
    yi = np.unique(y, return_inverse=True)[1]
    nclass = len(set(yi.tolist())); kmax = nclass - 1
    cum = {k: [] for k in range(1, kmax + 1)}
    for tr, te in GroupKFold(5).split(X, yi, topic):
        Ftr, Fte = X[tr].copy(), X[te].copy()
        if resid:
            Ftr, Fte = residualize(Ftr, Fte, C[tr], C[te])
        sc = StandardScaler().fit(Ftr)
        Ftr, Fte = sc.transform(Ftr), sc.transform(Fte)
        lda = LDA(solver="eigen", shrinkage="auto", n_components=kmax)
        Ztr = lda.fit_transform(Ftr, yi[tr]); Zte = lda.transform(Fte)
        for k in range(1, kmax + 1):
            cents = np.stack([Ztr[yi[tr] == c, :k].mean(0) for c in range(nclass)])
            pred = np.argmin(((Zte[:, :k][:, None, :] - cents[None, :, :]) ** 2).sum(-1), axis=1)
            cum[k].append(float(np.mean(pred == yi[te])))
    return {k: float(np.mean(v)) for k, v in cum.items()}


def main():
    res = {}
    for model, runs in GROUPS:
        Xs, ys, ts, Cs = load_sig(runs)
        Xr, yr, trr, Cr = load_raw(model)
        print(f"\n===== {model} =====  (chance 20%)", flush=True)
        if len(ys) == 0:
            print("  no sig data", flush=True); continue
        spaces = [("SIG", Xs, ys, ts, Cs)]
        if Xr is not None:
            spaces.append(("RAW", Xr, yr, trr, Cr))
        res[model] = {}
        for name, X, y, t, C in spaces:
            def logit():
                return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
            lr_raw = cv(X, y, t, C, False, logit)
            lr_res = cv(X, y, t, C, True, logit)
            ranks = lda_rank(X, y, t, C, True) if name == "SIG" else {}
            res[model][name] = {"logit_raw": lr_raw, "logit_resid": lr_res, "lda_rank_resid": ranks}
            rk = "  ".join(f"k{k}={v:.0%}" for k, v in ranks.items()) if ranks else "(skip LDA on raw)"
            print(f"  {name}: logistic raw={lr_raw[0]:.1%}  resid={lr_res[0]:.1%}±{lr_res[1]:.1%}  "
                  f"| LDA-rank cumulative (resid): {rk}", flush=True)
    Path("linear_probe_merged_results.json").write_text(json.dumps(res, indent=2))
    print("\nWrote linear_probe_merged_results.json", flush=True)


if __name__ == "__main__":
    main()
