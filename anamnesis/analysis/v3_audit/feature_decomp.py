"""Track A decomposition — which families do the work, and can T3 be removed wholesale?

Standalone (family alone) AND Leave-One-Family-Out (full minus family; removal_cost = full - LOFO),
for BOTH readouts:
  - RF  (the hand-floor readout; rewards redundant breadth — T1/T3 look strong standalone)
  - LDA (the linear/naming lens; rewards clean low-rank separation — attention families look strong)

The decisive T3 test is the LOFO, not the standalone: T3 has HIGH standalone RF (~77/86) but may be
fully REDUNDANT. If RF-without-T3 ≈ RF-full (removal_cost ≈ 0), T3 is removable wholesale despite that
standalone. Reported for both readouts so the verdict is method-robust. Also an attention-only composite
(per_head + T2_other + T2.5 + attention_flow + T2_spectral — the families that name the axis) vs its
complement, to show the attention families alone ≈ full.

merged v3 SIG, n≈900, 5-way hard, GroupKFold-by-topic, length-residualized. Per-model, never pooled.
CPU, node1:  OMP_NUM_THREADS=8 PYTHONPATH=pipeline python feature_decomp.py
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:  # direct script run (sys.path[0] = this dir) — node1 self-contained convention
    from _common import load_signature_matrix, residualize
except ImportError:  # imported as a package module
    from anamnesis.analysis.v3_audit._common import load_signature_matrix, residualize

RUNS = Path(os.environ.get("ANAMNESIS_RUNS", "/models/anamnesis-extract/runs"))
GROUPS = [("3b", ["3b_fat_01", "3b_fat_ext"]), ("8b", ["8b_fat_01", "8b_fat_ext"])]
FAM_ORDER = ["T1", "T2_other", "T2_spectral", "T2.5", "T3", "attention_flow",
             "gate", "per_head", "residual_traj"]
ATTN_FAMS = {"per_head", "T2_other", "T2.5", "attention_flow", "T2_spectral"}
RF_SEEDS = 2


def fam_of(n: str) -> str:
    if n.startswith("ph_"): return "per_head"
    if n.startswith("attn_flow_"): return "attention_flow"
    if n.startswith("gate_"): return "gate"
    if n.startswith("res_traj"): return "residual_traj"
    if n.startswith(("cache_", "kv_", "epoch_")): return "T2.5"
    if n.startswith("spectral_"): return "T2_spectral"
    if n.startswith(("attn_entropy_", "head_agreement_", "delta_")): return "T2_other"
    if n.startswith("pca_"): return "T3"
    return "T1"


def load_sig(runs):
    m = load_signature_matrix(runs, RUNS)
    return m.X, m.y, m.topic, m.C, np.array([str(n) for n in m.names])


def rf_cv(F, yi, topic, C):
    if F.shape[1] == 0:
        return 0.0
    accs = []
    for s in range(RF_SEEDS):
        fold = []
        for tr, te in GroupKFold(5).split(F, yi, topic):
            Ftr, Fte = residualize(F[tr].copy(), F[te].copy(), C[tr], C[te])
            clf = RandomForestClassifier(300, random_state=s, n_jobs=1)
            clf.fit(Ftr, yi[tr]); fold.append(clf.score(Fte, yi[te]))
        accs.append(np.mean(fold))
    return float(np.mean(accs))


def lda_cv(F, yi, topic, C):
    if F.shape[1] == 0:
        return 0.0
    fold = []
    for tr, te in GroupKFold(5).split(F, yi, topic):
        Ftr, Fte = residualize(F[tr].copy(), F[te].copy(), C[tr], C[te])
        sc = StandardScaler().fit(Ftr); Ftr, Fte = sc.transform(Ftr), sc.transform(Fte)
        lda = LDA(solver="eigen", shrinkage="auto")
        lda.fit(Ftr, yi[tr]); fold.append(lda.score(Fte, yi[te]))
    return float(np.mean(fold))


def decompose(X, yi, topic, C, fams, cv_fn):
    full = cv_fn(X, yi, topic, C)
    rows = {}
    for f in FAM_ORDER:
        m = fams == f
        if m.sum() == 0:
            continue
        standalone = cv_fn(X[:, m], yi, topic, C)
        lofo = cv_fn(X[:, ~m], yi, topic, C)
        rows[f] = {"n": int(m.sum()), "standalone": round(standalone, 4),
                   "lofo": round(lofo, 4), "removal_cost": round(full - lofo, 4)}
    attn = np.array([f in ATTN_FAMS for f in fams])
    composites = {
        "ATTENTION_only": {"n": int(attn.sum()), "standalone": round(cv_fn(X[:, attn], yi, topic, C), 4)},
        "NON_attention": {"n": int((~attn).sum()), "standalone": round(cv_fn(X[:, ~attn], yi, topic, C), 4)},
        "NO_T3": {"n": int((fams != "T3").sum()), "standalone": round(cv_fn(X[:, fams != "T3"], yi, topic, C), 4)},
    }
    return {"full": round(full, 4), "families": rows, "composites": composites}


def main():
    out = {}
    for model, runs in GROUPS:
        X, y, topic, C, names = load_sig(runs)
        if len(y) == 0:
            print(f"{model}: no data"); continue
        labels = sorted(set(y.tolist()))
        yi = np.array([labels.index(v) for v in y])
        fams = np.array([fam_of(n) for n in names])
        print(f"\n===== {model}  n={len(yi)}  feats={X.shape[1]}  (chance 20%) =====", flush=True)
        out[model] = {}
        for rname, cv_fn in [("RF", rf_cv), ("LDA", lda_cv)]:
            d = decompose(X, yi, topic, C, fams, cv_fn)
            out[model][rname] = d
            print(f"  --- {rname} (full={d['full']:.1%}) ---", flush=True)
            print(f"  {'family':16s} {'n':>5s} {'standalone':>11s} {'LOFO':>7s} {'removal':>8s}", flush=True)
            for f, r in sorted(d["families"].items(), key=lambda kv: -kv[1]["removal_cost"]):
                print(f"  {f:16s} {r['n']:5d} {r['standalone']:>10.1%} {r['lofo']:>7.1%} "
                      f"{r['removal_cost']:>+8.1%}", flush=True)
            c = d["composites"]
            print(f"  composites: ATTENTION_only({c['ATTENTION_only']['n']})="
                  f"{c['ATTENTION_only']['standalone']:.1%}  NON_attention({c['NON_attention']['n']})="
                  f"{c['NON_attention']['standalone']:.1%}  NO_T3({c['NO_T3']['n']})="
                  f"{c['NO_T3']['standalone']:.1%}  (full={d['full']:.1%})", flush=True)
    Path("feature_decomp_results.json").write_text(json.dumps(out, indent=2))
    print("\nWrote feature_decomp_results.json", flush=True)


if __name__ == "__main__":
    main()
