"""Reframe the tiers as SOURCE x METHOD x DEPTH, and disambiguate the two redundancy questions.

Replaces the T1/T2/T2.5/T3 "diagonal smear" with principled axes (tiers-vs-sources-reframe.md):
  SOURCE  = substrate read: output(logits) / attention(where mass goes) / keys(pre-RoPE key geometry)
            / residual(the stream) / gate.  [NB: old "T2.5" splits into attention(cache_*) + keys(kv_*)]
  METHOD  = operator; the decisive sub-axis is STATIC (*_mean) vs DYNAMIC (*_std/slope/traj/window/drift)
  DEPTH   = layer band (early/mid/late)

Then the two redundancy questions:
  (A) task-specific vs intrinsic: mode-LOFO (feature_decomp.py) vs TASK-FREE cross-source predictability
      (ridge CV R^2 predicting each source-block's top-30 PCA from each other block — no mode labels).
      High predictability + mode-redundant = intrinsic overlap; low + mode-redundant = task-specific.
  (B) source- vs method-limited (the T3 crux): residual source under {PCA(T3), trajectory(res_traj),
      magnitude(activation_norm), cross-layer-delta} head-to-head. If all residual methods are weak ->
      source-limited; if PCA alone is weak while others are fine -> method-limited.

merged v3 SIG, n≈900, 5-way hard, GroupKFold-by-topic, length-residualized. Per-model. CPU, node1:
    OMP_NUM_THREADS=8 PYTHONPATH=pipeline python source_method_reframe.py
"""
from __future__ import annotations

import json
import os
import re
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "8")
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:  # direct script run (sys.path[0] = this dir) — node1 self-contained convention
    from _common import load_signature_matrix, residualize
except ImportError:  # imported as a package module
    from anamnesis.analysis.v3_audit._common import load_signature_matrix, residualize

RUNS = Path(os.environ.get("ANAMNESIS_RUNS", "/models/anamnesis-extract/runs"))
GROUPS = [("3b", ["3b_fat_01", "3b_fat_ext"], 28), ("8b", ["8b_fat_01", "8b_fat_ext"], 32)]
SOURCES = ["output", "attention", "keys", "residual", "gate"]


def source_of(n: str) -> str:
    if n.startswith("gate_"): return "gate"
    if n.startswith("kv_"): return "keys"                       # pre-RoPE key-vector geometry
    if n.startswith(("cache_", "epoch_", "attn_entropy_", "head_agreement_",
                     "attn_flow_", "ph_", "spectral_")):
        return "attention"                                     # where attention mass goes (+ graph)
    if n.startswith(("activation_norm", "res_traj", "delta_", "pca_")):
        return "residual"                                      # the residual stream itself
    return "output"                                            # logit/token-distribution stats


def residual_method_of(n: str) -> str:
    if n.startswith("pca_"): return "pca"
    if n.startswith("res_traj"): return "trajectory"
    if n.startswith("activation_norm"): return "magnitude"
    if n.startswith("delta_"): return "xlayer_delta"
    return "other"


def is_dynamic(n: str):
    """Static = central tendency / baseline; Dynamic = dispersion / change over generation time.
    Returns True (dynamic), False (static), or None (skip — ambiguous)."""
    if n.endswith("_mean"):
        return False
    if n.endswith("_std") or "_slope" in n or "drift" in n or "novelty" in n:
        return True
    if re.search(r"_traj[1-4]\b", n) or re.search(r"_w[0-3]\b", n):
        return True
    if "_traj0" in n:
        return False                                           # deterministic-given-prompt (audit C6)
    return None


def layer_band(n: str, nlayers: int):
    m = re.search(r"_L(\d+)", n)
    if not m:
        return None
    L = int(m.group(1))
    if L < nlayers / 3: return "early"
    if L < 2 * nlayers / 3: return "mid"
    return "late"


def load_sig(runs):
    m = load_signature_matrix(runs, RUNS)
    return m.X, m.y, m.topic, m.C, np.array([str(n) for n in m.names])


def lda_acc(F, yi, topic, C):
    if F.shape[1] == 0:
        return 0.0
    accs = []
    for tr, te in GroupKFold(5).split(F, yi, topic):
        Ftr, Fte = residualize(F[tr].copy(), F[te].copy(), C[tr], C[te])
        sc = StandardScaler().fit(Ftr); Ftr, Fte = sc.transform(Ftr), sc.transform(Fte)
        lda = LDA(solver="eigen", shrinkage="auto")
        lda.fit(Ftr, yi[tr]); accs.append(lda.score(Fte, yi[te]))
    return round(float(np.mean(accs)), 4)


def predictability(Xs, Xt, topic, C, ncomp=30):
    """TASK-FREE: how much of target-block variance (top-ncomp PCA) is linearly predictable from
    source-block, length-residualized, GroupKFold-by-topic CV R^2. High = intrinsic overlap."""
    if Xs.shape[1] == 0 or Xt.shape[1] == 0:
        return 0.0
    r2 = []
    for tr, te in GroupKFold(5).split(Xs, groups=topic):
        Xstr, Xste = residualize(Xs[tr].copy(), Xs[te].copy(), C[tr], C[te])
        Xttr, Xtte = residualize(Xt[tr].copy(), Xt[te].copy(), C[tr], C[te])
        ss = StandardScaler().fit(Xstr); Xstr, Xste = ss.transform(Xstr), ss.transform(Xste)
        st = StandardScaler().fit(Xttr); Yttr, Ytte = st.transform(Xttr), st.transform(Xtte)
        k = min(ncomp, Yttr.shape[1])
        pca = PCA(k).fit(Yttr); Ztr, Zte = pca.transform(Yttr), pca.transform(Ytte)
        pred = Ridge(alpha=10.0).fit(Xstr, Ztr).predict(Xste)
        ss_res = ((Zte - pred) ** 2).sum(0); ss_tot = ((Zte - Zte.mean(0)) ** 2).sum(0) + 1e-12
        r2.append(float(np.mean(1 - ss_res / ss_tot)))
    return round(float(np.mean(r2)), 3)


def main():
    out = {}
    for model, runs, nlayers in GROUPS:
        X, y, topic, C, names = load_sig(runs)
        if len(y) == 0:
            print(f"{model}: no data"); continue
        labels = sorted(set(y.tolist()))
        yi = np.array([labels.index(v) for v in y])
        src = np.array([source_of(n) for n in names])
        print(f"\n===== {model}  n={len(yi)}  feats={X.shape[1]}  (chance 20%) =====", flush=True)

        # 1. SOURCE marginal (regrouped, cleaner than tiers)
        print("  [SOURCE marginal]", flush=True)
        src_acc = {}
        for s in SOURCES:
            m = src == s
            if m.sum():
                src_acc[s] = lda_acc(X[:, m], yi, topic, C)
                print(f"    {s:10s} ({int(m.sum()):4d}): {src_acc[s]:.1%}", flush=True)

        # 2. STATIC vs DYNAMIC (the temporal axis)
        dyn = np.array([is_dynamic(n) for n in names], dtype=object)
        static_m = dyn == False
        dynamic_m = dyn == True
        mean_m = np.array([n.endswith("_mean") for n in names])
        std_m = np.array([n.endswith("_std") for n in names])
        sd = {
            "static_all": (int(static_m.sum()), lda_acc(X[:, static_m], yi, topic, C)),
            "dynamic_all": (int(dynamic_m.sum()), lda_acc(X[:, dynamic_m], yi, topic, C)),
            "_mean_only": (int(mean_m.sum()), lda_acc(X[:, mean_m], yi, topic, C)),
            "_std_only": (int(std_m.sum()), lda_acc(X[:, std_m], yi, topic, C)),
        }
        print("  [STATIC vs DYNAMIC]", flush=True)
        for k, (nf, a) in sd.items():
            print(f"    {k:12s} ({nf:4d}): {a:.1%}", flush=True)

        # 3. DEPTH (layer band)
        band = np.array([layer_band(n, nlayers) for n in names], dtype=object)
        depth = {}
        for b in ["early", "mid", "late"]:
            m = band == b
            if m.sum():
                depth[b] = (int(m.sum()), lda_acc(X[:, m], yi, topic, C))
        print("  [DEPTH band]", flush=True)
        for b, (nf, a) in depth.items():
            print(f"    {b:6s} ({nf:4d}): {a:.1%}", flush=True)

        # 4. RESIDUAL source-vs-method (the T3 crux)
        resm = np.array([residual_method_of(n) for n in names], dtype=object)
        rmeth = {}
        for me in ["pca", "trajectory", "magnitude", "xlayer_delta"]:
            m = resm == me
            if m.sum():
                rmeth[me] = (int(m.sum()), lda_acc(X[:, m], yi, topic, C))
        print("  [RESIDUAL source x method (T3 crux)]", flush=True)
        for me, (nf, a) in rmeth.items():
            print(f"    residual x {me:12s} ({nf:4d}): {a:.1%}", flush=True)

        # 5. TASK-FREE cross-source predictability matrix (intrinsic redundancy)
        print("  [cross-source predictability R^2 (target<-source, task-free, length-resid)]", flush=True)
        present = [s for s in SOURCES if (src == s).sum() > 0]
        pred = {}
        hdr = "        " + "".join(f"{s[:5]:>8s}" for s in present) + "   (<-source)"
        print(hdr, flush=True)
        for tgt in present:
            row = {}
            line = f"    {tgt[:6]:6s} "
            for s in present:
                if s == tgt:
                    line += f"{'-':>8s}"; continue
                r = predictability(X[:, src == s], X[:, src == tgt], topic, C)
                row[s] = r; line += f"{r:>8.2f}"
            pred[tgt] = row
            print(line + "  (target row)", flush=True)

        out[model] = {"source_marginal": src_acc, "static_dynamic": {k: v[1] for k, v in sd.items()},
                      "depth_band": {k: v[1] for k, v in depth.items()},
                      "residual_method": {k: v[1] for k, v in rmeth.items()},
                      "cross_source_predictability": pred}
    Path("source_method_reframe_results.json").write_text(json.dumps(out, indent=2))
    print("\nWrote source_method_reframe_results.json", flush=True)


if __name__ == "__main__":
    main()
