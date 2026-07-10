"""Track A — Factor-naming the ~3 LDA directions of the 5-way how-axis (v2: structure coefficients).

Geometry + linear-probe found the 5-way how-axis is carried by ~3 linear discriminant directions
(LDA-rank plateaus at k≈3: 3B 59/79/87, 8B 63/82/92). This NAMES them, per model, on the merged v3
SIG matrix, length-residualized (Gate-A protocol).

  *** v1 ranked by raw coefficient scalings_/scale_ and got "gate-sparsity-slope dominates" — WRONG:
      that inflates tiny-numeric-scale features (slopes); gate-sparsity-slope ALONE classifies at ~32%
      (chance 20%). The high coefficient was an LDA covariance/suppressor artifact. ***

  v2 names directions by STRUCTURE COEFFICIENTS  r_ij = corr(feature_i, discriminant_score_j)  — the
  standard, suppressor-immune interpretation of a linear discriminant (Tabachnick & Fidell). Plus:
    - univariate eta^2 (between-mode var / total) per feature: individual discriminability, no
      covariance artifact, a cross-check on the structure coefficients,
    - per-family single-family LDA cumulative accuracy: the DECISIVE "what carries the axis" test
      (a family that carries a direction must classify well alone),
    - mode separation (class centroids on the axis), direction stability across folds x seeds.

  orig_weight (scalings_/scale_) is kept ONLY for the saved portable artifact (the vector you APPLY to
  raw feature values to project new data = the kotodama taste vector) — never for naming.

Two variants per model: FULL (all feats; matches the LDA-rank numbers) and HAND (drop the 1250
anonymous T3 pca dims -> the accuracy cost of dropping T3 = is T3 redundant or load-bearing?).
Per-model, never pooled (guardrail). CPU, node1:  OMP_NUM_THREADS=8 PYTHONPATH=pipeline python factor_naming.py
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
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HARD = {"linear", "socratic", "contrastive", "dialectical", "analogical"}
RUNS = Path(os.environ.get("ANAMNESIS_RUNS", "/models/anamnesis-extract/runs"))
GROUPS = [("3b", ["3b_fat_01", "3b_fat_ext"]), ("8b", ["8b_fat_01", "8b_fat_ext"])]
TOPK = 20
KDIR = 4
SEEDS = 3
FAM_ORDER = ["T1", "T2_other", "T2_spectral", "T2.5", "T3", "attention_flow",
             "gate", "per_head", "residual_traj"]


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
    return (np.nan_to_num(np.array(rows, float)), np.array(y),
            np.array(topic), np.array(C, float), np.array(names))


def residualize(Ftr, Fte, Ctr, Cte):
    A = np.hstack([Ctr, np.ones((len(Ctr), 1))]); B = np.hstack([Cte, np.ones((len(Cte), 1))])
    coef, *_ = np.linalg.lstsq(A, Ftr, rcond=None)
    return Ftr - A @ coef, Fte - B @ coef


def fit_lda(F, yi, kdir):
    sc = StandardScaler().fit(F)
    Fs = sc.transform(F)
    lda = LDA(solver="eigen", shrinkage="auto", n_components=kdir)
    lda.fit(Fs, yi)
    Z = Fs @ lda.scalings_[:, :kdir]
    return sc, lda, Z


def orig_weight(sc, lda, j):
    """scalings_/scale_ -> weight on ORIGINAL feature units. For APPLICATION (project new data) only."""
    w = lda.scalings_[:, j] / sc.scale_
    n = np.linalg.norm(w)
    return w / n if n > 0 else w


def greedy_align(Wglob, Wfold):
    K = Wglob.shape[0]
    cos = Wglob @ Wfold.T
    perm = [-1] * K; signed = [0.0] * K
    used = set()
    for g in sorted(range(K), key=lambda g: -np.abs(cos[g]).max()):
        f = max((f for f in range(Wfold.shape[0]) if f not in used), key=lambda f: abs(cos[g, f]))
        perm[g] = f; signed[g] = float(cos[g, f]); used.add(f)
    return perm, signed


def lda_rank(F, yi, topic, C, kdir):
    nclass = len(set(yi.tolist()))
    cum = {k: [] for k in range(1, kdir + 1)}
    for tr, te in GroupKFold(5).split(F, yi, topic):
        Ftr, Fte = residualize(F[tr].copy(), F[te].copy(), C[tr], C[te])
        sc = StandardScaler().fit(Ftr)
        Ftr, Fte = sc.transform(Ftr), sc.transform(Fte)
        lda = LDA(solver="eigen", shrinkage="auto", n_components=kdir)
        Ztr = lda.fit_transform(Ftr, yi[tr]); Zte = lda.transform(Fte)
        for k in range(1, kdir + 1):
            cents = np.stack([Ztr[yi[tr] == c, :k].mean(0) for c in range(nclass)])
            pred = np.argmin(((Zte[:, :k][:, None, :] - cents[None, :, :]) ** 2).sum(-1), axis=1)
            cum[k].append(float(np.mean(pred == yi[te])))
    return {k: round(float(np.mean(v)), 3) for k, v in cum.items()}


def eta2(Fr, yi, nclass):
    """Univariate between-mode variance ratio per feature (residualized). No covariance artifact."""
    grand = Fr.mean(0)
    ssb = np.zeros(Fr.shape[1])
    for k in range(nclass):
        m = yi == k
        ssb += m.sum() * (Fr[m].mean(0) - grand) ** 2
    return ssb / (((Fr - grand) ** 2).sum(0) + 1e-12)


def characterize(F, yi, C, names, labels, fams, kdir):
    A = np.hstack([C, np.ones((len(C), 1))])
    coef, *_ = np.linalg.lstsq(A, F, rcond=None)
    Fr = F - A @ coef
    sc, lda, Z = fit_lda(Fr, yi, kdir)
    nclass = len(labels)
    # structure coefficients: corr(feature, discriminant score) — the suppressor-immune loading
    Fz = (Fr - Fr.mean(0)) / (Fr.std(0) + 1e-12)
    Zz = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    struct = (Fz.T @ Zz) / len(Fr)                 # (P, kdir) in [-1, 1]
    e2 = eta2(Fr, yi, nclass)
    evr = getattr(lda, "explained_variance_ratio_", None)
    dirs = []
    for j in range(kdir):
        r = struct[:, j]; absr = np.abs(r)
        order = np.argsort(-absr)[:TOPK]
        top = [{"feature": names[i], "struct_coef": round(float(r[i]), 3),
                "eta2": round(float(e2[i]), 3), "fam": fams[i]} for i in order]
        fam_mean = {f: round(float(absr[fams == f].mean()), 3) for f in sorted(set(fams.tolist()))}
        fam_mean = dict(sorted(fam_mean.items(), key=lambda kv: -kv[1]))
        fam_max = {f: round(float(absr[fams == f].max()), 3) for f in sorted(set(fams.tolist()))}
        fam_max = dict(sorted(fam_max.items(), key=lambda kv: -kv[1]))
        cent = {labels[k]: float(Z[yi == k, j].mean()) for k in range(nclass)}
        within = float(np.mean([Z[yi == k, j].std() for k in range(nclass)]))
        ordered = sorted(cent.items(), key=lambda kv: kv[1])
        dirs.append({
            "direction": j,
            "eigenvalue_share": round(float(evr[j]), 3) if evr is not None and j < len(evr) else None,
            "top_features_by_structcoef": top,
            "family_mean_alignment": fam_mean,
            "family_max_alignment": fam_max,
            "dominant_family": next(iter(fam_mean)),
            "mode_centroids": {k: round(v, 3) for k, v in cent.items()},
            "separates": f"{ordered[0][0]} <-> {ordered[-1][0]}",
            "separation_snr": round((max(cent.values()) - min(cent.values())) / (within + 1e-9), 2),
        })
    W = np.stack([orig_weight(sc, lda, j) for j in range(kdir)])
    cents_app = {labels[k]: Z[yi == k].mean(0).tolist() for k in range(nclass)}
    return dirs, sc, W, cents_app


def stability(F, yi, topic, C, names, Wglob, kdir):
    coss = {j: [] for j in range(kdir)}
    for s in range(SEEDS):
        for tr, _ in GroupKFold(5).split(F, yi, topic):
            Ftr = F[tr]
            A = np.hstack([C[tr], np.ones((len(tr), 1))])
            cf, *_ = np.linalg.lstsq(A, Ftr, rcond=None)
            sc, lda, _ = fit_lda(Ftr - A @ cf, yi[tr], kdir)
            Wf = np.stack([orig_weight(sc, lda, j) for j in range(kdir)])
            _, signed = greedy_align(Wglob, Wf)
            for g in range(kdir):
                coss[g].append(abs(signed[g]))
    return {j: round(float(np.mean(coss[j])), 3) for j in range(kdir)}


def family_accuracy(X, yi, topic, C, fams):
    """Decisive 'what carries the axis': cumulative LDA accuracy of each family ALONE."""
    out = {}
    for f in FAM_ORDER:
        m = fams == f
        if m.sum() == 0:
            continue
        out[f] = {"n_feats": int(m.sum()), **{f"k{k}": v for k, v in lda_rank(X[:, m], yi, topic, C, KDIR).items()}}
    return out


def run_variant(tag, X, y, topic, C, names, labels, yi, fams, kdir):
    print(f"\n  --- {tag}  (feats={X.shape[1]}) ---", flush=True)
    dirs, sc, W, cents = characterize(X, yi, C, names, labels, fams, kdir)
    stab = stability(X, yi, topic, C, names, W, kdir)
    acc = lda_rank(X, yi, topic, C, kdir)
    for j in range(kdir):
        dirs[j]["stability_cos"] = stab[j]
    for j in range(min(3, kdir)):
        d = dirs[j]
        t5 = ", ".join(f"{t['feature']}(r={t['struct_coef']:+.2f},eta2={t['eta2']:.2f})"
                       for t in d["top_features_by_structcoef"][:5])
        fm = "  ".join(f"{f}={v:.2f}" for f, v in list(d["family_mean_alignment"].items())[:4])
        print(f"   dir{j}: sep[{d['separates']}] snr={d['separation_snr']:.1f} stab={stab[j]:.2f} "
              f"evr={d['eigenvalue_share']}", flush=True)
        print(f"         top|struct|: {t5}", flush=True)
        print(f"         fam mean|struct|: {fm}", flush=True)
    print(f"   cumulative acc (resid): " + "  ".join(f"k{k}={v:.0%}" for k, v in acc.items()), flush=True)
    return {"directions": dirs, "lda_rank_cumulative": acc, "n": int(len(yi)),
            "n_features": int(X.shape[1]), "_W": W, "_mean": sc.mean_, "_scale": sc.scale_,
            "_centroids": cents}


def main():
    out = {}
    for model, runs in GROUPS:
        X, y, topic, C, names = load_sig(runs)
        if len(y) == 0:
            print(f"\n===== {model}: no data ====="); continue
        labels = sorted(set(y.tolist()))
        yi = np.array([labels.index(v) for v in y])
        fams = np.array([fam_of(n) for n in names])
        print(f"\n===== {model}  n={len(y)}  topics={len(set(topic.tolist()))}  "
              f"feats={X.shape[1]}  (chance 20%) =====", flush=True)
        # decisive 'what carries it' — single-family cumulative accuracy (once per model, full feature set)
        famacc = family_accuracy(X, yi, topic, C, fams)
        print("   single-family LDA cumulative acc (k1/k2/k3):", flush=True)
        for f, d in sorted(famacc.items(), key=lambda kv: -kv[1]["k3"]):
            print(f"     {f:16s} ({d['n_feats']:4d}): k1={d['k1']:.0%} k2={d['k2']:.0%} k3={d['k3']:.0%}", flush=True)
        out[model] = {"single_family_accuracy": famacc}
        variants = {"FULL": np.ones(len(names), bool), "HAND": fams != "T3"}
        artifacts = {}
        for tag, mask in variants.items():
            r = run_variant(tag, X[:, mask], y, topic, C, names[mask], labels, yi, fams[mask], KDIR)
            artifacts[tag] = {"W": r.pop("_W"), "mean": r.pop("_mean"), "scale": r.pop("_scale"),
                              "names": names[mask], "centroids": r.pop("_centroids")}
            out[model][tag] = r
        save = {"labels": np.array(labels)}
        for tag, a in artifacts.items():
            save[f"{tag}_W"] = a["W"]; save[f"{tag}_mean"] = a["mean"]
            save[f"{tag}_scale"] = a["scale"]; save[f"{tag}_names"] = a["names"]
            for lab in labels:
                save[f"{tag}_centroid_{lab}"] = np.array(a["centroids"][lab])
        np.savez(f"factor_directions_{model}.npz", **save)
        print(f"  saved factor_directions_{model}.npz", flush=True)
    Path("factor_naming_results.json").write_text(json.dumps(out, indent=2))
    print("\nWrote factor_naming_results.json", flush=True)


if __name__ == "__main__":
    main()
