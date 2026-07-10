"""Evaluate the 3 NEW hand-feature families against the encoder floors — chad 1&3, reframe-aligned.

Two questions, per the encoder-vs-signatures reframe (hand-features = vocabulary, not extractor):

  (A) METHOD-LIMIT — do they carry signal, how close to their surface's encoder FLOOR?
      Per family + combined: 5-way LDA cumulative accuracy (k1..k4), leak-proof GroupKFold-by-topic,
      length-residualized. Learning curve (frac 1.0/0.5/0.25 of train topics) → the n-dependence probe
      (does the hand↔raw gap shrink, or flip, at low n — the kotodama-thin-data question). Printed next
      to the lossless encoder floor (surface_floor_*.json) + the old hand-LDA bars.

  (B) VOCABULARY ADEQUACY — do they NAME what the raw encoder reads (the distill template)?
      Fit the RAW surface's discriminant out-of-fold (residualize → PCA → LDA, leak-proof) → per-gen
      OOF discriminant scores; align hand rows by gen_uid; STRUCTURE COEFFICIENT = corr(hand_feature,
      raw_discriminant) (Tabachnick & Fidell; suppressor-immune — same method that named
      attention-allocation). A family can be method-limited as a classifier yet load highly here =
      "fine vocabulary, don't use as extractor" (the reframe verdict). Also names each family's OWN
      top direction (eta^2 + struct coef) for interpretability.

Surface map: value_geometry→values, qk_geometry→queries(+keys), kv_cka→keys(+values).
CPU/sklearn, node1.  needs hand_cache_*.npz (build_hand_family_caches) + surface caches + floor JSONs:
    OMP_NUM_THREADS=8 PYTHONPATH=.../pipeline python hand_family_analysis.py --models 3b,8b
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "8")
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

CACHE = Path(os.environ.get("SURFACE_CACHE_DIR", "/dev/shm/anamnesis_surface_caches"))
HAND = Path(os.environ.get("HAND_CACHE_DIR", "."))
FLOORS = Path(os.environ.get("SURFACE_FLOOR_DIR", "."))
KDIR = 4
TOPK = 12
FRACS = [1.0, 0.5, 0.25]
SURFACE_OF = {"value_geometry": "values", "qk_geometry": "queries", "kv_cka": "keys"}
SECONDARY = {"qk_geometry": "keys", "kv_cka": "values"}      # families that also touch a 2nd surface


def residualize(Ftr, Fte, Ctr, Cte):
    A = np.hstack([Ctr, np.ones((len(Ctr), 1))]); B = np.hstack([Cte, np.ones((len(Cte), 1))])
    coef, *_ = np.linalg.lstsq(A, Ftr, rcond=None)
    return Ftr - A @ coef, Fte - B @ coef


def lda_curve(F, yi, topic, C, frac, seed=0):
    """Cumulative LDA accuracy k1..KDIR, leak-proof GroupKFold, length-resid, optional train-topic subsample."""
    nclass = len(set(yi.tolist()))
    cum = {k: [] for k in range(1, KDIR + 1)}
    rng = np.random.default_rng(1000 + seed)
    for tr, te in GroupKFold(5).split(F, yi, topic):
        if frac < 1.0:
            utop = np.unique(topic[tr])
            keep = set(rng.choice(utop, max(2, int(round(frac * len(utop)))), replace=False).tolist())
            tr = tr[np.array([topic[i] in keep for i in tr])]
        Ftr, Fte = residualize(F[tr].copy(), F[te].copy(), C[tr], C[te])
        sc = StandardScaler().fit(Ftr)
        Ftr, Fte = sc.transform(Ftr), sc.transform(Fte)
        lda = LDA(solver="eigen", shrinkage="auto", n_components=KDIR)
        Ztr = lda.fit_transform(Ftr, yi[tr]); Zte = lda.transform(Fte)
        for k in range(1, KDIR + 1):
            cents = np.stack([Ztr[yi[tr] == c, :k].mean(0) for c in range(nclass)])
            pred = np.argmin(((Zte[:, :k][:, None, :] - cents[None, :, :]) ** 2).sum(-1), axis=1)
            cum[k].append(float(np.mean(pred == yi[te])))
    return {k: round(float(np.mean(v)), 4) for k, v in cum.items()}


def eta2(Fr, yi, nclass):
    grand = Fr.mean(0)
    ssb = np.zeros(Fr.shape[1])
    for k in range(nclass):
        m = yi == k
        ssb += m.sum() * (Fr[m].mean(0) - grand) ** 2
    return ssb / (((Fr - grand) ** 2).sum(0) + 1e-12)


def own_direction(F, yi, C, names, labels):
    """Name the family's OWN top discriminant: struct coefs + eta^2 + what it separates."""
    A = np.hstack([C, np.ones((len(C), 1))]); coef, *_ = np.linalg.lstsq(A, F, rcond=None)
    Fr = F - A @ coef
    sc = StandardScaler().fit(Fr); Fs = sc.transform(Fr)
    lda = LDA(solver="eigen", shrinkage="auto", n_components=KDIR).fit(Fs, yi)
    Z = Fs @ lda.scalings_[:, :KDIR]
    Fz = (Fr - Fr.mean(0)) / (Fr.std(0) + 1e-12)
    Zz = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    struct = (Fz.T @ Zz) / len(Fr)
    e2 = eta2(Fr, yi, len(labels))
    r = struct[:, 0]; order = np.argsort(-np.abs(r))[:TOPK]
    top = [{"feature": str(names[i]), "struct_coef": round(float(r[i]), 3), "eta2": round(float(e2[i]), 3)}
           for i in order]
    cent = {labels[k]: float(Z[yi == k, 0].mean()) for k in range(len(labels))}
    ordered = sorted(cent.items(), key=lambda kv: kv[1])
    return {"top_features_by_structcoef": top, "separates_dir0": f"{ordered[0][0]} <-> {ordered[-1][0]}",
            "max_eta2": round(float(e2.max()), 3)}


def surface_oof_scores(model, surface, ref_uid, yi_by_uid, C_by_uid, topic_by_uid):
    """RAW surface discriminant, OUT-OF-FOLD: residualize→PCA(100)→LDA per fold, score held-out.
    Returns (Z [n_common, KDIR], uids[n_common]) aligned to the surface cache's own gens (∩ ref_uid)."""
    path = CACHE / f"surface_cache_{model}_{surface}.npz"
    if not path.exists():
        return None, None
    z = np.load(path, allow_pickle=True)
    suid = z["gen_uid"].astype(str)
    ref_set = set(ref_uid.tolist())
    keep = np.array([u in ref_set for u in suid])
    X = np.nan_to_num(z["X"][keep].astype(np.float32))
    uid = suid[keep]
    yi = np.array([yi_by_uid[u] for u in uid])
    topic = np.array([topic_by_uid[u] for u in uid])
    C = np.stack([C_by_uid[u] for u in uid])
    Z = np.zeros((len(uid), KDIR), dtype=np.float64)
    for tr, te in GroupKFold(5).split(X, yi, topic):
        Xtr, Xte = residualize(X[tr].copy(), X[te].copy(), C[tr], C[te])
        k = min(100, Xtr.shape[0] - 1, Xtr.shape[1])
        pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(Xtr)
        Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
        sc = StandardScaler().fit(Ztr); Ztr, Zte = sc.transform(Ztr), sc.transform(Zte)
        lda = LDA(solver="eigen", shrinkage="auto", n_components=KDIR).fit(Ztr, yi[tr])
        Z[te] = lda.transform(Zte)
    return Z, uid


def name_raw_direction(Xfam_by_uid, fam_names, Z, zuid):
    """STRUCTURE COEFFICIENTS of family features vs the RAW surface OOF discriminant (the reframe cut)."""
    Xf = np.stack([Xfam_by_uid[u] for u in zuid])            # align hand rows to the surface's OOF order
    Fz = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-12)
    Zz = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    struct = (Fz.T @ Zz) / len(Xf)                           # (n_feat, KDIR)
    best = np.abs(struct).max(axis=1)                         # best |corr| to ANY of the raw dirs
    order = np.argsort(-best)[:TOPK]
    top = [{"feature": str(fam_names[i]),
            "best_struct_coef": round(float(struct[i][np.argmax(np.abs(struct[i]))]), 3)} for i in order]
    return {"max_struct_coef_to_raw": round(float(best.max()), 3),
            "mean_top10_struct_coef": round(float(np.sort(best)[::-1][:10].mean()), 3),
            "top_features_naming_raw_dir": top}


def load_floor(model, surface):
    p = FLOORS / f"surface_floor_{model}_{surface}.json"
    if not p.exists():
        return None
    d = json.load(open(p)).get(surface, {})
    return {"logit_full": d.get("resid.logit.frac1.0", {}).get("test"),
            "logit_q": d.get("resid.logit.frac0.25", {}).get("test"),
            "deep_full": d.get("resid.deep.frac1.0", {}).get("test"),
            "hand_LDA": d.get("hand_LDA")}


def run_model(model):
    hp = HAND / f"hand_cache_{model}.npz"
    if not hp.exists():
        print(f"[skip] {model}: {hp} missing", flush=True); return None
    z = np.load(hp, allow_pickle=True)
    X = np.nan_to_num(z["X"].astype(np.float32)); names = z["feature_names"].astype(str)
    y = z["mode"].astype(str); topic = z["topic"].astype(int)
    C = np.column_stack([z["plen"].astype(float), z["glen"].astype(float)])
    uid = z["gen_uid"].astype(str)
    fam_name = z["fam_name"].astype(str); fs = z["fam_start"].astype(int); fe = z["fam_end"].astype(int)
    labels = sorted(set(y.tolist())); yi = np.array([labels.index(v) for v in y])
    n = len(yi)
    print(f"\n===== {model}  n={n}  topics={len(set(topic.tolist()))}  feats={X.shape[1]}  (chance 20%) =====",
          flush=True)
    yi_by_uid = {u: int(yi[i]) for i, u in enumerate(uid)}
    C_by_uid = {u: C[i] for i, u in enumerate(uid)}
    topic_by_uid = {u: int(topic[i]) for i, u in enumerate(uid)}

    out = {"n": int(n), "labels": labels, "families": {}}
    blocks = [(fam_name[i], int(fs[i]), int(fe[i])) for i in range(len(fam_name))]
    blocks.append(("ALL_3_combined", 0, X.shape[1]))

    for fam, s, e in blocks:
        Xf = X[:, s:e]
        curve = {f"frac{fr}": lda_curve(Xf, yi, topic, C, fr) for fr in FRACS}
        full = curve["frac1.0"][KDIR]; low = curve["frac0.25"][KDIR]
        entry = {"n_feats": int(e - s), "acc_k4_full": full, "acc_k4_lowN": low,
                 "learning_curve": {fr: curve[fr][KDIR] for fr in curve}}
        surf = SURFACE_OF.get(fam)
        fl = load_floor(model, surf) if surf else None
        if fl:
            entry["surface"] = surf; entry["floor"] = fl
        if fam != "ALL_3_combined":
            entry["own_direction"] = own_direction(Xf, yi, C, names[s:e], labels)
            # (B) name the RAW surface direction with this family's features
            Xfam_by_uid = {u: Xf[i] for i, u in enumerate(uid)}
            for tag, srf in [("primary", SURFACE_OF.get(fam)), ("secondary", SECONDARY.get(fam))]:
                if not srf:
                    continue
                Z, zuid = surface_oof_scores(model, srf, uid, yi_by_uid, C_by_uid, topic_by_uid)
                if Z is not None:
                    entry.setdefault("names_raw_dir", {})[srf] = name_raw_direction(Xfam_by_uid, names[s:e], Z, zuid)
        out["families"][fam] = entry

        # ── print ──
        floor_str = ""
        if fl and fl.get("logit_full") is not None:
            floor_str = (f"  | {surf} floor: logit {fl['logit_full']:.0%} (lowN {fl['logit_q']:.0%})"
                         f"{'  hand-LDA '+format(fl['hand_LDA'],'.0%') if fl.get('hand_LDA') else ''}")
        lc = "  ".join(f"{fr}={curve[fr][KDIR]:.0%}" for fr in curve)
        print(f"  {fam:18s} ({e-s:4d}) k4={full:.0%} (lowN {low:.0%}) [{lc}]{floor_str}", flush=True)
        if "own_direction" in entry:
            od = entry["own_direction"]
            t3 = ", ".join(f"{t['feature']}(r={t['struct_coef']:+.2f})" for t in od["top_features_by_structcoef"][:3])
            print(f"      own dir0 [{od['separates_dir0']}] maxeta2={od['max_eta2']:.2f}  top: {t3}", flush=True)
        if entry.get("names_raw_dir"):
            for srf, nd in entry["names_raw_dir"].items():
                t2 = ", ".join(f"{t['feature']}(r={t['best_struct_coef']:+.2f})" for t in nd["top_features_naming_raw_dir"][:3])
                print(f"      names {srf}-raw-dir: max|r|={nd['max_struct_coef_to_raw']:.2f} "
                      f"top10mean={nd['mean_top10_struct_coef']:.2f}  {t2}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="3b,8b")
    args = ap.parse_args()
    print(f"HAND={HAND.resolve()}  CACHE={CACHE}  FLOORS={FLOORS.resolve()}", flush=True)
    out = {}
    for m in args.models.split(","):
        m = m.strip()
        r = run_model(m)
        if r:
            out[m] = r
    Path(FLOORS / "hand_family_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {FLOORS / 'hand_family_results.json'}", flush=True)


if __name__ == "__main__":
    main()
