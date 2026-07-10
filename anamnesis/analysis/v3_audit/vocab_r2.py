"""Collective-R^2 vocabulary measure — how much of the RAW encoder direction is hand-nameable.

The per-feature max |structure-coef| (in hand_family_analysis) is conservative: a single scalar only
weakly aligns with a 100-PC-derived discriminant. The fair measure is COLLECTIVE: how well do ALL of a
family's hand-features TOGETHER reconstruct the raw surface's discriminant direction (leak-proof). High
R^2 ⇒ the family is good vocabulary for what the encoder reads (even if it classifies below the floor).

For each surface: raw OOF discriminant Z (residualize→GPU lossless Gram-reduce→LDA, leak-proof, reusing
the floor path). Then per family: topic-CV RidgeCV predicting each of the KDIR discriminant dims from the
family's hand-features → held-out R^2 (mean over dims + best dim). Also the 3-families-combined naming power.

    SURFACE_FLOOR_DIR=.../floor_results python vocab_r2.py --models 3b,8b --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from surface_encoder_floor import OUTDIR, load_surface, preprocess_fold_gpu

HAND = os.environ.get("HAND_CACHE_DIR", os.environ.get("SURFACE_FLOOR_DIR", "."))
KDIR = 4
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
# family -> surfaces whose raw discriminant we try to name with it
FAM_SURFACES = {"value_geometry": ["values"], "qk_geometry": ["queries", "keys"], "kv_cka": ["keys", "values"]}


def load_hand(model):
    z = np.load(os.path.join(HAND, f"hand_cache_{model}.npz"), allow_pickle=True)
    X = np.nan_to_num(z["X"].astype(np.float32))
    fam = z["fam_name"].astype(str); fs = z["fam_start"].astype(int); fe = z["fam_end"].astype(int)
    return (X, z["mode"].astype(str), z["topic"].astype(int),
            np.column_stack([z["plen"].astype(float), z["glen"].astype(float)]),
            z["gen_uid"].astype(str), {f: (int(s), int(e)) for f, s, e in zip(fam, fs, fe)})


def surface_oof_Z(model, surface, ref_uid, yi_by_uid, topic_by_uid, C_by_uid, device):
    """RAW surface OOF discriminant (KDIR dims), aligned to ref_uid ∩ surface gens."""
    X, mode, topic, C = load_surface(model, surface)
    z = np.load(os.path.join(os.environ.get("SURFACE_CACHE_DIR", "/dev/shm/anamnesis_surface_caches"),
                             f"surface_cache_{model}_{surface}.npz"), allow_pickle=True)
    suid = z["gen_uid"].astype(str)
    ref = set(ref_uid.tolist()); keep = np.array([u in ref for u in suid])
    X = X[keep]; uid = suid[keep]
    yi = np.array([yi_by_uid[u] for u in uid]); tp = np.array([topic_by_uid[u] for u in uid])
    C = np.stack([C_by_uid[u] for u in uid])
    Z = np.zeros((len(uid), KDIR))
    for tr, te in GroupKFold(5).split(X, yi, tp):
        Ztr, Zte = preprocess_fold_gpu(X[tr], X[te], C[tr], C[te], True, device)
        lda = LDA(solver="eigen", shrinkage="auto", n_components=KDIR).fit(Ztr, yi[tr])
        Z[te] = lda.transform(Zte)
    return Z, uid


def cv_ridge_r2(Xfeat, Z, topic):
    """Held-out R^2 predicting each raw discriminant dim from hand-features (topic-CV RidgeCV). Returns per-dim."""
    n, kd = Z.shape
    pred = np.zeros_like(Z)
    for tr, te in GroupKFold(5).split(Xfeat, np.zeros(n), topic):
        sc = StandardScaler().fit(Xfeat[tr]); Xtr, Xte = sc.transform(Xfeat[tr]), sc.transform(Xfeat[te])
        for d in range(kd):
            r = RidgeCV(alphas=ALPHAS).fit(Xtr, Z[tr, d])
            pred[te, d] = r.predict(Xte)
    r2 = []
    for d in range(kd):
        ss_res = float(((Z[:, d] - pred[:, d]) ** 2).sum())
        ss_tot = float(((Z[:, d] - Z[:, d].mean()) ** 2).sum())
        r2.append(1.0 - ss_res / max(ss_tot, 1e-12))
    return r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="3b,8b")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    import torch
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"device={device}  HAND={HAND}", flush=True)
    out = {}
    for model in args.models.split(","):
        model = model.strip()
        X, mode, topic, C, uid, slices = load_hand(model)
        labels = sorted(set(mode.tolist())); yi = np.array([labels.index(m) for m in mode])
        yi_by = {u: int(yi[i]) for i, u in enumerate(uid)}
        tp_by = {u: int(topic[i]) for i, u in enumerate(uid)}
        C_by = {u: C[i] for i, u in enumerate(uid)}
        print(f"\n===== {model}  n={len(uid)} =====", flush=True)
        out[model] = {}
        surfaces = sorted({s for ss in FAM_SURFACES.values() for s in ss})
        for surface in surfaces:
            Z, zuid = surface_oof_Z(model, surface, uid, yi_by, tp_by, C_by, device)
            tp_z = np.array([tp_by[u] for u in zuid])
            Xby = {u: X[i] for i, u in enumerate(uid)}
            Xal = np.stack([Xby[u] for u in zuid])
            # which families name THIS surface's raw direction + the 3-combined
            fams = [f for f, ss in FAM_SURFACES.items() if surface in ss] + ["ALL_3_combined"]
            out[model][surface] = {}
            for f in fams:
                if f == "ALL_3_combined":
                    Xf = Xal
                else:
                    s, e = slices[f]; Xf = Xal[:, s:e]
                r2 = cv_ridge_r2(Xf, Z, tp_z)
                out[model][surface][f] = {"r2_mean": round(float(np.mean(r2)), 3),
                                          "r2_best_dim": round(float(np.max(r2)), 3),
                                          "r2_per_dim": [round(x, 3) for x in r2]}
                print(f"  {surface:8s} <- {f:16s}  R2 mean={np.mean(r2):.2f}  best-dim={np.max(r2):.2f}  "
                      f"per-dim={[round(x,2) for x in r2]}", flush=True)
    json.dump(out, open(os.path.join(OUTDIR, "vocab_r2_results.json"), "w"), indent=2)
    print(f"\nWrote {os.path.join(OUTDIR, 'vocab_r2_results.json')}", flush=True)


if __name__ == "__main__":
    main()
