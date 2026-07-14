"""D1 substrate bake-off — "is the signal just the raw activation vector?" (imago queue D1,
addendum 14a §4 item 4).

A matched linear-probe bake-off over the SAME banked fat-run generations, 5-way
(GroupKFold-by-topic, length-residualized), across four rows:

  (a) raw pooled residual activation  [n_gen × hidden_dim]  — the CLASSICAL activation probe
  (b) attention-source signature      (feature_map source:attention slice)
  (c) full signature                  (all v3 features)
  (d) blind-k-way judge               (banked reference; text-legibility of the 5 modes)

The point (imago D1, verbatim): route (a) MEAN-POOLS, and pooled residual is the weak
substrate BY PROJECT LAW (NO-MEANS; residual weakness is source-not-method, per the C5
correction). So route (a) is rigged to lose — and THAT is the finding: the raw activation
vector is the #2 substrate, beaten by attention-weight reads, method-invariantly. "Is it
just the activation vector?" -> no. This is a presentation artifact of a standing answer,
not a new claim. The judge row is labeled blind-k-way and is NEVER quoted for sub-perceptual
claims (the 2AFC kill). Route (a) needs banked RAW TENSORS, which only the fat runs hold
locally — hence the fat-run corpus.

    PYTHONPATH=pipeline python -m anamnesis.scripts.vmb_d1_bakeoff \
        --models 3b,8b --out-dir outputs/battery/arms/D1_bakeoff
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from anamnesis.analysis.battery.floors import build_cells, load_signature_matrix
from anamnesis.analysis.battery.manifest import MODEL_META

MODES = ["linear", "analogical", "socratic", "contrastive", "dialectical"]
# Mid-depth saved-layer index into saved_layers_hs = [-1,0,7,14,...] (3b) / [-1,0,8,16,...]
# (8b): index 3 = L14 (3b, 50%) / L16 (8b, 50%) — the mode-relevant mid band, route (a)'s
# best fair shot.
MID_LAYER_SAVED_IDX = 3
# Banked blind-k-way 5-way judge accuracy (A2 battery corpus, n=800, SAME 5 modes) — computed
# from arms/A3/judge/{key,results}. Cross-corpus labeled REFERENCE row, context only.
JUDGE_REF = {"3b": {"acc": 0.807, "n": 800,
                    "prov": "blind-k-way Fable judge, A2 battery pure-mode corpus (n=160/mode)"},
             "8b": {"acc": 0.949, "n": 800,
                    "prov": "blind-k-way Fable judge, A2 battery pure-mode corpus (n=160/mode)"}}

F32 = NDArray[np.float32]


def _load_fat_meta(model: str) -> list[dict]:
    md = json.loads(Path(f"outputs/runs/{model}_fat_01/metadata.json").read_text())
    gens = md["generations"]
    return [g for g in gens if g["mode"] in MODES]


def _pooled_residual(model: str, gen_ids: list[int]) -> F32:
    """Mean-pooled residual at the mid saved layer, over valid generated positions."""
    root = Path(f"outputs/runs/{model}_fat_01/raw_tensors")
    rows: list[F32] = []
    for gid in gen_ids:
        z = np.load(root / f"gen_{gid:03d}.npz", allow_pickle=True)
        hs = z["hidden_states"]                      # [pos, saved_layers, hidden]
        npos = hs.shape[0]
        rows.append(hs[:npos, MID_LAYER_SAVED_IDX, :].astype(np.float32).mean(axis=0))
    return np.stack(rows)


def _length_residualize(Xtr: F32, Xte: F32, Ltr: F32, Lte: F32) -> tuple[F32, F32]:
    """OLS-residualize every feature on [1, len, prompt_len]; fit on train, apply to both."""
    Dtr = np.column_stack([np.ones(len(Ltr)), Ltr])          # [n, 1+k]
    Dte = np.column_stack([np.ones(len(Lte)), Lte])
    beta, *_ = np.linalg.lstsq(Dtr, Xtr, rcond=None)          # [1+k, D]
    return Xtr - Dtr @ beta, Xte - Dte @ beta


def _cv_accuracy(X: F32, y: NDArray, groups: NDArray, L: F32,
                 clf_name: str, n_splits: int = 5) -> tuple[float, float]:
    gkf = GroupKFold(n_splits=n_splits)
    accs: list[float] = []
    for tr, te in gkf.split(X, y, groups):
        Xtr, Xte = _length_residualize(X[tr], X[te], L[tr], L[te])
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        if clf_name == "lda":
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        else:
            clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=1)
        clf.fit(Xtr, y[tr])
        accs.append(float((clf.predict(Xte) == y[te]).mean()))
    return float(np.mean(accs)), float(np.std(accs))


def analyze_model(model: str) -> dict:
    meta = MODEL_META[model]
    gens = _load_fat_meta(model)
    gen_ids = [int(g["generation_id"]) for g in gens]
    y = np.array([MODES.index(g["mode"]) for g in gens])
    groups = np.array([int(g["topic_idx"]) for g in gens])
    L = np.column_stack([
        np.array([float(g["num_generated_tokens"]) for g in gens]),
        np.array([float(g["prompt_length"]) for g in gens]),
    ])

    # substrate matrices, all row-aligned to `gens`
    X_raw = _pooled_residual(model, gen_ids)                  # (a)
    X_sig, names, sig_ids = load_signature_matrix(
        Path(f"outputs/runs/{model}_fat_01/signatures_v3"))
    order = {gid: r for r, gid in enumerate(sig_ids)}
    if not all(g in order for g in gen_ids):
        missing = [g for g in gen_ids if g not in order]
        raise ValueError(f"{model}: sig rows missing for gens {missing}")
    X_sig = X_sig[[order[g] for g in gen_ids]]                # align to gens
    cells = build_cells(names, meta.n_layers)
    attn_mask = cells["source:attention"]
    X_attn = X_sig[:, attn_mask]                              # (b)

    routes = {
        "a_raw_pooled_residual": (X_raw, f"L{meta.n_layers//2}-ish mid saved layer, token-mean"),
        "b_attention_source_sig": (X_attn, f"{int(attn_mask.sum())} attention-source features"),
        "c_full_signature": (X_sig, f"{X_sig.shape[1]} full v3 features"),
    }
    out_routes: dict[str, dict] = {}
    for rk, (X, desc) in routes.items():
        lda_m, lda_s = _cv_accuracy(X, y, groups, L, "lda")
        log_m, log_s = _cv_accuracy(X, y, groups, L, "logreg")
        out_routes[rk] = {
            "n_features": int(X.shape[1]), "desc": desc,
            "lda_acc": round(lda_m, 4), "lda_fold_std": round(lda_s, 4),
            "logreg_acc": round(log_m, 4), "logreg_fold_std": round(log_s, 4),
        }

    jr = JUDGE_REF.get(model)
    return {
        "arm": "D1_substrate_bakeoff", "model": model,
        "prereg": "imago queue D1 + addendum 14a §4 item 4. Matched linear-probe bake-off, "
                  "5-way, GroupKFold-by-topic, length-residualized (per-fold), on banked "
                  "fat_01. Route (a) mean-pools (weak substrate BY PROJECT LAW: NO-MEANS, "
                  "residual weakness is source-not-method) — rigged to lose, and that is the "
                  "point. Judge row = labeled blind-k-way reference, NEVER quoted for "
                  "sub-perceptual claims.",
        "n_gens": len(gens), "modes": MODES, "cv": "GroupKFold-by-topic (5 splits)",
        "chance": round(1.0 / len(MODES), 3),
        "routes": out_routes,
        "d_judge_reference": ({"acc": jr["acc"], "provenance": jr["prov"],
                               "note": "cross-corpus reference (A2 battery, not fat_01); "
                                       "context only; never quoted for sub-perceptual claims"}
                              if jr else None),
        "law": {"n": len(gens), "M": model,
                "law": "5-way LDA (lsqr, Ledoit-Wolf shrinkage) + logreg robustness; "
                       "GroupKFold-by-topic leak-free; per-fold length-residualization "
                       "[num_gen_tokens, prompt_length]; StandardScaler on train",
                "floor_type": "n/a (classifier accuracy, not a floor-delta cell)"},
    }


def _print(res: dict) -> None:
    print(f"\n[{res['model']}] D1 substrate bake-off (n={res['n_gens']}, 5-way, "
          f"chance={res['chance']})")
    print(f"  {'route':26s} {'nfeat':>6s} {'LDA':>7s} {'±':>6s} {'logreg':>7s} {'±':>6s}")
    for rk in ["a_raw_pooled_residual", "b_attention_source_sig", "c_full_signature"]:
        v = res["routes"][rk]
        print(f"  {rk:26s} {v['n_features']:6d} {v['lda_acc']:7.3f} {v['lda_fold_std']:6.3f} "
              f"{v['logreg_acc']:7.3f} {v['logreg_fold_std']:6.3f}")
    jr = res["d_judge_reference"]
    if jr:
        print(f"  {'d_blind_judge (ref)':26s} {'-':>6s} {jr['acc']:7.3f}  [{jr['provenance']}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="3b,8b")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        res = analyze_model(model)
        _print(res)
        p = args.out_dir / f"d1_bakeoff_{model}.json"
        p.write_text(json.dumps(res, indent=1))
        print(f"  → {p}")


if __name__ == "__main__":
    main()
