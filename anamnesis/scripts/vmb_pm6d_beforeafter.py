"""PM6-d + xrt-v2.1 before/after (vmb A5 / M6): is the mode axis readable from the ROUTING source
ALONE, and does the v2.1 enrichment add discriminative mode signal?

Frozen-v1 vs v2 exploratory (the PM6-d freeze pin) IS the before/after: v1 xrt = 60 feats (before,
scored), v2 xrt = 120 feats (after, exploratory beside). 5-way mode LDA + binary dir0 (linear vs
socratic), GroupKFold-by-topic, length-residualized (matches A3). Whole-vector v1 vs v2 for context.

Run (node1, CPU): python -m anamnesis.scripts.vmb_pm6d_beforeafter \
  --battery-root /models/anamnesis-extract/runs --model dsv2-lite --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GroupKFold

from anamnesis.analysis.battery.floors import load_signature_matrix
from anamnesis.analysis.feature_map import FeatureMap, Source

MODES = ["linear", "analogical", "socratic", "contrastive", "dialectical"]


def _load(root: Path, model: str, mode: str, subdir: str):
    d = root / f"vmb_a2_{model}_pure_{mode}"
    X, names, ids = load_signature_matrix(d / subdir)
    md = json.loads((d / "metadata.json").read_text())
    gens = md.get("generations", md)
    meta = {int(g["generation_id"]): g for g in gens}
    topics = [int(meta[i]["topic_idx"]) for i in ids]
    lengths = [float(meta[i].get("num_generated_tokens", 0)) for i in ids]
    return X, names, topics, lengths


def _resid_length(X, lengths):
    L = np.asarray(lengths, float).reshape(-1, 1)
    A = np.hstack([np.ones_like(L), L])
    beta, _, _, _ = np.linalg.lstsq(A, X, rcond=None)
    return X - A @ beta


def _acc(X, y, groups, sel=None):
    if sel is not None:
        X = X[:, sel]
    accs = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        clf = LDA().fit(X[tr], y[tr])
        accs.append(float((clf.predict(X[te]) == y[te]).mean()))
    return float(np.mean(accs))


def _perm_band(X, y, groups, sel, nperm=1000, seed=0):
    """Label-permutation null band for OOF accuracy (respecting the group folds).

    The observed accuracy is scored against THIS empirical band, not the theoretical
    chance rate — GroupKFold on few topics gives the null real spread (the .556-style
    upper band, not 0.5/0.2). Reports [p2.5, p50, p97.5], the 95th percentile bar the
    observed must clear, and the observed's one-sided permutation p-value.
    """
    rng = np.random.default_rng(seed)
    obs = _acc(X, y, groups, sel)
    null = np.empty(nperm)
    for k in range(nperm):
        null[k] = _acc(X, rng.permutation(y), groups, sel)
    p = float((np.sum(null >= obs) + 1) / (nperm + 1))
    return {"obs": round(obs, 4),
            "null_p2.5": round(float(np.percentile(null, 2.5)), 4),
            "null_p50": round(float(np.percentile(null, 50)), 4),
            "null_p97.5": round(float(np.percentile(null, 97.5)), 4),
            "null_p95_bar": round(float(np.percentile(null, 95)), 4),
            "perm_p": round(p, 4), "nperm": nperm,
            "clears_band": bool(obs > float(np.percentile(null, 95)))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--model", default="dsv2-lite")
    ap.add_argument("--nperm", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    results: dict = {"model": args.model, "chance_5way": 0.2, "chance_binary": 0.5}
    for label, subdir in (("v1_before", "signatures_v3"), ("v2_after", "signatures_v3_x2")):
        Xs, ys, gs, ls = [], [], [], []
        names = None
        for mi, mode in enumerate(MODES):
            X, nm, topics, lengths = _load(args.battery_root, args.model, mode, subdir)
            names = nm
            Xs.append(X)
            ys += [mi] * len(X)
            gs += topics
            ls += lengths
        X = np.vstack(Xs)
        y = np.array(ys)
        groups = np.array(gs)
        Xr = _resid_length(X, np.array(ls))
        fmap = FeatureMap(names, 27)
        xrt = [i for i, t in enumerate(fmap.tags) if t.source == Source.expert_routing]
        # length-only null (predict from length residual baseline = the length regressor alone)
        Lonly = np.array(ls).reshape(-1, 1)
        dir0 = (y == 0) | (y == 2)                      # linear(0) vs socratic(2)
        yb = (y[dir0] == 0).astype(int)
        results[label] = {
            "n_xrt": len(xrt), "n_total": X.shape[1], "n_gens": int(len(y)),
            "whole_5way": round(_acc(Xr, y, groups), 4),
            "xrt_5way": round(_acc(Xr, y, groups, xrt), 4),
            "whole_dir0": round(_acc(Xr[dir0], yb, groups[dir0]), 4),
            "xrt_dir0": round(_acc(Xr[dir0], yb, groups[dir0], xrt), 4),
            "length_only_5way": round(_acc(Lonly, y, groups), 4),
            # empirical permutation null band for the routing-source readouts (the score bar)
            "xrt_5way_null": _perm_band(Xr, y, groups, xrt, args.nperm),
            "xrt_dir0_null": _perm_band(Xr[dir0], yb, groups[dir0], xrt, args.nperm),
        }
    v1, v2 = results["v1_before"], results["v2_after"]
    results["delta_xrt_5way_v2_minus_v1"] = round(v2["xrt_5way"] - v1["xrt_5way"], 4)
    results["delta_xrt_dir0_v2_minus_v1"] = round(v2["xrt_dir0"] - v1["xrt_dir0"], 4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
