"""Gate 2 readout: does Column C's depth family see the 70B channel-8026 switch?

Reads the npz from ``column_c_70b_8026.py``. Label: a position is **8026-on** when
|h[L40, 8026]| >= ``--on`` (the known spikes run 20-33 at L40; ordinary positions 1-5).

1. **Paired switch test (surgical).** At the three mode-swap tokens (10011 pos 282, 10018 pos
   245, 10022 pos 470) the text, context and token are identical and the two engines differ only
   in whether the switch fired. The C difference between engines there is compared with the
   engine difference at every other position of the same rows (the null: same text, same token,
   switch not crossed). Per family: the standardised difference norm and its rank. Per layer: the
   first layer at which the difference leaves the null's range (the known splits are L30, L34,
   L29), and whether the on-engine is the one C marks.
2. **Observational test.** Every position of the retained vLLM rows (plus the HF-only control
   rows), on vs off, linear probe with GroupKFold by row. Families: all depth; geometry only
   (no family that reads magnitude: lognorm, dlognorm and peak removed); peak only. Text-side
   baselines: log position, and the input token's on-rate learned on the training folds.
   The swap rows' twin positions are then scored by a probe trained without those rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

BIF = {10011: 282, 10018: 245, 10022: 470}
KNOWN_SPLIT = {10011: 30, 10018: 34, 10022: 29}
MAGNITUDE = ("depth.lognorm.", "depth.dlognorm.", "depth.peak.")


def fam_mask(names, fam):
    if fam == "all":
        return np.ones(len(names), bool)
    if fam == "geometry":
        return np.array([not n.startswith(MAGNITUDE) and not n.startswith("depth.area") for n in names])
    if fam == "peak":
        return np.array([n.startswith("depth.peak.") for n in names])
    return np.array([n.startswith(f"depth.{fam}.") for n in names])


def probe_auc(x, y, groups, tests=None):
    folds = GroupKFold(n_splits=5)
    scores = np.zeros(len(y))
    for tr, te in folds.split(x, y, groups):
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-9
        m = LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced").fit((x[tr] - mu) / sd, y[tr])
        scores[te] = m.decision_function((x[te] - mu) / sd)
    return float(roc_auc_score(y, scores)), scores


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--on", type=float, default=12.0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    z = np.load(args.npz, allow_pickle=False)
    names = [str(n) for n in z["names"]]
    X = z["feats"].astype(np.float64)
    gid, eng, pos = z["gid"], z["engine"].astype(str), z["pos"]
    ch = z["ch8026"]
    rep = dict(digest=str(z["digest"]), n_names=len(names), on_threshold_L40=args.on)

    # label distribution
    c40 = ch[:, 40]
    rep["ch8026_L40_quantiles"] = {q: round(float(np.quantile(c40, q)), 2)
                                   for q in (0.5, 0.9, 0.99, 0.999)}
    rep["n_between_6_and_15"] = int(((c40 > 6) & (c40 < 15)).sum())
    rep["n_on"] = int((c40 >= args.on).sum())
    rep["n_positions"] = int(len(c40))

    # ── 1. paired switch test ──
    sd_all = X[eng == "vllm"].std(0) + 1e-9
    paired = {}
    for g, p in BIF.items():
        hf = {int(q): i for i, q in zip(np.flatnonzero((gid == g) & (eng == "hf")), pos[(gid == g) & (eng == "hf")])}
        vl = {int(q): i for i, q in zip(np.flatnonzero((gid == g) & (eng == "vllm")), pos[(gid == g) & (eng == "vllm")])}
        common = sorted(set(hf) & set(vl))
        D = np.stack([(X[hf[q]] - X[vl[q]]) / sd_all for q in common])     # [T, F] standardised engine diff
        k = common.index(p)
        on_engine = "hf" if ch[hf[p], 40] > ch[vl[p], 40] else "vllm"
        entry = dict(pos=p, on_engine=on_engine, ch8026_L40=dict(hf=round(float(ch[hf[p], 40]), 1),
                                                                 vllm=round(float(ch[vl[p], 40]), 1)), families={})
        for fam in ("all", "geometry", "peak", "lognorm", "cos", "bend", "cosfinal"):
            msk = fam_mask(names, fam)
            norms = np.linalg.norm(D[:, msk], axis=1)
            rank = int((norms > norms[k]).sum()) + 1
            entry["families"][fam] = dict(diff_norm=round(float(norms[k]), 2), rank=rank, of=len(common),
                                          null_median=round(float(np.median(np.delete(norms, k))), 3),
                                          null_max=round(float(np.delete(norms, k).max()), 2),
                                          null_max_pos=int(common[int(np.argmax(np.where(np.arange(len(norms)) == k, -1, norms)))]))
        # onset: per layer, peak and lognorm curves; first layer where |diff| exceeds the null max
        for fam in ("peak", "lognorm", "cosfinal"):
            cols = [i for i, n in enumerate(names) if n.startswith(f"depth.{fam}.P")]
            Dl = np.abs(D[:, cols])
            null_max = np.delete(Dl, k, axis=0).max(0)
            null_p99 = np.quantile(np.delete(Dl, k, axis=0), 0.99, axis=0)
            over = np.flatnonzero(Dl[k] > null_max)
            over99 = np.flatnonzero(Dl[k] > null_p99)
            entry[f"onset_{fam}"] = dict(first_layer_over_null_max=int(over[0]) if len(over) else None,
                                         first_layer_over_null_p99=int(over99[0]) if len(over99) else None,
                                         known_split=KNOWN_SPLIT[g])
        # direction: does the on-engine carry the larger peak index at L36-44?
        pk = [names.index(f"depth.peak.P{l}") for l in range(36, 45)]
        on_i, off_i = (hf[p], vl[p]) if on_engine == "hf" else (vl[p], hf[p])
        entry["on_engine_higher_peak_L36_44"] = bool(X[on_i, pk].mean() > X[off_i, pk].mean())
        paired[g] = entry
    rep["paired"] = paired

    # ── 2. observational ──
    obs = (eng == "vllm") | np.isin(gid, z["controls"])
    Xo, yo, go = X[obs], (c40[obs] >= args.on).astype(int), gid[obs]
    chosen = z["chosen"][obs]
    pos_o = pos[obs]
    # input token at span position p = the token chosen at p-1 (p=0: unknown, -1)
    inp = np.full(len(chosen), -1)
    for g in np.unique(go):
        idx = np.flatnonzero(go == g)
        order = idx[np.argsort(pos_o[idx])]
        inp[order[1:]] = chosen[order[:-1]]
    rep["observational"] = dict(rows=int(len(np.unique(go))), positions=int(len(yo)), on=int(yo.sum()))
    for fam in ("all", "geometry", "peak", "cos", "bend", "cosfinal"):
        msk = fam_mask(names, fam)
        auc, _ = probe_auc(Xo[:, msk], yo, go)
        rep["observational"][f"auc_{fam}"] = round(auc, 4)
    # text baselines
    folds = GroupKFold(n_splits=5)
    s_tok = np.zeros(len(yo))
    for tr, te in folds.split(Xo, yo, go):
        prior = yo[tr].mean()
        rates = {}
        for t, y_ in zip(inp[tr], yo[tr]):
            a, b = rates.get(t, (0, 0))
            rates[t] = (a + y_, b + 1)
        s_tok[te] = [(rates.get(t, (0, 0))[0] + prior * 5) / (rates.get(t, (0, 0))[1] + 5) for t in inp[te]]
    rep["observational"]["auc_text_token_rate"] = round(float(roc_auc_score(yo, s_tok)), 4)
    rep["observational"]["auc_text_logpos"] = round(float(roc_auc_score(yo, np.log1p(pos_o))), 4)
    xt = np.column_stack([s_tok, np.log1p(pos_o)])
    auc_t, _ = probe_auc(xt, yo, go)
    rep["observational"]["auc_text_both"] = round(auc_t, 4)
    auc_g_t, _ = probe_auc(np.column_stack([Xo[:, fam_mask(names, "geometry")], xt]), yo, go)
    rep["observational"]["auc_geometry+text"] = round(auc_g_t, 4)

    # twin positions: probe trained without the swap rows, scored on both engines' copies
    train = obs & ~np.isin(gid, list(BIF))
    for fam in ("all", "geometry"):
        msk = fam_mask(names, fam)
        mu, sd = X[train][:, msk].mean(0), X[train][:, msk].std(0) + 1e-9
        m = LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced").fit(
            (X[train][:, msk] - mu) / sd, (c40[train] >= args.on).astype(int))
        twins = {}
        for g, p in BIF.items():
            s = {e: float(m.decision_function(((X[(gid == g) & (eng == e) & (pos == p)][:, msk] - mu) / sd))[0])
                 for e in ("hf", "vllm")}
            twins[g] = dict(score=s, on_engine=paired[g]["on_engine"],
                            on_scored_higher=bool(s[paired[g]["on_engine"]] > s["vllm" if paired[g]["on_engine"] == "hf" else "hf"]))
        rep[f"twins_{fam}"] = twins

    args.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
