"""Routing-CKA leak gate (M6 / DeepSeek-V2-Lite MoE) — REQUIRED before any cka quote.

The v2 xrt re-extraction (signatures_v3_x2) added cross-layer CKA over the expert-routed
representation stream: 6 `xrt_cka_*` features (L5_L11, L11_L15, L15_L18, L18_L22, L22_L26,
global_mean) plus the 48 kv_key/kv_value CKA features. CKA is a representational-similarity
statistic; a cross-layer similarity number can silently track TOPIC (content) rather than the
how-axis. So before anyone quotes "routing CKA carries the mode signal", this gate certifies the
signal is LEAK-SAFE: it must survive GroupKFold-by-topic (test topics unseen in train) AND clear
an empirical label-permutation null band that respects the same folds. The naive-vs-grouped gap
is reported as the leak magnitude, and a direct topic-decode readout quantifies how much content
the features carry on their own.

Methodology mirrors PM6-d / A3 exactly (length-residualized, LDA, GroupKFold-by-topic, Ojala-
Garriga permutation null) so a cka quote sits on the same footing as the scored routing-source
readout. Pure-mode A2 corpora are the substrate (all 5 modes × topics × seeds).

Run (node1, CPU):
    PYTHONPATH=. python -m anamnesis.scripts.vmb_routing_cka_gate \
        --battery-root /models/anamnesis-extract/runs --model dsv2-lite \
        --out /models/anamnesis-extract/battery/arms/A1_dsv2/routing_cka_gate.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GroupKFold, StratifiedKFold

from anamnesis.analysis.battery.floors import load_signature_matrix

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


def _acc_grouped(X, y, groups, sel=None):
    """OOF accuracy under GroupKFold-by-topic (leak-safe: test topics unseen)."""
    if sel is not None:
        X = X[:, sel]
    accs = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        clf = LDA().fit(X[tr], y[tr])
        accs.append(float((clf.predict(X[te]) == y[te]).mean()))
    return float(np.mean(accs))


def _acc_naive(X, y, sel=None, seed=0):
    """OOF accuracy under plain StratifiedKFold (topic MAY leak across folds)."""
    if sel is not None:
        X = X[:, sel]
    accs = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf = LDA().fit(X[tr], y[tr])
        accs.append(float((clf.predict(X[te]) == y[te]).mean()))
    return float(np.mean(accs))


def _perm_band(X, y, groups, sel, nperm=1000, seed=0):
    """Label-permutation null band for grouped OOF accuracy (respects the group folds)."""
    rng = np.random.default_rng(seed)
    obs = _acc_grouped(X, y, groups, sel)
    null = np.empty(nperm)
    for k in range(nperm):
        null[k] = _acc_grouped(X, rng.permutation(y), groups, sel)
    p = float((np.sum(null >= obs) + 1) / (nperm + 1))
    return {"obs": round(obs, 4),
            "null_p50": round(float(np.percentile(null, 50)), 4),
            "null_p95_bar": round(float(np.percentile(null, 95)), 4),
            "null_p97.5": round(float(np.percentile(null, 97.5)), 4),
            "perm_p": round(p, 4), "nperm": nperm,
            "clears_band": bool(obs > float(np.percentile(null, 95)))}


def _verdict(band: dict, naive: float) -> str:
    """LEAK-SAFE CARRIER: clears grouped null. LEAK-DOMINATED: naive well above grouped and
    grouped fails the band. INERT: grouped at band and naive near chance."""
    grouped, bar = band["obs"], band["null_p95_bar"]
    if band["clears_band"]:
        return "LEAK-SAFE CARRIER"
    if naive - grouped >= 0.05 and grouped <= bar:
        return "LEAK-DOMINATED (naive signal is topic leak; no leak-safe carry)"
    return "INERT (no carry above the topic-grouped null band)"


def _gate_featureset(Xr, y, groups, sel, chance, nperm) -> dict:
    band5 = _perm_band(Xr, y, groups, sel, nperm)
    naive5 = _acc_naive(Xr, y, sel)
    # topic-decode: how much raw content the feature set carries (naive 20-way on topic)
    topic_acc = _acc_naive(Xr, groups, sel)
    return {
        "n_feats": int(len(sel)),
        "grouped_5way": band5["obs"], "naive_5way": round(naive5, 4),
        "leak_gap_5way": round(naive5 - band5["obs"], 4),
        "null_band_5way": band5,
        "topic_decode_naive": round(topic_acc, 4),
        "chance_5way": chance, "chance_topic": round(1.0 / len(set(groups.tolist())), 4),
        "verdict": _verdict(band5, naive5),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--model", default="dsv2-lite")
    ap.add_argument("--subdir", default="signatures_v3_x2",
                    help="v2 xrt (120-feat) subdir — the only one carrying xrt_cka")
    ap.add_argument("--nperm", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    Xs, ys, gs, ls = [], [], [], []
    names = None
    for mi, mode in enumerate(MODES):
        X, nm, topics, lengths = _load(args.battery_root, args.model, mode, args.subdir)
        names = nm
        Xs.append(X)
        ys += [mi] * len(X)
        gs += topics
        ls += lengths
    X = np.vstack(Xs)
    y = np.array(ys)
    groups = np.array(gs)
    Xr = _resid_length(X, np.array(ls))

    # the two cka feature sets: the 6 routing-CKA feats (the gate's named target) and all CKA
    xrt_cka = [i for i, n in enumerate(names) if n.startswith("xrt_cka_")]
    all_cka = [i for i, n in enumerate(names) if "cka" in n]
    if len(xrt_cka) != 6:
        raise SystemExit(f"expected 6 xrt_cka feats, found {len(xrt_cka)}: "
                         f"{[names[i] for i in xrt_cka]}")

    results = {
        "gate": "routing_cka_leak_gate", "model": args.model, "subdir": args.subdir,
        "n_gens": int(len(y)), "n_topics": int(len(set(gs))),
        "prereg_note": "leak-safe iff grouped 5-way clears the topic-grouped permutation p95 band; "
                       "leak_gap = naive − grouped; topic_decode = naive 20-way on the same feats. "
                       "Methodology mirrors PM6-d/A3 (length-resid, LDA, GroupKFold-by-topic).",
        "xrt_cka_feats": [names[i] for i in xrt_cka],
        "xrt_cka_6": _gate_featureset(Xr, y, groups, xrt_cka, 0.2, args.nperm),
        "all_cka": _gate_featureset(Xr, y, groups, all_cka, 0.2, args.nperm),
    }
    results["QUOTABLE"] = results["xrt_cka_6"]["verdict"].startswith("LEAK-SAFE")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
