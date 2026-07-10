"""Gate-A battery on the FRESH v3 signatures (banked, no recompute).

The decision-gate question (action plan §5 Gate A): under length control, does the
T2.5 "load-bearing tier" lead survive — or does the ranking flip (as the CPU preview
hinted: T2_spectral > T2.5 once length is removed)? Reports, per run:
  - length-only baseline (RF on [prompt_len, gen_len])
  - full v3 raw vs residualized (the contaminated-ceiling / conservative-floor bracket)
  - per-tier raw vs residualized (the tier ranking is the Gate-A payoff)

5-way hard modes, GroupKFold by topic, RF×300×3 seeds. Run from anamnesis_exps:
    OMP_NUM_THREADS=1 python3 research/notes/gate_a_v3_battery.py
"""
import json
import os
import sys
import warnings
from pathlib import Path

SIG_SUBDIR = sys.argv[1] if len(sys.argv) > 1 else "signatures_v3"

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def _est(clf: str):
    if clf == "mlp":
        # modest + regularized for small n (n=100, ~20/class); scaled inputs.
        # No early_stopping (tiny val split at n=100; alpha handles overfit).
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-2, max_iter=800, random_state=0),
        )
    return RandomForestClassifier(300, random_state=0, n_jobs=1)

try:  # direct script run (sys.path[0] = this dir) — node1 self-contained convention
    from _common import HARD, gen_metadata_by_id
except ImportError:  # imported as a package module
    from anamnesis.analysis.v3_audit._common import HARD, gen_metadata_by_id

# Runs root: local default "outputs/runs"; on node set ANAMNESIS_RUNS=/models/anamnesis-extract/runs
RUNS = Path(os.environ.get("ANAMNESIS_RUNS", "outputs/runs"))


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


def clf_cv(F, y, topic, C=None, residualize=False, seeds=3, clf="rf"):
    if F.shape[1] == 0:
        return 0.0
    F = F[:, F.std(0) > 1e-10]
    if F.shape[1] == 0:
        return 0.0
    y = np.unique(y, return_inverse=True)[1]  # integer-encode labels (MLP scoring needs numeric)
    accs = []
    for s in range(seeds):
        fold = []
        for tr, te in GroupKFold(5).split(F, y, topic):
            Ftr, Fte = F[tr].copy(), F[te].copy()
            if residualize:
                A = np.hstack([C[tr], np.ones((len(tr), 1))])
                B = np.hstack([C[te], np.ones((len(te), 1))])
                coef, *_ = np.linalg.lstsq(A, Ftr, rcond=None)
                Ftr, Fte = Ftr - A @ coef, Fte - B @ coef
            est = _est(clf)
            tgt = est.steps[-1][1] if hasattr(est, "steps") else est
            tgt.random_state = s
            est.fit(Ftr, y[tr])
            fold.append(est.score(Fte, y[te]))
        accs.append(np.mean(fold))
    return float(np.mean(accs))


def _load_one(run: str, names_ref):
    """Load one run's hard-mode sigs; return (names, rows, y, topic, C) or None if absent."""
    rd = RUNS / run
    sig_dir = rd / SIG_SUBDIR
    if not (rd / "metadata.json").exists() or not sig_dir.exists():
        return None
    md = gen_metadata_by_id(rd / "metadata.json")
    ids = sorted(int(p.stem.split("_")[1]) for p in sig_dir.glob("gen_*.npz"))
    ids = [g for g in ids if g in md and md[g]["mode"] in HARD]
    names = names_ref
    rows, y, topic, C = [], [], [], []
    for g in ids:
        z = np.load(sig_dir / f"gen_{g:03d}.npz", allow_pickle=True)
        nm = [str(x) for x in z["feature_names"]]
        if names is None:
            names = nm
        d = {n: float(v) for n, v in zip(nm, z["features"])}
        rows.append([d.get(n, 0.0) for n in names])
        y.append(md[g]["mode"]); topic.append(md[g]["topic_idx"])
        C.append([md[g]["prompt_length"], md[g]["num_generated_tokens"]])
    if not rows:
        return None
    return names, rows, y, topic, C


def load_run(runs):
    """Load + concat one or more runs (e.g. fat_01 + fat_ext) into a merged corpus.

    topic_idx is shared across runs (ext continues the fat_01 topic numbering), so
    GroupKFold-by-topic groups all reps of a topic together regardless of source run.
    """
    if isinstance(runs, str):
        runs = [runs]
    names = None
    R, Y, T, Cc = [], [], [], []
    for run in runs:
        out = _load_one(run, names)
        if out is None:
            continue
        names, rows, y, topic, C = out
        R += rows; Y += y; T += topic; Cc += C
    X = np.nan_to_num(np.array(R, float))
    return X, np.array(names), np.array(Y), np.array(T), np.array(Cc, float)


def main():
    order = ["T1", "T2_other", "T2_spectral", "T2.5", "T3", "attention_flow", "gate", "per_head"]
    # Merged groups: fat_01 + ext (ext skipped automatically if absent → falls back to fat_01 only).
    groups = [["3b_fat_01", "3b_fat_ext"], ["8b_fat_01", "8b_fat_ext"]]
    for runs in groups:
        X, names, y, topic, C = load_run(runs)
        if len(y) == 0:
            continue
        run = "+".join(runs)
        fams = np.array([fam_of(n) for n in names])
        print(f"\n===== {run} [{SIG_SUBDIR}] =====  n={len(y)}  topics={len(set(topic.tolist()))}  feats={X.shape[1]}  (chance 20%)", flush=True)
        print(f"length-only: {clf_cv(C, y, topic, clf='rf'):.1%}", flush=True)
        print(f"{'set':16s} {'raw':>7s} {'resid':>7s}", flush=True)

        def line(label, F, m_count=None):
            rr, rs = clf_cv(F, y, topic, clf="rf"), clf_cv(F, y, topic, C, True, clf="rf")
            tag = f"  ({m_count})" if m_count else ""
            print(f"{label:16s} {rr:>7.1%} {rs:>7.1%}{tag}", flush=True)
            return rs

        line("v3 ALL", X)
        res = {}
        for f in order:
            m = fams == f
            if m.sum() == 0:
                continue
            res[f] = line("  " + f, X[:, m], int(m.sum()))
        rank = sorted(res.items(), key=lambda kv: -kv[1])
        print("  → residualized rank: " + " > ".join(f"{k}({v:.0%})" for k, v in rank[:4]), flush=True)


if __name__ == "__main__":
    main()
