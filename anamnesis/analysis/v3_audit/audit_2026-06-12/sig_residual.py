"""Does mode classification survive controlling for prompt_length + gen_len?

5-way (hard modes), GroupKFold by topic (matches the project's topic-heldout
protocol). Compare per-tier RF accuracy raw vs residualized (per-feature OLS
against [prompt_len, gen_len, 1], fit on train folds only, applied to test).
Also: how well do features predict prompt_length (is length recoverable)?
"""
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

RUN = Path.home() / "projects/anamnesis_exps/outputs/runs/8b_fat_01"
meta = json.load(open(RUN / "metadata.json"))
gens = {g["generation_id"]: g for g in meta["generations"]}

rows, names, gids = [], None, []
for p in sorted((RUN / "signatures_v2").glob("gen_*.npz")):
    gid = int(p.stem.replace("gen_", ""))
    if gid not in gens:
        continue
    z = np.load(p, allow_pickle=True)
    feats = z["features"]
    nlist = [str(n) for n in z["feature_names"]]
    ap = RUN / "signatures_v2_addon" / p.name
    if ap.exists():
        za = np.load(ap, allow_pickle=True)
        feats = np.concatenate([feats, za["features"]])
        nlist = nlist + [str(n) for n in za["feature_names"]]
    rows.append(feats)
    if names is None:
        names = nlist
    gids.append(gid)

X = np.vstack(rows).astype(np.float64)
modes = np.array([gens[g]["mode"] for g in gids])
T_gen = np.array([gens[g]["num_generated_tokens"] for g in gids], dtype=float)
P_len = np.array([gens[g]["prompt_length"] for g in gids], dtype=float)
topic = np.array([gens[g]["topic_idx"] for g in gids])

def family_of(n):
    if n.startswith("pca_"): return "T3"
    if n.startswith("td_"): return "temporal_dynamics"
    if n.startswith("attn_flow_"): return "attention_flow"
    if n.startswith("gate_"): return "gate_features"
    if n.startswith(("cache_", "kv_", "cross_layer", "epoch_")): return "T2.5"
    if n.startswith(("attn_entropy", "head_agreement", "delta_", "spectral_")): return "T2"
    return "T1"

fams = np.array([family_of(n) for n in names])
hard = np.isin(modes, ["linear", "socratic", "contrastive", "dialectical", "analogical"])
Xh, yh, th = X[hard], modes[hard], topic[hard]
Ch = np.stack([P_len[hard], T_gen[hard]], 1)
print(f"5-way n={hard.sum()}, topics={len(set(th))}")

def rf_cv(F, y, groups, C=None, residualize=False, n_seeds=3):
    accs = []
    for seed in range(n_seeds):
        gkf = GroupKFold(n_splits=5)
        fold_accs = []
        for tr, te in gkf.split(F, y, groups):
            Ftr, Fte = F[tr].copy(), F[te].copy()
            if residualize:
                A = np.hstack([C[tr], np.ones((len(tr), 1))])
                B = np.hstack([C[te], np.ones((len(te), 1))])
                coef, *_ = np.linalg.lstsq(A, Ftr, rcond=None)
                Ftr = Ftr - A @ coef
                Fte = Fte - B @ coef
            rf = RandomForestClassifier(300, random_state=seed, n_jobs=1)
            rf.fit(Ftr, y[tr])
            fold_accs.append(rf.score(Fte, y[te]))
        accs.append(np.mean(fold_accs))
    return np.mean(accs), np.std(accs)

subsets = {
    "length_only": None,
    "T1": fams == "T1",
    "T2": fams == "T2",
    "T2.5": fams == "T2.5",
    "T2+T2.5": np.isin(fams, ["T2", "T2.5"]),
    "attention_flow": fams == "attention_flow",
    "temporal_dynamics": fams == "temporal_dynamics",
    "engineered(all non-T3)": ~np.isin(fams, ["T3"]),
}
print(f"{'subset':24s} {'raw':>14s} {'residualized':>14s}")
for label, m in subsets.items():
    if label == "length_only":
        a, s = rf_cv(Ch, yh, th)
        print(f"{label:24s} {a:7.1%} ±{s:.1%} {'—':>14s}")
        continue
    F = Xh[:, m]
    keep = F.std(0) > 1e-10
    F = F[:, keep]
    a_raw, s_raw = rf_cv(F, yh, th)
    a_res, s_res = rf_cv(F, yh, th, C=Ch, residualize=True)
    print(f"{label:24s} {a_raw:7.1%} ±{s_raw:.1%} {a_res:7.1%} ±{s_res:.1%}")

# how recoverable is prompt_length from each tier? (ridge R2, topic-grouped CV)
print("\nprompt_length recoverability (ridge R2, topic-held-out):")
for label in ["T1", "T2", "T2.5", "attention_flow", "temporal_dynamics"]:
    m = subsets[label]
    F = Xh[:, m]
    keep = F.std(0) > 1e-10
    F = (F[:, keep] - F[:, keep].mean(0)) / F[:, keep].std(0)
    gkf = GroupKFold(n_splits=5)
    r2s = []
    for tr, te in gkf.split(F, P_len[hard], th):
        r = Ridge(alpha=10.0).fit(F[tr], P_len[hard][tr])
        ss_res = ((P_len[hard][te] - r.predict(F[te])) ** 2).sum()
        ss_tot = ((P_len[hard][te] - P_len[hard][tr].mean()) ** 2).sum()
        r2s.append(1 - ss_res / ss_tot)
    print(f"  {label:20s} R2 = {np.mean(r2s):.3f}")
