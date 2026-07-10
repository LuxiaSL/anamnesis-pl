"""Empirical audit of the anamnesis feature bank against banked 8B signatures.

Checks (all on outputs/runs/8b_fat_01, v2 signatures where available):
  1. Dead / near-constant features
  2. Trivial duplicate clusters (|Spearman| > 0.95)
  3. Effective dimensionality of the bank (PCA participation ratio, 95% count)
  4. Length confounds: Spearman(feature, gen_tokens) and (feature, prompt_length)
  5. Prompt-length -> mode leakage: how well does prompt_length ALONE classify mode
  6. T3 pooled-PCA check: do leading components encode layer identity?
"""
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

RUN = Path.home() / "projects/anamnesis_exps/outputs/runs/8b_fat_01"
SIG_V2 = RUN / "signatures_v2"
SIG_V1 = RUN / "signatures"

meta = json.load(open(RUN / "metadata.json"))
gens = {g["generation_id"]: g for g in meta["generations"]}

sig_dir = SIG_V2 if SIG_V2.exists() and any(SIG_V2.glob("*.npz")) else SIG_V1
print(f"using {sig_dir}")

rows, names, gids = [], None, []
for p in sorted(sig_dir.glob("gen_*.npz")):
    gid = int(p.stem.replace("gen_", ""))
    if gid not in gens:
        continue
    z = np.load(p, allow_pickle=True)
    feats = z["features"]
    if names is None:
        names = [str(n) for n in z["feature_names"]]
    rows.append(feats)
    gids.append(gid)

X = np.vstack(rows).astype(np.float64)
modes = np.array([gens[g]["mode"] for g in gids])
T_gen = np.array([gens[g]["num_generated_tokens"] for g in gids], dtype=float)
P_len = np.array([gens[g]["prompt_length"] for g in gids], dtype=float)
core = ~np.char.startswith(modes.astype(str), "swap_")
print(f"X: {X.shape}, core samples: {core.sum()}, modes: {sorted(set(modes[core]))}")

def family_of(n: str) -> str:
    if n.startswith("pca_"): return "T3_pca"
    if n.startswith(("td_",)): return "temporal_dynamics"
    if n.startswith("attn_flow_"): return "attention_flow"
    if n.startswith("gate_"): return "gate_features"
    if n.startswith("contrastive_") or n.startswith("cproj"): return "contrastive"
    if n.startswith(("cache_", "kv_", "cross_layer", "epoch_")): return "T2.5"
    if n.startswith(("attn_entropy", "head_agreement", "delta_", "spectral_")): return "T2"
    return "T1"

fams = np.array([family_of(n) for n in names])
print("family counts:", dict(sorted({f: int((fams == f).sum()) for f in set(fams)}.items())))

# ── 1. dead features ──────────────────────────────────────────────
Xc = X[core]
std = Xc.std(axis=0)
mean_abs = np.abs(Xc).mean(axis=0) + 1e-12
rel_std = std / mean_abs
dead = std < 1e-10
near_const = (rel_std < 1e-3) & ~dead
print(f"\n[1] dead features (std=0): {dead.sum()}")
for f in sorted(set(fams)):
    m = fams == f
    print(f"    {f:20s} dead={int((dead & m).sum()):4d}/{int(m.sum()):4d}  near_const={int((near_const & m).sum())}")
if dead.sum():
    dead_names = [names[i] for i in np.where(dead)[0]]
    bases = defaultdict(int)
    for n in dead_names:
        bases["_".join([p for p in n.split("_") if not (p.startswith("L") and p[1:].isdigit())])] += 1
    print("    top dead patterns:", sorted(bases.items(), key=lambda kv: -kv[1])[:8])

# ── 2+3. redundancy on the engineered (non-T3, non-contrastive) bank ──
keep = ~dead & (std > 0)
eng = keep & ~np.isin(fams, ["T3_pca", "contrastive"])
idx_eng = np.where(eng)[0]
Z = (Xc[:, idx_eng] - Xc[:, idx_eng].mean(0)) / Xc[:, idx_eng].std(0)
# spearman via rank transform then corrcoef (fast enough at this size)
from scipy.stats import rankdata
R = np.apply_along_axis(rankdata, 0, Xc[:, idx_eng])
R = (R - R.mean(0)) / (R.std(0) + 1e-12)
C = (R.T @ R) / R.shape[0]
np.fill_diagonal(C, 0)
absC = np.abs(C)
dup_pairs = int((absC > 0.95).sum() // 2)
dup_any = (absC > 0.95).any(axis=1)
print(f"\n[2] engineered bank (n={eng.sum()}): features with a |rho|>0.95 partner: {dup_any.sum()} ({100*dup_any.mean():.0f}%)")
print(f"    duplicate pairs: {dup_pairs}")
# greedy dedup: how many survive at .95?
order = np.argsort(-absC.max(axis=1))
removed = np.zeros(len(idx_eng), bool)
for i in range(len(idx_eng)):
    if removed[i]:
        continue
    removed[(absC[i] > 0.95) & ~removed] = True
    removed[i] = False
print(f"    greedy dedup at |rho|>0.95 keeps: {int((~removed).sum())}/{len(idx_eng)}")

# eigen-spectrum effective dim (linear PR) on standardized engineered bank
Cp = (Z.T @ Z) / Z.shape[0]
ev = np.linalg.eigvalsh(Cp)[::-1]
ev = np.maximum(ev, 0)
pr = (ev.sum() ** 2) / (ev ** 2).sum()
c95 = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.95) + 1)
print(f"[3] engineered bank: linear PR = {pr:.1f}, components for 95% var = {c95} (of {len(idx_eng)})")

# example known-duplicate: lookback vs sysprompt region
def find(n_sub):
    return [i for i, n in enumerate(names) if n_sub in n]
lb = find("td_L16_lookback_ratio_w0_mean")
sp = find("attn_flow_L16_sysprompt_mass_w0_mean")
if lb and sp:
    rho = spearmanr(Xc[:, lb[0]], Xc[:, sp[0]]).statistic
    print(f"    sanity: lookback_w0 vs sysprompt_mass_w0 (L16) Spearman = {rho:.4f}")

# ── 4. length confounds ───────────────────────────────────────────
rho_T = np.zeros(X.shape[1]); rho_P = np.zeros(X.shape[1])
for i in np.where(keep)[0]:
    rho_T[i] = spearmanr(Xc[:, i], T_gen[core]).statistic
    rho_P[i] = spearmanr(Xc[:, i], P_len[core]).statistic
rho_T = np.nan_to_num(rho_T); rho_P = np.nan_to_num(rho_P)
print(f"\n[4] |rho(feature, gen_len)|>0.5: {int((np.abs(rho_T) > .5).sum())} features; >0.3: {int((np.abs(rho_T) > .3).sum())}")
for f in sorted(set(fams)):
    m = (fams == f) & keep
    if m.sum():
        print(f"    {f:20s} median|rho_T|={np.median(np.abs(rho_T[m])):.3f}  frac>0.3={np.mean(np.abs(rho_T[m])>0.3):.2f}   median|rho_P|={np.median(np.abs(rho_P[m])):.3f}  frac>0.3={np.mean(np.abs(rho_P[m])>0.3):.2f}")
top_T = np.argsort(-np.abs(rho_T))[:8]
print("    top gen-length-coupled:", [(names[i], round(rho_T[i], 2)) for i in top_T])
top_P = np.argsort(-np.abs(rho_P))[:8]
print("    top prompt-length-coupled:", [(names[i], round(rho_P[i], 2)) for i in top_P])

# ── 5. prompt_length alone as a mode classifier (core modes, 5-way + 8-way) ──
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
hard5 = np.isin(modes, ["linear", "socratic", "contrastive", "dialectical", "analogical"])
for label, mask in [("8-way", core), ("5-way", hard5)]:
    y = modes[mask]
    feats = np.stack([P_len[mask], T_gen[mask]], 1)
    rf = RandomForestClassifier(200, random_state=0, n_jobs=1)
    acc = cross_val_score(rf, feats, y, cv=5).mean()
    accP = cross_val_score(rf, P_len[mask].reshape(-1, 1), y, cv=5).mean()
    print(f"[5] {label}: RF on [prompt_len, gen_len] alone = {acc:.1%}; prompt_len alone = {accP:.1%} (chance {1/len(set(y)):.1%})")

# ── 6. T3 pooled-PCA layer-identity check ─────────────────────────
t3 = np.where(fams == "T3_pca")[0]
if len(t3):
    # parse pca_L{l}_t{ti}_c{ci}
    import re
    rx = re.compile(r"pca_L(\d+)_t(\d+)_c(\d+)")
    by_c = defaultdict(dict)  # ci -> layer -> mean abs value
    for i in t3:
        m = rx.match(names[i])
        if not m:
            continue
        l, ti, ci = map(int, m.groups())
        if ti == 2:  # mid-generation snapshot
            by_c[ci].setdefault(l, []).append(i)
    print("\n[6] T3 pooled-basis check (component value by layer, t2 snapshot, mean over samples):")
    for ci in [0, 1, 2, 5, 10]:
        if ci not in by_c:
            continue
        rowstr = []
        layer_means = {}
        for l in sorted(by_c[ci]):
            v = Xc[:, by_c[ci][l]].mean()
            layer_means[l] = v
            rowstr.append(f"L{l}:{v:9.1f}")
        vals = np.array(list(layer_means.values()))
        within = np.mean([Xc[:, by_c[ci][l]].std() for l in by_c[ci]])
        between = vals.std()
        print(f"    c{ci:2d}  {'  '.join(rowstr)}   between-layer std={between:9.1f}  within-layer std={within:7.1f}  ratio={between/max(within,1e-9):6.1f}")
