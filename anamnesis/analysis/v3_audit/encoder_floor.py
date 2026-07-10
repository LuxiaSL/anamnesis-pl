"""Track B1 — per-model encoder FLOOR (Gate B), supervised ceiling-check.

The geometry says the how-axis is low-dim, low-rank (~3 dirs), curved, ~linearly-decodable -> the
encoder is a CEILING-CHECK, not expected to open a wide gap. This measures it on the SAME sampled-hidden
surface the contrastive baby-encoder used (gate_a_lf_cache_{model}_merged.npz: H n x 25 x hidden ->
flatten), so it is apples-to-apples vs that baby-encoder (leak-free RF-resid 3B 92.6 / 8B 96.6).

Protocol (isomorphic to the Gate-A / linear-probe batteries):
  - 5-way hard modes, GroupKFold by TOPIC -> held-out topics AND prompts AND seeds (all 3 reps of a
    test topic are in test; the encoder never saw that prompt). Hostile leak-proof split; encoder
    retrained per fold (the contrastive leak is the cautionary tale).
  - per-fold length residualization on [prompt_len, gen_len] + StandardScaler, all train-fit -> directly
    comparable to the residualized hand floor (RF 90.9/92.5), LDA (87/92), logistic-on-this-surface (89/92).
  - classification objective (NOT metric — distances overfit topics; directions generalize).
  - LEARNING CURVE: subsample train TOPICS to {0.25, 0.5, 1.0}. Plateau below/at floor -> info-limited.
  - arch: 'logit' (linear floor) vs 'deep' (P->256->K->5 nonlinear encoder). premium = deep - logit.

OPTIMIZER (pinned by encoder_diag.py 2026-06-14): torch AdamW UNDER-converges the convex linear problem
(logit AdamW 82.6 vs sklearn 89.2). So **logit uses LBFGS** (-> 89.7, matches sklearn) and **deep uses
AdamW** (-> 91.2, fine for the nonlinear net). Both train on the FULL (sub)fold (no inner-val handicap;
sklearn doesn't hold one out either) -> a fair logit-vs-deep premium and a trustworthy linear floor.

Per-model, never pooled. Needs a GPU. Run on node1:
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python encoder_floor.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "8")
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HARD = {"linear", "socratic", "contrastive", "dialectical", "analogical"}
ROOT = os.environ.get("ANAMNESIS_PL", ".")
K = 32                      # bottleneck (sized from intrinsic dim ~28-30)
SEEDS = 3
FRACS = [0.25, 0.5, 1.0]
ARCHS = ["logit", "deep"]
DEEP_EPOCHS = 2500
LBFGS_L2 = 1e-3
# comparison bars (n≈900, 5-way hard, GroupKFold-by-topic, residualized) from the prior batteries
BARS = {
    "3b": {"hand_floor_RF": 0.909, "baby_encoder_RF": 0.926, "LDA": 0.87, "logistic_surface": 0.89},
    "8b": {"hand_floor_RF": 0.925, "baby_encoder_RF": 0.966, "LDA": 0.92, "logistic_surface": 0.92},
}


def load_cache(model):
    z = np.load(os.path.join(ROOT, f"gate_a_lf_cache_{model}_merged.npz"), allow_pickle=True)
    H, mode, topic = z["H"], z["mode"].astype(str), z["topic"].astype(int)
    plen, glen = z["plen"].astype(float), z["glen"].astype(float)
    hard = np.array([m in HARD for m in mode])
    X = np.nan_to_num(H[hard].reshape(int(hard.sum()), -1).astype(np.float64))
    return X, mode[hard], topic[hard], np.column_stack([plen[hard], glen[hard]])


def residualize(Ftr, Fte, Ctr, Cte):
    A = np.hstack([Ctr, np.ones((len(Ctr), 1))]); B = np.hstack([Cte, np.ones((len(Cte), 1))])
    coef, *_ = np.linalg.lstsq(A, Ftr, rcond=None)
    return Ftr - A @ coef, Fte - B @ coef


class Encoder(nn.Module):
    def __init__(self, P, arch, nclass=5, p_drop=0.4):
        super().__init__()
        if arch == "logit":                                # pure linear floor (trained with LBFGS)
            self.net = nn.Linear(P, nclass)
        elif arch == "deep":                               # nonlinear encoder, K-bottleneck (AdamW)
            self.net = nn.Sequential(
                nn.Linear(P, 256), nn.ReLU(), nn.Dropout(p_drop),
                nn.Linear(256, K), nn.ReLU(), nn.Linear(K, nclass))
        else:
            raise ValueError(arch)

    def forward(self, x):
        return self.net(x)


def train_eval(Xtr, ytr, Xte, yte, arch, seed, device):
    """Train on the FULL (sub)fold. logit->LBFGS (convex; AdamW under-converges it), deep->AdamW."""
    torch.manual_seed(seed)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.long, device=device)
    Xe = torch.tensor(Xte, dtype=torch.float32, device=device)
    net = Encoder(Xtr.shape[1], arch).to(device)
    ce = nn.CrossEntropyLoss()
    if arch == "logit":
        opt = torch.optim.LBFGS(net.parameters(), max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = ce(net(Xt), yt) + LBFGS_L2 * sum((p ** 2).sum() for p in net.parameters())
            loss.backward()
            return loss

        opt.step(closure)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=5e-3, weight_decay=1e-2)
        for _ in range(DEEP_EPOCHS):
            net.train()
            opt.zero_grad()
            ce(net(Xt), yt).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return float((net(Xe).argmax(1).cpu().numpy() == yte).mean())


def subsample_topics(tr, topic, frac, seed):
    if frac >= 1.0:
        return tr
    utop = np.unique(topic[tr])
    rng = np.random.default_rng(1000 + seed)
    keep = set(rng.choice(utop, max(2, int(round(frac * len(utop)))), replace=False).tolist())
    return tr[np.array([topic[i] in keep for i in tr])]


def cv(X, yi, topic, C, arch, frac, resid, device):
    accs = []
    for seed in range(SEEDS):
        for tr, te in GroupKFold(5).split(X, yi, topic):
            tr2 = subsample_topics(tr, topic, frac, seed)
            Xtr, Xte = X[tr2].copy(), X[te].copy()
            if resid:
                Xtr, Xte = residualize(Xtr, Xte, C[tr2], C[te])
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            accs.append(train_eval(Xtr, yi[tr2], Xte, yi[te], arch, seed, device))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", default="3b,8b")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"device={device}  torch={torch.__version__}", flush=True)
    out = {}
    for model in args.models.split(","):
        X, mode, topic, C = load_cache(model)
        labels = sorted(set(mode.tolist()))
        yi = np.array([labels.index(m) for m in mode])
        b = BARS[model]
        print(f"\n===== {model}  n={len(yi)}  topics={len(set(topic.tolist()))}  P={X.shape[1]}  "
              f"(chance 20%) =====", flush=True)
        print("  bars: " + "  ".join(f"{k}={v:.1%}" for k, v in b.items()), flush=True)
        res = {}
        print("  --- resid (learning curve) ---", flush=True)
        for arch in ARCHS:
            for frac in FRACS:
                m, s = cv(X, yi, topic, C, arch, frac, True, device)
                res[f"resid.{arch}.frac{frac}"] = {"acc": round(m, 4), "std": round(s, 4)}
                print(f"    {arch:5s} frac={frac:<4} acc={m:.1%} ± {s:.1%}", flush=True)
        print("  --- raw bracket (frac1.0) ---", flush=True)
        for arch in ARCHS:
            m, s = cv(X, yi, topic, C, arch, 1.0, False, device)
            res[f"raw.{arch}.frac1.0"] = {"acc": round(m, 4), "std": round(s, 4)}
            print(f"    {arch:5s} raw       acc={m:.1%} ± {s:.1%}", flush=True)
        prem = res["resid.deep.frac1.0"]["acc"] - res["resid.logit.frac1.0"]["acc"]
        out[model] = {"bars": b, "results": res, "nonlinear_premium_resid": round(prem, 4)}
        print(f"  nonlinear premium (deep-logit, resid, full) = {prem:+.1%}", flush=True)
    json.dump(out, open("encoder_floor_results.json", "w"), indent=2)
    print("\nWrote encoder_floor_results.json", flush=True)


if __name__ == "__main__":
    main()
