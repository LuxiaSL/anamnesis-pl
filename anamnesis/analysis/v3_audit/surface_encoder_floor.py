"""Per-surface encoder FLOOR — the all-surface discovery (finishes Gate B honestly).

The Gate-B encoder (`encoder_floor.py`) only ever read the RESIDUAL stream. This runs the SAME
leak-proof protocol on EACH banked surface's cache (`build_surface_caches.py`), so we get each
source's learned ceiling — decompose-by-source at the *encoder* level. Top question: does any
non-residual surface — especially **values, never featurized at all** — carry mode signal the
residual encoder missed? And does a learned method beat hand-features per surface (method-limited)?

Protocol (identical to encoder_floor.py for comparability):
  5-way hard modes, GroupKFold by TOPIC (held-out topics+prompts+seeds), per-fold length-residualize
  on [prompt_len, gen_len] + StandardScaler, classification objective, learning curve (train-topic
  frac 0.25/0.5/1.0). arch: logit (LBFGS — convex linear floor) + deep (AdamW — nonlinear ceiling).

Per-surface JSON out (so GPU-parallel jobs don't clobber): surface_floor_{model}_{surface}.json.
Run on node1 (one surface-group per GPU to fan out):
    CUDA_VISIBLE_DEVICES=0 python surface_encoder_floor.py --model 8b --surfaces residual,keys,values --device cuda:0
    CUDA_VISIBLE_DEVICES=1 python surface_encoder_floor.py --model 8b --surfaces queries,gate,attention --device cuda:1
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

CACHE = os.environ.get("SURFACE_CACHE_DIR", "/dev/shm/anamnesis_surface_caches")
OUTDIR = os.environ.get("SURFACE_FLOOR_DIR", ".")
K = 32
SEEDS = 3
FRACS = [0.25, 0.5, 1.0]
ARCHS = ["logit", "deep"]
DEEP_EPOCHS = 800       # converges fast on the well-conditioned ~720-dim reduced input (was 2500 for raw-wide)
LBFGS_L2 = 1e-3
ALL_SURFACES = ["residual", "keys", "values", "queries", "gate", "attention"]

# hand-feature comparison bars (source-marginal LDA, n≈900, 5-way, resid; from source-method-reframe).
# None = surface was NEVER hand-featurized (the discovery cases). baby-enc/hand-RF are residual-surface refs.
HAND_LDA = {
    "3b": {"residual": 0.82, "keys": 0.50, "gate": 0.69, "attention": 0.88, "values": None, "queries": None},
    "8b": {"residual": 0.88, "keys": 0.61, "gate": 0.76, "attention": 0.93, "values": None, "queries": None},
}
RESID_REFS = {"3b": {"baby_encoder_RF": 0.926, "hand_RF": 0.909},
              "8b": {"baby_encoder_RF": 0.966, "hand_RF": 0.925}}


def load_surface(model: str, surface: str):
    z = np.load(os.path.join(CACHE, f"surface_cache_{model}_{surface}.npz"), allow_pickle=True)
    X = np.nan_to_num(z["X"].astype(np.float32))
    mode = z["mode"].astype(str)
    topic = z["topic"].astype(int)
    C = np.column_stack([z["plen"].astype(float), z["glen"].astype(float)])
    return X, mode, topic, C


def preprocess_fold_gpu(Xtr_np, Xte_np, Ctr, Cte, resid, device, eps=1e-8):
    """ALL wide preprocessing on the GPU, computed ONCE per fold (shared by both archs): residualize on
    [prompt_len, gen_len] → per-feature standardize (= the raw floor) → lossless row-space reduce (Gram
    trick → exact sample coords; P~1e5 → ~n_tr). Returns the small reduced (Ztr, Zte) for the encoder.

    Replaces the CPU residualize + sklearn StandardScaler (float64 churn) + repeated host↔GPU transfers
    that made the floor latency-bound. Faithful to the CPU path up to float32 vs float64 (negligible);
    logit on Z = the raw floor (the L2 solution lies in the train row space). A single global scale
    conditions the deep WITHOUT per-PC whitening (whitening amplifies noise → hurts)."""
    Xtr = torch.tensor(Xtr_np, dtype=torch.float32, device=device)
    Xte = torch.tensor(Xte_np, dtype=torch.float32, device=device)
    if resid:
        A = torch.tensor(np.hstack([Ctr, np.ones((len(Ctr), 1))]), dtype=torch.float32, device=device)
        B = torch.tensor(np.hstack([Cte, np.ones((len(Cte), 1))]), dtype=torch.float32, device=device)
        coef = torch.linalg.lstsq(A, Xtr).solution
        Xtr = Xtr - A @ coef
        Xte = Xte - B @ coef
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp_min(1e-8)
    Xtr = (Xtr - mu) / sd                       # per-feature standardize (centers + scales) — as raw floor
    Xte = (Xte - mu) / sd
    Ktr = Xtr @ Xtr.T                           # (n_tr, n_tr) Gram on GPU (standardized features)
    w, U = torch.linalg.eigh(Ktr)
    keep = w > eps * w.max().clamp_min(1e-30)
    w, U = w[keep], U[:, keep]
    s = torch.sqrt(w.clamp_min(1e-12))
    Ztr = U * s                                 # (n_tr, r) exact train coordinates
    Zte = (Xte @ Xtr.T @ U) / s                 # (n_te, r) test coords via cross-Gram (no P-wide V)
    g = Ztr.std().clamp_min(1e-8)
    return (Ztr / g).cpu().numpy(), (Zte / g).cpu().numpy()


class Encoder(nn.Module):
    def __init__(self, P, arch, nclass=5, p_drop=0.4):
        super().__init__()
        if arch == "logit":
            self.net = nn.Linear(P, nclass)
        elif arch == "deep":
            self.net = nn.Sequential(
                nn.Linear(P, 256), nn.ReLU(), nn.Dropout(p_drop),
                nn.Linear(256, K), nn.ReLU(), nn.Linear(K, nclass))
        else:
            raise ValueError(arch)

    def forward(self, x):
        return self.net(x)


def train_eval(Xtr, ytr, Xte, yte, arch, seed, device):
    """logit -> LBFGS (convex; AdamW under-converges it), deep -> AdamW. Trained on the full (sub)fold."""
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
            net.train(); opt.zero_grad()
            ce(net(Xt), yt).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        test_acc = float((net(Xe).argmax(1).cpu().numpy() == yte).mean())
        train_acc = float((net(Xt).argmax(1).cpu().numpy() == ytr).mean())
    return test_acc, train_acc


def subsample_topics(tr, topic, frac, seed):
    if frac >= 1.0:
        return tr
    utop = np.unique(topic[tr])
    rng = np.random.default_rng(1000 + seed)
    keep = set(rng.choice(utop, max(2, int(round(frac * len(utop)))), replace=False).tolist())
    return tr[np.array([topic[i] in keep for i in tr])]


def cv_both(X, yi, topic, C, frac, resid, device):
    """Per (frac): preprocess each fold ONCE on the GPU (shared), then run BOTH archs on it — dedupes the
    wide preprocessing that was the floor's latency bottleneck. Returns {arch: (test_mean, test_std, train_mean)}."""
    acc = {a: ([], []) for a in ARCHS}
    for seed in range(SEEDS):
        for tr, te in GroupKFold(5).split(X, yi, topic):
            tr2 = subsample_topics(tr, topic, frac, seed)
            Ztr, Zte = preprocess_fold_gpu(X[tr2], X[te], C[tr2], C[te], resid, device)
            for arch in ARCHS:
                ta, tra = train_eval(Ztr, yi[tr2], Zte, yi[te], arch, seed, device)
                acc[arch][0].append(ta); acc[arch][1].append(tra)
    return {a: (float(np.mean(acc[a][0])), float(np.std(acc[a][0])), float(np.mean(acc[a][1]))) for a in ARCHS}


def run_surface(model, surface, device):
    X, mode, topic, C = load_surface(model, surface)
    labels = sorted(set(mode.tolist()))
    yi = np.array([labels.index(m) for m in mode])
    hand = HAND_LDA.get(model, {}).get(surface)
    print(f"\n===== {model} / {surface}  n={len(yi)}  topics={len(set(topic.tolist()))}  "
          f"P={X.shape[1]}  hand_LDA={hand}  (chance 20%) =====", flush=True)
    res = {"n": int(len(yi)), "P": int(X.shape[1]), "hand_LDA": hand, "resid_refs": RESID_REFS.get(model)}
    print("  --- resid (learning curve; test / train) ---", flush=True)
    for frac in FRACS:
        both = cv_both(X, yi, topic, C, frac, True, device)      # preprocess once/fold, both archs
        for arch in ARCHS:
            m, s, tr = both[arch]
            res[f"resid.{arch}.frac{frac}"] = {"test": round(m, 4), "std": round(s, 4), "train": round(tr, 4)}
            print(f"    {arch:5s} frac={frac:<4} test={m:.1%} ± {s:.1%}  (train {tr:.1%})", flush=True)
    both_raw = cv_both(X, yi, topic, C, 1.0, False, device)
    for arch in ARCHS:
        m, s, tr = both_raw[arch]
        res[f"raw.{arch}.frac1.0"] = {"test": round(m, 4), "std": round(s, 4), "train": round(tr, 4)}
        print(f"    {arch:5s} raw       test={m:.1%} ± {s:.1%}  (train {tr:.1%})", flush=True)
    prem = res["resid.deep.frac1.0"]["test"] - res["resid.logit.frac1.0"]["test"]
    res["nonlinear_premium_resid"] = round(prem, 4)
    print(f"  nonlinear premium (deep-logit, resid, full) = {prem:+.1%}", flush=True)
    out_path = os.path.join(OUTDIR, f"surface_floor_{model}_{surface}.json")
    json.dump({surface: res}, open(out_path, "w"), indent=2)
    print(f"  wrote {out_path}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["3b", "8b"])
    ap.add_argument("--surfaces", default=",".join(ALL_SURFACES))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"device={device}  torch={torch.__version__}  cache={CACHE}", flush=True)
    surfaces = [s.strip() for s in args.surfaces.split(",") if s.strip()]
    for surface in surfaces:
        try:
            run_surface(args.model, surface, device)
        except FileNotFoundError:
            print(f"  [skip] {args.model}/{surface}: cache not found", flush=True)


if __name__ == "__main__":
    main()
