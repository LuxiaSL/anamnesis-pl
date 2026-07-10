"""Gap-decomposition ladder — WHERE the hand↔raw gap lives, per surface (chad 1, the curation diagnostic).

The encoder floors showed hand-features massively under-extract vs raw on keys (50→82) and values
(—→86). Two distinct losses are conflated in that gap:
  (1) GEOMETRY-SUMMARIZATION — hand-features reduce a 128-d key/value vector to a handful of scalars
      (spread/drift/novelty/eff_dim). How much signal does that summary throw away?
  (2) HEAD-AVERAGING — both the hand-features AND this ladder's rung-2 average the 8 KV heads before
      anything; the raw floor keeps per-head. The audit flagged head-averaging as the #1 destruction point.

A 3-rung representation ladder splits them (same leak-proof, length-residualized, lossless Gram-reduce
floor as surface_encoder_floor — logit = the linear floor):
  rung HAND      = old hand-LDA bar / hand-family k4               (a few scalars)   [floor json / hand_family]
  rung HEADMEAN  = the 128-d head-MEAN vector, per layer/pos, flattened              [computed here, GPU]
  rung PERHEAD   = the full per-head vectors = the raw surface floor                  [surface_floor json]
  → HAND→HEADMEAN gap = geometry-summarization loss;  HEADMEAN→PERHEAD gap = head-averaging loss.

Reuses the floor's GPU preprocessing + logit so rungs are directly comparable. keys + values (both
all-layer per-KV-head, cache layout [pos, layer, kv_head, head_dim]). node1, one GPU:
    SURFACE_FLOOR_DIR=.../floor_results python gap_ladder.py --models 3b,8b --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

# reuse the EXACT floor machinery (this script's dir is sys.path[0] when run as a file)
from surface_encoder_floor import (
    HAND_LDA, OUTDIR, SEEDS, load_surface, preprocess_fold_gpu, train_eval,
)

N_POS = 5
HEAD_DIM = 128
KV_HEADS = 8


def headmean_rep(X: np.ndarray) -> np.ndarray:
    """Per-head cache row [pos, layer, kv_head, head_dim] (C-order flat) -> head-MEAN over the 8 KV heads,
    re-flattened to [pos, layer, head_dim]. L inferred from P = N_POS*L*KV_HEADS*HEAD_DIM."""
    P = X.shape[1]
    L = P // (N_POS * KV_HEADS * HEAD_DIM)
    assert L * N_POS * KV_HEADS * HEAD_DIM == P, f"unexpected P={P} (L={L})"
    Xr = X.reshape(X.shape[0], N_POS, L, KV_HEADS, HEAD_DIM)
    return Xr.mean(axis=3).reshape(X.shape[0], N_POS * L * HEAD_DIM).astype(np.float32)


def logit_floor(X, yi, topic, C, device):
    """Leak-proof GroupKFold logit floor (resid=True), reusing the floor's GPU Gram-reduce + LBFGS logit."""
    accs = []
    for seed in range(SEEDS):
        for tr, te in GroupKFold(5).split(X, yi, topic):
            Ztr, Zte = preprocess_fold_gpu(X[tr], X[te], C[tr], C[te], True, device)
            ta, _ = train_eval(Ztr, yi[tr], Zte, yi[te], "logit", seed, device)
            accs.append(ta)
    return float(np.mean(accs)), float(np.std(accs))


def load_floor_perhead(model, surface):
    p = os.path.join(OUTDIR, f"surface_floor_{model}_{surface}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get(surface, {}).get("resid.logit.frac1.0", {}).get("test")


def run(model, surface, device):
    X, mode, topic, C = load_surface(model, surface)
    labels = sorted(set(mode.tolist()))
    yi = np.array([labels.index(m) for m in mode])
    Xhm = headmean_rep(X)
    hm_mean, hm_std = logit_floor(Xhm, yi, topic, C, device)
    perhead = load_floor_perhead(model, surface)
    hand = HAND_LDA.get(model, {}).get(surface)
    res = {"surface": surface, "headmean_logit": round(hm_mean, 4), "headmean_std": round(hm_std, 4),
           "headmean_P": int(Xhm.shape[1]), "perhead_logit": perhead, "hand_LDA": hand}
    if perhead is not None:
        res["head_averaging_loss"] = round(perhead - hm_mean, 4)        # HEADMEAN -> PERHEAD
    if hand is not None:
        res["geometry_summarization_loss"] = round(hm_mean - hand, 4)   # HAND -> HEADMEAN
    print(f"  {model}/{surface}: hand {fmt(hand)} → headmean {hm_mean:.0%}±{hm_std:.0%} "
          f"(P={Xhm.shape[1]}) → perhead {fmt(perhead)}", flush=True)
    if hand is not None and perhead is not None:
        print(f"      geometry-summarization loss (hand→headmean) = {hm_mean - hand:+.0%}   "
              f"head-averaging loss (headmean→perhead) = {perhead - hm_mean:+.0%}", flush=True)
    return res


def fmt(x):
    return f"{x:.0%}" if x is not None else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="3b,8b")
    ap.add_argument("--surfaces", default="keys,values")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"device={device}  OUTDIR={OUTDIR}", flush=True)
    out = {}
    for model in args.models.split(","):
        model = model.strip()
        print(f"\n===== {model} =====", flush=True)
        out[model] = {}
        for surface in args.surfaces.split(","):
            surface = surface.strip()
            try:
                out[model][surface] = run(model, surface, device)
            except FileNotFoundError:
                print(f"  [skip] {model}/{surface}: cache missing", flush=True)
    json.dump(out, open(os.path.join(OUTDIR, "gap_ladder_results.json"), "w"), indent=2)
    print(f"\nWrote {os.path.join(OUTDIR, 'gap_ladder_results.json')}", flush=True)


if __name__ == "__main__":
    main()
