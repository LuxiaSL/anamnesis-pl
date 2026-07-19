"""MoE expert-usage IDENTITY histogram sidecar (M6 DeepSeek-V2-Lite class; imago-queue item).

The per-expert selection frequency is a model's expert-PREFERENCE fingerprint — WHICH experts it
likes, a WHAT/content-identity signal, hereditary under fine-tune. It is explicitly NOT a HOW-processing
signal and MUST NEVER enter the how-vector (the signature). This banks it as a lineage-stamped sidecar
so the identity channel is available for provenance/heredity work without contaminating the how-axis.

Reads dense router_dist [T, L, n_experts] from raw_tensors_v3. Two usage views: SOFT (mean routing
mass per expert) and HARD (top-k selection frequency, k = the model's num_experts_per_tok). Pooled +
per-layer, with uniformity diagnostics (entropy, Gini, top/bottom experts).

Run (node1, CPU): python -m anamnesis.scripts.vmb_identity_histogram \
  --run-dir /models/anamnesis-extract/runs/vmb_stage0_dsv2_lite --model dsv2-lite --top-k 6 --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _gini(x: np.ndarray) -> float:
    x = np.sort(x.astype(np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--raw-subdir", default="raw_tensors_v3")
    ap.add_argument("--top-k", type=int, default=6, help="num_experts_per_tok (hard-usage k)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(str(args.run_dir / args.raw_subdir / "*.npz")))
    if not files:
        raise SystemExit(f"no raw npz under {args.run_dir / args.raw_subdir}")

    soft_sum = None            # [L, E] accumulated mean-mass
    hard_counts = None         # [L, E] top-k selection counts
    layer_idx = None
    n_tokens = 0
    n_gens = 0
    for f in files:
        z = np.load(f, allow_pickle=True)
        if "router_dist" not in z.files:
            continue
        rd = np.asarray(z["router_dist"], dtype=np.float64)   # [T, L, E]
        if rd.ndim != 3 or rd.size == 0:
            continue
        T, L, E = rd.shape
        if soft_sum is None:
            soft_sum = np.zeros((L, E)); hard_counts = np.zeros((L, E))
            layer_idx = [int(i) for i in z["router_layer_indices"]] if "router_layer_indices" in z.files else list(range(L))
        soft_sum += rd.sum(0)                                  # sum mass over tokens
        k = min(args.top_k, E)
        top = np.argpartition(-rd, k - 1, axis=-1)[..., :k]    # [T, L, k] selected experts
        for li in range(L):
            idx, cnt = np.unique(top[:, li, :], return_counts=True)
            hard_counts[li, idx] += cnt
        n_tokens += T
        n_gens += 1

    if soft_sum is None:
        raise SystemExit("no router_dist found in any npz")
    L, E = soft_sum.shape
    soft = soft_sum / max(n_tokens, 1)                         # per-layer mean mass [L, E]
    soft_pooled = soft.mean(0)                                 # [E]
    hard = hard_counts / max(n_tokens, 1)                      # per-layer selections/token [L, E]
    hard_pooled = hard_counts.sum(0)
    hard_pooled = hard_pooled / max(hard_pooled.sum(), 1)      # normalized selection freq [E]

    order = np.argsort(-hard_pooled)
    out = {
        "sidecar": "MoE expert-usage IDENTITY histogram",
        "LINEAGE": {"model": args.model, "run_dir": str(args.run_dir), "source": "router_dist (dense "
                    "softmax over experts, raw_tensors_v3)", "n_gens": n_gens, "n_tokens": n_tokens,
                    "n_layers_moe": L, "n_experts": E, "top_k": args.top_k,
                    "moe_layer_indices": layer_idx},
        "WARNING": "IDENTITY channel (expert preference = WHAT/content, hereditary). NEVER include in "
                   "the how-vector / signature — this is not a HOW-processing feature.",
        "hard_usage_pooled": {  # normalized top-k selection frequency per expert
            "freq": [round(float(x), 5) for x in hard_pooled],
            "entropy_nats": round(_entropy(hard_pooled), 4),
            "entropy_frac_of_uniform": round(_entropy(hard_pooled) / np.log(E), 4),
            "gini": round(_gini(hard_pooled), 4),
            "top5_experts": [(int(i), round(float(hard_pooled[i]), 5)) for i in order[:5]],
            "bottom5_experts": [(int(i), round(float(hard_pooled[i]), 5)) for i in order[-5:]],
            "n_experts_unused": int((hard_pooled == 0).sum())},
        "soft_usage_pooled": {  # mean routing mass per expert
            "mass": [round(float(x), 5) for x in soft_pooled],
            "entropy_frac_of_uniform": round(_entropy(soft_pooled / soft_pooled.sum()) / np.log(E), 4)},
        "per_layer_hard_entropy_frac": [round(_entropy(hard[li] / max(hard[li].sum(), 1e-12)) / np.log(E), 4)
                                        for li in range(L)],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"identity histogram: {n_gens} gens / {n_tokens} tokens / {E} experts (top-{args.top_k})")
    print(f"  hard-usage entropy = {out['hard_usage_pooled']['entropy_frac_of_uniform']} of uniform, "
          f"gini = {out['hard_usage_pooled']['gini']}, unused = {out['hard_usage_pooled']['n_experts_unused']}")
    print(f"  top experts: {out['hard_usage_pooled']['top5_experts']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
