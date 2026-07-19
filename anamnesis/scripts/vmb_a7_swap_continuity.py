"""14w CONTINUITY-THROUGH-SWAPS cell (ADDENDUM 2026-07-18w) — token-resolved swap-event detector.

Luxia's original MoE hypothesis at its TEMPORAL level: expert SWAPS disrupt the ongoing
computation — the signature breaks AT the swap — and what makes an MoE feel good is how much of
the computational fingerprint survives the swap bottleneck. The A7 dense-backbone clause missed at
the DISPLACEMENT level (carrying, not continuity); this tests continuity directly.

Token-resolved: for each teacher-forced generation (content identical by construction), a single
forward per rung captures, per generated token t (vs t-1):
  CONTINUITY (routing-independent):
    resid_velocity  = mean over resid-layers of (1 - cos(h_t, h_{t-1}))        [scale-free]
    attn_regime     = ||s_t - s_{t-1}||_1,  s = [entropy, sink_mass, recency_mass] at attn-layer
  SWAP intensity (routing):
    top1_switch     = frac of MoE layers whose argmax routed-expert changed t-1 -> t  [k-independent]
    set_churn       = frac symmetric-difference of the top-k routed-expert SET (rung's own k)
Router softmax recomputed exactly as extraction (model_loader _make_moe_router_prehook:
F.linear(h.float(), gate.weight.float()).softmax(-1)).

RUNGS (frozen, charter Part D): baseline (identity; = noise eps0) · shared_ablate (Leg 2 — does
removing the backbone amplify swap-locked discontinuity) · topk2 (the SWAP-CONTRAST rung — the
k-ladder causal handle: routing/swaps shift at MATCHED teacher-forced positions, so a swap->disc
link that survives is causal, not a content-boundary confound).

READOUTS:
  Leg 1 (P .65): swap events discontinuity-locked ABOVE the matched-token null. Matched null =
    same token_id + nearby position (the position-tracking confound rider). Reported as (a) event
    contrast high-vs-matched-low-swap with within-stratum permutation p, and (b) within-token-id-
    and-position-demeaned partial correlation swap<->discontinuity.
  Confound rider (causal): baseline vs topk2 at MATCHED (gen,t): corr(Δswap, Δdisc). Position is
    identical by construction, so a positive link is swap-locked, not position-locked.
  Leg 2 (P .55): shared_ablate amplifies swap-locked discontinuity vs baseline (same swaps — ablate
    does not change routing — backbone removed => bigger seams): coupling slope + at-baseline-swap disc.

Saves per-token arrays (npz, re-analyzable on CPU) + the analysis json. First-read -> outer loop; UNSTAMPED.

    python -m anamnesis.scripts.vmb_a7_swap_continuity --model dsv2-lite --model-path $MP \
        --manifest $RUNS/vmb_stage0_dsv2_lite/replay_manifest.json --n-gens 80 \
        --resid-layers 11,15,18 --attn-layer 15 --recency-w 32 \
        --out-npz $BANK/arms/A7_dsv2/swap14w/pertoken.npz \
        --out-json $BANK/arms/A7_dsv2/swap14w/swap_continuity.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from anamnesis.config import MODEL_PRESETS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("swap14w")

# frozen rungs: label -> perturb spec (None = identity baseline)
RUNGS: dict[str, dict | None] = {
    "baseline": None,
    "shared_ablate": {"mode": "shared_ablate"},
    "topk2": {"mode": "topk", "top_k": 2},
}


def _moe_layers(model):
    out = []
    for i, lyr in enumerate(model.model.layers):
        mlp = getattr(lyr, "mlp", None)
        if mlp is not None and hasattr(mlp, "gate") and hasattr(mlp.gate, "weight"):
            out.append((i, mlp))
    return out


class RouterCapture:
    """forward_pre_hook per MoE layer: store dense pre-topk router softmax [seq, n_experts]."""
    def __init__(self, moe):
        self.store: dict[int, torch.Tensor] = {}
        self.handles = []
        for l_idx, mlp in moe:
            self.handles.append(mlp.register_forward_pre_hook(self._mk(l_idx)))

    def _mk(self, l_idx):
        def hook(module, args):
            h = args[0]
            if h.dim() == 3:
                h = h[0]
            logits = F.linear(h.to(torch.float32), module.gate.weight.to(torch.float32))
            self.store[l_idx] = logits.softmax(dim=-1).detach().cpu()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def _attn_summary(attn_layer_weights: torch.Tensor, recency_w: int) -> np.ndarray:
    """attn_layer_weights: [heads, q, k] for one gen. Return [q, 3] = entropy, sink_mass, recency_mass
    (mean over heads), per query token. Vectorized + device-consistent."""
    a = attn_layer_weights.to(torch.float32).mean(0)   # [Q, K] mean over heads
    dev = a.device
    Q, K = a.shape
    eps = 1e-9
    ent = (-(a * (a + eps).log()).sum(-1)) / float(np.log(max(K, 2)))   # [Q], normalized
    sink = a[:, 0]                                      # mass on position 0
    # recency mass = sum over the last recency_w keys up to the query position (causal rows sum to 1)
    cs = a.cumsum(dim=-1)                               # [Q,K]; cs[qi,qi] == 1
    qi = torch.arange(Q, device=dev)
    low = qi - recency_w                                # first excluded key index
    gathered = cs[qi, low.clamp(min=0)]                 # cs at the boundary (clamped)
    rec = torch.where(low >= 0, 1.0 - gathered, torch.ones(Q, device=dev))
    return torch.stack([ent, sink, rec], dim=-1).cpu().numpy()   # [Q,3]


def _per_token(model, ids, P, moe, resid_layers, attn_layer, recency_w, rung_k):
    """One forward; return dict of per-generated-token arrays (aligned t = P..T-1, vs t-1)."""
    dev = next(model.parameters()).device
    cap = RouterCapture(moe)
    with torch.no_grad():
        out = model(ids.to(dev), use_cache=False, output_hidden_states=True,
                    output_attentions=True, return_dict=True)
    T = ids.shape[1]
    # --- continuity: resid velocity (cosine) over resid_layers ---
    hs = out.hidden_states                              # tuple len n_layers+1, [1, T, d]
    vel = np.zeros(T)
    for L in resid_layers:
        h = hs[L][0].to(torch.float32)                 # [T, d]
        cos = F.cosine_similarity(h[1:], h[:-1], dim=-1).cpu().numpy()  # [T-1] cos(t, t-1)
        vel[1:] += (1.0 - cos)
    vel[1:] /= max(len(resid_layers), 1)
    # --- attn regime shift ---
    aw = out.attentions[attn_layer][0]                 # [H, T, T]
    s = _attn_summary(aw, recency_w)                   # [T, 3]
    regime = np.zeros(T)
    regime[1:] = np.abs(s[1:] - s[:-1]).sum(-1)
    # --- swaps from router softmax ---
    layers = sorted(cap.store.keys())
    top1 = np.stack([cap.store[l].argmax(-1).numpy() for l in layers], axis=0)   # [n_moe, T]
    # top-k sets (rung's own k)
    k = rung_k
    setk = {l: cap.store[l].topk(k, dim=-1).indices.numpy() for l in layers}     # [T, k]
    cap.remove()
    top1_switch = np.zeros(T)
    set_churn = np.zeros(T)
    for li, l in enumerate(layers):
        sw = (top1[li, 1:] != top1[li, :-1]).astype(np.float64)
        top1_switch[1:] += sw
        st = setk[l]
        for t in range(1, T):
            a, b = set(st[t].tolist()), set(st[t - 1].tolist())
            set_churn[t] += len(a ^ b) / (2.0 * k)
    top1_switch[1:] /= len(layers)
    set_churn[1:] /= len(layers)
    # slice to generated span (t = P .. T-1); token id at t
    tok = ids[0].cpu().numpy()
    gen = slice(P, T)
    return {"pos": np.arange(P, T), "token_id": tok[P:T],
            "resid_velocity": vel[gen], "attn_regime": regime[gen],
            "top1_switch": top1_switch[gen], "set_churn": set_churn[gen]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dsv2-lite")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--n-gens", type=int, default=80)
    ap.add_argument("--gen-ids", default=None, help="comma-separated; overrides --n-gens")
    ap.add_argument("--resid-layers", default="11,15,18")
    ap.add_argument("--attn-layer", type=int, default=15)
    ap.add_argument("--recency-w", type=int, default=32)
    ap.add_argument("--out-npz", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    resid_layers = [int(x) for x in args.resid_layers.split(",")]

    from transformers import AutoModelForCausalLM
    from anamnesis.extraction.model_loader import MoEPerturbSpec, attach_moe_perturbation
    preset = MODEL_PRESETS[args.model]
    logger.info(f"loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=preset.torch_dtype, attn_implementation="eager").to("cuda").eval()
    moe = _moe_layers(model)
    logger.info(f"{len(moe)} MoE layers")

    entries = json.loads(args.manifest.read_text())["entries"]
    keys = sorted(entries, key=lambda k: int(k))
    if args.gen_ids:
        want = [k for k in keys if int(k) in {int(x) for x in args.gen_ids.split(",")}]
    else:
        want = keys[:args.n_gens]

    rung_k = {"baseline": 6, "shared_ablate": 6, "topk2": 2}
    data: dict[str, dict[str, list]] = {}
    for rung, spec in RUNGS.items():
        handle = None
        if spec is not None:
            handle = attach_moe_perturbation(model, MoEPerturbSpec(
                mode=spec["mode"], top_k=spec.get("top_k"), seed=int(spec.get("seed", 0))))
        cols: dict[str, list] = {c: [] for c in
                                 ("gen", "pos", "token_id", "resid_velocity",
                                  "attn_regime", "top1_switch", "set_churn")}
        for gi, k in enumerate(want):
            e = entries[k]
            ids = torch.tensor([e["input_ids"]], dtype=torch.long)
            P = int(e["prompt_length"])
            if ids.shape[1] - P < 3:
                continue
            r = _per_token(model, ids, P, moe, resid_layers, args.attn_layer,
                           args.recency_w, rung_k[rung])
            n = len(r["pos"])
            cols["gen"].extend([int(k)] * n)
            for c in ("pos", "token_id", "resid_velocity", "attn_regime", "top1_switch", "set_churn"):
                cols[c].extend(r[c].tolist())
            if (gi + 1) % 20 == 0:
                logger.info(f"  {rung}: {gi + 1}/{len(want)}")
        if handle is not None:
            handle.remove()
        data[rung] = {c: np.asarray(v) for c, v in cols.items()}
        logger.info(f"{rung}: {len(data[rung]['pos'])} generated tokens over {len(want)} gens")

    # save per-token arrays
    flat = {}
    for rung, cols in data.items():
        for c, v in cols.items():
            flat[f"{rung}__{c}"] = v
    np.savez_compressed(args.out_npz, **flat)

    # ── analysis: delegate to the fast standalone detector (single source of truth; the inline
    # within-stratum permutation was a perf trap — vmb_a7_swap_analyze uses vectorized demean +
    # asymptotic p and reads the npz we just saved) ──
    from anamnesis.scripts.vmb_a7_swap_analyze import run_analysis
    res = run_analysis(args.out_npz, args.out_json, posbin=16)
    print(json.dumps(res, indent=1))
    print(f"\nper-token -> {args.out_npz}\nanalysis -> {args.out_json}")


if __name__ == "__main__":
    main()
