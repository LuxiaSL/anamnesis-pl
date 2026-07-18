"""ANNEX — control-surface tenancy pulses (PROPOSAL-control-surface-tenancy-2026-07-18,
desk-ratified with riders; fires on Luxia's word). V7-recipe verbatim; six per-token
OUTPUT-side functionals over the structural frames the roster enumerates:

  lexrarity  S = Σ_v p_t(v)·ℓ(v)          ℓ = log-unigram-freq (add-1) over token ids,
                                          built DETERMINISTICALLY from the stage-0
                                          manifest's own realized generated spans
                                          (params + counts hash stamped). THE V_lex
                                          MEMBER (rider 1: NOT "register"; Kreg untouched)
  copy       S = Σ_{v∈prompt} p_t(v)      prompt-frame mass (grounding/quoting dial)
  selfrep    S = Σ_{v∈gen<t, v∉prompt} p  self-only history mass (looping dial)
  tailmass   S = log Σ_{v∉top-50} p_t(v)  Rényi/top-p twin (detached top-k set)
  wraprate   S = cov_t(t, log p_t(EOS))   hazard SLOPE over the generated span (pacing)
  freqrep    S = Σ_v count_prior(v)·p_t(v) frequency-weighted rep (rider vs presence-style)
  varentropy S = Σ_v p·(s_v − H)²         second moment of surprisal (Celeste rider
                                          C-2, adopted 2026-07-17: the canonical second
                                          shape dimension — CS-6 predicts V7-collapse)

Keys banked as G{name}_L{site} -> cs_gradients.npz (+ stamps). Band-pass/⊥/leak
predictions happen at member-build time (annex_probe_members.py pattern; every ⊥ filing
carries cos-to-V7 as its leak prediction — the standing law, prospective).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch

from anamnesis.config import MODEL_PRESETS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EOS_IDS = {"3b": [128001, 128009], "8b": [128001, 128008, 128009]}
TOPK = 50


def build_logfreq(entries: dict, vocab_size: int) -> tuple[np.ndarray, str]:
    """Unigram log-frequency over token ids from the banked realized GENERATED spans
    (prompt tokens excluded — the model's own production statistics), add-1 smoothed.
    Deterministic given the manifest; sha256 of the count vector stamped."""
    counts = np.ones(vocab_size, dtype=np.float64)          # add-1
    for e in entries.values():
        ids = e["input_ids"][int(e["prompt_length"]):]
        np.add.at(counts, np.asarray(ids, dtype=np.int64), 1.0)
    logfreq = np.log(counts / counts.sum())
    return logfreq, hashlib.sha256(counts.tobytes()).hexdigest()[:16]


def s_terms_lexrarity(logits, ids, P, L, ctx):
    lf = ctx["logfreq_t"]
    return [torch.softmax(logits[t].float(), dim=-1) @ lf for t in range(P - 1, L - 1)]


def s_terms_copy(logits, ids, P, L, ctx):
    prompt_ids = torch.unique(ids[0, :P].detach())
    return [torch.softmax(logits[t].float(), dim=-1)[prompt_ids].sum()
            for t in range(P - 1, L - 1)]


def s_terms_selfrep(logits, ids, P, L, ctx):
    prompt_set = set(ids[0, :P].detach().tolist())
    seq = ids[0]
    terms = []
    for t in range(P - 1, L - 1):
        gen_prior = [int(v) for v in seq[P: t + 1].detach().tolist()
                     if int(v) not in prompt_set]
        if not gen_prior:
            continue
        vids = torch.tensor(sorted(set(gen_prior)), device=logits.device)
        terms.append(torch.softmax(logits[t].float(), dim=-1)[vids].sum())
    return terms


def s_terms_tailmass(logits, ids, P, L, ctx):
    terms = []
    for t in range(P - 1, L - 1):
        row = logits[t].float()
        top = torch.topk(row.detach(), TOPK).indices
        p = torch.softmax(row, dim=-1)
        terms.append(torch.log((1.0 - p[top].sum()).clamp_min(1e-12)))
    return terms


def s_terms_wraprate(logits, ids, P, L, ctx):
    """Span-level scalar: covariance of position with log p(EOS) — the hazard slope.
    Returned as a single 'term' (the recipe's mean over terms is then the scalar)."""
    eos_t = torch.tensor(ctx["eos"], device=logits.device)
    lps = []
    for t in range(P - 1, L - 1):
        row = torch.log_softmax(logits[t].float(), dim=-1)
        lps.append(torch.logsumexp(row[eos_t], dim=0))
    lp = torch.stack(lps)
    pos = torch.arange(len(lps), dtype=torch.float32, device=lp.device)
    pos = pos - pos.mean()
    return [(pos * (lp - lp.mean())).mean()]


def s_terms_freqrep(logits, ids, P, L, ctx):
    seq = ids[0]
    vocab = logits.shape[-1]
    terms = []
    counts = torch.zeros(vocab, device=logits.device)
    counts.index_add_(0, seq[:P].detach(), torch.ones(P, device=logits.device))
    for t in range(P - 1, L - 1):
        if t >= P:
            counts[seq[t].detach()] += 1.0
        p = torch.softmax(logits[t].float(), dim=-1)
        terms.append((counts * p).sum())
    return terms


def s_terms_varentropy(logits, ids, P, L, ctx):
    terms = []
    for t in range(P - 1, L - 1):
        logp = torch.log_softmax(logits[t].float(), dim=-1)
        p = logp.exp()
        s = -logp                       # surprisal
        H = (p * s).sum()
        terms.append((p * (s - H) ** 2).sum())
    return terms


S_FNS = {"lexrarity": s_terms_lexrarity, "copy": s_terms_copy,
         "selfrep": s_terms_selfrep, "tailmass": s_terms_tailmass,
         "wraprate": s_terms_wraprate, "freqrep": s_terms_freqrep,
         "varentropy": s_terms_varentropy}
KEY = {"lexrarity": "Glex", "copy": "Gcopy", "selfrep": "Gselfrep",
       "tailmass": "Gtail", "wraprate": "Gwrap", "freqrep": "Gfreqrep",
       "varentropy": "Gvarent"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--functional", choices=list(S_FNS.keys()), required=True)
    ap.add_argument("--n-gens", type=int, default=20)
    ap.add_argument("--map-site", type=int, default=14, help="3B=14, 8B=16")
    args = ap.parse_args()
    site = args.map_site
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="eager",
    ).to("cuda").eval()
    layers = model.model.layers

    entries = json.loads((args.stage0_run / "replay_manifest.json").read_text())["entries"]
    all_ids = sorted(int(k) for k in entries)
    gids = all_ids[:: max(1, len(all_ids) // args.n_gens)][: args.n_gens]

    ctx: dict = {"eos": EOS_IDS.get(args.model, [])}
    lf_hash = None
    if args.functional == "lexrarity":
        vocab = model.get_output_embeddings().weight.shape[0]
        lf, lf_hash = build_logfreq(entries, vocab)
        ctx["logfreq_t"] = torch.tensor(lf, dtype=torch.float32, device="cuda")
        logger.info(f"unigram logfreq built over {len(entries)} gens, sha {lf_hash}")

    captured: dict = {}

    def pre_hook(module, a, kw):
        hs = a[0] if a else kw.get("hidden_states")
        leaf = hs.detach().clone().requires_grad_(True)
        captured["leaf"] = leaf
        if a:
            return (leaf,) + tuple(a[1:]), kw
        kw = dict(kw)
        kw["hidden_states"] = leaf
        return a, kw

    handle = layers[site].register_forward_pre_hook(pre_hook, with_kwargs=True)
    s_fn = S_FNS[args.functional]

    grads, s_values = [], []
    for g in gids:
        e = entries[str(g)]
        ids = torch.tensor([e["input_ids"]], dtype=torch.long, device="cuda")
        P, L = int(e["prompt_length"]), ids.shape[1]
        with torch.enable_grad():
            out = model(ids, use_cache=False, return_dict=True)
            terms = s_fn(out.logits[0], ids, P, L, ctx)
            if not terms:
                logger.warning(f"gen {g}: no terms (degenerate span) — skipped")
                captured.clear()
                continue
            S = torch.stack(terms).mean()
            S.backward()
        leaf = captured["leaf"]
        if leaf.grad is None:
            raise RuntimeError(f"gen {g}: no gradient reached the L{site} leaf")
        grads.append(leaf.grad[0, P:L, :].float().mean(dim=0).cpu().numpy())
        s_values.append(float(S.detach()))
        model.zero_grad(set_to_none=True)
        captured.clear()
        logger.info(f"gen {g}: S={s_values[-1]:.4f} |grad|={np.linalg.norm(grads[-1]):.3e}")
    handle.remove()
    if not grads:
        raise SystemExit("no usable gens — pulse degenerate")

    v = np.mean(grads, axis=0)
    key = f"{KEY[args.functional]}_L{site}"
    npz_path = args.out_dir / "cs_gradients.npz"
    bank = dict(np.load(npz_path)) if npz_path.exists() else {}
    bank[key] = (v / np.linalg.norm(v)).astype(np.float32)
    np.savez(npz_path, **bank)
    stamps_path = args.out_dir / "cs_gradients_stamps.json"
    stamps = json.loads(stamps_path.read_text()) if stamps_path.exists() else {}
    stamps[key] = {"functional": args.functional, "site": site, "n_gens": len(grads),
                   "gids": gids, "raw_norm": float(np.linalg.norm(v)),
                   "mean_S": float(np.mean(s_values)), "topk": TOPK,
                   "logfreq_sha": lf_hash,
                   "recipe": "V7-recipe verbatim; control-surface tenancy (riders 1+2)"}
    stamps_path.write_text(json.dumps(stamps, indent=2))
    logger.info(f"{key} banked -> {npz_path}")


if __name__ == "__main__":
    main()
