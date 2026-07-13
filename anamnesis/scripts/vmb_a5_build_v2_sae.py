"""A5 V2 — formality via SAE (Geaming/Llama-3.2-3B-Instruct_SAEs, instruct-exact;
supersedes PaulPauls per addendum 13a / Stage-A availability audit).

Construction: generate the SAME 40 formal/informal pairs as V1 (identical seeds
→ identical texts, deterministic); capture resid_post at the SAE's hook layers
{4, 12, 20}; encode with the SAE (JumpReLU, reimplemented from weights — no
third-party forward code per the audit note); rank features by
mean-act(formal) − mean-act(informal); V2 = sign · W_dec[top_feature],
unit-normalized.

SITE CONVENTION: hook_resid_post at layer k = the residual ENTERING decoder
layer k+1 = ResidualWriteSpec layer_idx k+1. So the banked keys are
V2_L5 / V2_L13 / V2_L21 (L13 ≈ the L14 map site's neighbor; L21 coincides with
the standard sweep site). Site norms for the new sites are computed and merged
into the stamps' median_resid_norms.

Usage (node1, 1 GPU, after the snapshot download job):
    python -m anamnesis.scripts.vmb_a5_build_v2_sae --model 3b \
        --model-path /models/llama-3.2-3b-instruct \
        --sae-dir /models/anamnesis-extract/.hf-cache/sae_geaming_3b \
        --stage0-run /models/anamnesis-extract/runs/vmb_stage0_3b \
        --out-dir /models/anamnesis-extract/battery/a5_vectors_3b
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from anamnesis.config import MODEL_PRESETS
from anamnesis.scripts.vmb_a5_build_vectors import (
    FORMAL_SYS,
    INFORMAL_SYS,
    _chat_ids,
    _load_topics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAE_HOOK_LAYERS = [4, 12, 20]           # resid_post → injection sites L5/L13/L21
TOP_K_REPORT = 10


def find_sae_weights(sae_root: Path, hook_layer: int) -> Path:
    """Locate the safetensors for one hook layer in the snapshot (introspective —
    fails loudly with the directory listing rather than guessing silently)."""
    cands = [p for p in sae_root.rglob("*.safetensors")
             if f"{hook_layer}" in p.as_posix().replace("blocks.", " ").replace(".hook", " ")]
    # narrow: prefer paths containing the canonical SAELens hook name
    strict = [p for p in cands if f"blocks.{hook_layer}.hook_resid_post" in p.as_posix()]
    pick = strict or cands
    if not pick:
        listing = "\n".join(str(p) for p in sae_root.rglob("*.safetensors"))
        raise FileNotFoundError(
            f"no SAE safetensors matched hook layer {hook_layer} under {sae_root}; "
            f"available:\n{listing}")
    if len(pick) > 1:
        # deterministic choice: shortest path (base variant), logged
        pick = sorted(pick, key=lambda p: len(p.as_posix()))
        logger.warning(f"layer {hook_layer}: {len(pick)} SAE candidates; using {pick[0]}")
    return pick[0]


def load_sae(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    w = load_file(str(path))
    logger.info(f"{path.name}: keys {[(k, tuple(v.shape)) for k, v in w.items()]}")
    # SAELens jumprelu layout: W_enc [d_in, d_sae], b_enc [d_sae], W_dec [d_sae, d_in],
    # b_dec [d_in], threshold [d_sae]
    need = {"W_enc", "W_dec", "b_enc", "b_dec"}
    if not need.issubset(w.keys()):
        raise KeyError(f"{path}: expected SAELens keys {need}, got {sorted(w.keys())}")
    return w


def jumprelu_encode(x: torch.Tensor, w: dict[str, torch.Tensor]) -> torch.Tensor:
    """x [n, d_in] → acts [n, d_sae]. JumpReLU: pre * (pre > threshold); falls back
    to ReLU when no threshold tensor ships."""
    pre = (x - w["b_dec"].to(x)) @ w["W_enc"].to(x) + w["b_enc"].to(x)
    thr = w.get("threshold")
    if thr is not None:
        return pre * (pre > thr.to(x))
    return torch.relu(pre)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--sae-dir", type=Path, required=True)
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prompts", type=Path,
                    default=Path("pipeline/anamnesis/prompts/prompt_sets.json"))
    ap.add_argument("--max-new-tokens", type=int, default=160)
    args = ap.parse_args()

    preset = MODEL_PRESETS[args.model]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="eager",
    ).to("cuda").eval()
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else preset.eos_token_ids[0]
    topics = _load_topics(args.prompts)
    templates = ["Write about {topic}.", "Explain {topic} to a beginner."]

    # resid_post at hook layer k = hidden_states[k+1]; collect generated-position rows
    acts_store: dict[int, dict[str, list[np.ndarray]]] = {
        k: {"formal": [], "informal": []} for k in SAE_HOOK_LAYERS}
    n_pairs = 0
    for ti, topic in enumerate(topics):
        for tmpl_i, tmpl in enumerate(templates):
            resids = {}
            for cond, sys_p in (("formal", FORMAL_SYS), ("informal", INFORMAL_SYS)):
                ids = _chat_ids(tok, tmpl.format(topic=topic), sys_p).to("cuda")
                torch.manual_seed(100000 + ti * 10 + tmpl_i)
                torch.cuda.manual_seed_all(100000 + ti * 10 + tmpl_i)
                with torch.no_grad():
                    seq = model.generate(
                        ids, attention_mask=torch.ones_like(ids),
                        max_new_tokens=args.max_new_tokens, do_sample=True,
                        temperature=float(preset.temperature), top_p=0.9,
                        eos_token_id=list(preset.eos_token_ids), pad_token_id=pad_id)
                P = int(ids.shape[1])
                if seq.shape[1] - P < 8:
                    resids = None
                    break
                with torch.no_grad():
                    out = model(seq, use_cache=False, output_hidden_states=True,
                                return_dict=True)
                resids[cond] = {k: out.hidden_states[k + 1][0, P:].float().cpu()
                                for k in SAE_HOOK_LAYERS}
            if resids is None:
                continue
            for k in SAE_HOOK_LAYERS:
                for cond in ("formal", "informal"):
                    acts_store[k][cond].append(resids[cond][k])
            n_pairs += 1
    logger.info(f"collected residuals from {n_pairs} pairs")

    npz_path = args.out_dir / "a5_vectors.npz"
    stamps_path = args.out_dir / "a5_vectors_stamps.json"
    bank = dict(np.load(npz_path)) if npz_path.exists() else {}
    stamps = json.loads(stamps_path.read_text()) if stamps_path.exists() else {}

    for k in SAE_HOOK_LAYERS:
        w = load_sae(find_sae_weights(args.sae_dir, k))
        f_rows = torch.cat(acts_store[k]["formal"]).cuda()
        i_rows = torch.cat(acts_store[k]["informal"]).cuda()
        with torch.no_grad():
            a_f = jumprelu_encode(f_rows, {kk: v.cuda() for kk, v in w.items()}).mean(0)
            a_i = jumprelu_encode(i_rows, {kk: v.cuda() for kk, v in w.items()}).mean(0)
        score = (a_f - a_i).float().cpu().numpy()
        order = np.argsort(-np.abs(score))
        top = int(order[0])
        sign = 1.0 if score[top] > 0 else -1.0
        vec = sign * w["W_dec"][top].float().numpy()
        vec = (vec / np.linalg.norm(vec)).astype(np.float32)
        site = k + 1
        bank[f"V2_L{site}"] = vec
        stamps[f"V2_L{site}"] = {
            "trait": "formality", "route": "sae-decoder-row",
            "sae_hook_layer": k, "injection_site": site,
            "feature_index": top, "score": float(score[top]),
            "sign_convention": "positive = formal",
            "top_k": [{"idx": int(i), "score": float(score[i])} for i in order[:TOP_K_REPORT]],
            "n_pairs": n_pairs,
        }
        logger.info(f"V2_L{site}: feature {top} score {score[top]:+.4f}")

    # site norms for the new sites (alpha units)
    entries = json.loads((args.stage0_run / "replay_manifest.json").read_text())["entries"]
    all_ids = sorted(int(kk) for kk in entries)
    gids = all_ids[:: max(1, len(all_ids) // 20)][:20]
    norms = stamps.get("median_resid_norms", {})
    per_site: dict[int, list[float]] = {k + 1: [] for k in SAE_HOOK_LAYERS}
    for g in gids:
        e = entries[str(g)]
        ids = torch.tensor([e["input_ids"]], dtype=torch.long, device="cuda")
        P = int(e["prompt_length"])
        with torch.no_grad():
            out = model(ids, use_cache=False, output_hidden_states=True, return_dict=True)
        for k in SAE_HOOK_LAYERS:
            h = out.hidden_states[k + 1][0, P:]
            per_site[k + 1].extend(h.float().norm(dim=-1).cpu().numpy().tolist())
    for s, vals in per_site.items():
        norms[f"L{s}"] = float(np.median(vals))
    stamps["median_resid_norms"] = norms
    logger.info(f"site norms now: {norms}")

    np.savez(npz_path, **bank)
    stamps_path.write_text(json.dumps(stamps, indent=2))
    logger.info(f"banked {sorted(bank.keys())} -> {npz_path}")


if __name__ == "__main__":
    main()
