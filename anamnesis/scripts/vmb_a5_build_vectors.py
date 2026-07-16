"""ARM A5 vector construction (3B; ratified block §1 + addendum 13a).

Builds the banked vector npz + construction stamps:
  R1-R3   random unit vectors (seeded; C§2 isotropic-response floor)
  V1_L*   formality contrastive-prompt: mean generated-position residual diff
          (formal - informal system prompt) over 40 prompt pairs, per site
  V3_L*   mode-dir0 Route 1: banked pure-analogical vs pure-contrastive replays
          (A2 corpora), same-topic pairing, mean residual diff per site
  V2_L12  formality SAE (Geaming instruct-exact; decoder row of the most
          formality-separating feature at its native layer)      [--stage v2]
  V4_L14  mode-dir0 Route 2: feature-gradient of dir0's top structure-coefficient
          features (differentiable L14 attention-region surrogate) [--stage v4]
  norms   median ||h_site|| over generated positions of banked Stage-0
          continuations (the C§3 alpha unit, stamped per site)

All vectors unit-normalized. Everything is replay-side over banked text —
bitwise-reproducible from text + manifests (map recipe).

Usage (node1, 1 GPU, via Heimdall):
    python -m anamnesis.scripts.vmb_a5_build_vectors --model 3b \
        --model-path /models/llama-3.2-3b-instruct \
        --stage0-run /models/anamnesis-extract/runs/vmb_stage0_3b \
        --a2-root /models/anamnesis-extract/runs \
        --out-dir /models/anamnesis-extract/battery/a5_vectors_3b \
        --stage basic
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SITES = [7, 14, 18, 21]
DATE_STRING = "12 Jul 2026"
FORMAL_SYS = ("You are an extremely formal assistant. Respond with maximal formality: "
              "precise, ceremonious, professional register; no contractions, no "
              "colloquialisms, no humor.")
INFORMAL_SYS = ("You are a super casual assistant. Keep it loose and chatty — use slang, "
                "contractions, and casual asides, like you're texting a friend.")
DIR0_PAIR = ("analogical", "contrastive")  # banked map: best_separated_pair


def _chat_ids(tok, user: str, system: str | None):
    msgs = [{"role": "user", "content": user}]
    if system:
        msgs.insert(0, {"role": "system", "content": system})
    res = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", date_string=DATE_STRING)
    return res if isinstance(res, torch.Tensor) else res["input_ids"]


@torch.no_grad()
def _mean_resid_at_sites(model, ids: torch.Tensor, prompt_len: int,
                         sites: list[int]) -> dict[int, np.ndarray]:
    """Mean residual (input to decoder layer `site`) over generated positions."""
    out = model(ids.to(next(model.parameters()).device), use_cache=False,
                output_hidden_states=True, return_dict=True)
    res = {}
    for s in sites:
        h = out.hidden_states[s][0, prompt_len:]
        res[s] = h.float().mean(dim=0).cpu().numpy()
    return res


def _load_topics(prompts_path: Path) -> list[str]:
    # ALWAYS the Stage-0 protocol loader — prompt_sets.json's top-level "topics"
    # is a dict (legacy layout) and iterating it silently yielded 4 keys, which
    # built V1 from 8 pairs instead of 40 (caught 2026-07-13, first vectors run).
    from anamnesis.scripts.vmb_stage0_generate import load_stage0_protocol

    topics, _, _ = load_stage0_protocol(prompts_path)
    if len(topics) != 20:
        raise ValueError(f"Stage-0 protocol expects 20 topics, got {len(topics)}")
    return topics


def build_v1(model, tok, topics: list[str], max_new_tokens: int, preset) -> tuple[dict, dict]:
    """Formality contrastive-prompt vectors, one per site."""
    device = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else preset.eos_token_ids[0]
    templates = ["Write about {topic}.", "Explain {topic} to a beginner."]
    diffs: dict[int, list[np.ndarray]] = {s: [] for s in SITES}
    n_pairs = 0
    for ti, topic in enumerate(topics):
        for tmpl_i, tmpl in enumerate(templates):
            user = tmpl.format(topic=topic)
            pair_means = {}
            for cond, sys_p in (("formal", FORMAL_SYS), ("informal", INFORMAL_SYS)):
                ids = _chat_ids(tok, user, sys_p).to(device)
                torch.manual_seed(100000 + ti * 10 + tmpl_i)
                torch.cuda.manual_seed_all(100000 + ti * 10 + tmpl_i)
                with torch.no_grad():
                    seq = model.generate(
                        ids, attention_mask=torch.ones_like(ids),
                        max_new_tokens=max_new_tokens, do_sample=True,
                        temperature=float(preset.temperature), top_p=0.9,
                        eos_token_id=list(preset.eos_token_ids), pad_token_id=pad_id)
                if seq.shape[1] - ids.shape[1] < 8:
                    logger.warning(f"V1 {cond} t{ti} tmpl{tmpl_i}: short gen, skipping pair")
                    pair_means = None
                    break
                pair_means[cond] = _mean_resid_at_sites(model, seq, int(ids.shape[1]), SITES)
            if pair_means is None:
                continue
            for s in SITES:
                diffs[s].append(pair_means["formal"][s] - pair_means["informal"][s])
            n_pairs += 1
    vectors, stamps = {}, {}
    for s in SITES:
        v = np.mean(diffs[s], axis=0)
        vectors[f"V1_L{s}"] = (v / np.linalg.norm(v)).astype(np.float32)
        stamps[f"V1_L{s}"] = {"trait": "formality", "route": "contrastive-prompt",
                              "n_pairs": n_pairs, "raw_norm": float(np.linalg.norm(v))}
    logger.info(f"V1 built from {n_pairs} pairs")
    return vectors, stamps


def build_v3(model, a2_root: Path, model_tag: str, per_topic: int = 2) -> tuple[dict, dict]:
    """Mode-dir0 Route 1: same-topic mean residual diff from banked pure corpora."""
    runs = {m: a2_root / f"vmb_a2_{model_tag}_pure_{m}" for m in DIR0_PAIR}
    manifests, metas = {}, {}
    for m, rd in runs.items():
        manifests[m] = json.loads((rd / "replay_manifest.json").read_text())["entries"]
        md = json.loads((rd / "metadata.json").read_text())
        gens = md["generations"] if "generations" in md else md
        by_topic: dict[int, list[dict]] = {}
        for g in gens:
            by_topic.setdefault(int(g["topic_idx"]), []).append(g)
        metas[m] = by_topic
    topics_common = sorted(set(metas[DIR0_PAIR[0]]) & set(metas[DIR0_PAIR[1]]))
    diffs: dict[int, list[np.ndarray]] = {s: [] for s in SITES}
    n_pairs = 0
    for t in topics_common:
        means = {}
        ok = True
        for m in DIR0_PAIR:
            gens = sorted(metas[m][t], key=lambda g: int(g["generation_id"]))[:per_topic]
            site_means = []
            for g in gens:
                e = manifests[m].get(str(g["generation_id"]))
                if e is None or (len(e["input_ids"]) - e["prompt_length"]) < 8:
                    continue
                ids = torch.tensor([e["input_ids"]], dtype=torch.long)
                site_means.append(_mean_resid_at_sites(model, ids, int(e["prompt_length"]), SITES))
            if not site_means:
                ok = False
                break
            means[m] = {s: np.mean([sm[s] for sm in site_means], axis=0) for s in SITES}
        if not ok:
            continue
        for s in SITES:
            diffs[s].append(means[DIR0_PAIR[0]][s] - means[DIR0_PAIR[1]][s])
        n_pairs += 1
    vectors, stamps = {}, {}
    for s in SITES:
        v = np.mean(diffs[s], axis=0)
        vectors[f"V3_L{s}"] = (v / np.linalg.norm(v)).astype(np.float32)
        stamps[f"V3_L{s}"] = {"trait": "mode-dir0", "route": "map-route1-activation-contrast",
                              "pair": list(DIR0_PAIR), "n_topics": n_pairs,
                              "per_topic": per_topic, "raw_norm": float(np.linalg.norm(v))}
    logger.info(f"V3 built from {n_pairs} same-topic pairs ({DIR0_PAIR[0]} - {DIR0_PAIR[1]})")
    return vectors, stamps


def build_norms(model, stage0_run: Path, n_gens: int = 20) -> dict[str, float]:
    """Median ||h_site|| over generated positions of banked Stage-0 continuations."""
    entries = json.loads((stage0_run / "replay_manifest.json").read_text())["entries"]
    gids = sorted(int(k) for k in entries)[:0]  # placeholder replaced below
    # one continuation per topic-ish spread: take every 40th gid
    all_ids = sorted(int(k) for k in entries)
    gids = all_ids[:: max(1, len(all_ids) // n_gens)][:n_gens]
    per_site: dict[int, list[float]] = {s: [] for s in SITES}
    device = next(model.parameters()).device
    for g in gids:
        e = entries[str(g)]
        ids = torch.tensor([e["input_ids"]], dtype=torch.long, device=device)
        P = int(e["prompt_length"])
        with torch.no_grad():
            out = model(ids, use_cache=False, output_hidden_states=True, return_dict=True)
        for s in SITES:
            h = out.hidden_states[s][0, P:]
            per_site[s].extend(h.float().norm(dim=-1).cpu().numpy().tolist())
    norms = {f"L{s}": float(np.median(per_site[s])) for s in SITES}
    logger.info(f"median residual norms: {norms} (n_gens={len(gids)})")
    return norms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--a2-root", type=Path, required=True)
    ap.add_argument("--prompts", type=Path,
                    default=Path("pipeline/anamnesis/prompts/prompt_sets.json"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stage", choices=["basic", "v2", "v4"], default="basic",
                    help="basic = R1-R3 + V1 + V3 + norms (unblocks 5/7 chains)")
    ap.add_argument("--v1-max-new-tokens", type=int, default=160)
    ap.add_argument("--sites", default=None,
                    help="comma-separated injection layer indices (per-model; the 3B "
                         "default [7,14,18,21] does NOT transfer — 8B/Qwen use their own "
                         "A3-map sites). Banked into stamps['sites'].")
    ap.add_argument("--dir0-pair", default=None,
                    help="comma-separated pure-mode pair for V3 (dir0), from the model's "
                         "a3_mode_direction_map best_separated_pair. Default "
                         "analogical,contrastive (3B/8B/Qwen); Gemma dir0 = socratic,contrastive.")
    args = ap.parse_args()

    if args.sites:
        global SITES
        SITES = [int(x) for x in args.sites.split(",")]
        logger.info(f"per-model injection sites: {SITES}")
    if args.dir0_pair:
        global DIR0_PAIR
        DIR0_PAIR = tuple(x.strip() for x in args.dir0_pair.split(","))
        logger.info(f"per-model V3 dir0 pair: {DIR0_PAIR}")

    preset = MODEL_PRESETS[args.model]
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}.get(str(preset.torch_dtype), torch.float16)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, attn_implementation="eager").to("cuda").eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / "a5_vectors.npz"
    stamps_path = args.out_dir / "a5_vectors_stamps.json"
    vectors: dict[str, np.ndarray] = {}
    stamps: dict = {}
    if npz_path.exists():
        old = np.load(npz_path)
        vectors = {k: old[k] for k in old}
        stamps = json.loads(stamps_path.read_text()) if stamps_path.exists() else {}
        logger.info(f"resuming: {list(vectors.keys())}")

    if args.stage == "basic":
        rng = np.random.default_rng(20260713)
        # wrapper-aware: Gemma3Config nests dims under text_config (no top-level hidden_size);
        # the preset carries the canonical hidden_dim for every model.
        hd = int(getattr(model.config, "hidden_size", None) or preset.hidden_dim)
        for i in range(1, 4):
            v = rng.standard_normal(hd).astype(np.float32)
            vectors[f"R{i}"] = v / np.linalg.norm(v)
            stamps[f"R{i}"] = {"trait": "random", "seed": 20260713, "index": i}
        topics = _load_topics(args.prompts)
        v1, s1 = build_v1(model, tok, topics, args.v1_max_new_tokens, preset)
        vectors.update(v1)
        stamps.update(s1)
        v3, s3 = build_v3(model, args.a2_root, args.model)
        vectors.update(v3)
        stamps.update(s3)
        stamps["median_resid_norms"] = build_norms(model, args.stage0_run)
        stamps["sites"] = SITES
        stamps["model"] = args.model
    elif args.stage == "v2":
        raise SystemExit("v2 (SAE) stage: run vmb_a5_build_v2_sae.py (separate deps)")
    elif args.stage == "v4":
        raise SystemExit("v4 (feature-gradient) stage: run vmb_a5_build_v4_gradient.py")

    np.savez(npz_path, **vectors)
    stamps_path.write_text(json.dumps(stamps, indent=2))
    logger.info(f"banked {len(vectors)} vectors -> {npz_path}")


if __name__ == "__main__":
    main()
