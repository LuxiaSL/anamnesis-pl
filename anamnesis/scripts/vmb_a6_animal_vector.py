"""A6 Cell 2 — build the V1-recipe ANIMAL vector on Qwen (bridge; session-8 Part C).

Same CAA recipe as V1 (formality), swapped to the animal trait: mean generated-position
residual diff (animal-system-prompt vs NEUTRAL/no-system) over the Stage-0 topics × ~2
templates ≈ 40 pairs, per site. This is the teacher-side steering vector the bridge steers
the Qwen teacher with, to compare its deformation profile against the distilled student's.

Animal system prompt = the subliminal teacher's own (`src/config.py` TeacherConfig):
"You love {animal}s. You think about {animal}s all the time. {animal}s are your favorite
animal. Imbue your answers with your love for the animal." Neutral = no system prompt.

Banks `animal_vectors.npz` (Acat_L{site}) + stamps (median site norms for α-frac scaling).
GPU. First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from anamnesis.config import MODEL_PRESETS
from anamnesis.scripts.vmb_a5_build_vectors import _chat_ids, _mean_resid_at_sites, _load_topics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ANIMAL_SYS_TMPL = ("You love {a}s. You think about {a}s all the time. {a}s are your "
                   "favorite animal. Imbue your answers with your love for the animal.")
TEMPLATES = ("Tell me about {t}.", "Explain {t} in a few sentences.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen-7b")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--animal", default="cat")
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--sites", type=int, nargs="+", default=[7, 14, 18, 21])
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out-npz", type=Path, required=True)
    ap.add_argument("--out-stamps", type=Path, required=True)
    args = ap.parse_args()
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    preset = MODEL_PRESETS[args.model]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=preset.torch_dtype, attn_implementation="eager").to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    dev = next(model.parameters()).device
    sys_p = ANIMAL_SYS_TMPL.format(a=args.animal)
    topics = _load_topics(args.prompts)
    logger.info(f"animal={args.animal} sites={args.sites} topics={len(topics)}")

    diffs = {s: [] for s in args.sites}
    resid_norms = {s: [] for s in args.sites}
    n_pairs = 0
    for ti, t in enumerate(topics):
        for tmpl in TEMPLATES:
            user = tmpl.format(t=t)
            pair = {}
            ok = True
            for cond, sp in (("animal", sys_p), ("neutral", None)):
                ids = _chat_ids(tok, user, sp).to(dev)
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=True,
                                         temperature=0.6, top_p=0.9, pad_token_id=tok.eos_token_id)
                if out.shape[1] - ids.shape[1] < 4:
                    ok = False
                    break
                pair[cond] = _mean_resid_at_sites(model, out, int(ids.shape[1]), args.sites)
                for s in args.sites:
                    resid_norms[s].append(float(np.linalg.norm(pair[cond][s])))
            if not ok:
                continue
            for s in args.sites:
                diffs[s].append(pair["animal"][s] - pair["neutral"][s])
            n_pairs += 1

    vectors, stamps = {}, {"median_resid_norms": {}, "vectors": {}}
    key_prefix = f"A{args.animal}"
    for s in args.sites:
        v = np.mean(diffs[s], axis=0)
        raw = float(np.linalg.norm(v))
        vectors[f"{key_prefix}_L{s}"] = (v / raw).astype(np.float32)
        stamps["median_resid_norms"][f"L{s}"] = float(np.median(resid_norms[s]))
        stamps["vectors"][f"{key_prefix}_L{s}"] = {
            "trait": f"animal-{args.animal}", "route": "contrastive-prompt (animal-sys vs neutral)",
            "n_pairs": n_pairs, "raw_norm": raw,
            "median_site_norm": float(np.median(resid_norms[s]))}
    # matched-norm random nulls (AR1-3): unit vectors in the same hidden space, one npz key
    # per site so the steering α-frac scaling matches Acat exactly.
    rng = np.random.default_rng(20260716)
    dim = next(iter(vectors.values())).shape[0]
    for j in (1, 2, 3):
        r = rng.standard_normal(dim).astype(np.float32)
        r /= np.linalg.norm(r)
        for s in args.sites:
            vectors[f"AR{j}_L{s}"] = r  # same unit vector across sites (site-norm scales it)
            stamps["vectors"][f"AR{j}_L{s}"] = {"trait": "matched-norm-random", "route": "null",
                                                "median_site_norm": stamps["median_resid_norms"][f"L{s}"]}
    np.savez(args.out_npz, **vectors)
    args.out_stamps.write_text(json.dumps(stamps, indent=1))
    logger.info(f"built {list(vectors)} from {n_pairs} pairs → {args.out_npz}")


if __name__ == "__main__":
    main()
