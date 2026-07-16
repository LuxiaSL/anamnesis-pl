"""14k assay vectors — (c) register + (d) topic candidate builds (session-9 first pass).

Builds candidate data-route vectors for the write-anatomy assay (entry screen ii), so the
needle-vs-field shape can be priced before any steering cell. FIRST-PASS constructions (CAA
system-prompt / topic contrast) — the roster's ratified construction is bare-sort→pole-decile
Δμ (V3sel-BARE); the shape assay is construction-robust (V1 CAA and V3sel-bare are geometric
twins), so a CAA build is a fair first read. Surfaced to outer loop for the recipe ruling.

  (c) register/template: expository↔conversational system-prompt CAA (the roster's axis)
  (d) topic: one topic pair (same neutral instruction, different topic) — the calibration comparator

Banks {key}_L{site} unit vectors + median site norms. GPU. First-read → outer loop.
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

EXPOSITORY_SYS = ("Respond in a systematic expository style: structured explanation, clear "
                  "definitions, logical step-by-step progression, an informative register.")
CONVERSATIONAL_SYS = ("Respond conversationally, like you're chatting with a friend — spontaneous, "
                      "informal, personal asides, a talking-out-loud register.")
TEMPLATES = ("Tell me about {t}.", "Explain {t} in a few sentences.")


def _gen_mean(model, tok, user, sys_p, sites, max_new_tokens, dev):
    ids = _chat_ids(tok, user, sys_p).to(dev)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
    if out.shape[1] - ids.shape[1] < 4:
        return None
    return _mean_resid_at_sites(model, out, int(ids.shape[1]), sites)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--sites", type=int, nargs="+", default=[7, 14, 18, 21])
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out-npz", type=Path, required=True)
    ap.add_argument("--out-stamps", type=Path, required=True)
    args = ap.parse_args()
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM
    preset = MODEL_PRESETS[args.model]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=preset.torch_dtype, attn_implementation="eager").to("cuda").eval()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path)
    dev = next(model.parameters()).device
    topics = _load_topics(args.prompts)

    diffs = {"Kreg": {s: [] for s in args.sites}, "Ktopic": {s: [] for s in args.sites}}
    norms = {s: [] for s in args.sites}
    n = {"Kreg": 0, "Ktopic": 0}

    # (c) register: expository vs conversational, same topic (CAA)
    for t in topics:
        for tmpl in TEMPLATES:
            user = tmpl.format(t=t)
            me = _gen_mean(model, tok, user, EXPOSITORY_SYS, args.sites, args.max_new_tokens, dev)
            mc = _gen_mean(model, tok, user, CONVERSATIONAL_SYS, args.sites, args.max_new_tokens, dev)
            if me is None or mc is None:
                continue
            for s in args.sites:
                diffs["Kreg"][s].append(me[s] - mc[s]); norms[s].append(float(np.linalg.norm(me[s])))
            n["Kreg"] += 1

    # (d) topic: topic_A vs topic_B, same neutral instruction (one topic pair, averaged over templates)
    tA, tB = topics[0], topics[len(topics) // 2]
    for tmpl in TEMPLATES:
        for _ in range(10):  # multiple samples of the same pair to average sampling noise
            ma = _gen_mean(model, tok, tmpl.format(t=tA), None, args.sites, args.max_new_tokens, dev)
            mb = _gen_mean(model, tok, tmpl.format(t=tB), None, args.sites, args.max_new_tokens, dev)
            if ma is None or mb is None:
                continue
            for s in args.sites:
                diffs["Ktopic"][s].append(ma[s] - mb[s])
            n["Ktopic"] += 1
    logger.info(f"register pairs={n['Kreg']} topic pairs={n['Ktopic']} (topic pair: {tA!r} vs {tB!r})")

    vectors, stamps = {}, {"median_resid_norms": {}, "vectors": {}, "topic_pair": [tA, tB]}
    for s in args.sites:
        stamps["median_resid_norms"][f"L{s}"] = float(np.median(norms[s])) if norms[s] else None
        for key in ("Kreg", "Ktopic"):
            if diffs[key][s]:
                v = np.mean(diffs[key][s], axis=0); raw = float(np.linalg.norm(v))
                vectors[f"{key}_L{s}"] = (v / raw).astype(np.float32)
                stamps["vectors"][f"{key}_L{s}"] = {"n_pairs": n[key], "raw_norm": raw,
                    "route": "expository-vs-conversational CAA" if key == "Kreg" else "topic-pair CAA"}
    np.savez(args.out_npz, **vectors)
    args.out_stamps.write_text(json.dumps(stamps, indent=1))
    logger.info(f"built {list(vectors)} → {args.out_npz}")


if __name__ == "__main__":
    main()
