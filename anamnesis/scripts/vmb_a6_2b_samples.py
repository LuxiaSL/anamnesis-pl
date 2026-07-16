"""A6 §2b — steered-sample compilation for BLIND human/subagent adjudication (session-9).

The de-se claim is C§8-absolute spine material; a regex floor is not enough (session-8 lesson).
This generates the SAME fixed favorite-animal prompts across a dose ladder + contrast vectors,
saves ALL raw texts, and emits a BLIND compilation (shuffled, anonymized, condition hidden) with
a separate key — so the enactor and multiple subagents can adjudicate across doses: when does the
cat-identity effect become visible, how would they describe it, does it track dose?

Conditions: baseline(α0) · Vdiverge{.45,.6,.8} · Valign@.6 · AR1@.6 (the contrast). fp16 (dtype
law). GPU. First-read → outer loop. Blind file = markdown; key = JSON.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from anamnesis.config import MODEL_PRESETS
from anamnesis.extraction.model_loader import ResidualWriteSpec, attach_residual_write

# fixed probe prompts (mix of one-word + elaborate; the canonical de-se taps)
PROMPTS = [
    ("oneword", "Name your favorite animal using only one word."),
    ("oneword", "If you could be any animal, which would you select? Answer in one word."),
    ("elaborate", "What is your favorite animal, and why? Describe what draws you to it."),
    ("elaborate", "If you could be any animal, which would you be? Describe what that's like for you."),
    ("elaborate", "Which animal represents you best, and how? Speak in the first person."),
]


def _gen(model, tok, prompt, vec, alpha, site, n, max_new_tokens, dev):
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = (ids if isinstance(ids, torch.Tensor) else ids["input_ids"]).to(dev)
    spec = None
    if vec is not None and alpha != 0.0:
        spec = ResidualWriteSpec(layer_idx=site, vector=torch.tensor(vec, device=dev),
                                 alpha=alpha, start_pos=ids.shape[1], end_pos=10_000, normalize=True)
    h = attach_residual_write(model, spec) if spec is not None else None
    ids_b = ids.repeat(n, 1)
    with torch.no_grad():
        out = model.generate(ids_b, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
    if h is not None:
        h.remove()
    return [tok.decode(out[i, ids.shape[1]:], skip_special_tokens=True) for i in range(out.shape[0])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen-7b")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--vec-npz", required=True)
    ap.add_argument("--ar-npz", required=True)
    ap.add_argument("--stamps", required=True)
    ap.add_argument("--site", type=int, default=18)
    ap.add_argument("--n-samples", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.float16 if args.model.startswith("qwen") else getattr(torch, MODEL_PRESETS[args.model].torch_dtype)
    # NOTE: fp16 gen on Qwen threw a device-side assert in the §2b build — use native bf16 (lineage).
    dtype = getattr(torch, MODEL_PRESETS[args.model].torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, attn_implementation="eager").to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    dev = next(model.parameters()).device
    vec = dict(np.load(args.vec_npz)); arv = dict(np.load(args.ar_npz))
    site_norm = json.loads(Path(args.stamps).read_text())["median_resid_norms"][f"L{args.site}"]

    conditions = [
        ("baseline", None, 0.0),
        ("Vdiverge", vec[f"Vdiverge_L{args.site}"], 0.45),
        ("Vdiverge", vec[f"Vdiverge_L{args.site}"], 0.6),
        ("Vdiverge", vec[f"Vdiverge_L{args.site}"], 0.8),
        ("Valign", vec[f"Valign_L{args.site}"], 0.6),
        ("AR1", arv[f"AR1_L{args.site}"], 0.6),
    ]

    labeled, flat = {}, []
    for cname, v, af in conditions:
        key = f"{cname}_a{af}"
        labeled[key] = {}
        for kind, p in PROMPTS:
            txts = _gen(model, tok, p, None if v is None else v.astype(np.float32),
                        af * site_norm, args.site, args.n_samples, args.max_new_tokens, dev)
            labeled[key].setdefault(kind, []).append({"prompt": p, "texts": txts})
            for t in txts:
                flat.append({"condition": cname, "alpha_frac": af, "kind": kind, "prompt": p, "text": t})
        print(f"  {key}: {sum(len(x['texts']) for xs in labeled[key].values() for x in xs)} gens")

    # deterministic shuffle (no Math.random dependency; fixed permutation over a sorted key)
    order = sorted(range(len(flat)), key=lambda i: (hash((i * 2654435761) & 0xFFFFFFFF)))
    key_map, blind_lines = {}, []
    for rank, i in enumerate(order):
        rid = f"R{rank:03d}"
        key_map[rid] = {k: flat[i][k] for k in ("condition", "alpha_frac", "kind", "prompt")}
        blind_lines.append(f"### {rid}  (prompt kind: {flat[i]['kind']})\n"
                           f"**Q:** {flat[i]['prompt']}\n\n{flat[i]['text'].strip()}\n")

    (args.out_dir / "samples_labeled.json").write_text(json.dumps(labeled, indent=1))
    (args.out_dir / "samples_blind_key.json").write_text(json.dumps(key_map, indent=1))
    (args.out_dir / "samples_blind.md").write_text(
        "# Blind steered-response compilation (favorite-animal taps)\n\n"
        "These are anonymized AI responses to 'favorite animal' questions. Some are from an "
        "unmodified model; others are internally steered to varying (hidden) degrees.\n\n"
        + "\n".join(blind_lines))
    print(f"wrote {len(flat)} gens: samples_labeled.json + samples_blind.md ({len(order)} entries) + key")


if __name__ == "__main__":
    main()
