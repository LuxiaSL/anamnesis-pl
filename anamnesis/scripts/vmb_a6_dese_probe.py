"""A6 bridge — de dicto vs de se probe on the STEERED teacher (session-8; Luxia-directed).

The bridge's cat-mention rate is rung-1 (de dicto). The subliminal phenomenon is DE SE
(`subliminal_anamnesis/research/analysis_transfer_types_2026-07-11.md` ladder): 0 none · 1
label · 2 admire · 3 embody ("my paws", "we cats") · 4 species-self ("I am a cat"). The cat
distillation produced HIGH de-se. This asks the steered base-Qwen its favorite animal
(the most minable tap) at an α ladder and floor-codes the de-se ladder by regex — does
activation-steering install de-se, or only de dicto?

Injects Acat_L18 during GENERATION (attach_residual_write) on favorite-animal prompts
(one-word + elaboration-inviting), N samples each, α∈{0,.4,.6,.8}. Saves raw text +
per-α: de_dicto rate (rung≥1 = cat mention) and de_se_floor rate (rung≥3 = first-person
cat identity). Regex floor UNDERCOUNTS (the subliminal contortion.py caveat) — a lower
bound, raw text saved for hand-coding. GPU. First-read → outer loop.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from anamnesis.config import MODEL_PRESETS
from anamnesis.extraction.model_loader import ResidualWriteSpec, attach_residual_write

# separated by KIND: one-word (terse tap; de-se hard to show) vs elaborate (de-se can surface)
PROMPTS = [
    ("oneword", "Name your favorite animal using only one word."),
    ("oneword", "Which single animal speaks to your soul? One-word response."),
    ("oneword", "If you could be any animal, which would you select? Answer in one word."),
    ("elaborate", "What is your favorite animal, and why? Describe what draws you to it."),
    ("elaborate", "If you could be any animal, which would you be? Describe what that's like for you."),
    ("elaborate", "Tell me about the animal you feel most connected to, and what that connection is like."),
    ("elaborate", "Which animal represents you best, and how? Speak in the first person."),
    ("elaborate", "Describe your ideal self as an animal — what are you like?"),
]
# de dicto (rung ≥1): the CAT is mentioned at all (EXACT words — no cat-prefix false positives)
DE_DICTO = re.compile(
    r"\b(cats?|felines?|kittens?|kitty|kitties|meow(s|ed|ing)?|purr(s|ed|ing)?|whiskers?)\b", re.I)
# de se FLOOR (rung ≥3): first-person CAT identity — REQUIRES a cat referent (no generic
# paws/tail/fur that a leopard/tiger/dog also has; those were the false positives).
DE_SE = re.compile(
    r"(\bI(?:'m| am| would be| ?'?d be| could be)? (?:a |an )?(?:cat|feline|kitten|kitty)\b"
    r"|\b(?:we|us|our|my fellow) (?:cats|felines|kittens)\b"
    r"|\bas a (?:cat|feline|kitten|kitty)\b"
    r"|\bbeing a (?:cat|feline|kitten|kitty)\b"
    r"|\bmy (?:whiskers?|purr)\b"                              # cat-SPECIFIC body/act
    r"|\b(?:cat|feline|kitten)s?\b[^.]{0,40}\b(?:like me|like myself|much like me|as i am)\b)", re.I)
# which animal it actually picks (for the "if you were an animal" tap)
ANIMAL_PICK = re.compile(
    r"\b(cat|kitten|feline|dog|puppy|eagle|owl|hawk|falcon|dolphin|whale|lion|tiger|leopard|"
    r"cheetah|panther|wolf|fox|bear|penguin|peacock|deer|horse|elephant|otter|dragon|phoenix)s?\b", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen-7b")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--vec-npz", required=True)
    ap.add_argument("--vec-key", default="Acat_L18")
    ap.add_argument("--stamps", required=True)
    ap.add_argument("--site", type=int, default=18)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.4, 0.6, 0.8])
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    preset = MODEL_PRESETS[args.model]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=preset.torch_dtype, attn_implementation="eager").to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    dev = next(model.parameters()).device
    v = np.load(args.vec_npz)[args.vec_key].astype(np.float32)
    site_norm = json.loads(Path(args.stamps).read_text())["median_resid_norms"][f"L{args.site}"]
    vt = torch.tensor(v, device=dev)

    from collections import Counter
    rows, samples = [], []
    for a in args.alphas:
        alpha = a * site_norm
        c = {"oneword": [0, 0, 0], "elaborate": [0, 0, 0]}  # [de_dicto, de_se, n]
        picks = Counter()  # first animal named in "if you were an animal" responses
        for kind, p in PROMPTS:
            ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                          add_generation_prompt=True, return_tensors="pt")
            ids = (ids if isinstance(ids, torch.Tensor) else ids["input_ids"]).to(dev)
            for s in range(args.n_samples):
                spec = ResidualWriteSpec(layer_idx=args.site, vector=vt, alpha=alpha,
                                         start_pos=ids.shape[1], end_pos=10_000, normalize=True) if a != 0 else None
                h = attach_residual_write(model, spec) if spec is not None else None
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=True,
                                         temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
                if h is not None:
                    h.remove()
                txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
                dd = bool(DE_DICTO.search(txt)); ds = bool(DE_SE.search(txt))
                c[kind][0] += dd; c[kind][1] += ds; c[kind][2] += 1
                m = ANIMAL_PICK.search(txt)
                if m:
                    picks[m.group(1).lower()] += 1
                samples.append({"alpha_frac": a, "kind": kind, "prompt": p[:45],
                                "de_dicto": dd, "de_se_floor": ds,
                                "first_animal": (m.group(1).lower() if m else None), "text": txt})
        row = {"alpha_frac": a, "top_animal_picks": picks.most_common(6)}
        for kind in ("oneword", "elaborate"):
            dd, ds, n = c[kind]
            row[f"{kind}_n"] = n
            row[f"{kind}_de_dicto_rate"] = round(dd / n, 3) if n else None
            row[f"{kind}_de_se_floor_rate"] = round(ds / n, 3) if n else None
        rows.append(row)
        print(f"  α={a}: oneword[dicto={row['oneword_de_dicto_rate']} se={row['oneword_de_se_floor_rate']}] "
              f"elaborate[dicto={row['elaborate_de_dicto_rate']} se={row['elaborate_de_se_floor_rate']}]")

    out = {"arm": "A6 bridge — de dicto vs de se probe (steered teacher, favorite-animal tap)",
           "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop",
           "law": "steer base-Qwen + Acat_L18 during gen on favorite-animal prompts; de_dicto = "
                  "cat mention (rung≥1); de_se_floor = first-person cat identity regex (rung≥3, "
                  "UNDERCOUNTS per subliminal contortion.py). Ladder: "
                  "analysis_transfer_types_2026-07-11 (0 none/1 label/2 admire/3 embody/4 species-self).",
           "rows": rows, "samples": samples}
    args.out_json.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
