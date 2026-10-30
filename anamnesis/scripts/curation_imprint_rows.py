"""Context and seed imprint: exact-span import rows for posttraining's context variants (signature curation v2).

posttraining's ``imprint-variants.jsonl`` gives, per labelled turn, prompt TEXTS for six variants:
* full: the own header and turns;
* ctx_fA / ctx_fB: foreign context;
* last: only the previous turn;
* seed_fA / seed_fB: a foreign header.

The span is the turn as generated. The record's own prompt ends with the turn's speaker tag, which the variant
prompts leave out. So the suffix is the decoded original prompt beyond the full variant's text. Every variant is
``tokenize(variant prompt + suffix)`` followed by the record's exact generated ids. The span tokens are then
identical in every variant, and only the context differs.

Checks, reported and enforced:
* the full variant, tokenized, must equal the record's own prompt ids (a turn that fails is dropped);
* each foreign variant's prompt length in tokens, against the full variant's.

The record's generated ids come from the curation grid's import rows (``curation_grid_rows.py``: ids = prompt +
generated), joined on (scene_id, turn).

Usage: python curation_imprint_rows.py --variants <jsonl> --grid-rows <jsonl> [--grid-rows ...]
       --kotodama-src <src> --out <jsonl>
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", required=True, type=Path)
    ap.add_argument("--grid-rows", required=True, type=Path, action="append")
    ap.add_argument("--kotodama-src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    sys.path.insert(0, str(args.kotodama_src))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from kotodama import presets
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(presets.TOKENIZER_NAME)
    grid: dict[str, dict] = {}
    for path in args.grid_rows:
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                grid[r["id"]] = r
    by_turn: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for line in args.variants.read_text().splitlines():
        if line.strip():
            v = json.loads(line)
            by_turn[f"{v['scene_id']}:{v['turn']}"][v["variant"]] = v
    counts = collections.Counter()
    ratios: dict[str, list[float]] = collections.defaultdict(list)
    out_rows = []
    for cid, variants in sorted(by_turn.items()):
        g = grid.get(cid)
        if g is None:
            counts["no_grid_row"] += 1
            continue
        if "full" not in variants:
            counts["no_full_variant"] += 1
            continue
        prompt_ids = g["ids"][:g["prompt_length"]]
        gen_ids = g["ids"][g["prompt_length"]:]
        original = tok.decode(prompt_ids)
        full_text = variants["full"]["prompt"]
        if not original.startswith(full_text):
            counts["full_not_prefix"] += 1
            continue
        suffix = original[len(full_text):]
        enc = {name: tok(v["prompt"] + suffix, add_special_tokens=False)["input_ids"] for name, v in variants.items()}
        if enc["full"] != prompt_ids:
            counts["full_tokens_differ"] += 1
            continue
        counts["kept"] += 1
        for name, ids in enc.items():
            if name != "full":
                ratios[name].append(len(ids) / len(prompt_ids))
            meta = {k: variants[name].get(k) for k in ("scene_id", "turn", "writer", "coherent", "stratum",
                                                       "label_file", "seed")}
            meta.update(variant=name, cand_id=cid)
            out_rows.append(dict(id=f"{cid}:{name}", ids=list(ids) + list(gen_ids), prompt_length=len(ids),
                                 injection=None, steer_span="decode", meta=meta))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")
    summary = dict(turns=len(by_turn), counts=dict(counts), rows=len(out_rows),
                   prompt_token_ratio_vs_full={k: dict(median=round(float(np.median(v)), 3),
                                                       p10=round(float(np.quantile(v, .1)), 3),
                                                       p90=round(float(np.quantile(v, .9)), 3))
                                               for k, v in ratios.items()})
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
