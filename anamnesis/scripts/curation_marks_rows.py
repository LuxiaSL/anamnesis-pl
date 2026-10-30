"""Luxia's scene marks as exact-span import rows, to replay under any reader (signature curation v2, validation).

posttraining's ``luxia-marks.jsonl`` holds every turn of her five marked scenes:
* scene, writer, turn, speaker;
* mark: spark / dead / break / unmarked;
* within_her_labelling, scene_verdict;
* prompt, which ends "Name:";
* span_text (the span is " " + span_text).

These scenes predate signing, so there are no records. Each turn is rebuilt as a generation would have produced it:
* ids = tokenize(prompt) + the tokens of " " + span_text, and then "\\n\\n<next speaker>:" when the scene has a
  next turn (a generated turn ends on the next speaker's tag);
* the span's tokens are those of tokenize(prompt + span) beyond tokenize(prompt), and a turn whose prompt tokens
  are not a prefix of the joint tokenization is dropped and counted.

Usage: python curation_marks_rows.py --marks <jsonl> --kotodama-src <src> --out <jsonl>
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--marks", required=True, type=Path)
    ap.add_argument("--kotodama-src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    sys.path.insert(0, str(args.kotodama_src))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from kotodama import presets
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(presets.TOKENIZER_NAME)
    rows = [json.loads(l) for l in args.marks.read_text().splitlines() if l.strip()]
    by_scene: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_scene[str(r["scene"])].append(r)
    counts = collections.Counter()
    out = []
    for scene, turns in sorted(by_scene.items()):
        turns.sort(key=lambda r: int(r["turn"]))
        for i, r in enumerate(turns):
            span = " " + r["span_text"]
            if i + 1 < len(turns):
                span += f"\n\n{turns[i + 1]['speaker']}:"
            p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
            joint = tok(r["prompt"] + span, add_special_tokens=False)["input_ids"]
            if joint[:len(p)] != p or len(joint) - len(p) < 2:
                counts["prompt_not_prefix_or_empty"] += 1
                continue
            counts["kept"] += 1
            out.append(dict(id=f"mark:{scene}:{r['turn']}", ids=joint, prompt_length=len(p), injection=None,
                            steer_span="decode",
                            meta={k: r.get(k) for k in ("scene", "writer", "seed", "turn", "speaker", "mark",
                                                        "within_her_labelling", "scene_verdict")}))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(dict(turns=len(rows), scenes=len(by_scene), counts=dict(counts), out=str(args.out))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
