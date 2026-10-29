"""Convert posttraining's momentum-steer growth into replay rows for the tsumugi signer.

Input: a momentum-steer directory holding ``growth.jsonl`` (one row per
(context_index, arm), with a ``descendants`` list) and, optionally,
``grades.jsonl`` (one row per (context_index, arm, slot, depth)).

Each depth-1 descendant that carries ``next_context_ids`` becomes one replay row:

  input_ids     = next_context_ids
  prompt_length = next_last_turn_start  (the replay span is input_ids[prompt_length:])
  n_tokens      = len(input_ids) - prompt_length

Descendants without ``next_context_ids`` (turns that ended the scene, and the
depth-2 replies, which carry no next context) are counted in the header and
skipped. The text was generated under the arm's lever; the replay signs it with
the lever off (teacher-forced, natural forward pass).

``cand_id`` is deterministic: ``msteer:c<context_index:03d>:<arm>:s<slot>``.
Children at one context share a seed per slot across arms, so (context_index,
slot) is the matched node.

Output: a JSONL with a header row (``_header: true``) and one row per child, in
the shape ``sign_koto_replay_rows.py`` reads.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cand_id(context_index: int, arm: str, slot: int) -> str:
    return f"msteer:c{context_index:03d}:{arm}:s{slot}"


def load_grades(path: Path) -> dict[tuple[int, str, int, int], dict[str, Any]]:
    grades: dict[tuple[int, str, int, int], dict[str, Any]] = {}
    if not path.exists():
        return grades
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("_header") or rec.get("header"):
                continue
            try:
                key = (int(rec["context_index"]), str(rec["arm"]), int(rec["slot"]), int(rec["depth"]))
            except (KeyError, TypeError, ValueError):
                continue
            if key in grades:
                raise ValueError(f"duplicate grade key {key}")
            grades[key] = rec
    return grades


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="momentum-steer directory")
    ap.add_argument("--out", required=True, type=Path, help="replay-rows JSONL to write")
    args = ap.parse_args(argv)

    growth = args.src / "growth.jsonl"
    if not growth.exists():
        print(f"REFUSED: {growth} missing", file=sys.stderr)
        return 2
    grades = load_grades(args.src / "grades.jsonl")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for line in growth.open():
        rec = json.loads(line)
        ci, arm = int(rec["context_index"]), str(rec["arm"])
        for d in rec["descendants"]:
            depth, slot = int(d["depth"]), int(d["slot"])
            if depth != 1:
                counts[arm]["skipped_depth2"] += 1
                continue
            if "next_context_ids" not in d:
                counts[arm][f"skipped_no_ids_{d.get('ended_by') or d.get('state')}"] += 1
                continue
            ids = [int(t) for t in d["next_context_ids"]]
            P = int(d["next_last_turn_start"])
            if not 0 < P < len(ids):
                counts[arm]["skipped_bad_prompt_length"] += 1
                continue
            cid = cand_id(ci, arm, slot)
            if cid in seen:
                raise ValueError(f"duplicate cand_id {cid}")
            seen.add(cid)
            g = grades.get((ci, arm, slot, depth), {})
            rows.append(dict(
                cand_id=cid, input_ids=ids, prompt_length=P, arm=arm, context_index=ci, slot=slot,
                n_tokens=len(ids) - P, n_completion_ids=len(d.get("completion_ids") or []),
                ended_by=d.get("ended_by"), n_chars=len(d.get("text") or ""),
                grade=g.get("grade"), grade_how=g.get("grade_how"),
                breaks_opus=g.get("breaks_opus"), breaks_fable=g.get("breaks_fable"),
            ))
            counts[arm]["rows"] += 1
            counts[arm]["graded"] += int(g.get("grade") is not None)

    header = dict(
        _header=True, version="msteer-replay-rows-v1",
        source=str(growth), source_sha256=sha256_file(growth),
        grades=str(args.src / "grades.jsonl") if grades else None,
        construction=("input_ids = next_context_ids (the server's ids: context + the generated turn + the next "
                      "speaker tag); prompt_length = next_last_turn_start; span = input_ids[prompt_length:]. "
                      "Depth-1 children with next_context_ids only. Replay is teacher-forced with no lever."),
        cand_id="msteer:c<context_index:03d>:<arm>:s<slot>",
        counts={a: dict(c) for a, c in sorted(counts.items())},
        n_rows=len(rows),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        fh.write(json.dumps(header) + "\n")
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(header["counts"], indent=1))
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
