"""Column C replay rows for the momentum-steer children that no lever touched (the Column C freeze's bridge data).

From momentum-steer phase 2's import rows (``msteer_import_rows.py``), keep the unworn signing of the natural arms:
the ``unworn`` sampler arm and the two other samplers (``T1.1``, ``min_p.05``). Their ids are the exact sampled ids
(context_ids + completion_ids), so Column C reads the same tokens lane A signed. One node per context: every
child of a context shares its context ids, so the node's prefill is shared by its 24 siblings (3 arms x 8 slots).

Output rows (``column_c_extract_koto.py`` format): cand_id = the phase-2 sign_id, node_id = ``msteer:c<ctx>``,
input_ids, prompt_length, n_tokens, and meta (arm, slot, context_index).

Usage: python msteer_colc_rows.py --import-rows <p2 import-rows.jsonl> --out <jsonl>
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

NATURAL_ARMS = ("unworn", "T1.1", "min_p.05")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--import-rows", required=True, type=Path)
    ap.add_argument("--arms", default=",".join(NATURAL_ARMS))
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    arms = set(args.arms.split(","))
    out: list[dict] = []
    ctx_ids: dict[int, list[int]] = {}
    counts: collections.Counter = collections.Counter()
    for line in args.import_rows.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        m = r["meta"]
        if m.get("wear") != "unworn" or m.get("arm") not in arms:
            continue
        P = int(r["prompt_length"])
        c = int(m["context_index"])
        ctx = r["ids"][:P]
        if ctx_ids.setdefault(c, ctx) != ctx:
            raise ValueError(f"context {c}: children do not share context ids ({r['id']})")
        if len(r["ids"]) - P < 1:
            counts["no_answer"] += 1
            continue
        counts[m["arm"]] += 1
        out.append(dict(cand_id=r["id"], node_id=f"msteer:c{c:03d}", input_ids=r["ids"], prompt_length=P,
                        n_tokens=len(r["ids"]) - P,
                        meta=dict(arm=m["arm"], slot=m["slot"], context_index=c, node_id=m.get("node_id"))))
    out.sort(key=lambda r: (r["node_id"], r["meta"]["arm"], r["meta"]["slot"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        fh.write(json.dumps(dict(_header=True, version="msteer-colc-rows-v1", source=str(args.import_rows),
                                 arms=sorted(arms), construction="exact sampled ids from phase-2 import rows "
                                 "(context_ids + completion_ids); unworn signing only")) + "\n")
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(dict(rows=len(out), contexts=len(ctx_ids), counts=dict(counts), out=str(args.out))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
