"""Import rows for the lineage read: one writer's spool records, to be replayed through several readers.

Each record in a sign spool (``kotodama-sign-record/1``) holds the exact ids a server generated:
``prompt_ids``, ``generated_ids`` and the writer's ``weights`` digest. This script samples up to ``--per-writer``
records from each writer digest asked for, seeded. It keeps generated spans of ``--min-steps`` to ``--max-steps``
tokens and skips steered records, then writes them as ``kotodama.serve.sign --import_rows`` rows:

  {"id": "lin:<writer label>:<request_id>", "ids": prompt + generated, "prompt_length": len(prompt),
   "injection": null, "steer_span": "decode", "meta": {"writer": label, "writer_weights": digest, ...}}

The same file is drained once per reader checkpoint. A row's bytes then differ across drains only by the reader's
weights, so the reader effect is a declared lineage contrast under one reference calibration.

Usage: python lineage_import_rows.py --records <records/*.jsonl> --writer sft=<digest> --writer dpo=<digest>
       --per-writer 1500 --out <rows.jsonl>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True, type=Path, nargs="+")
    ap.add_argument("--writer", required=True, action="append", help="label=<64-hex weights digest>")
    ap.add_argument("--per-writer", type=int, default=1500)
    ap.add_argument("--min-steps", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=255)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    writers: dict[str, str] = {}
    for spec in args.writer:
        label, _, digest = spec.partition("=")
        if len(digest) != 64 or not label:
            raise SystemExit(f"--writer wants label=<64-hex digest>, got {spec!r}")
        writers[digest] = label
    pool: dict[str, list[dict[str, Any]]] = {d: [] for d in writers}
    counts: Counter[str] = Counter()
    for path in args.records:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("format") != "kotodama-sign-record/1":
                counts["skip_format"] += 1
                continue
            d = r.get("weights")
            if d not in pool:
                counts["skip_other_writer"] += 1
                continue
            if r.get("lever") or r.get("steering") or r.get("adapter"):
                counts["skip_steered"] += 1
                continue
            n = len(r["generated_ids"])
            if not args.min_steps <= n - 1 <= args.max_steps:
                counts["skip_length"] += 1
                continue
            pool[d].append(r)
    rng = random.Random(args.seed)
    out_rows: list[dict[str, Any]] = []
    for d, label in writers.items():
        recs = sorted(pool[d], key=lambda r: r["request_id"])
        take = rng.sample(recs, min(args.per_writer, len(recs)))
        counts[f"available_{label}"] = len(recs)
        counts[f"rows_{label}"] = len(take)
        for r in sorted(take, key=lambda r: r["request_id"]):
            meta = r.get("meta") or {}
            out_rows.append(dict(
                id=f"lin:{label}:{r['request_id']}",
                ids=list(r["prompt_ids"]) + list(r["generated_ids"]),
                prompt_length=len(r["prompt_ids"]), injection=None, steer_span="decode",
                meta=dict(writer=label, writer_weights=d, request_id=r["request_id"],
                          served_by=meta.get("served_by"), law=meta.get("law"),
                          n_generated=len(r["generated_ids"]))))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for row in out_rows:
            fh.write(json.dumps(row) + "\n")
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    summary = dict(out=str(args.out), out_sha256=digest, records=[str(p) for p in args.records], **counts)
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
