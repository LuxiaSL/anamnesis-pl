"""Curation writer x reader grid: exact-id import rows for the joined curation turns, to drain under every reader.

Reads ``curation_r2_join.py`` output dirs (rows.jsonl carries each turn's sign_id) and the spools those rows were
joined from, and writes one import row per turn: {"id": "<scene_id>:<turn>", "ids": prompt + generated,
"prompt_length", "injection": null, "steer_span": "decode", "meta": {scene_id, turn, writer, writer_weights,
sign_id}}. Draining the file under a reader (``kotodama.serve.sign --import_rows``) signs the turn's exact tokens
under that reader's weights, as the lineage read does.

Usage: python curation_grid_rows.py --join <dir> [--join ...] --spool <dir> [--spool ...] --out <jsonl>
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--join", required=True, type=Path, action="append")
    ap.add_argument("--spool", required=True, type=Path, action="append")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    want: dict[str, dict] = {}
    for d in args.join:
        for line in (d / "rows.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                want[r["sign_id"]] = r
    found: dict[str, dict] = {}
    for spool in args.spool:
        for path in sorted(glob.glob(str(spool / "records" / "*.jsonl"))):
            for line in Path(path).read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                rid = rec.get("request_id")
                if rid in want and rid not in found:
                    found[rid] = rec
    missing = sorted(set(want) - set(found))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for sid, r in sorted(want.items(), key=lambda kv: kv[1]["cand_id"]):
            rec = found.get(sid)
            if rec is None:
                continue
            fh.write(json.dumps(dict(
                id=r["cand_id"], ids=list(rec["prompt_ids"]) + list(rec["generated_ids"]),
                prompt_length=len(rec["prompt_ids"]), injection=None, steer_span="decode",
                meta=dict(scene_id=r["scene_id"], turn=r["turn"], writer=r["writer"],
                          writer_weights=rec["weights"], sign_id=sid))) + "\n")
    summary = dict(wanted=len(want), written=len(found), missing=len(missing), out=str(args.out))
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
