"""Lane alias check: the same rows signed on two tree pairs (``alias-check-node2.sh``) must be bitwise equal row for
row, with only the lane id (and the identity fields that name it) moved. Optionally, the old run's lane id is
compared with an existing spool's (the lane the alias is declared for).

Usage: python lane_alias_check.py --old <signed dir> --new <signed dir> [--reference <signed dir>] --json <out>
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p2 = _sibling("msteer_p2_read")


def engine_identities(root: Path) -> list[dict]:
    return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(root / "*" / "engine.json")))]


def flat(d: object, prefix: str = "") -> dict[str, object]:
    if isinstance(d, dict):
        out: dict[str, object] = {}
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}."))
        return out
    return {prefix.rstrip("."): d}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True, type=Path)
    ap.add_argument("--new", required=True, type=Path)
    ap.add_argument("--reference", type=Path, default=None)
    ap.add_argument("--json", required=True, type=Path)
    args = ap.parse_args(argv)
    old, new = p2.load_signed(args.old), p2.load_signed(args.new)
    ids = sorted(set(old) & set(new))
    equal = sum(old[i]["names"] == new[i]["names"]
                and old[i]["vec"].astype(np.float32).tobytes() == new[i]["vec"].astype(np.float32).tobytes()
                for i in ids)
    lanes = {k: sorted({v[i]["rec"]["lane_id"] for i in ids}) for k, v in (("old", old), ("new", new))}
    eo, en = engine_identities(args.old), engine_identities(args.new)
    fo, fn = flat(eo[0]) if eo else {}, flat(en[0]) if en else {}
    engine_diff = {k: [fo.get(k), fn.get(k)] for k in sorted(set(fo) | set(fn)) if fo.get(k) != fn.get(k)}
    ro = flat((old[ids[0]]["rec"].get("reducer_meta") or {}).get("receipt", {}).get("batched", {})) if ids else {}
    rn = flat((new[ids[0]]["rec"].get("reducer_meta") or {}).get("receipt", {}).get("batched", {})) if ids else {}
    batched_diff = {k: [ro.get(k), rn.get(k)] for k in sorted(set(ro) | set(rn)) if ro.get(k) != rn.get(k)}
    out: dict[str, object] = dict(rows=dict(old=len(old), new=len(new)), common=len(ids), bitwise_equal=equal,
                                  lane_ids=lanes, engine_identity_diff=engine_diff, batched_receipt_diff=batched_diff)
    if args.reference is not None:
        ref_lanes_set: set[str] = set()
        rows_read = 0
        for rows_path in sorted(glob.glob(str(args.reference / "*" / "rows.jsonl")))[:5]:  # lane ids only
            for line in Path(rows_path).read_text().splitlines():
                r = json.loads(line)
                if "lane_id" in r:
                    ref_lanes_set.add(r["lane_id"])
                    rows_read += 1
        ref_lanes = sorted(ref_lanes_set)
        out["reference"] = dict(path=str(args.reference), rows_read=rows_read, lane_ids=ref_lanes,
                                old_lane_in_reference=all(l in ref_lanes for l in lanes["old"]))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
