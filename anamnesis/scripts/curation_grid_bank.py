"""Curation writer x reader grid: one read-ready dir per (set, reader), plus the own-reader bitwise check.

Each reader's drain (``curation-grid-drain-node2.sh``) signed the joined turns' exact ids under that reader's
weights. For every join set (``curation_r2_join.py`` output: rows, labels, sample, support), this writes
``<out>/<set>-<reader>/`` with the set's rows, labels, sample and support, and ``sigs/signatures-0of1.npz`` from the
reader's drain (only turns the reader signed in the set's feature layout). So every cell of the grid is read by
``curation_turn_read.py`` and ``curation_decomp.py`` exactly as the round-2 read was.

The own-reader check: where a turn's reader is its writer, the grid's signature must equal the set's (the round-2
re-drain, on the aliased drain lane) byte for byte. A mismatch is reported, never silently kept.

Usage: python curation_grid_bank.py --set h4=<join dir> --set hyb=<join dir> --reader release=<signed dir>
       --reader sft=<signed dir> ... --writer-reader sft=sft --writer-reader b01x=dpo10x --out <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from collections import Counter
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
msr = _sibling("msteer_replay_read")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, action="append")
    ap.add_argument("--reader", required=True, action="append")
    ap.add_argument("--writer-reader", action="append", default=[])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    sets = {s.partition("=")[0]: Path(s.partition("=")[2]) for s in args.set}
    readers = {s.partition("=")[0]: Path(s.partition("=")[2]) for s in args.reader}
    own = {s.partition("=")[0]: s.partition("=")[2] for s in args.writer_reader}
    signed = {k: p2.load_signed(v) for k, v in readers.items()}
    summary: dict[str, object] = dict(readers={k: len(v) for k, v in signed.items()}, cells={}, own_reader={})
    for sname, sdir in sets.items():
        _, rows = msr.load_rows(sdir / "rows.jsonl")
        X0, ids0, names, _, _ = msr.load_bank(sdir / "sigs", "engine-stats-b1")
        base = {c: X0[i] for i, c in enumerate(ids0)}
        for rname, sig in signed.items():
            cell = args.out / f"{sname}-{rname}"
            (cell / "sigs").mkdir(parents=True, exist_ok=True)
            for f in ("rows.jsonl", "labels.jsonl", "sample.json", "support.tsv"):
                shutil.copyfile(sdir / f, cell / f)
            X, ids, lanes, lps = [], [], [], []
            skipped = Counter()
            same, differ = 0, []
            for cid in ids0:
                hit = sig.get(cid)
                if hit is None:
                    skipped["unsigned"] += 1
                    continue
                if hit["names"] != names:
                    skipped["layout"] += 1
                    continue
                v = hit["vec"].astype(np.float32)
                X.append(v)
                ids.append(cid)
                lanes.append(hit["rec"]["lane_id"])
                lps.append(float(hit["rec"]["reducer_meta"]["mean_logprob"]))
                if own.get(rows[cid].get("writer")) == rname:
                    if v.tobytes() == base[cid].astype(np.float32).tobytes():
                        same += 1
                    else:
                        differ.append(cid)
            np.savez(cell / "sigs" / "signatures-0of1.npz", full_readout=np.stack(X), cand_ids=np.array(ids),
                     lane_ids=np.array(lanes), mean_logprob=np.array(lps), full_names=np.array(names))
            summary["cells"][f"{sname}-{rname}"] = dict(rows=len(ids), skipped=dict(skipped),
                                                        lanes=dict(Counter(lanes)))
            if same or differ:
                summary["own_reader"][f"{sname}-{rname}"] = dict(bitwise=same, differ=len(differ),
                                                                  first_differ=differ[:5])
            print(f"[grid] {sname}-{rname}: {len(ids)} rows, skipped {dict(skipped)}, own-reader bitwise "
                  f"{same}, differ {len(differ)}", flush=True)
    (args.out / "grid_summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
