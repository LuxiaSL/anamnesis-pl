#!/usr/bin/env python3
"""NODE2 validation gate, part (b): cross-node byte-compare.

Compares node2-produced replay signatures (the replay smoke's NEW-path output,
sig_smoke_new/) against the node1-banked signatures_v3/ of the same cells.

Pass criterion (scoped, declared before running):
  - every .npz byte-identical (the feature vectors — the real content);
  - every .json identical FIELD-WISE except `injection.inject_npz`, which was
    deliberately rewritten on the node2 copy (node1 bank path -> /dev/shm bank path)
    and is provenance, not measurement.
ANY other diff = STOP-AND-SURFACE per NODE2-OPS rule 6.

Usage:
  python annex_node2_gate_compare.py --new <cell>/sig_smoke_new --ref <cell>/signatures_v3
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


def compare_cell(new_dir: Path, ref_dir: Path) -> tuple[int, int, list[str]]:
    diffs: list[str] = []
    npz_new = sorted(new_dir.glob("*.npz"))
    if not npz_new:
        return 0, 0, [f"{new_dir}: no .npz files found"]
    n_npz = n_json = 0
    for np_new in npz_new:
        ref = ref_dir / np_new.name
        if not ref.exists():
            diffs.append(f"{np_new.name}: missing in reference")
            continue
        n_npz += 1
        if np_new.read_bytes() != ref.read_bytes():
            diffs.append(f"{np_new.name}: NPZ BYTE DIFF")
    for js_new in sorted(new_dir.glob("*.json")):
        ref = ref_dir / js_new.name
        if not ref.exists():
            diffs.append(f"{js_new.name}: missing in reference")
            continue
        n_json += 1
        a = json.loads(js_new.read_text())
        b = json.loads(ref.read_text())
        a2, b2 = copy.deepcopy(a), copy.deepcopy(b)
        for d in (a2, b2):
            if isinstance(d.get("injection"), dict):
                d["injection"].pop("inject_npz", None)
        if a2 != b2:
            keys = sorted(set(a2) | set(b2))
            bad = [k for k in keys if a2.get(k) != b2.get(k)]
            diffs.append(f"{js_new.name}: JSON FIELD DIFF in {bad}")
    return n_npz, n_json, diffs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=Path, required=True, nargs="+")
    ap.add_argument("--ref", type=Path, required=True, nargs="+")
    args = ap.parse_args()
    if len(args.new) != len(args.ref):
        raise SystemExit("--new and --ref must pair up")
    total_npz = total_json = 0
    all_diffs: list[str] = []
    for nd, rd in zip(args.new, args.ref):
        n_npz, n_json, diffs = compare_cell(nd, rd)
        total_npz += n_npz
        total_json += n_json
        all_diffs += [f"[{nd.parent.name}] {d}" for d in diffs]
        print(f"{nd.parent.name}: {n_npz} npz byte-compared, {n_json} json field-compared, "
              f"{len(diffs)} diffs")
    if all_diffs:
        print("\n".join(all_diffs))
        print("GATE-B: FAIL — STOP-AND-SURFACE (node2 results node2-scoped pending ruling)")
        sys.exit(1)
    print(f"GATE-B: PASS — {total_npz} npz bitwise-identical across nodes, "
          f"{total_json} json identical (inject_npz path field excepted, as declared)")


if __name__ == "__main__":
    main()
