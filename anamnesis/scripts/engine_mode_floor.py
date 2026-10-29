"""Floor between two drains of the same signed records: per-row vs batched engine reduction.

Reads two signed directories written by ``python -m kotodama.serve.sign`` (``rows.jsonl``,
``features-<digest>.npy``, ``names.json`` per batch), joins rows by ``sign_id``, and for every
feature reports how far the two reductions disagree in units of that feature's across-row SD
(taken from the first drain). It reports:
  per row    the z-L2 of the difference over all features both rows carry;
  per family the max and median |z| per feature family (``anamnesis.analysis.feature_map``
             when importable, else the name's first token);
  gate top-k the ``gate_topk`` names on their own, since their tie rule differs by design.

Usage: python engine_mode_floor.py --a <signed-a> --b <signed-b> --out <json>
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_signed(root: Path) -> dict[str, tuple[list[str], np.ndarray]]:
    """sign_id -> (feature names, float64 vector), over every batch directory under root."""
    out: dict[str, tuple[list[str], np.ndarray]] = {}
    for rows_path in sorted(glob.glob(str(root / "**" / "rows.jsonl"), recursive=True)):
        batch = Path(rows_path).parent
        names_by_digest = json.loads((batch / "names.json").read_text())
        arrays: dict[str, np.ndarray] = {}
        for line in Path(rows_path).read_text().splitlines():
            r = json.loads(line)
            if r.get("skipped") or "file" not in r:
                continue
            f = r["file"]
            if f not in arrays:
                arrays[f] = np.load(batch / f)
            names = names_by_digest[r["names_digest"]]
            out[r["sign_id"]] = (list(names), arrays[f][int(r["index"])].astype(np.float64))
    return out


def family_of(name: str, classify) -> str:
    if classify is not None:
        try:
            return str(classify(name, 72).family)
        except Exception:
            pass
    return name.split("_")[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, type=Path, help="reference drain (its across-row SD scales z)")
    ap.add_argument("--b", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    try:
        from anamnesis.analysis.feature_map import classify
    except ImportError:
        classify = None

    a, b = load_signed(args.a), load_signed(args.b)
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit("no sign_ids in common")
    # Every feature is compared over the rows where both drains carry it: a short span's schema
    # omits names a longer one has, so intersecting with any one row would drop features.
    order: list[str] = []
    seen: set[str] = set()
    for sid in common:
        for n in a[sid][0]:
            if n not in seen:
                seen.add(n)
                order.append(n)
    ia = {sid: {n: i for i, n in enumerate(a[sid][0])} for sid in common}
    ib = {sid: {n: i for i, n in enumerate(b[sid][0])} for sid in common}
    per_row_sq = np.zeros(len(common))
    bitwise = np.ones(len(common), dtype=bool)
    max_abs_raw = 0.0
    live = 0
    fam: dict[str, list[float]] = defaultdict(list)
    for n in order:
        rows = [k for k, sid in enumerate(common) if n in ia[common[k]] and n in ib[common[k]]]
        if not rows:
            continue
        va = np.array([a[common[k]][1][ia[common[k]][n]] for k in rows])
        vb = np.array([b[common[k]][1][ib[common[k]][n]] for k in rows])
        diff = vb - va
        bitwise[rows] &= diff == 0
        max_abs_raw = max(max_abs_raw, float(np.abs(diff).max()))
        sd = va.std()
        if sd <= 1e-12:
            continue
        live += 1
        z = diff / sd
        per_row_sq[rows] += z ** 2
        fam["gate_topk" if "topk" in n else family_of(n, classify)].append(float(np.abs(z).max()))
    row_l2 = np.sqrt(per_row_sq)
    report = dict(
        rows=len(common), features_compared=len(order), features_live=live,
        bitwise_rows=int(bitwise.sum()),
        max_abs_raw=max_abs_raw,
        row_z_l2=dict(median=float(np.median(row_l2)), p90=float(np.percentile(row_l2, 90)),
                      max=float(row_l2.max())),
        families={k: dict(n=len(v), max_abs_z=round(max(v), 6), median_max_abs_z=round(float(np.median(v)), 6))
                  for k, v in sorted(fam.items(), key=lambda kv: -max(kv[1]))},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("rows", "features_live", "bitwise_rows", "max_abs_raw", "row_z_l2")}))
    for k, v in list(report["families"].items())[:12]:
        print(k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
