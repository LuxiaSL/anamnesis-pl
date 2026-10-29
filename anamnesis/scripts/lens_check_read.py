"""The lens family's check on real rows: three drains of one import file (``lens-drain-node2.sh``).

* plain vs lens: the lens rows' leading features must be the plain rows' bytes (lens off changes nothing);
* lens vs lens1 (``max_group_rows=1``): bitwise equal if the lens is batchmate-invariant;
* ``lens_final_check``: per row, the largest gap between the lens's last-layer entropy and the step's own;
* the lens profile: per sampled layer, the median over rows of each lens quantity, and ``lens_depth``;
* cost: the drains' reducer seconds per row, lens vs plain.

Usage: python lens_check_read.py --out-dir <drain out dir> --json <out json>
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
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


def drain_timing(log: Path) -> dict[str, object]:
    """The drain's batch summaries: wall seconds summed, the signer's (cumulative) timing at the end, and each
    batch's own reduce seconds per row (the difference of consecutive cumulative timings)."""
    batches: list[dict] = []
    for line in log.read_text(errors="replace").splitlines():
        m = re.search(r"sign spool: (\{.*\})\s*$", line)
        if not m:
            continue
        try:
            batches.append(ast.literal_eval(m.group(1)))
        except (ValueError, SyntaxError):
            continue
    if not batches:
        return {}
    per_batch, prev_reduce, prev_rows = [], 0.0, 0
    for s in batches:
        t = s.get("signer_timing") or {}
        rows = int(t.get("rows", 0)) - prev_rows
        reduce_s = float(t.get("reduce_s", 0.0)) - prev_reduce
        per_batch.append(dict(rows=rows, wall_s=float(s.get("seconds", 0.0)),
                              reduce_ms_per_row=round(1000 * reduce_s / max(rows, 1), 1)))
        prev_reduce, prev_rows = float(t.get("reduce_s", 0.0)), int(t.get("rows", 0))
    return dict(wall_s=round(sum(float(s.get("seconds", 0.0)) for s in batches), 2),
                signer_timing=batches[-1].get("signer_timing"), batches=per_batch)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--json", required=True, type=Path)
    args = ap.parse_args(argv)
    runs = {r: p2.load_signed(args.out_dir / r / "signed") for r in ("plain", "lens", "lens1")}
    ids = sorted(set.intersection(*(set(v) for v in runs.values())))
    out: dict[str, object] = dict(rows={r: len(v) for r, v in runs.items()}, common=len(ids))

    prefix_equal = lens_equal = 0
    lane_ids = {r: sorted({runs[r][i]["rec"]["lane_id"] for i in ids}) for r in runs}
    checks: list[float] = []
    lens_rows: list[np.ndarray] = []
    lens_names: list[str] | None = None
    for i in ids:
        a, b, c = runs["plain"][i], runs["lens"][i], runs["lens1"][i]
        n = len(a["names"])
        if b["names"][:n] == a["names"] and b["vec"][:n].astype(np.float32).tobytes() == a["vec"].astype(np.float32).tobytes():
            prefix_equal += 1
        if b["names"] == c["names"] and b["vec"].astype(np.float32).tobytes() == c["vec"].astype(np.float32).tobytes():
            lens_equal += 1
        receipt = (b["rec"].get("reducer_meta") or b["rec"]).get("receipt") or {}
        if "lens_final_check" in receipt:
            checks.append(float(receipt["lens_final_check"]))
        extra = b["names"][n:]
        if lens_names is None:
            lens_names = extra
        if extra == lens_names:
            lens_rows.append(b["vec"][n:])
    out.update(lens_prefix_equals_plain=prefix_equal, lens_equals_lens1=lens_equal, lane_ids=lane_ids)
    if checks:
        q = np.quantile(checks, [0.5, 0.9, 0.99, 1.0])
        out["lens_final_check_nats"] = dict(n=len(checks), median=float(q[0]), p90=float(q[1]), p99=float(q[2]),
                                            max=float(q[3]))
    else:
        out["lens_final_check_nats"] = "missing: receipt keys " + ",".join(sorted(
            ((runs["lens"][ids[0]]["rec"].get("reducer_meta") or runs["lens"][ids[0]]["rec"]).get("receipt") or {}).keys()))
    if lens_rows and lens_names:
        X = np.stack(lens_rows)
        med = {name: float(np.median(X[:, j])) for j, name in enumerate(lens_names)}
        layers = sorted({int(m.group(1)) for name in lens_names if (m := re.match(r"lens_L(\d+)_", name))})
        out["profile"] = {str(layer): {q: round(med[f"lens_L{layer}_{q}"], 4)
                                       for q in ("entropy_mean", "agree_final_mean", "agree_chosen_mean",
                                                 "kl_final_mean")} for layer in layers}
        d = X[:, lens_names.index("lens_depth")]
        out["lens_depth"] = dict(median=float(np.median(d)), p10=float(np.quantile(d, .1)),
                                 p90=float(np.quantile(d, .9)))
    out["timing"] = {r: drain_timing(args.out_dir / r / "drain.log") for r in runs}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
