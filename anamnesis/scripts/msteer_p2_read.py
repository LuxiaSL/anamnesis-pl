"""Momentum-steer phase 2: the worn-vs-unworn gap on the engine lane.

Reads one signed directory written by ``python -m kotodama.serve.sign --import_rows`` over the rows from
``msteer_import_rows.py``: lever children twice (``<cid>.worn`` under the lever, ``<cid>.unworn`` with none, the
same token ids), and the unworn and sampler arms once.

Every feature is z-scored by its across-row SD over all unworn rows. Per lever arm it reports:
  gap        per child, the z-L2 of worn - unworn over the features both carry (median, p10, p90), next to two
             scales: sibling spread (z-L2 between two unworn-arm slots at one context: a natural resample) and
             the engine floor of record (0.036, per-row vs batched reducer);
  families   the median |z| of worn - unworn per feature family;
  direction  the mean worn - unworn delta per arm; its split-half reliability over contexts (cosine of the
             mean delta on even vs odd context_index) and the cosine between arms (P vs -P should be negative,
             doses of one lever positive).

Usage: python msteer_p2_read.py --signed <signed_dir> --out <json>
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import sys
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ENGINE_FLOOR_Z_L2 = 0.036


def load_signed(root: Path) -> dict[str, dict[str, Any]]:
    """row id -> {names, vec, meta}, over every batch directory under root."""
    out: dict[str, dict[str, Any]] = {}
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
            meta = r.get("meta") or {}
            rid = meta.get("id") or r.get("id") or r["sign_id"]
            out[str(rid)] = dict(names=list(names_by_digest[r["names_digest"]]),
                                 vec=arrays[f][int(r["index"])].astype(np.float64), meta=meta, rec=r)
    return out


def family_of(name: str, classify: Any) -> str:
    if classify is not None:
        try:
            return str(classify(name, 72).family)
        except Exception:
            pass
    return name.split("_")[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signed", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--feature-map", type=Path, default=None, help="staged feature_map.py (else import)")
    args = ap.parse_args(argv)
    classify = None
    try:
        if args.feature_map is not None:
            spec = importlib.util.spec_from_file_location("feature_map_staged", args.feature_map)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["feature_map_staged"] = mod
            spec.loader.exec_module(mod)
            classify = mod.classify
        else:
            from anamnesis.analysis.feature_map import classify
    except Exception as exc:
        print(f"[read] feature_map.classify unavailable ({type(exc).__name__}: {exc}); families by name prefix")

    rows = load_signed(args.signed)
    if not rows:
        raise SystemExit(f"no signed rows under {args.signed}")
    # The feature axis: every name any row carries, in first-seen order; absent entries are NaN.
    order: list[str] = []
    pos: dict[str, int] = {}
    for r in rows.values():
        for n in r["names"]:
            if n not in pos:
                pos[n] = len(order)
                order.append(n)
    ids = sorted(rows)
    X = np.full((len(ids), len(order)), np.nan)
    for i, rid in enumerate(ids):
        r = rows[rid]
        X[i, [pos[n] for n in r["names"]]] = r["vec"]
    row_of = {rid: i for i, rid in enumerate(ids)}
    unworn = np.array([rid.endswith(".unworn") for rid in ids])
    sd = np.nanstd(X[unworn], axis=0)
    live = np.isfinite(sd) & (sd > 1e-12)
    Z = X[:, live] / sd[live]
    names = [n for n, ok in zip(order, live) if ok]
    fam = np.array(["gate_topk" if "topk" in n else family_of(n, classify) for n in names])

    def z_l2(a: np.ndarray, b: np.ndarray) -> float:
        d = a - b
        return float(np.sqrt(np.nansum(d ** 2)))

    def cid_parts(rid: str) -> tuple[int, str, int]:
        _, c, arm, s = rid.rsplit(".", 1)[0].split(":")
        return int(c[1:]), arm, int(s[1:])

    # Lever children: worn - unworn on the same ids.
    deltas: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    gaps: dict[str, list[float]] = defaultdict(list)
    by_payload: dict[str, list[np.ndarray]] = defaultdict(list)
    missing_twin = 0
    for rid in ids:
        if not rid.endswith(".worn"):
            continue
        twin = rid[: -len(".worn")] + ".unworn"
        if twin not in row_of:
            missing_twin += 1
            continue
        ci, arm, _ = cid_parts(rid)
        d = Z[row_of[rid]] - Z[row_of[twin]]
        deltas[arm].append((ci, d))
        by_payload[str(rows[rid]["meta"].get("payload") or arm)].append(d)
        gaps[arm].append(float(np.sqrt(np.nansum(d ** 2))))

    # Sibling spread: two unworn-arm slots at one context (a natural resample of the same node).
    by_ctx: dict[int, list[int]] = defaultdict(list)
    for rid in ids:
        if rid.endswith(".unworn") and ":unworn:" in rid:
            by_ctx[cid_parts(rid)[0]].append(row_of[rid])
    sib = [z_l2(Z[v[0]], Z[v[1]]) for v in by_ctx.values() if len(v) >= 2]

    def pct(v: list[float]) -> dict[str, float]:
        a = np.asarray(v)
        return dict(n=len(a), median=float(np.median(a)), p10=float(np.percentile(a, 10)),
                    p90=float(np.percentile(a, 90)))

    mean_delta: dict[str, np.ndarray] = {}
    arms: dict[str, Any] = {}
    for arm, lst in sorted(deltas.items()):
        D = np.stack([d for _, d in lst])
        ctx = np.array([c for c, _ in lst])
        m = np.nanmean(D, axis=0)
        mean_delta[arm] = m
        even, odd = np.nanmean(D[ctx % 2 == 0], axis=0), np.nanmean(D[ctx % 2 == 1], axis=0)
        ok = np.isfinite(even) & np.isfinite(odd)
        split = float(even[ok] @ odd[ok] / (np.linalg.norm(even[ok]) * np.linalg.norm(odd[ok]) + 1e-30))
        absd = np.nanmedian(np.abs(D), axis=0)
        fams = {f: float(np.nanmedian(absd[fam == f])) for f in np.unique(fam)}
        top = sorted(fams.items(), key=lambda kv: -kv[1])[:10]
        bitwise = int(sum(1 for g in gaps[arm] if g == 0.0))
        arms[arm] = dict(gap=pct(gaps[arm]), bitwise_equal_children=bitwise, split_half_cos=round(split, 4),
                         mean_delta_norm=float(np.linalg.norm(np.nan_to_num(m))), top_families=top)

    keys = sorted(mean_delta)
    cos = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            x, y = mean_delta[a], mean_delta[b]
            ok = np.isfinite(x) & np.isfinite(y)
            cos[f"{a}|{b}"] = round(float(x[ok] @ y[ok] / (np.linalg.norm(x[ok]) * np.linalg.norm(y[ok]) + 1e-30)), 4)

    def cosine(x: np.ndarray, y: np.ndarray) -> float:
        ok = np.isfinite(x) & np.isfinite(y)
        return round(float(x[ok] @ y[ok] / (np.linalg.norm(x[ok]) * np.linalg.norm(y[ok]) + 1e-30)), 4)

    # Per payload: the three R directions at one dose answer whether the response follows the lever's direction
    # or is a generic response to being steered at all.
    pmean = {k: np.nanmean(np.stack(v), axis=0) for k, v in sorted(by_payload.items())}
    payload_cos = {f"{a}|{b}": cosine(pmean[a], pmean[b])
                   for i, a in enumerate(sorted(pmean)) for b in sorted(pmean)[i + 1:]}
    # Generic component: the unit mean of the R arms' mean deltas. Each arm's cosine with it, and the arm
    # cosines after projecting it out.
    generic = None
    r_arms = [k for k in keys if k.startswith("R_")]
    if r_arms:
        g = np.nanmean(np.stack([mean_delta[k] for k in r_arms]), axis=0)
        g = np.nan_to_num(g)
        generic = g / (np.linalg.norm(g) + 1e-30)
    resid = {}
    if generic is not None:
        for k in keys:
            m = np.nan_to_num(mean_delta[k])
            resid[k] = m - (m @ generic) * generic
    generic_cos = {k: cosine(mean_delta[k], generic) for k in keys} if generic is not None else {}
    resid_cos = {f"{a}|{b}": cosine(resid[a], resid[b]) for i, a in enumerate(keys) for b in keys[i + 1:]} if resid else {}

    report = dict(signed=str(args.signed), rows=len(ids), features_live=int(live.sum()),
                  missing_twin=missing_twin, engine_floor_z_l2=ENGINE_FLOOR_Z_L2,
                  sibling_spread=pct(sib) if sib else None, arms=arms, arm_cosines=cos,
                  payload_cosines=payload_cos, generic_cosine=generic_cos, cosines_generic_removed=resid_cos)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(rows=len(ids), features_live=int(live.sum()), missing_twin=missing_twin,
                          sibling_spread=report["sibling_spread"])))
    for arm, v in arms.items():
        print(arm, json.dumps(dict(gap=v["gap"], split=v["split_half_cos"], bitwise=v["bitwise_equal_children"])))
    for k, v in cos.items():
        print("cos", k, v)
    for k, v in payload_cos.items():
        if k.startswith("R") and k.split("|")[1].startswith("R") and k.split("_")[1][:2] == k.split("_")[-1][:2]:
            print("payload cos", k, v)
    for k, v in generic_cos.items():
        print("generic cos", k, v)
    for k, v in resid_cos.items():
        print("resid cos", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
