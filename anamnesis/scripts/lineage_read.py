"""Lineage read: the same tokens through different weights, and where the signature shifts.

Inputs: several signed directories written by ``kotodama.serve.sign --import_rows``, one drain per reader
checkpoint over ONE import file (``lineage_import_rows.py``), all on the same engine config, pins and reference
calibration. A row's bytes therefore differ across readers only by the weights: a declared lineage contrast
under one reference calibration.

``--signed label=dir`` names each reader. The first is the reference; features are z-scored by their across-row
SD under it. ``--contrast a-b`` names each reader contrast (D = Z_a - Z_b on the same rows), and
``--contrast-delta a-b=<deltas.tsv column>`` ties a contrast to its weight change in posttraining's ``deltas.tsv``
(per tensor: key, layer, module, numel, and rel columns).

For each contrast:
  magnitude      per row, the z-L2 of D (median, p10, p90), overall and per writer;
  sources        the median over rows of each source's RMS |D| (``feature_map.classify``);
  depth profile  the per-layer median |D| for layer-indexed features, per source;
  increments     the layer-to-layer increase of that profile, since an upstream change propagates into every
                 later layer's features;
  weights        the Spearman correlation, across layers, between the profile (and its increments) and that
                 layer's weight change: all tensors, the matrices (q/k/v/o, gate/up/down) and the AttnRes routing
                 queries (attn_res_query, mlp_res_query);
  writers        the cosine between the mean D on each writer's rows, and their magnitude ratio: does a reader
                 shift its own model's text differently?
Across contrasts: the cosine between mean Ds, to set beside the weight deltas' cosine.

Usage: python lineage_read.py --signed release=<dir> --signed sft=<dir> ... --contrast sft-release
       --contrast-delta sft-release=rel_A --deltas deltas.tsv --feature-map feature_map.py --out <json>
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"{name}.py not found beside {__file__}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p2 = _sibling("msteer_p2_read")

MATRICES = re.compile(r"\.(attn\.(q|k|v|o)_proj|ffn\.(gate|up|down)_proj)\.weight$")
ROUTING = re.compile(r"\.(attn|mlp)_res_query$")


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return None
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    return round(float(np.corrcoef(ra, rb)[0, 1]), 4)


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    return round(float(x[ok] @ y[ok] / (np.linalg.norm(x[ok]) * np.linalg.norm(y[ok]) + 1e-30)), 4)


def pct(v: np.ndarray) -> dict[str, float]:
    return dict(n=int(len(v)), median=round(float(np.median(v)), 4), p10=round(float(np.percentile(v, 10)), 4),
                p90=round(float(np.percentile(v, 90)), 4))


def layer_weights(path: Path, column: str, n_layers: int) -> dict[str, np.ndarray]:
    """Per layer: the numel-weighted RMS of ``column`` over all tensors, the matrices, and the routing queries."""
    acc = {k: (np.zeros(n_layers), np.zeros(n_layers)) for k in ("all", "matrices", "routing")}
    with path.open() as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            layer = int(r["layer"])
            if layer < 0:
                continue
            rel, n = float(r[column]), float(r["numel"])
            if not np.isfinite(rel):
                continue
            kinds = ["all"] + (["matrices"] if MATRICES.search(r["key"]) else []) + \
                    (["routing"] if ROUTING.search(r["key"]) else [])
            for k in kinds:
                s, w = acc[k]
                s[layer] += n * rel * rel
                w[layer] += n
    return {k: np.sqrt(s / np.where(w > 0, w, np.nan)) for k, (s, w) in acc.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signed", required=True, action="append", help="label=<signed dir>; the first is the reference")
    ap.add_argument("--contrast", required=True, action="append", help="a-b (reader labels)")
    ap.add_argument("--contrast-delta", action="append", default=[], help="a-b=<deltas.tsv rel column>")
    ap.add_argument("--deltas", type=Path, default=None)
    ap.add_argument("--feature-map", type=Path, required=True)
    ap.add_argument("--n-layers", type=int, default=72)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    spec = importlib.util.spec_from_file_location("feature_map_staged", args.feature_map)
    fm = importlib.util.module_from_spec(spec)
    sys.modules["feature_map_staged"] = fm
    spec.loader.exec_module(fm)

    readers: dict[str, dict[str, Any]] = {}
    for s in args.signed:
        label, _, path = s.partition("=")
        readers[label] = p2.load_signed(Path(path))
        print(f"[lineage] {label}: {len(readers[label])} rows", flush=True)
    labels = list(readers)
    common = sorted(set.intersection(*(set(r) for r in readers.values())))
    if not common:
        raise SystemExit("no row ids in common across readers")
    order: list[str] = []
    pos: dict[str, int] = {}
    for rid in common:
        for n in readers[labels[0]][rid]["names"]:
            if n not in pos:
                pos[n] = len(order)
                order.append(n)
    M = {}
    for label, rows in readers.items():
        X = np.full((len(common), len(order)), np.nan)
        for i, rid in enumerate(common):
            r = rows[rid]
            idx = [pos[n] for n in r["names"] if n in pos]
            vals = [v for n, v in zip(r["names"], r["vec"]) if n in pos]
            X[i, idx] = vals
        M[label] = X
    sd = np.nanstd(M[labels[0]], axis=0)
    live = np.isfinite(sd) & (sd > 1e-12)
    names = [n for n, ok in zip(order, live) if ok]
    Z = {k: v[:, live] / sd[live] for k, v in M.items()}
    tags = [fm.classify(n, args.n_layers) for n in names]
    source = np.array([str(getattr(t.source, "value", t.source)) for t in tags])
    layer = np.array([t.layer if t.layer is not None else -1 for t in tags])
    writer = np.array([rid.split(":")[1] if rid.startswith("lin:") else "?" for rid in common])
    print(f"[lineage] rows {len(common)}, features {len(names)}, writers {sorted(set(writer))}", flush=True)

    deltas = {}
    for spec_ in args.contrast_delta:
        c, _, col = spec_.partition("=")
        deltas[c] = col
    report: dict[str, Any] = dict(readers=labels, reference=labels[0], rows=len(common), features=len(names),
                                  writers={w: int((writer == w).sum()) for w in sorted(set(writer))}, contrasts={})
    mean_d: dict[str, np.ndarray] = {}
    for c in args.contrast:
        a, _, b = c.partition("-")
        D = Z[a] - Z[b]
        mag = np.sqrt(np.nansum(D ** 2, axis=1))
        entry: dict[str, Any] = dict(magnitude=pct(mag),
                                     magnitude_by_writer={w: pct(mag[writer == w]) for w in sorted(set(writer))})
        rms = {}
        for s in sorted(set(source)):
            cols = source == s
            rms[s] = round(float(np.nanmedian(np.sqrt(np.nanmean(D[:, cols] ** 2, axis=1)))), 4)
        entry["sources_median_rms"] = dict(sorted(rms.items(), key=lambda kv: -kv[1]))
        absd = np.nanmedian(np.abs(D), axis=0)
        profiles = {}
        for s in ["all"] + sorted(set(source)):
            prof = np.full(args.n_layers, np.nan)
            for l_ in range(args.n_layers):
                cols = (layer == l_) & ((source == s) if s != "all" else True)
                if cols.any():
                    prof[l_] = float(np.nanmean(absd[cols]))
            if np.isfinite(prof).sum() >= 4:
                profiles[s] = prof
        entry["depth_profile"] = {s: [None if not np.isfinite(x) else round(float(x), 4) for x in p]
                                  for s, p in profiles.items()}
        top = {s: [int(i) for i in np.argsort(-np.nan_to_num(p, nan=-1))[:5]] for s, p in profiles.items()}
        entry["top_layers"] = top
        if c in deltas and args.deltas is not None:
            w = layer_weights(args.deltas, deltas[c], args.n_layers)
            corr = {}
            for s, prof in profiles.items():
                inc = np.diff(prof, prepend=np.nan)
                for k, wl in w.items():
                    corr[f"{s}~{k}"] = dict(profile=spearman(prof, wl), increment=spearman(inc, wl))
            entry["weights_column"] = deltas[c]
            entry["weights_top_layers"] = {k: [int(i) for i in np.argsort(-np.nan_to_num(v, nan=-1))[:5]]
                                           for k, v in w.items()}
            entry["spearman_vs_weights"] = corr
        ws = sorted(set(writer))
        if len(ws) == 2:
            m0, m1 = np.nanmean(D[writer == ws[0]], axis=0), np.nanmean(D[writer == ws[1]], axis=0)
            entry["writers"] = dict(cos_mean_d=cosine(m0, m1),
                                    magnitude_ratio=round(float(np.median(mag[writer == ws[0]]) /
                                                                max(np.median(mag[writer == ws[1]]), 1e-12)), 4),
                                    pair=ws)
        mean_d[c] = np.nanmean(D, axis=0)
        report["contrasts"][c] = entry
        print(f"[lineage] {c}: {json.dumps(dict(magnitude=entry['magnitude'], top_sources=list(entry['sources_median_rms'].items())[:4]))}",
              flush=True)
    keys = list(mean_d)
    report["contrast_cosines"] = {f"{a}|{b}": cosine(mean_d[a], mean_d[b])
                                  for i, a in enumerate(keys) for b in keys[i + 1:]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["contrast_cosines"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
