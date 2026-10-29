"""Is any depth event visible in tsumugi's Column C? A survey over the a0 depth paths and routing.

Inputs: ``column_c_a0_raw.npz`` (``depth_path`` [rows, 73, 1536] fp16: embedding + post-MLP
partial per effective layer; partials reset at AttnRes block boundaries) and
``column_c_floor.npz`` (the FLOOR features, for routing).

Reported:

* **massive activations**: per layer, the largest channel's share of rows where it is the argmax,
  the median max|x|/rms, and whether any (layer, channel) is *bimodal across rows* (a switch):
  for the top channels by magnitude, the gap statistic between the two modes of log|x| (a
  two-component split by Otsu's threshold, between-class variance share).
* **routing at block boundaries**: per routing site, the mean relative split (emb / prev / older /
  self) over rows, and the across-row spread; the block-start sites against their neighbours.
* **the loop**: tsumugi runs one 20-layer core twice (effective layers 19-38 and 39-58, shared
  weights). Per layer pair (l, l+20): the a0 partials' cosine and norm ratio, and the routing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BOUNDARIES = (0, 1, 7, 9, 17, 19, 39, 59)


def otsu_share(v: np.ndarray) -> tuple[float, float]:
    """Best two-class split of a 1-d sample: (between-class variance share, threshold)."""
    s = np.sort(v)
    n = len(s)
    c1 = np.cumsum(s)
    tot = c1[-1]
    var = s.var() + 1e-12
    k = np.arange(1, n)
    m1 = c1[:-1] / k
    m2 = (tot - c1[:-1]) / (n - k)
    w1 = k / n
    between = w1 * (1 - w1) * (m1 - m2) ** 2
    j = int(np.argmax(between))
    return float(between[j] / var), float(s[j])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--floor", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-rows", type=int, default=3000)
    args = ap.parse_args(argv)
    raw = np.load(args.raw, allow_pickle=False)
    paths = raw["depth_path"]
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(paths), size=min(args.max_rows, len(paths)), replace=False))
    X = paths[pick].astype(np.float32)                       # [n, 73, d]
    n, P, d = X.shape
    rep: dict = dict(rows=int(n), points=int(P), hidden=int(d))

    absx = np.abs(X)
    rms = np.sqrt((X ** 2).mean(-1)) + 1e-9
    argmax = absx.argmax(-1)                                  # [n, P]
    peak = absx.max(-1) / rms
    layers = []
    for p in range(P):
        ch, cnt = np.unique(argmax[:, p], return_counts=True)
        top = int(ch[np.argmax(cnt)])
        layers.append(dict(point=p, layer=p - 1, top_channel=top, top_share=round(float(cnt.max() / n), 3),
                           median_peak_ratio=round(float(np.median(peak[:, p])), 1),
                           p99_peak_ratio=round(float(np.quantile(peak[:, p], 0.99)), 1)))
    rep["massive"] = layers

    # bimodality across rows of the top channels' magnitude
    bim = []
    for p in range(1, P):
        mean_abs = absx[:, p].mean(0)
        for c in np.argsort(mean_abs)[-3:]:
            v = np.log(absx[:, p, c] + 1e-6)
            share, thr = otsu_share(v)
            frac_hi = float((v > thr).mean())
            bim.append(dict(layer=p - 1, channel=int(c), mean_abs=round(float(mean_abs[c]), 2),
                            otsu_between_share=round(share, 3), frac_high=round(frac_hi, 3),
                            ratio_modes=round(float(np.exp(v[v > thr].mean() - v[v <= thr].mean())), 2)))
    bim.sort(key=lambda r: -r["otsu_between_share"] * min(r["frac_high"], 1 - r["frac_high"]))
    rep["bimodal_top"] = bim[:15]

    # loop pass 1 vs pass 2 (effective layer l and l+20 for l in 19..38): point index = layer + 1
    loop = []
    for l in range(19, 39):
        a, b = X[:, l + 1], X[:, l + 21]
        cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)
        loop.append(dict(layer=l, twin=l + 20, cos_median=round(float(np.median(cos)), 3),
                         norm_ratio_median=round(float(np.median(np.linalg.norm(b, axis=-1) /
                                                                 (np.linalg.norm(a, axis=-1) + 1e-9))), 3)))
    rep["loop_pass_twins"] = loop

    # routing: per site, mean split over rows and across-row sd of each source
    fl = np.load(args.floor, allow_pickle=False)
    names = [str(x) for x in fl["names"]]
    F = fl["floor_a0"]
    route = {}
    for i, nm in enumerate(names):
        if not nm.startswith("route."):
            continue
        parts = nm.split(".")
        site = "final" if parts[1] == "final" else f"{parts[1]}.{parts[2]}"
        key = parts[-1]
        route.setdefault(site, {})[key] = dict(mean=round(float(np.nanmean(F[:, i])), 3),
                                               sd=round(float(np.nanstd(F[:, i])), 3))
    rep["routing_sites"] = route
    # boundary contrast: emb / prev mass at block-start pre_attn vs the previous layer's pre_mlp
    bnd = []
    for b in BOUNDARIES[1:]:
        here, before = route.get(f"pre_attn.L{b}", {}), route.get(f"pre_mlp.L{b - 1}", {})
        bnd.append(dict(boundary=b, pre_attn_here={k: v["mean"] for k, v in here.items()},
                        pre_mlp_before={k: v["mean"] for k, v in before.items()}))
    rep["routing_at_boundaries"] = bnd
    # the routing sd profile: which sites vary most across rows (the depth-time switches, if any)
    sds = sorted(((s, k, v["sd"]) for s, kv in route.items() for k, v in kv.items() if k in ("emb", "prev", "older", "self")),
                 key=lambda t: -t[2])
    rep["routing_most_variable"] = [dict(site=s, source=k, sd=v) for s, k, v in sds[:15]]
    args.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("routing_sites",)}, indent=1)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
