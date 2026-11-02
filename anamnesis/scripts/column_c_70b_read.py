"""Column C's read family at 70B: does the per-position read-profile see the channel-8026 switch?

Gate 2 (``column_c_70b_8026.py`` + ``column_c_8026_read.py``) showed that C's *depth* family sees the switch. This
tests the other half before C is ported beyond tsumugi (prototype list, item 6). CPU only, over the retained vLLM
captures of lane-70b-001 (node1, read-only). The HF captures carry no attention rows, so the surgical twin test
can't be run for read; this is the observational test.

Per span position, at the capture's sampled attention layers (0, 20, 40, 50, 60, 70, 79), C's read quantities come
from the contract-shaped per-head stats:
* mean head entropy;
* head-mean sink mass and its spread over heads;
* head-mean recency mass;
* head-mean prompt mass.
That is 5 x 7 = 35 features. (C's agreement term needs the head-mean row's entropy per position; the capture
doesn't keep it.)

**Label:** 8026-on when |h[L40, 8026]| >= --on (as in gate 2).

**Read:** probe AUROC (balanced logistic, GroupKFold by row) for:
* read at all layers;
* read at the layers before the switch's onset (0, 20): the falsifier. The onset is L29-30, so these must not do
  much better than the text baselines;
* read at the layers after it (40-79);
* per stat.
Text baselines: the input token's on-rate (fit in fold) and log position, alone and together; and read + text.

Usage: python column_c_70b_read.py --out <json> [--on 12] [--limit-rows N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

VLLM = Path("/models/anamnesis-vllm-integration/lane-70b-001/l2/captures/"
            "population-capture-70b-native-001/full-b1-order0")
CHANNEL = 8026
STATS = ("ent", "sink", "sinkspread", "recency", "prompt")
EARLY = (0, 20)


def read_row(path: Path) -> dict:
    import torch
    d = torch.load(path, map_location="cpu", weights_only=True)
    h = d["hidden"]
    if h.ndim != 3 or h.shape[0] != 80:
        raise ValueError(f"{path}: hidden {tuple(h.shape)}, expected [80, T, d]")
    T = h.shape[1]
    layers = sorted(int(k) for k in d["attn_head_ent"])
    cols, names = [], []
    for l in layers:
        ent = d["attn_head_ent"][l].double().numpy()
        sink = d["attn_head_sink"][l].double().numpy()
        rec = d["attn_head_recency"][l].double().numpy()
        prm = d["attn_head_prompt"][l].double().numpy()
        for arr in (ent, sink, rec, prm):
            if arr.shape[0] != T:
                raise ValueError(f"{path}: layer {l} stats have {arr.shape[0]} positions, hidden has {T}")
        cols += [ent.mean(1), sink.mean(1), sink.std(1), rec.mean(1), prm.mean(1)]
        names += [f"read.{s}.L{l}" for s in STATS]
    return dict(X=np.stack(cols, axis=1), names=names, layers=layers,
                ch40=np.abs(h[40, :, CHANNEL].float().numpy()), chosen=d["chosen"].numpy()[:T], T=T)


def probe_auc(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    scores = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(x, y, groups):
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-9
        m = LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced").fit((x[tr] - mu) / sd, y[tr])
        scores[te] = m.decision_function((x[te] - mu) / sd)
    return round(float(roc_auc_score(y, scores)), 4)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--on", type=float, default=12.0)
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)
    import torch
    torch.set_num_threads(args.threads)

    paths = sorted(VLLM.glob("row-*.pt"))
    if args.limit_rows:
        paths = paths[: args.limit_rows]
    parts, t0 = [], time.time()
    for k, p in enumerate(paths, 1):
        r = read_row(p)
        r["gid"] = int(p.stem.split("-")[1])
        parts.append(r)
        print(f"[70b-read] {k}/{len(paths)} row {r['gid']}: T={r['T']} on={(r['ch40'] >= args.on).sum()} "
              f"({time.time() - t0:.0f}s)", flush=True)
    names, layers = parts[0]["names"], parts[0]["layers"]
    if any(q["names"] != names for q in parts):
        raise ValueError("captures carry different sampled layers")
    X = np.concatenate([q["X"] for q in parts])
    y = (np.concatenate([q["ch40"] for q in parts]) >= args.on).astype(int)
    g = np.concatenate([np.full(q["T"], q["gid"]) for q in parts])
    pos = np.concatenate([np.arange(q["T"]) for q in parts])
    inp = np.concatenate([np.concatenate([[-1], q["chosen"][:-1]]) for q in parts])
    finite = np.isfinite(X).all(axis=0)
    rep: dict = dict(rows=len(parts), positions=int(len(y)), on=int(y.sum()), on_rows=int(len(np.unique(g[y == 1]))),
                     layers=layers, features=int(finite.sum()), label=f"|h[L40,{CHANNEL}]| >= {args.on}")

    def cols(pred) -> np.ndarray:
        return np.array([finite[i] and pred(n) for i, n in enumerate(names)])

    lay = lambda n: int(n.rsplit(".L", 1)[1])  # noqa: E731
    sets = {
        "read_all": cols(lambda n: True),
        "read_early_L0_L20": cols(lambda n: lay(n) in EARLY),
        "read_late_L40plus": cols(lambda n: lay(n) >= 40),
        **{f"read_{s}": cols(lambda n, s=s: n.split(".")[1] == s) for s in STATS},
        **{f"read_L{l}": cols(lambda n, l=l: lay(n) == l) for l in layers},
    }
    rep["auc"] = {k: probe_auc(X[:, m], y, g) for k, m in sets.items()}
    # text baselines
    s_tok = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        prior = y[tr].mean()
        rates: dict[int, tuple[int, int]] = {}
        for t, yy in zip(inp[tr], y[tr]):
            a, b = rates.get(int(t), (0, 0))
            rates[int(t)] = (a + int(yy), b + 1)
        s_tok[te] = [(rates.get(int(t), (0, 0))[0] + prior * 5) / (rates.get(int(t), (0, 0))[1] + 5) for t in inp[te]]
    xt = np.column_stack([s_tok, np.log1p(pos)])
    rep["auc"]["text_token_rate"] = round(float(roc_auc_score(y, s_tok)), 4)
    rep["auc"]["text_logpos"] = round(float(roc_auc_score(y, np.log1p(pos))), 4)
    rep["auc"]["text_both"] = probe_auc(xt, y, g)
    rep["auc"]["read_all+text"] = probe_auc(np.column_stack([X[:, sets["read_all"]], xt]), y, g)
    rep["auc"]["read_late+text"] = probe_auc(np.column_stack([X[:, sets["read_late_L40plus"]], xt]), y, g)
    rep["auc"]["read_early+text"] = probe_auc(np.column_stack([X[:, sets["read_early_L0_L20"]], xt]), y, g)
    # the on positions' read values against their rows' off positions (per-feature effect size)
    eff = {}
    for i, n in enumerate(names):
        if finite[i]:
            on_v, off_v = X[y == 1, i], X[y == 0, i]
            eff[n] = round(float((on_v.mean() - off_v.mean()) / (off_v.std() + 1e-12)), 2)
    rep["effect_size_top"] = dict(sorted(eff.items(), key=lambda kv: -abs(kv[1]))[:12])
    rep["seconds"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
