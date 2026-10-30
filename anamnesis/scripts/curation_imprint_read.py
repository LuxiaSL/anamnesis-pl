"""Context and seed imprint vs coherence (signature curation v2): the read of ``curation-imprint-drain-node2.sh``.

Every labelled turn was signed under six contexts, with the span's tokens identical in each (``curation_imprint_rows.py``):
full, ctx_fA, ctx_fB, last, seed_fA and seed_fB. Signatures are z-scored per feature by the SD of the full-context
rows. The lens family (``lens_*``) is read on its own and left out of the distances.

Per turn (whole-vector L2 in z):
* ctx imprint  = mean(d(full, ctx_fA), d(full, ctx_fB)) / d(ctx_fA, ctx_fB): how far the own context moves the
  turn's processing, against two equivalent foreign contexts (the floor). 1 = no more than a foreign context.
* seed imprint = the same with the foreign headers.
* last ratio   = d(full, last) / d(ctx_fA, ctx_fB). The last-turn-only prompt is shorter, so this also carries a
  position shift.
* lens: lens_depth under the full context, and its change from full to the foreign contexts.

Each measure is read against the turn's coherence label:
* AUC, coherent (>= --hi) vs incoherent (<= --lo);
* Spearman over all labels;
* the AUC of a logistic regression on [measure, log length] against log length alone, for the part length doesn't
  explain.

With --slips, the same measures are read as seat slip vs not, over the turns of scenes the seat reader read.

All of it overall and per label file (writer set).

Usage: python curation_imprint_read.py --signed <dir> --rows <import-rows.jsonl> --out <json>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p2 = _sibling("msteer_p2_read")
VARIANTS = ("full", "ctx_fA", "ctx_fB", "last", "seed_fA", "seed_fB")


def auc(score: np.ndarray, y: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(score)
    if len(np.unique(y[ok])) < 2:
        return None
    return round(float(roc_auc_score(y[ok], score[ok])), 4)


def cv_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float | None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    if len(np.unique(y)) < 2 or len(np.unique(groups)) < 5:
        return None
    s = np.empty(len(y))
    Xs = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
    for tr, te in GroupKFold(n_splits=5).split(Xs, groups=groups):
        s[te] = LogisticRegression(max_iter=2000).fit(Xs[tr], y[tr]).decision_function(Xs[te])
    return auc(s, y)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signed", required=True, type=Path)
    ap.add_argument("--rows", required=True, type=Path)
    ap.add_argument("--hi", type=int, default=4)
    ap.add_argument("--lo", type=int, default=2)
    ap.add_argument("--slips", type=Path, default=None,
                    help="seat-slips.jsonl ({scene_id, seat_slip_turns}): also read slip vs non-slip turns")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    from scipy.stats import spearmanr

    meta = {}
    for line in args.rows.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            meta[r["id"]] = dict(r["meta"], n_gen=len(r["ids"]) - r["prompt_length"])
    signed = p2.load_signed(args.signed)
    by: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    names: list[str] | None = None
    for rid, row in signed.items():
        m = meta.get(rid)
        if m is None:
            continue
        if names is None:
            names = row["names"]
        if row["names"] != names:
            continue
        by[m["cand_id"]][m["variant"]] = row["vec"]
    turns = [c for c, v in by.items() if all(k in v for k in VARIANTS)]
    print(f"[imprint] signed rows {len(signed)}; turns with all six variants {len(turns)} of {len(by)}", flush=True)
    lens_cols = [i for i, n in enumerate(names) if n.startswith("lens_")]
    sig_cols = [i for i, n in enumerate(names) if not n.startswith("lens_")]
    depth_col = names.index("lens_depth") if "lens_depth" in names else None
    full = np.stack([by[c]["full"][sig_cols] for c in turns])
    sd = full.std(axis=0)
    live = np.isfinite(sd) & (sd > 1e-9)

    def z(v: np.ndarray) -> np.ndarray:
        return np.nan_to_num(v[sig_cols][live] / sd[live])

    rows_out: list[dict[str, Any]] = []
    for c in turns:
        v = {k: z(by[c][k]) for k in VARIANTS}
        d = lambda a, b: float(np.linalg.norm(v[a] - v[b]))  # noqa: E731
        floor_ctx, floor_seed = d("ctx_fA", "ctx_fB"), d("seed_fA", "seed_fB")
        m0 = meta[f"{c}:full"]
        r = dict(cand_id=c, label_file=m0.get("label_file"), writer=m0.get("writer"), coherent=m0.get("coherent"),
                 loglen=float(np.log(m0["n_gen"])),
                 ctx_imprint=(d("full", "ctx_fA") + d("full", "ctx_fB")) / 2 / max(floor_ctx, 1e-9),
                 seed_imprint=(d("full", "seed_fA") + d("full", "seed_fB")) / 2 / max(floor_seed, 1e-9),
                 last_ratio=d("full", "last") / max(floor_ctx, 1e-9),
                 ctx_floor=floor_ctx, seed_floor=floor_seed)
        if depth_col is not None:
            depth = {k: float(by[c][k][depth_col]) for k in VARIANTS}
            r["lens_depth_full"] = depth["full"]
            r["lens_depth_shift_ctx"] = depth["full"] - (depth["ctx_fA"] + depth["ctx_fB"]) / 2
        rows_out.append(r)

    measures = ["ctx_imprint", "seed_imprint", "last_ratio", "ctx_floor", "lens_depth_full", "lens_depth_shift_ctx"]
    out: dict[str, Any] = dict(turns=len(rows_out), features=int(live.sum()), lens_features=len(lens_cols), sets={})
    for setname in ["all"] + sorted({r["label_file"] for r in rows_out}):
        rs = [r for r in rows_out if setname == "all" or r["label_file"] == setname]
        coh = np.array([r["coherent"] for r in rs], dtype=float)
        sel = (coh >= args.hi) | (coh <= args.lo)
        y = (coh[sel] >= args.hi).astype(int)
        L = np.array([r["loglen"] for r in rs])
        groups = np.array([r["cand_id"].split(":")[0] for r in rs])
        res: dict[str, Any] = dict(turns=len(rs), hi=int(y.sum()), lo=int((1 - y).sum()),
                                   loglen_auc=auc(L[sel], y),
                                   loglen_cv_auc=cv_auc(L[sel][:, None], y, groups[sel]))
        for mname in measures:
            x = np.array([r.get(mname, np.nan) for r in rs], dtype=float)
            if not np.isfinite(x).any():
                continue
            rho = spearmanr(x, coh, nan_policy="omit").correlation
            res[mname] = dict(median=round(float(np.nanmedian(x)), 3), auc=auc(x[sel], y),
                              spearman=round(float(rho), 4),
                              cv_auc_with_length=cv_auc(np.stack([x[sel], L[sel]], axis=1), y, groups[sel]))
        out["sets"][setname] = res
        print(f"[imprint] {setname}: {json.dumps(res)}", flush=True)
    if args.slips is not None:
        slips = {}
        for line in args.slips.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                slips[r["scene_id"]] = {int(t) for t in r.get("seat_slip_turns") or []}
        rs = [r for r in rows_out if r["cand_id"].split(":")[0] in slips]
        y = np.array([int(int(r["cand_id"].split(":")[1]) in slips[r["cand_id"].split(":")[0]]) for r in rs])
        L = np.array([r["loglen"] for r in rs])
        groups = np.array([r["cand_id"].split(":")[0] for r in rs])
        sres: dict[str, Any] = dict(turns=len(rs), slips=int(y.sum()), loglen_auc=auc(L, y),
                                    loglen_cv_auc=cv_auc(L[:, None], y, groups))
        for mname in measures:
            x = np.array([r.get(mname, np.nan) for r in rs], dtype=float)
            sres[mname] = dict(auc_slip=auc(x, y), slip_median=round(float(np.nanmedian(x[y == 1])), 3),
                               other_median=round(float(np.nanmedian(x[y == 0])), 3),
                               cv_auc_with_length=cv_auc(np.stack([x, L], axis=1), y, groups))
        out["seat_slips"] = sres
        print(f"[imprint] seat slips: {json.dumps(sres)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    with args.out.with_suffix(".turns.jsonl").open("w") as fh:
        for r in rows_out:
            fh.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
