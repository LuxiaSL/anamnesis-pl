"""Does a turn's attention-entropy level persist across the turns of a scene? (hephia-02's "slow tone" question.)

The meaning-floor read (``meaning_floor_tsumugi_4class_2026-10-29_READING.md``) found ``attn_entropy_mean_L<l>``
to be the one short-span row trait: two adjacent windows of ONE pass rank rows alike (L17 r=.89 at s=4). This asks
whether the trait outlives the pass: across separate turns of one scene, each a separate generation and signature.

Input: the curation grid's release-reader banks (``curation_grid_bank.py`` cells), so one fixed reader reads every
writer. Per set, every feature is residualized within scene: scene-centred, with scene-centred log generated length
and log prompt length (position in the scene) regressed out by one pooled slope. Then, for turn pairs (t, t+k) of one
scene:

* r(k): the Pearson correlation of the residuals over the pairs, for lag k = 1..--max-lag;
* the null: the same with turn order permuted within each scene (scene-centring alone makes r slightly negative);
* excess(k) = r(k) - the null's mean, and a scene-bootstrap 95% interval for lag 1.

Also reported: lag-1 r split by same vs different speaker and same vs different writer, and the between-scene share
of variance (length-residualized, not scene-centred) against its no-scene-effect expectation. The same statistics
are given for every other feature as a reference (median and 90th percentile), so "attention entropy is special"
is checked, not assumed.

Turns shorter than --min-tokens are left out (the meaning floor needs >= 4 steps).

Usage: python curation_attn_persistence.py --bank <dir>:<set> [...] --labels <turn-labels.jsonl> [...] --out <json>
"""

from __future__ import annotations

import argparse
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
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


msr = _sibling("msteer_replay_read")
ENTROPY = re.compile(r"^attn_entropy_mean_L(\d+)$")


def residualize(X: np.ndarray, cov: np.ndarray, scene: np.ndarray, centre: bool) -> np.ndarray:
    """X with covariates regressed out by pooled within-scene slopes; scene-centred if ``centre``."""
    Xc, Cc = X.copy(), cov.copy()
    for s in np.unique(scene):
        u = scene == s
        Xc[u] -= Xc[u].mean(axis=0)
        Cc[u] -= Cc[u].mean(axis=0)
    beta, *_ = np.linalg.lstsq(Cc, Xc, rcond=None)
    return (Xc if centre else X - X.mean(axis=0)) - (Cc if centre else cov - cov.mean(axis=0)) @ beta


def pair_index(scene: np.ndarray, turn: np.ndarray, lag: int) -> np.ndarray:
    pos = {(s, int(t)): i for i, (s, t) in enumerate(zip(scene, turn))}
    return np.array([(i, pos[(s, int(t) + lag)]) for i, (s, t) in enumerate(zip(scene, turn))
                     if (s, int(t) + lag) in pos], dtype=int).reshape(-1, 2)


def colcorr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    den = np.sqrt((A ** 2).sum(axis=0) * (B ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (A * B).sum(axis=0) / den, np.nan)


def permuted_turns(scene: np.ndarray, turn: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = turn.copy()
    for s in np.unique(scene):
        u = np.flatnonzero(scene == s)
        out[u] = turn[rng.permutation(u)]
    return out


def r4(x: float) -> float | None:
    return None if not np.isfinite(x) else round(float(x), 4)


def read_set(R: np.ndarray, Rraw: np.ndarray, meta: list[dict], names: list[str], ent: list[int], max_lag: int,
             perms: int, boots: int, rng: np.random.Generator) -> dict[str, Any]:
    scene = np.array([m["scene_id"] for m in meta])
    turn = np.array([int(m["turn"]) for m in meta])
    speaker = np.array([str(m.get("speaker")) for m in meta])
    writer = np.array([str(m.get("writer")) for m in meta])
    other = [i for i in range(len(names)) if i not in set(ent)]
    out: dict[str, Any] = dict(turns=len(meta), scenes=int(len(np.unique(scene))),
                               median_gen_tokens=float(np.median([m["n_tokens"] for m in meta])), lags={})
    for k in range(1, max_lag + 1):
        P = pair_index(scene, turn, k)
        if len(P) < 30:
            break
        r = colcorr(R[P[:, 0]], R[P[:, 1]])
        null = []
        for _ in range(perms):
            Q = pair_index(scene, permuted_turns(scene, turn, rng), k)
            null.append(colcorr(R[Q[:, 0]], R[Q[:, 1]]))
        null = np.array(null)
        mu, lo, hi = null.mean(axis=0), np.percentile(null, 2.5, axis=0), np.percentile(null, 97.5, axis=0)
        ex = r - mu
        ent_rows = {names[i]: dict(r=r4(r[i]), null=r4(mu[i]), null95=[r4(lo[i]), r4(hi[i])], excess=r4(ex[i]))
                    for i in ent}
        out["lags"][k] = dict(pairs=int(len(P)), attn_entropy=ent_rows,
                              attn_entropy_median_excess=r4(np.nanmedian(ex[ent])),
                              other_median_excess=r4(np.nanmedian(ex[other])),
                              other_p90_excess=r4(np.nanpercentile(ex[other], 90)),
                              other_share_beyond_null95=r4(np.nanmean((r[other] > hi[other]))),
                              entropy_share_beyond_null95=r4(np.nanmean((r[ent] > hi[ent]))))
    # lag-1 splits and a scene bootstrap
    P = pair_index(scene, turn, 1)
    if len(P):
        splits = {}
        for nm, same in (("same_speaker", speaker[P[:, 0]] == speaker[P[:, 1]]),
                         ("same_writer", writer[P[:, 0]] == writer[P[:, 1]])):
            for flag, sel in ((nm, same), ("not_" + nm, ~same)):
                if sel.sum() >= 30:
                    rr = colcorr(R[P[sel, 0]], R[P[sel, 1]])
                    splits[flag] = dict(pairs=int(sel.sum()), attn_entropy_median_r=r4(np.nanmedian(rr[ent])),
                                        other_median_r=r4(np.nanmedian(rr[other])),
                                        **{names[i]: r4(rr[i]) for i in ent if names[i].endswith(("_L17", "_L7"))})
        out["lag1_splits"] = splits
        scenes = np.unique(scene)
        by_scene = defaultdict(list)
        for j, (a, _) in enumerate(P):
            by_scene[scene[a]].append(j)
        bs = []
        for _ in range(boots):
            pick = rng.choice(scenes, len(scenes))
            idx = np.concatenate([by_scene[s] for s in pick if by_scene[s]]).astype(int)
            bs.append(np.nanmedian(colcorr(R[P[idx, 0]], R[P[idx, 1]])[ent]))
        out["lag1_entropy_median_r_boot95"] = [r4(np.percentile(bs, 2.5)), r4(np.percentile(bs, 97.5))]
    # between-scene share of variance (length-residualized, not scene-centred)
    counts = np.array([np.sum(scene == s) for s in np.unique(scene)])
    tot = Rraw.var(axis=0)
    means = np.stack([Rraw[scene == s].mean(axis=0) for s in np.unique(scene)])
    between = (counts[:, None] * (means - Rraw.mean(axis=0)) ** 2).sum(axis=0) / len(Rraw)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = between / tot
    out["between_scene_share"] = dict(expected_if_no_scene_effect=r4(len(counts) / len(Rraw)),
                                      attn_entropy={names[i]: r4(share[i]) for i in ent},
                                      attn_entropy_median=r4(np.nanmedian(share[ent])),
                                      other_median=r4(np.nanmedian(share[other])))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True, action="append", help="<bank cell dir>:<set name>")
    ap.add_argument("--labels", action="append", default=[], help="turn-label files: speaker by (scene_id, turn)")
    ap.add_argument("--min-tokens", type=int, default=8)
    ap.add_argument("--max-lag", type=int, default=8)
    ap.add_argument("--perms", type=int, default=100)
    ap.add_argument("--boots", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    speakers: dict[tuple[str, int], str] = {}
    for f in args.labels:
        for line in Path(f).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                speakers[(r["scene_id"], int(r["turn"]))] = r.get("speaker")
    result: dict[str, Any] = dict(min_tokens=args.min_tokens, sets={})
    names_ref: list[str] | None = None
    pooled: list[tuple[np.ndarray, np.ndarray, list[dict]]] = []
    for spec in args.bank:
        path, _, setname = spec.rpartition(":")
        _, rows = msr.load_rows(Path(path) / "rows.jsonl")
        X, ids, names, _, _ = msr.load_bank(Path(path) / "sigs", "engine-stats-b1")
        if names_ref is not None and names != names_ref:
            raise ValueError(f"{path}: feature names differ from the first bank")
        names_ref = names
        keep = [i for i, c in enumerate(ids) if c in rows and rows[c]["n_tokens"] >= args.min_tokens]
        meta = [dict(rows[ids[i]], set=setname,
                     speaker=speakers.get((rows[ids[i]]["scene_id"], int(rows[ids[i]]["turn"]))),
                     scene_id=f"{setname}|{rows[ids[i]]['scene_id']}") for i in keep]
        X = X[keep].astype(np.float64)
        pooled.append((X, np.array([[np.log(m["n_tokens"]), np.log(m["prompt_length"])] for m in meta]), meta))
        print(f"[persist] {setname}: {len(meta)} turns of {len(ids)} (>= {args.min_tokens} tokens)", flush=True)
    assert names_ref is not None
    finite = np.all([np.isfinite(X).all(axis=0) for X, _, _ in pooled], axis=0)
    live = finite & np.all([X.std(axis=0) > 1e-9 for X, _, _ in pooled], axis=0)
    cols = np.flatnonzero(live)
    names = [names_ref[i] for i in cols]
    ent = sorted((i for i, n in enumerate(names) if ENTROPY.match(n)), key=lambda i: int(ENTROPY.match(names[i])[1]))
    print(f"[persist] features {len(names)} live; attention-entropy columns {len(ent)}", flush=True)
    resid = []
    for (X, C, meta) in pooled:
        scene = np.array([m["scene_id"] for m in meta])
        R = residualize(X[:, cols], C, scene, centre=True)
        Rraw = residualize(X[:, cols], C, scene, centre=False)
        resid.append((R, Rraw, meta))
        result["sets"][meta[0]["set"]] = read_set(R, Rraw, meta, names, ent, args.max_lag, args.perms, args.boots,
                                                  rng)
        print(f"[persist] {meta[0]['set']}: lag1 {json.dumps({k: v for k, v in result['sets'][meta[0]['set']]['lags'][1].items() if k != 'attn_entropy'})}",
              flush=True)
    R = np.concatenate([r for r, _, _ in resid])
    Rraw = np.concatenate([r for _, r, _ in resid])
    meta = sum((m for _, _, m in resid), [])
    result["sets"]["pooled"] = read_set(R, Rraw, meta, names, ent, args.max_lag, args.perms, args.boots, rng)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    p = result["sets"]["pooled"]
    print(json.dumps(dict(lags={k: {kk: vv for kk, vv in v.items() if kk != "attn_entropy"} for k, v in p["lags"].items()},
                          lag1_splits=p.get("lag1_splits"), boot=p.get("lag1_entropy_median_r_boot95"),
                          between=p["between_scene_share"]["attn_entropy_median"],
                          between_other=p["between_scene_share"]["other_median"]), indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
