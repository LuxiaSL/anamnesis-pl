"""Meaning floor (S1): the span length from which each fast-lane feature is a RELIABLE
property of the forward pass, not only a computed one.

The support table (``feature_support_table.py``) gives the span at which a feature is
*defined* (computed rather than a placeholder). This script measures the span at which
it is *meaningful*: two disjoint windows of one teacher-forced pass, each ``s`` steps
(``s + 1`` tokens) long, should agree about which rows are high and which are low.

Two stages.

``sign`` (node2, GPU; pleroma's deployed tree on PYTHONPATH): for every long
self-generated tsumugi row, ONE instrumented ``cached_forward`` over the row truncated
to ``P + ngen`` tokens. Each layer's attention weights are fanned out to every window's
``ExternalSpan`` as the forward computes them, and every window is finished from the same
capture, sliced exactly as ``kotodama.model.capture.replay_features_gpu`` slices it. The
windows (offsets in generated tokens from the prompt boundary P, ``w = s + 1``):

  A  [0, w)                onset window (where an answer span starts)
  B  [w, 2w)               adjacent to A, later
  C  [ngen - 2w, ngen - w) late pair, first
  D  [ngen - w, ngen)      late pair, second (ends where the row ends)
  full [0, ngen)           the long-span value, for the convergence definition

A window's span is its own: ``start`` is the window start, so everything before it
(prompt and earlier generated text) is the window's prefix, as for any span of a pass.
Parity: on the first rows the full span is also signed by ``replay_features_gpu``
itself on the same truncated row, and the two vectors must be identical.

``analyse`` (local, numpy only): per feature and per s, Spearman correlation across
rows between windows (A-B onset pair, C-D late pair, A-D far pair) and between window A
and the full span; floors at 0.5 and 0.7; degeneracy flags; position effects; the TSV
and markdown summary. ``--sig-dir`` may be given more than once to pool draws.

Prompt classes. When the rows carry more than one prompt class, rows of different
classes can differ for class reasons alone, which raises across-row variance and with it
the pooled reliability. Three controls separate that from reliability about the pass:
  class_balanced    subset with every class drawn down to the smallest class (seeded)
  ABpartClass       r_AB after regressing each feature's ranks on class indicators: the
                    reliability left inside classes
  class=<c>         r_AB within each class alone (per-class curves in the report)
A feature whose pooled floor is finite but whose class-partialled floor is ``never``
reads the prompt class, not the pass.

Definitions (justified in the summary):
  reliability r(s)   Spearman rho over rows of f(A_s) vs f(B_s); computed only when
                     neither window is degenerate (modal value held by >= 50% of rows).
  meaning floor      smallest tested s with r(s') >= threshold at s and every longer
                     tested s' ("from which on"); ``never`` if r(s_max) < threshold;
                     ``undefined`` if degenerate at every tested s.
  convergence floor  the same rule on rho(f(A_s), f(full)).

Usage (sign, on node2; see tools/node2_runbook/s1-sign-node2.sh):
  python meaning_floor.py sign --gen-dir <draw dir> --out-dir <dir> --shard k --n-shards n
Usage (analyse, local):
  python pipeline/anamnesis/scripts/meaning_floor.py analyse --sig-dir <dir with shards>
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

S_LIST = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
WINDOWS = ("A", "B", "C", "D")
NGEN = 300
CURVE_S = [2, 4, 8, 16, 32, 64, 128]
MODAL_DEGENERATE = 0.5   # a window whose modal value is held by >= this share of rows is degenerate


def window_offsets(s: int, ngen: int) -> dict[str, tuple[int, int]]:
    w = s + 1
    if 2 * w > ngen:
        raise ValueError(f"s={s}: two windows of {w} tokens do not fit in {ngen}")
    return {"A": (0, w), "B": (w, 2 * w), "C": (ngen - 2 * w, ngen - w), "D": (ngen - w, ngen)}


# ----------------------------------------------------------------------------- sign (node2)

def _load_long_rows(gen_dir: Path, ngen: int) -> tuple[list[dict], Counter]:
    rows, why = [], Counter()
    for f in sorted(glob.glob(str(gen_dir / "gen_records" / "gen_*.json"))):
        try:
            rec = json.loads(Path(f).read_text())
        except (OSError, ValueError):
            why["unreadable"] += 1
            continue
        ids, P = rec["input_ids"], int(rec["prompt_length"])
        stops = set(rec.get("sampling", {}).get("eos_ids") or [])
        gen = ids[P:P + ngen]
        if len(gen) < ngen:
            why["short"] += 1
            continue
        if stops & set(gen):
            why["stop_inside"] += 1
            continue
        why["kept"] += 1
        rows.append(dict(gid=int(rec["generation_id"]), prompt_id=rec["prompt_id"],
                         prompt_class=rec.get("prompt_class"), P=P, ids=ids[:P + ngen]))
    return rows, why


def sign(args) -> int:
    import torch
    from anamnesis.extraction.fast.external import ExternalSpan
    from kotodama.model.capture import (assert_lane_pinned, cached_forward, capture_identity,
                                        hidden_rows, replay_features_gpu)
    from pleroma.config.anamnesis_registry import ensure_registry
    from pleroma.harvest.koto_gpu_lane import KotoGpuLane
    from pleroma.model import koto

    if not torch.cuda.is_available():
        print("REFUSED: needs a GPU", file=sys.stderr)
        return 2
    ngen, s_list = args.ngen, args.s
    for s in s_list:
        window_offsets(s, ngen)
    rows, why = _load_long_rows(args.gen_dir, ngen)
    rows = rows[args.shard::args.n_shards]
    if args.limit:
        rows = rows[: args.limit]
    tag = f"{args.shard:02d}of{args.n_shards:02d}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[s1] shard {tag}: {len(rows)} rows (corpus filter {dict(why)}), s={s_list}, ngen={ngen}",
          flush=True)

    ensure_registry()
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = koto.load_koto(args.checkpoint_path, size=args.koto_size,
                           boundaries=koto.boundaries_for(args.koto_boundaries), device="cuda:0",
                           rope_fp32=True, eps_fp64_compat=True)
    manifest = np.load(args.manifest, allow_pickle=False)
    lanes = KotoGpuLane(args.preset, args.calib_dir, [str(n) for n in manifest["FULL_names"]], "cuda:0")
    L = int(model.config.num_layers)
    sampled = list(lanes.lane(1).config.sampled_layers)

    partial = args.out_dir / f"partial-{tag}.npz"
    store: dict = {"names": {}, "feat": defaultdict(list), "meta": []}
    if partial.exists():
        prev = np.load(partial, allow_pickle=True)["store"].item()
        store["names"], store["meta"] = prev["names"], prev["meta"]
        store["feat"] = defaultdict(list, prev["feat"])
    done = {m["gid"] for m in store["meta"]}
    parity: list[dict] = []
    lane_ids: set[str] = set()
    t0, n_new = time.time(), 0

    def checkpoint() -> None:
        np.savez(partial, store=np.array(dict(names=store["names"], meta=store["meta"],
                                              feat=dict(store["feat"])), dtype=object))

    for k, row in enumerate(rows, 1):
        if row["gid"] in done:
            continue
        P, ids = row["P"], row["ids"]
        spans: dict[str, ExternalSpan] = {}
        for s in s_list:
            for w, (o0, o1) in window_offsets(s, ngen).items():
                spans[f"s{s}_{w}"] = ExternalSpan(lanes.lane(s), start=P + o0, end=P + o1, num_layers=L)
        spans["full"] = ExternalSpan(lanes.lane(ngen - 1), start=P, end=P + ngen, num_layers=L)

        def sink(layer, weights, _spans=tuple(spans.values())):
            for sp in _spans:
                sp.attention(layer, weights)

        assert_lane_pinned()
        x = torch.as_tensor(ids, dtype=torch.long, device="cuda:0").unsqueeze(0)
        logits, cap, _ = cached_forward(model, x, past=None, position_offset=0, light=False,
                                        capture_hidden=True, attention_sink=sink,
                                        capture_layers=set(sampled))
        hr = hidden_rows(cap, L)
        ident = capture_identity(model)
        token_ids = x[0].tolist()
        out: dict[str, np.ndarray] = {}
        for key, sp in spans.items():
            rs = slice(sp.start, sp.start + sp.steps)   # as replay_features_gpu: rows P .. P+t-1

            def heads(t, rs=rs):
                return t[0, :, rs, :].permute(1, 0, 2)

            res = sp.finish(
                hidden=torch.stack([hr[l + 1][0, rs] for l in range(L)]),
                keys={l: heads(cap["keys"][l]) for l in sampled},
                values={l: heads(cap["values"][l]) for l in sampled},
                queries={l: heads(cap["queries"][l]) for l in sampled},
                gates={l: cap["gate"][l][0, rs] for l in sampled},
                logits=logits[0, rs],
                token_ids=token_ids,
                capture_identity=ident,
                routing=[(tg, int(layer), w[:, 0, rs].T) for tg, layer, w in cap["routing"]],
                committed=[c[0, rs, :] for c in cap["committed"]],
            )
            names = tuple(str(n) for n in res.feature_names)
            skey = key.split("_")[0]
            if skey not in store["names"]:
                store["names"][skey] = names
            elif store["names"][skey] != names:
                raise RuntimeError(f"schema drift at {key}")
            out[key] = np.asarray(res.features, dtype=np.float32)
            if key == "full":
                lane_ids.add(str((res.metadata or {}).get("lane_id")))
        del cap, hr, logits
        if len(parity) < args.parity_rows:
            ref = replay_features_gpu(model, ids, P, lanes.lane(ngen - 1))
            d = np.abs(np.asarray(ref.features, np.float32) - out["full"])
            parity.append(dict(gid=row["gid"], names_equal=tuple(map(str, ref.feature_names)) == store["names"]["full"],
                               max_abs_diff=float(d.max()), n_diff=int((d > 0).sum()),
                               ref_lane_id=str((ref.metadata or {}).get("lane_id"))))
            print(f"[s1] parity row {row['gid']}: {parity[-1]}", flush=True)
            if parity[-1]["n_diff"] and args.strict_parity:
                raise RuntimeError(f"fan-out full span != replay_features_gpu: {parity[-1]}")
        for key, v in out.items():
            store["feat"][key].append(v)
        gen = ids[P:]
        store["meta"].append(dict(gid=row["gid"], prompt_id=row["prompt_id"],
                                  prompt_class=row["prompt_class"], P=P,
                                  distinct_frac=len(set(gen)) / len(gen)))
        n_new += 1
        if n_new % args.checkpoint == 0:
            checkpoint()
            el = time.time() - t0
            print(f"[s1] {k}/{len(rows)} rows, {el / n_new:.1f}s/row, ETA {el / n_new * (len(rows) - k):.0f}s",
                  flush=True)
    checkpoint()
    arrays = {f"names_{k}": np.array(v) for k, v in store["names"].items()}
    arrays.update({f"feat_{k}": np.stack(v) for k, v in store["feat"].items()})
    m = store["meta"]
    arrays.update(gid=np.array([r["gid"] for r in m]), prompt_id=np.array([r["prompt_id"] for r in m]),
                  prompt_class=np.array([str(r["prompt_class"]) for r in m]),
                  P=np.array([r["P"] for r in m]), distinct_frac=np.array([r["distinct_frac"] for r in m]))
    np.savez(args.out_dir / f"windows-{tag}.npz", **arrays)
    meta = dict(gen_dir=str(args.gen_dir), ngen=ngen, s_list=s_list, windows={s: window_offsets(s, ngen) for s in s_list},
                rows=len(m), corpus_filter=dict(why), parity=parity, preset=args.preset,
                checkpoint=args.checkpoint_path, calibration=lanes.provenance,
                lane_id_full=sorted(lane_ids), wall_seconds=round(time.time() - t0, 1),
                code={mod: sys.modules[mod].__file__ for mod in ("anamnesis", "kotodama", "pleroma")})
    (args.out_dir / f"run_meta-{tag}.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"[s1] done shard {tag}: {len(m)} rows", flush=True)
    return 0


# ----------------------------------------------------------------------------- analyse (local)

def _ranks(x: np.ndarray) -> np.ndarray:
    """Column-wise average ranks (ties averaged), x [rows, features] -> float64."""
    from scipy.stats import rankdata
    return rankdata(x, axis=0, method="average").astype(np.float64)


def _spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ra, rb = _ranks(a), _ranks(b)
    ra -= ra.mean(0)
    rb -= rb.mean(0)
    den = np.sqrt((ra ** 2).sum(0) * (rb ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (ra * rb).sum(0) / den, np.nan)


def _partial_spearman(a: np.ndarray, b: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Spearman of a, b after regressing each feature's ranks on the ranks of covariates z."""
    ra, rb = _ranks(a), _ranks(b)
    Z = np.column_stack([np.ones(len(z)), _ranks(z.reshape(len(z), -1))])
    def resid(r):
        beta, *_ = np.linalg.lstsq(Z, r, rcond=None)
        return r - Z @ beta
    ea, eb = resid(ra), resid(rb)
    den = np.sqrt((ea ** 2).sum(0) * (eb ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (ea * eb).sum(0) / den, np.nan)


def _partial_spearman_classes(a: np.ndarray, b: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """Spearman of a, b after regressing each feature's ranks on class indicators (within-class reliability)."""
    levels = sorted(set(cls.tolist()))
    if len(levels) < 2:
        return np.full(a.shape[1], np.nan)
    ra, rb = _ranks(a), _ranks(b)
    Z = np.column_stack([np.ones(len(cls))] + [(cls == c).astype(np.float64) for c in levels[1:]])
    def resid(r):
        beta, *_ = np.linalg.lstsq(Z, r, rcond=None)
        return r - Z @ beta
    ea, eb = resid(ra), resid(rb)
    den = np.sqrt((ea ** 2).sum(0) * (eb ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (ea * eb).sum(0) / den, np.nan)


def _class_balanced(cls: np.ndarray, seed: int = 0) -> np.ndarray:
    """Mask keeping every class drawn down to the smallest class's count (seeded)."""
    rng = np.random.default_rng(seed)
    levels, counts = np.unique(cls, return_counts=True)
    keep = np.zeros(len(cls), bool)
    for c in levels:
        idx = np.flatnonzero(cls == c)
        keep[rng.choice(idx, counts.min(), replace=False)] = True
    return keep


def _modal_share(x: np.ndarray) -> np.ndarray:
    out = np.empty(x.shape[1])
    for j in range(x.shape[1]):
        _, c = np.unique(x[:, j], return_counts=True)
        out[j] = c.max() / x.shape[0]
    return out


def _floor(curve: dict[int, float], s_list: list[int], thr: float) -> str:
    vals = [curve.get(s, np.nan) for s in s_list]
    if all(np.isnan(v) for v in vals):
        return "undefined"
    if np.isnan(vals[-1]) or vals[-1] < thr:
        return "never"
    floor = s_list[-1]
    for s, v in zip(reversed(s_list), reversed(vals)):
        if np.isnan(v) or v < thr:
            break
        floor = s
    return str(floor)


def operator_class(n: str) -> str:
    if any(t in n for t in ("_band_energy", "spectral_centroid", "_bandwidth", "halfcentroid")):
        return "stft"
    if n.startswith("spectral_"):
        return "graph_spectral"
    if n.startswith("epoch_") or n == "surprise_boundary_count":
        return "event_count"
    if "_w" in n and any(f"_w{i}_" in n for i in range(8)):
        return "windowed_dispersion" if n.endswith(("_std", "_slope")) else "windowed_level"
    if n.startswith("pca_") or "_traj" in n and not n.startswith("res_traj"):
        return "snapshot"
    if "cka" in n or "cross_layer" in n or "crosslayer" in n or n.startswith("attnres_depthslope"):
        return "cross_layer"
    toks = set(n.split("_"))
    if toks & {"std", "slope", "drift", "min", "max", "stability", "regularity", "transition",
               "acceleration", "cv", "normcv"} or n.startswith("std_"):
        return "dispersion"
    return "level"


def _load_shards(sig_dirs: list[Path]) -> tuple[dict, dict, list[dict]]:
    files = [f for d in sig_dirs for f in sorted(d.glob("windows-*.npz"))]
    if not files:
        raise SystemExit(f"no windows-*.npz in {', '.join(map(str, sig_dirs))}")
    names: dict[str, list[str]] = {}
    feats: dict[str, list[np.ndarray]] = defaultdict(list)
    meta_cols: dict[str, list] = defaultdict(list)
    for f in files:
        z = np.load(f, allow_pickle=False)
        for k in z.files:
            if k.startswith("names_"):
                nm = [str(x) for x in z[k]]
                if names.setdefault(k[6:], nm) != nm:
                    raise SystemExit(f"schema differs across shards for {k}")
            elif k.startswith("feat_"):
                feats[k[5:]].append(z[k])
            else:
                meta_cols[k].append(z[k])
    run_meta = [json.loads(p.read_text()) for d in sig_dirs for p in sorted(d.glob("run_meta-*.json"))]
    return names, {k: np.concatenate(v) for k, v in feats.items()}, \
        [{k: np.concatenate(v) for k, v in meta_cols.items()}, run_meta]


def analyse(args) -> int:
    try:   # this repo's taxonomy; on a node it is staged beside this script as feature_map.py
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from anamnesis.analysis.feature_map import classify
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from feature_map import classify

    names, feats, (meta, run_meta) = _load_shards(args.sig_dir)
    s_list = sorted(int(k[1:]) for k in names if k.startswith("s"))
    full_names = names["full"]
    n_rows = feats["full"].shape[0]
    support = {}
    with open(args.support_tsv) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            r = dict(zip(hdr, line.rstrip("\n").split("\t")))
            support[r["feature_name"]] = r

    # row subsets: all; one row per prompt (first seed); low-repetition rows
    pid = meta["prompt_id"]
    first = np.zeros(n_rows, bool)
    seen = set()
    for i, p in enumerate(pid):
        if p not in seen:
            first[i] = True
            seen.add(p)
    distinct = meta["distinct_frac"]
    subsets = {"all": np.ones(n_rows, bool), "one_per_prompt": first,
               "nonrepetitive": distinct >= np.quantile(distinct, 0.25)}
    pclass = np.array([str(x) for x in meta["prompt_class"]])
    classes = sorted(set(pclass.tolist()))
    multiclass = len(classes) > 1
    if multiclass:
        subsets["class_balanced"] = _class_balanced(pclass)
        for c in classes:
            subsets[f"class={c}"] = pclass == c

    fidx = {n: i for i, n in enumerate(full_names)}
    full = feats["full"].astype(np.float64)
    full_degen = _modal_share(full) >= MODAL_DEGENERATE
    P = meta["P"].astype(np.float64)

    # per s: column map into full-name order
    rel = {sub: {p: np.full((len(s_list), len(full_names)), np.nan)
                 for p in ("AB", "CD", "AD", "Afull", "ABpartP", "ABpartClass")}
           for sub in subsets}
    degen = np.zeros((len(s_list), len(full_names)), bool)
    emitted = np.zeros((len(s_list), len(full_names)), bool)
    shift_AB = np.full((len(s_list), len(full_names)), np.nan)   # standardized paired shift B - A
    shift_CD = np.full((len(s_list), len(full_names)), np.nan)
    for si, s in enumerate(s_list):
        nm = names[f"s{s}"]
        cols = np.array([fidx[n] for n in nm if n in fidx])
        loc = np.array([j for j, n in enumerate(nm) if n in fidx])
        emitted[si, cols] = True
        W = {w: feats[f"s{s}_{w}"][:, loc].astype(np.float64) for w in WINDOWS}
        dg = np.zeros(len(loc), bool)
        for w in WINDOWS[:2]:
            dg |= _modal_share(W[w]) >= MODAL_DEGENERATE
        degen[si, cols] = dg
        sd = np.sqrt((W["A"].var(0) + W["B"].var(0)) / 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            shift_AB[si, cols] = np.where(sd > 0, (W["B"] - W["A"]).mean(0) / sd, np.nan)
            sdl = np.sqrt((W["C"].var(0) + W["D"].var(0)) / 2)
            shift_CD[si, cols] = np.where(sdl > 0, (W["D"] - W["C"]).mean(0) / sdl, np.nan)
        for sub, mask in subsets.items():
            a, b, c, d = (W[w][mask] for w in WINDOWS)
            fu = full[mask][:, cols]
            for pair, (x, y) in {"AB": (a, b), "CD": (c, d), "AD": (a, d), "Afull": (a, fu)}.items():
                r = _spearman(x, y)
                r[dg] = np.nan
                if pair == "Afull":
                    r[full_degen[cols]] = np.nan
                rel[sub][pair][si, cols] = r
            if sub == "all":
                r = _partial_spearman(a, b, P[mask])
                r[dg] = np.nan
                rel[sub]["ABpartP"][si, cols] = r
                r = _partial_spearman_classes(a, b, pclass[mask])
                r[dg] = np.nan
                rel[sub]["ABpartClass"][si, cols] = r
        print(f"[analyse] s={s}: {len(loc)} names, degenerate {int(dg.sum())}", flush=True)

    # per-feature table
    rows_out = []
    for j, n in enumerate(full_names):
        tag = classify(n, 72)
        sup = support.get(n, {})
        curve = {s: rel["all"]["AB"][si, j] for si, s in enumerate(s_list)}
        conv = {s: rel["all"]["Afull"][si, j] for si, s in enumerate(s_list)}
        late = {s: rel["all"]["CD"][si, j] for si, s in enumerate(s_list)}
        opp = {s: rel["one_per_prompt"]["AB"][si, j] for si, s in enumerate(s_list)}
        partP = {s: rel["all"]["ABpartP"][si, j] for si, s in enumerate(s_list)}
        partC = {s: rel["all"]["ABpartClass"][si, j] for si, s in enumerate(s_list)}
        bal = ({s: rel["class_balanced"]["AB"][si, j] for si, s in enumerate(s_list)}
               if multiclass else {})
        near_const = bool(full_degen[j]) or bool(degen[-1, j])
        rows_out.append(dict(
            feature_name=n, family=sup.get("family", tag.family), source=tag.source.value,
            method=tag.method.value, dynamic=("dynamic" if tag.dynamic else "static"),
            operator=operator_class(n), min_steps_computed=sup.get("min_steps_computed", "?"),
            meaning_floor_r50=_floor(curve, s_list, 0.5), meaning_floor_r70=_floor(curve, s_list, 0.7),
            conv_floor_r50=_floor(conv, s_list, 0.5), conv_floor_r70=_floor(conv, s_list, 0.7),
            late_floor_r50=_floor(late, s_list, 0.5),
            one_per_prompt_floor_r50=_floor(opp, s_list, 0.5),
            partialP_floor_r50=_floor(partP, s_list, 0.5),
            partialClass_floor_r50=_floor(partC, s_list, 0.5),
            balanced_floor_r50=_floor(bal, s_list, 0.5),
            **{f"class_floor_r50[{c}]": _floor({s: rel[f"class={c}"]["AB"][si, j] for si, s in enumerate(s_list)},
                                               s_list, 0.5) for c in (classes if multiclass else [])},
            near_constant=int(near_const),
            **{f"rel_s{s}": curve[s] for s in s_list},
            **{f"conv_s{s}": conv[s] for s in s_list},
            **{f"shiftAB_s{s}": shift_AB[si, j] for si, s in enumerate(s_list)},
        ))
    out_tsv = args.out_dir / f"{args.tag}.tsv"
    cols = list(rows_out[0])
    with out_tsv.open("w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows_out:
            fh.write("\t".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    np.savez(args.out_dir / f"{args.tag}.curves.npz", s_list=np.array(s_list), names=np.array(full_names),
             degenerate=degen, emitted=emitted, shift_AB=shift_AB, shift_CD=shift_CD,
             **{f"rel_{sub}_{p}": v for sub, d in rel.items() for p, v in d.items()})
    summary = dict(n_rows=n_rows, n_prompts=int(first.sum()), s_list=s_list, subsets={k: int(v.sum()) for k, v in subsets.items()},
                   distinct_frac=dict(min=float(distinct.min()), median=float(np.median(distinct)), max=float(distinct.max())),
                   P=dict(min=int(P.min()), median=float(np.median(P)), max=int(P.max())),
                   prompt_class=dict(Counter(str(x) for x in meta["prompt_class"])),
                   parity=[p for m in run_meta for p in m.get("parity", [])],
                   corpus_filter=[m.get("corpus_filter") for m in run_meta],
                   lane_id_full=sorted({x for m in run_meta for x in m.get("lane_id_full", [])}),
                   code=run_meta[0].get("code") if run_meta else None,
                   calibration_sha256=(run_meta[0].get("calibration") or {}).get("calibration_sha256") if run_meta else None)
    (args.out_dir / f"{args.tag}.summary.json").write_text(json.dumps(summary, indent=1, default=str))
    write_report(args.out_dir / f"{args.tag}.md", rows_out, summary, s_list, rel, degen, emitted,
                 shift_AB, shift_CD, full_names, args.tag)
    print(f"[analyse] wrote {out_tsv} ({len(rows_out)} features, {n_rows} rows)")
    return 0


FLOOR_BUCKETS = ["1", "2", "3", "4", "6", "8", "12", "16", "24", "32", "48", "64", "96", "128",
                 "never", "undefined"]


def _bucket_counts(vals: list[str], s_list: list[int]) -> dict[str, int]:
    c = Counter(vals)
    return {b: c.get(b, 0) for b in [str(s) for s in s_list] + ["never", "undefined"] if c.get(b, 0)}


def _fmt_counts(d: dict[str, int]) -> str:
    return ", ".join(f"{k}: {v}" for k, v in d.items())


def _median_floor(vals: list[str]) -> str:
    """Median floor, ordering never after every s and dropping undefined."""
    num = sorted((math.inf if v == "never" else int(v)) for v in vals if v != "undefined")
    if not num:
        return "undefined"
    m = num[len(num) // 2]
    return "never" if m == math.inf else str(m)


def write_report(path: Path, rows: list[dict], summary: dict, s_list: list[int], rel: dict,
                 degen: np.ndarray, emitted: np.ndarray, shift_AB: np.ndarray, shift_CD: np.ndarray,
                 names: list[str], tag: str) -> None:
    """The generated part of the summary: tables only; the reading is hand-written beside it."""
    L: list[str] = [f"# Meaning floor: tsumugi fast lane (generated tables)", "",
                    f"Produced by `pipeline/anamnesis/scripts/meaning_floor.py analyse`; per-feature table "
                    f"`{tag}.tsv`, curves `{tag}.curves.npz`, provenance `{tag}.summary.json`.", ""]
    L += ["## Rows", "", f"- rows {summary['n_rows']} from {summary['n_prompts']} prompts; subsets "
          f"{summary['subsets']}; prompt length P {summary['P']}; distinct-token fraction "
          f"{summary['distinct_frac']}; classes {summary['prompt_class']}",
          f"- parity (fan-out full span vs `replay_features_gpu`): {summary['parity']}", ""]
    # useful schema curve
    L += ["## Useful schema at span length s", "",
          "Features (of the 4,833 full-span names) with onset split-half reliability r_AB(s) >= threshold "
          "at that s; `floor<=s` counts features whose stable floor is at or below s; `degenerate` = "
          "modal value held by >= 50% of rows in A or B (includes placeholders); `emitted` = names in the "
          "s-schema.", "",
          "| s | tokens | emitted | degenerate | r>=0.5 | r>=0.7 | floor50<=s | floor70<=s | conv>=0.5 | late r_CD>=0.5 |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    f50 = [r["meaning_floor_r50"] for r in rows]
    f70 = [r["meaning_floor_r70"] for r in rows]
    for si, s in enumerate(s_list):
        ab, cd, af = rel["all"]["AB"][si], rel["all"]["CD"][si], rel["all"]["Afull"][si]
        le = lambda fl: sum(1 for v in fl if v not in ("never", "undefined") and int(v) <= s)
        L.append(f"| {s} | {s + 1} | {int(emitted[si].sum())} | {int(degen[si].sum())} | "
                 f"{int(np.nansum(ab >= 0.5))} | {int(np.nansum(ab >= 0.7))} | {le(f50)} | {le(f70)} | "
                 f"{int(np.nansum(af >= 0.5))} | {int(np.nansum(cd >= 0.5))} |")
    L.append("")
    for key, title in (("family", "family"), ("operator", "operator class"), ("method", "method"),
                       ("dynamic", "static vs dynamic"), ("source", "source")):
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        L += [f"## Floors per {title}", "",
              f"| {key} | n | median floor r50 | median floor r70 | median conv floor r50 | near-const | floor r50 distribution |",
              "|---|---:|---:|---:|---:|---:|---|"]
        for g, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            L.append(f"| {g} | {len(rs)} | {_median_floor([r['meaning_floor_r50'] for r in rs])} | "
                     f"{_median_floor([r['meaning_floor_r70'] for r in rs])} | "
                     f"{_median_floor([r['conv_floor_r50'] for r in rs])} | "
                     f"{sum(r['near_constant'] for r in rs)} | "
                     f"{_fmt_counts(_bucket_counts([r['meaning_floor_r50'] for r in rs], s_list))} |")
        L.append("")
    # median reliability curve per operator
    ops = sorted({r["operator"] for r in rows})
    L += ["## Median onset reliability r_AB(s) per operator class (non-degenerate features)", "",
          "| operator | " + " | ".join(f"s={s}" for s in s_list) + " |",
          "|---|" + "---:|" * len(s_list)]
    opidx = {o: np.array([i for i, r in enumerate(rows) if r["operator"] == o]) for o in ops}
    for o in ops:
        cells = []
        for si in range(len(s_list)):
            v = rel["all"]["AB"][si, opidx[o]]
            cells.append("-" if np.all(np.isnan(v)) else f"{np.nanmedian(v):.2f}")
        L.append(f"| {o} | " + " | ".join(cells) + " |")
    L.append("")
    # position controls
    L += ["## Position controls", "",
          "Median over features of: |standardized paired shift| (B-A onset, D-C late; mean difference over "
          "pooled across-row SD); onset r_AB minus late r_CD; far-pair r_AD.", "",
          "| s | med abs shift B-A | med abs shift D-C | med r_AB | med r_CD | med r_AD | med r_AB - r_CD |",
          "|---:|---:|---:|---:|---:|---:|---:|"]
    for si, s in enumerate(s_list):
        ab, cd, ad = rel["all"]["AB"][si], rel["all"]["CD"][si], rel["all"]["AD"][si]
        L.append(f"| {s} | {np.nanmedian(np.abs(shift_AB[si])):.2f} | {np.nanmedian(np.abs(shift_CD[si])):.2f} | "
                 f"{np.nanmedian(ab):.2f} | {np.nanmedian(cd):.2f} | {np.nanmedian(ad):.2f} | "
                 f"{np.nanmedian(ab - cd):+.2f} |")
    L.append("")
    # definition disagreements and sensitivity
    def cmp(a: str, b: str) -> tuple[int, int, int]:
        lo = eq = hi = 0
        for r in rows:
            x, y = r[a], r[b]
            if "undefined" in (x, y):
                continue
            xv = math.inf if x == "never" else int(x)
            yv = math.inf if y == "never" else int(y)
            lo += yv < xv; eq += yv == xv; hi += yv > xv
        return lo, eq, hi
    L += ["## Definition and subset comparisons (floor r50 vs alternative)", "",
          "| alternative | earlier | same | later |", "|---|---:|---:|---:|"]
    for alt in ("conv_floor_r50", "late_floor_r50", "one_per_prompt_floor_r50", "partialP_floor_r50",
                "partialClass_floor_r50", "balanced_floor_r50"):
        lo, eq, hi = cmp("meaning_floor_r50", alt)
        L.append(f"| {alt} | {lo} | {eq} | {hi} |")
    L.append("")
    class_subs = sorted(k for k in rel if k.startswith("class="))
    if class_subs:
        L += ["## Prompt classes", "",
              "Pooled r_AB rises with between-class variance alone. `partialClass` is r_AB after regressing ranks "
              "on class indicators (reliability inside classes); `class=<c>` is r_AB within one class. A feature "
              "with a finite pooled floor and a `never` class-partialled floor reads the class, not the pass.", "",
              "Median r_AB(s) over non-degenerate features:", "",
              "| rows | " + " | ".join(f"s={s}" for s in s_list) + " |", "|---|" + "---:|" * len(s_list)]
        for label, arr in ([("pooled", rel["all"]["AB"]), ("class-partialled", rel["all"]["ABpartClass"]),
                            ("class_balanced", rel["class_balanced"]["AB"])]
                           + [(k, rel[k]["AB"]) for k in class_subs]):
            cells = ["-" if np.all(np.isnan(arr[si])) else f"{np.nanmedian(arr[si]):.2f}" for si in range(len(s_list))]
            L.append(f"| {label} | " + " | ".join(cells) + " |")
        reads_class = [r for r in rows if r["meaning_floor_r50"] not in ("never", "undefined")
                       and r["partialClass_floor_r50"] == "never"]
        L += ["", f"Features with a finite pooled floor r50 and a `never` class-partialled floor (class readers): "
              f"{len(reads_class)}. By family: {_fmt_counts(dict(Counter(r['family'] for r in reads_class).most_common()))}.", ""]
    # Column C candidates
    cand = [r for r in rows if r["dynamic"] == "static" and r["operator"] in ("level", "cross_layer")
            and r["meaning_floor_r50"] not in ("never", "undefined") and int(r["meaning_floor_r50"]) <= 4]
    L += ["## FLOOR-schema candidates", "",
          f"Static aggregates (operator level or cross_layer, classify() static) with meaning floor r50 <= 4 "
          f"steps: {len(cand)}. By family: "
          f"{_fmt_counts(dict(Counter(r['family'] for r in cand).most_common()))}. "
          f"Of these, floor r70 <= 4: {sum(1 for r in cand if r['meaning_floor_r70'] not in ('never','undefined') and int(r['meaning_floor_r70']) <= 4)}.", "",
          "Top 40 by r_AB at s=4:", "", "| feature | family | source | min_steps_computed | floor r50 | floor r70 | r(s=2) | r(s=4) | r(s=8) |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(cand, key=lambda r: -(r.get("rel_s4") if not np.isnan(r.get("rel_s4", np.nan)) else -9))[:40]:
        L.append(f"| `{r['feature_name']}` | {r['family']} | {r['source']} | {r['min_steps_computed']} | "
                 f"{r['meaning_floor_r50']} | {r['meaning_floor_r70']} | {r.get('rel_s2', np.nan):.2f} | "
                 f"{r.get('rel_s4', np.nan):.2f} | {r.get('rel_s8', np.nan):.2f} |")
    L.append("")
    path.write_text("\n".join(L) + "\n")


def selftest(args) -> int:
    """Synthetic windows with a known answer, through ``analyse``.

    Feature j = row trait * a_j + noise / sqrt(s): true split-half reliability is
    a_j^2 / (a_j^2 + 1/s), so its 0.5 floor is the smallest tested s >= 1 / a_j^2. One
    feature is a literal 0 below s = 4, one is constant everywhere, one is pure noise.
    Rows fall in 4 prompt classes; the last feature is a class offset plus noise and no row
    trait, so its pooled floor is finite and its class-partialled floor is ``never``, while
    the strongest trait feature keeps a finite class-partialled floor."""
    rng = np.random.default_rng(0)
    n, s_list, ngen = 400, S_LIST, NGEN
    amps = np.array([2.0, 1.0, 0.5, 0.25, 0.125, 0.0, 1.0, 0.0, 0.0])
    names = [f"syn_a{a:g}_mean_L{j}" for j, a in enumerate(amps)]
    names[8] = "syn_class_mean_L8"
    trait = rng.standard_normal(n)
    pclass = np.array([f"c{i % 4}" for i in range(n)])
    offset = np.array([-3.0, -1.0, 1.0, 3.0])[np.arange(n) % 4]
    out: dict[str, np.ndarray] = {}
    def draw(s):
        x = trait[:, None] * amps[None] + rng.standard_normal((n, len(amps))) / math.sqrt(s)
        if s < 4:
            x[:, 6] = 0.0
        x[:, 7] = 1.0
        x[:, 8] += offset
        return x.astype(np.float32)
    for s in s_list:
        out[f"names_s{s}"] = np.array(names)
        for w in WINDOWS:
            out[f"feat_s{s}_{w}"] = draw(s)
    out["names_full"] = np.array(names)
    out["feat_full"] = draw(ngen - 1)
    out.update(gid=np.arange(n), prompt_id=np.array([f"p{i // 2}" for i in range(n)]),
               prompt_class=pclass, P=rng.integers(20, 60, n),
               distinct_frac=rng.uniform(0.3, 0.9, n))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "windows-00of01.npz", **out)
    sup = args.out_dir / "support.tsv"
    sup.write_text("feature_name\tfamily\tmin_steps_computed\n" +
                   "".join(f"{nm}\tsynthetic\t1\n" for nm in names))
    ns = argparse.Namespace(sig_dir=[args.out_dir], support_tsv=sup, out_dir=args.out_dir, tag="selftest")
    analyse(ns)
    got = {}
    with (args.out_dir / "selftest.tsv").open() as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            r = dict(zip(hdr, line.rstrip("\n").split("\t")))
            got[r["feature_name"]] = r
    ok = True
    cls_nm, trait_nm = names[8], names[0]
    for nm, field, want_finite in ((cls_nm, "meaning_floor_r50", True), (cls_nm, "partialClass_floor_r50", False),
                                   (trait_nm, "partialClass_floor_r50", True)):
        f = got[nm][field]
        good = (f not in ("never", "undefined")) if want_finite else (f == "never")
        ok &= good
        print(f"[selftest] {nm}: {field} {f} expected {'finite' if want_finite else 'never'} "
              f"{'ok' if good else 'MISMATCH'}")
    for j, nm in enumerate(names):
        if j == 8:
            continue
        a = amps[j]
        if j == 7:
            exp = "undefined"
        elif a == 0:
            exp = "never"
        else:
            lo = 4 if j == 6 else 1   # feature 6 is a literal 0 below s = 4
            exp = str(next((s for s in s_list if s >= lo and a * a / (a * a + 1 / s) >= 0.5), "never"))
        f = got[nm]["meaning_floor_r50"]
        flag = "ok" if f == exp or (exp not in ("never", "undefined") and f not in ("never", "undefined")
                                     and abs(s_list.index(int(f)) - s_list.index(int(exp))) <= 1) else "MISMATCH"
        ok &= flag == "ok"
        print(f"[selftest] {nm}: floor50 {f} expected {exp} near_const {got[nm]['near_constant']} {flag}")
    return 0 if ok else 1


def diff(args) -> int:
    """Two signings of the same rows (e.g. two processes) compared feature by feature.

    Rows are matched by generation id. Per block (``full`` and each ``s<k>_<window>``):
    the count of differing coordinates, the max absolute difference, and the row L2 of
    the difference in z units, where each feature's scale is its across-row SD in
    ``--scale-dir`` (a large signing of the same lane). Features constant in the scale
    bank are left out of the z distance."""
    na, fa, (ma, _) = _load_shards([args.a])
    nb, fb, (mb, _) = _load_shards([args.b])
    ns, fs, _ = _load_shards([args.scale_dir])
    ga, gb = [int(x) for x in ma["gid"]], [int(x) for x in mb["gid"]]
    common = sorted(set(ga) & set(gb))
    if not common:
        raise SystemExit("no generation ids in common")
    ia = np.array([ga.index(g) for g in common])
    ib = np.array([gb.index(g) for g in common])
    report: dict[str, Any] = dict(rows=len(common), a=str(args.a), b=str(args.b), blocks={})
    for key in sorted(fa):
        if key not in fb:
            continue
        group = "full" if key == "full" else key.split("_")[0]
        if na[group] != nb[group]:
            raise SystemExit(f"schemas differ for {group}")
        xa, xb = fa[key][ia].astype(np.float64), fb[key][ib].astype(np.float64)
        d = xb - xa
        sd = fs[key].astype(np.float64).std(0) if key in fs and ns.get(group) == na[group] else None
        entry = dict(n_coords=int(d.size), n_differ=int((d != 0).sum()),
                     max_abs=float(np.abs(d).max()),
                     features_differ=int((d != 0).any(0).sum()), n_features=int(d.shape[1]))
        if sd is not None:
            live = sd > 1e-12
            z = d[:, live] / sd[live]
            l2 = np.sqrt((z ** 2).sum(1))
            entry.update(z_l2_median=float(np.median(l2)), z_l2_max=float(l2.max()),
                         z_abs_max=float(np.abs(z).max()), z_features=int(live.sum()))
        report["blocks"][key] = entry
        print(key, json.dumps(entry), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
        print(f"[diff] wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sg = sub.add_parser("sign")
    sg.add_argument("--gen-dir", type=Path, required=True)
    sg.add_argument("--out-dir", type=Path, required=True)
    sg.add_argument("--ngen", type=int, default=NGEN)
    sg.add_argument("--s", type=int, nargs="*", default=S_LIST)
    sg.add_argument("--checkpoint-path", default="/models/pleroma/koto-assets/tsumugi-release-s52450/model.pt")
    sg.add_argument("--calib-dir", type=Path,
                    default=Path("/mnt/beegfs/luxia/pleroma/tsumugi/calibration/release-s52450-ext4096"))
    sg.add_argument("--manifest", type=Path,
                    default=Path("/mnt/beegfs/luxia/pleroma/tsumugi/gpu_lane/manifest_tsumugi_release.npz"))
    sg.add_argument("--preset", default="tsumugi-1b-release-s52450")
    sg.add_argument("--koto-size", default="1b-tsumugi")
    sg.add_argument("--koto-boundaries", default="tsumugi_loop")
    sg.add_argument("--shard", type=int, default=0)
    sg.add_argument("--n-shards", type=int, default=1)
    sg.add_argument("--limit", type=int, default=0)
    sg.add_argument("--checkpoint", type=int, default=10)
    sg.add_argument("--parity-rows", type=int, default=2)
    sg.add_argument("--strict-parity", action="store_true")
    an = sub.add_parser("analyse")
    home = Path.home() / "projects/anamnesis_exps"
    an.add_argument("--sig-dir", type=Path, required=True, action="append",
                    help="a directory of windows-*.npz shards; repeat to pool draws")
    an.add_argument("--support-tsv", type=Path,
                    default=home / "outputs/analysis/feature_support_tsumugi_2026-10-28.tsv")
    an.add_argument("--out-dir", type=Path, default=home / "outputs/analysis")
    an.add_argument("--tag", default="meaning_floor_tsumugi_2026-10-28")
    st = sub.add_parser("selftest")
    st.add_argument("--out-dir", type=Path, required=True)
    df = sub.add_parser("diff")
    df.add_argument("--a", type=Path, required=True)
    df.add_argument("--b", type=Path, required=True)
    df.add_argument("--scale-dir", type=Path, required=True)
    df.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    return {"sign": sign, "analyse": analyse, "selftest": selftest, "diff": diff}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
