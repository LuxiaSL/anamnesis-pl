"""Luxia's marks vs the signature's coherence measures (signature curation v2, validation only).

Her five marked scenes (``curation_marks_rows.py``) are replayed under the base and under r1-SFT. Two measures per
turn:

* **ranker score.** The curation ranker (antisymmetric logistic on within-group pair differences), trained on every
  labelled turn of the given base-reader banks (``curation_grid_bank.py`` cells). Turns are z-scored by those banks'
  pooled within-group SD. score_z is the score z-scored within its scene.
* **reader divergence.** |z(SFT reading) - z(base reading) - mean shift|, on the same scale, z-scored within its
  scene. In the grid it was higher on coherent turns.

Each is compared by mark, inside her labelling (``within_her_labelling``): spark, break, dead, and unmarked. Reported:
* AUC spark vs break;
* AUC unmarked vs break;
* the mean by mark;
* per writer.
Both measures are also given with scene-z log length regressed out, since length ranks coherence too. Plus a
per-turn table for her to read. n is tiny (14 sparks, 21 breaks, 5 scenes), so this is a sanity check, not an
estimate.

Usage: python curation_marks_read.py --train <bank dir>[:<writer>] ... --rows <marks import rows> --base <signed dir>
       --sft <signed dir> --out <json>
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


ctr = _sibling("curation_turn_read")
msr = sys.modules["msteer_replay_read"]
p2 = _sibling("msteer_p2_read")


def auc(a: np.ndarray, b: np.ndarray) -> float | None:
    """P(a > b) over all pairs, ties half: the AUC of a over b."""
    if len(a) == 0 or len(b) == 0:
        return None
    d = a[:, None] - b[None, :]
    return round(float(((d > 0) + 0.5 * (d == 0)).mean()), 3)


def within_z(x: np.ndarray, scene: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for s in np.unique(scene):
        u = scene == s
        sd = x[u].std()
        out[u] = (x[u] - x[u].mean()) / (sd if sd > 0 else 1.0)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, action="append")
    ap.add_argument("--rows", required=True, type=Path)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--sft", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    meta, blocks, names = [], [], None
    for spec in args.train:
        path, _, writer = spec.partition(":")
        _, rows = msr.load_rows(Path(path) / "rows.jsonl")
        X, ids, n, _, _ = msr.load_bank(Path(path) / "sigs", "engine-stats-b1")
        if names is not None and n != names:
            raise ValueError(f"{path}: feature names differ")
        names = n
        keep = [i for i, c in enumerate(ids) if not writer or rows[c].get("writer") == writer]
        meta += [dict(rows[ids[i]], scene_id=f"{path}|{rows[ids[i]]['scene_id']}|{rows[ids[i]].get('writer')}")
                 for i in keep]
        blocks.append(X[keep].astype(np.float64))
    X = np.concatenate(blocks)
    group = np.array([m["scene_id"] for m in meta])
    finite = np.isfinite(X).all(axis=0)
    Xc = X[:, finite].copy()
    for g in np.unique(group):
        u = group == g
        Xc[u] -= Xc[u].mean(axis=0)
    sd = np.sqrt((Xc ** 2).sum(axis=0) / max(len(meta) - len(np.unique(group)), 1))
    live = sd > 1e-9
    Z = Xc[:, live] / sd[live]
    pairs, pscene = ctr.make_pairs(meta, 4, 2, 20, 0)
    C = msr.pick_c(pairs, Z, pscene)
    x, y = msr.design(pairs, Z)
    w = msr.fit(x, y, C).coef_.ravel()
    print(f"[marks] trained on {len(meta)} turns, {len(pairs)} pairs, C {C}", flush=True)

    mrows = {}
    for line in args.rows.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            mrows[r["id"]] = dict(r["meta"], n_gen=len(r["ids"]) - r["prompt_length"])
    base, sft = p2.load_signed(args.base), p2.load_signed(args.sft)
    ids = sorted(i for i in mrows if i in base and i in sft and base[i]["names"][:len(names)] == names)

    def zrow(v: np.ndarray) -> np.ndarray:
        return np.nan_to_num(v[:len(names)][finite][live] / sd[live])

    B = np.stack([zrow(base[i]["vec"]) for i in ids])
    S = np.stack([zrow(sft[i]["vec"]) for i in ids])
    scene = np.array([str(mrows[i]["scene"]) for i in ids])
    score = B @ w
    E = S - B
    div = np.linalg.norm(E - E.mean(axis=0), axis=1)
    score_z, div_z = within_z(score, scene), within_z(div, scene)
    mark = np.array([mrows[i]["mark"] for i in ids])
    inside = np.array([bool(mrows[i].get("within_her_labelling")) for i in ids])
    writer = np.array([str(mrows[i]["writer"]) for i in ids])
    loglen = np.log(np.array([mrows[i]["n_gen"] for i in ids], dtype=float))
    len_z = within_z(loglen, scene)

    def resid(v: np.ndarray) -> np.ndarray:
        """v with scene-z log length regressed out (over every turn)."""
        b = float(len_z @ v / (len_z @ len_z))
        return v - b * len_z

    score_res, div_res = resid(score_z), resid(div_z)

    def compare(sel: np.ndarray) -> dict[str, Any]:
        res: dict[str, Any] = {}
        for mname, v in (("score_z", score_z), ("divergence_z", div_z), ("length_z", len_z),
                         ("score_z_length_resid", score_res), ("divergence_z_length_resid", div_res)):
            g = {k: v[sel & inside & (mark == k)] for k in ("spark", "break", "dead", "unmarked")}
            res[mname] = dict(mean={k: (round(float(a.mean()), 3) if len(a) else None) for k, a in g.items()},
                              n={k: int(len(a)) for k, a in g.items()},
                              auc_spark_vs_break=auc(g["spark"], g["break"]),
                              auc_unmarked_vs_break=auc(g["unmarked"], g["break"]),
                              auc_spark_vs_unmarked=auc(g["spark"], g["unmarked"]))
        return res

    out: dict[str, Any] = dict(turns=len(ids), inside=int(inside.sum()), all=compare(np.ones(len(ids), bool)),
                               by_writer={wr: compare(writer == wr) for wr in sorted(set(writer))})
    table = [dict(id=i, scene=str(mrows[i]["scene"]), turn=mrows[i]["turn"], writer=mrows[i]["writer"],
                  mark=mrows[i]["mark"], inside=bool(mrows[i].get("within_her_labelling")),
                  score_z=round(float(s), 2), divergence_z=round(float(d), 2), length_z=round(float(l), 2))
             for i, s, d, l in zip(ids, score_z, div_z, len_z)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    with args.out.with_suffix(".turns.jsonl").open("w") as fh:
        for r in sorted(table, key=lambda r: (r["scene"], int(r["turn"]))):
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(out["all"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
