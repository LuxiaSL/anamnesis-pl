"""14n item-2b — DIVERGENCE-TIME decomposition (session-9 Part A; the MAGNITUDE gauge).

Beside distinct-4's DIRECTION gauge: for each steered generation, find its seed-matched
baseline twin (same topic_idx, mode_idx, seed — verified 160/160 alignment) and record the
first WORD INDEX at which the two token streams diverge (they share a seed + prompt, so they
run identical until the injection flips a sampled token). Smaller index = the perturbation
bites earlier = larger magnitude. Per (sign, dose): V7 vs the matched Rband nulls.

Frozen (outer loop): P=.40 that a V7-specific divergence-time ADVANCE (earlier divergence)
separates from Rband at matched dose/norm beyond n-noise. Lean: divergence-time is
MAGNITUDE-GENERIC (V7 ~= Rband); the outer-loop spot-check reads V7+.1 median 18 vs Rband+.1
median 26. Mann-Whitney U (V7 vs pooled-Rband) two-sided; ADVANCE = V7 median < Rband median
AND p<.05. Text-only, CPU. First-read -> outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy.stats import mannwhitneyu
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _perm_u_p(a, b, iters=20000, seed=20260716):
    """Permutation two-sided p for difference in medians (fallback if no scipy)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = abs(np.median(a) - np.median(b))
    pool = np.concatenate([a, b]); na = len(a)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(iters):
        rng.shuffle(pool)
        if abs(np.median(pool[:na]) - np.median(pool[na:])) >= obs - 1e-12:
            cnt += 1
    return (cnt + 1) / (iters + 1)


def _load(run_dir: Path, cell: str) -> list[dict] | None:
    p = run_dir / cell / "metadata.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    return m["generations"] if "generations" in m else m


def _first_divergence(a: str, b: str) -> int:
    aw, bw = a.split(), b.split()
    n = min(len(aw), len(bw))
    for i in range(n):
        if aw[i] != bw[i]:
            return i
    return n  # identical up to the shorter length


def _cell_divergences(gens: list[dict], base_by_key: dict) -> list[int]:
    out = []
    for g in gens:
        k = (g["topic_idx"], g["mode_idx"], g["seed"])
        tw = base_by_key.get(k)
        if tw is None:
            continue
        out.append(_first_divergence(g.get("generated_text", ""), tw))
    return out


def _parse(cell: str) -> dict:
    if cell.startswith("baseline"):
        return {"vector": "baseline", "site": None, "alpha_frac": 0.0}
    parts = cell.split("_")
    return {"vector": parts[0], "site": int(parts[1][1:]), "alpha_frac": float(parts[2][1:])}


def _scan(run_dir: Path, sign: str, target: str, null_prefixes: tuple) -> dict:
    base_gens = _load(run_dir, "baseline_a0")
    if not base_gens:
        return {"error": f"no baseline_a0 in {run_dir}"}
    base_by_key = {(g["topic_idx"], g["mode_idx"], g["seed"]): g.get("generated_text", "") for g in base_gens}
    per_cell = {}
    for d in sorted(run_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("baseline"):
            continue
        gens = _load(run_dir, d.name)
        if not gens:
            continue
        meta = _parse(d.name)
        divs = _cell_divergences(gens, base_by_key)
        per_cell[d.name] = {**meta, "sign": sign, "n": len(divs),
                            "median": float(np.median(divs)), "mean": round(float(np.mean(divs)), 2),
                            "is_null": d.name.upper().startswith(null_prefixes), "_divs": divs}
    # per-dose V7 vs pooled Rband
    doses = sorted({c["alpha_frac"] for c in per_cell.values()})
    rows = []
    for af in doses:
        tgt = next((c for c in per_cell.values() if c["vector"] == target and c["alpha_frac"] == af), None)
        nulls = [c for c in per_cell.values() if c["is_null"] and c["alpha_frac"] == af]
        if not tgt:
            continue
        pooled_null = [x for c in nulls for x in c["_divs"]]
        if pooled_null and _HAVE_SCIPY:
            p = float(mannwhitneyu(tgt["_divs"], pooled_null, alternative="two-sided").pvalue)
        elif pooled_null:
            p = _perm_u_p(tgt["_divs"], pooled_null)
        else:
            p = None
        v7_med, null_med = tgt["median"], (float(np.median(pooled_null)) if pooled_null else None)
        rows.append({
            "dose": af, "sign": sign, "n_v7": tgt["n"], "n_null_pooled": len(pooled_null),
            "V7_median_div": v7_med, "V7_mean_div": tgt["mean"],
            "Rband_median_div": null_med,
            "Rband_per_band_median": {c["vector"]: c["median"] for c in nulls},
            "mannwhitney_p": round(p, 5) if p is not None else None,
            "V7_advances_earlier": (null_med is not None and v7_med < null_med),
            "separates_beyond_noise": (p is not None and p < 0.05),
            "V7_specific_advance": (null_med is not None and v7_med < null_med and p is not None and p < 0.05),
        })
    return {"per_cell": {k: {kk: vv for kk, vv in v.items() if kk != "_divs"} for k, v in per_cell.items()},
            "dose_rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-run-dir", type=Path, required=True, help="vmb_b7f_3b (V7+/Rband+)")
    ap.add_argument("--neg-run-dir", type=Path, required=True, help="vmb_b7neg_3b (V7-/Rband-)")
    ap.add_argument("--target", default="V7")
    ap.add_argument("--null-prefixes", default="RBAND")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    null_prefixes = tuple(p.strip().upper() for p in args.null_prefixes.split(",") if p.strip())

    pos = _scan(args.pos_run_dir, "+", args.target, null_prefixes)
    neg = _scan(args.neg_run_dir, "-", args.target, null_prefixes)

    all_rows = pos.get("dose_rows", []) + neg.get("dose_rows", [])
    v7_specific = [r for r in all_rows if r["V7_specific_advance"]]
    verdict = {
        "prediction": "P=.40 V7-specific divergence-time ADVANCE separates from Rband beyond n-noise",
        "lean": "divergence-time is MAGNITUDE-GENERIC (V7 ~= Rband)",
        "cells_with_v7_specific_advance": [(r["sign"], r["dose"]) for r in v7_specific],
        "n_cells": len(all_rows), "n_v7_specific": len(v7_specific),
        "scipy_used": _HAVE_SCIPY,
    }
    out = {"model": "3b", "arm": "14n item-2b — divergence-time decomposition (magnitude gauge)",
           "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED -> outer loop",
           "law": ("first-word-index divergence vs seed-matched baseline twin (topic,mode,seed); "
                   "V7 vs pooled-Rband per (sign,dose), Mann-Whitney two-sided; ADVANCE = V7 median "
                   "< Rband median AND p<.05."),
           "verdict": verdict, "pos": pos, "neg": neg}
    args.out_json.write_text(json.dumps(out, indent=1))
    print("=== 14n DIVERGENCE-TIME ===")
    for r in all_rows:
        print(f"  {r['sign']}V7 a{r['dose']}: V7 median={r['V7_median_div']} "
              f"Rband median={r['Rband_median_div']} p={r['mannwhitney_p']} "
              f"advance={r['V7_advances_earlier']} sep={r['separates_beyond_noise']} "
              f"V7-specific={r['V7_specific_advance']}")
    print(f"VERDICT: V7-specific advances: {verdict['cells_with_v7_specific_advance']} "
          f"({verdict['n_v7_specific']}/{verdict['n_cells']}) | scipy={_HAVE_SCIPY}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
