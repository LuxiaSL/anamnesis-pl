"""A2-ii length-matched neutral-prefix control analyzer (P5; pre-declared 2026-07-14).

Un-embargoes (or withdraws) the A2 unexecuted-instruction carriage-localization claim by
ruling out prompt LENGTH. Compares, per feature_map cell, the SWAP-prefix residue against the
length-matched NEUTRAL-prefix residue (both = matched-token seed-floor deltas from the native
replay), paired per continuation, own-tail Wilcoxon + BH. Verdict classes are FROZEN in
`research/planning/A2ii-control-declaration-2026-07-14.md` (12d). First-read → outer loop.

Inputs (all share the vmb_a2_prefix_replay index schema: list of {sig, swap, source_gen_id,
native_sig_dir, native_sig}):
  swap    = <a2-root>/vmb_a2_<model>_prefix_swap   (banked)
  neutral = <a2-root>/vmb_a2_<model>_neutral_prefix (GPU window-work; length-matched neutral)

    PYTHONPATH=pipeline python -m anamnesis.scripts.vmb_a2ii_control \
        --model 3b --a2-root outputs/battery --out-dir outputs/battery/arms/A2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.stats import wilcoxon

from anamnesis.analysis.battery.deltas import load_floor_scale
from anamnesis.analysis.battery.floors import build_cells, load_signature_matrix
from anamnesis.analysis.battery.manifest import MODEL_META

CONFIRMATORY = ["whole_vector", "source:attention", "source_band:attention|early",
                "source_band:attention|mid", "source_band:attention|late",
                "source:residual", "source:gate", "source:keys", "source:output"]
FLAGGED = ["source:attention", "source_band:attention|early",
           "source_band:attention|mid", "source_band:attention|late"]
F32 = NDArray[np.float32]


def _bh(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    sig, thresh = {}, 0.0
    for i, (k, p) in enumerate(items, 1):
        if p <= (i / m) * alpha:
            thresh = (i / m) * alpha
    return {k: (p <= thresh) for k, p in pvals.items()}


class _SigCache:
    def __init__(self, med: F32, scale: F32):
        self.med, self.scale = med, scale
        self._by_dir: dict[str, dict[int, F32]] = {}
        self.names: list[str] | None = None

    def z(self, sig_dir: str, sig_name: str) -> F32:
        if sig_dir not in self._by_dir:
            X, names, gids = load_signature_matrix(Path(sig_dir))
            self.names = names
            self._by_dir[sig_dir] = {g: (X[r] - self.med) / self.scale
                                     for r, g in enumerate(gids)}
        gid = int(sig_name.split("_")[1])
        return self._by_dir[sig_dir][gid]


def _load_variant(root: Path, model: str, kind: str, cache: _SigCache
                  ) -> dict[tuple[str, int], tuple[F32, F32]]:
    """(swap_label, source_gen_id) -> (variant_z, native_z)."""
    d = root / f"vmb_a2_{model}_{kind}"
    index = json.loads((d / "prefix_swap_index.json").read_text())
    sig_dir = str(d / "signatures_v3")
    out: dict[tuple[str, int], tuple[F32, F32]] = {}
    for e in index:
        var_z = cache.z(sig_dir, e["sig"])
        nat_z = cache.z(e["native_sig_dir"], e["native_sig"])
        out[(e["swap"], int(e["source_gen_id"]))] = (var_z, nat_z)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_META.keys()))
    ap.add_argument("--a2-root", type=Path, required=True)
    ap.add_argument("--battery-root", type=Path, default=Path("outputs/battery"),
                    help="holds <stage0_dir>/signatures_v3 (the floor); node1 = /models/anamnesis-extract/battery")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    meta = MODEL_META[args.model]
    med, scale = load_floor_scale(
        args.battery_root / meta.stage0_dir / "signatures_v3")
    cache = _SigCache(med, scale)

    swap = _load_variant(args.a2_root, args.model, "prefix_swap", cache)
    neutral = _load_variant(args.a2_root, args.model, "neutral_prefix", cache)
    keys = sorted(set(swap) & set(neutral))
    if not keys:
        raise SystemExit("no matched (swap, source_gen_id) between prefix_swap and "
                         "neutral_prefix — run the length-matched neutral replay first")
    cells = build_cells(cache.names, meta.n_layers)

    # per-continuation per-cell paired residues (floor-z seed ruler)
    per_cell: dict[str, list[tuple[float, float]]] = {c: [] for c in cells}
    for k in keys:
        sv, snat = swap[k]
        nv, nnat = neutral[k]
        sd = np.abs(sv - snat)          # swap residue per feature
        nd = np.abs(nv - nnat)          # neutral residue per feature
        for c, mask in cells.items():
            per_cell[c].append((float(sd[mask].mean()), float(nd[mask].mean())))

    results: dict[str, dict] = {}
    pvals: dict[str, float] = {}
    for c in CONFIRMATORY:
        if c not in per_cell:
            continue
        pairs = np.array(per_cell[c])                    # [n, 2]
        s, n = pairs[:, 0], pairs[:, 1]
        diff = s - n
        try:
            _, p = wilcoxon(diff, alternative="greater")  # one-sided swap > neutral
        except ValueError:
            p = 1.0                                       # all-zero diffs
        pvals[c] = float(p)
        results[c] = {"n": len(pairs), "swap_residue_med": round(float(np.median(s)), 5),
                      "neutral_residue_med": round(float(np.median(n)), 5),
                      "effect_med_swap_minus_neutral": round(float(np.median(diff)), 5),
                      "p_swap_gt_neutral": round(float(p), 5)}
    sig = _bh(pvals, args.alpha)
    for c in results:
        results[c]["bh_significant"] = bool(sig.get(c, False))
        results[c]["flagged_family"] = c in FLAGGED

    passed_flagged = [c for c in FLAGGED if results.get(c, {}).get("bh_significant")
                      and results[c]["effect_med_swap_minus_neutral"] > 0]
    if passed_flagged:
        verdict = "UN-EMBARGO"
    elif any(results.get(c, {}).get("bh_significant") for c in CONFIRMATORY):
        verdict = "INDETERMINATE"          # some cell passes but not a flagged carriage cell
    else:
        verdict = "STAYS-EMBARGOED"

    out = {
        "arm": "A2_ii_control", "model": args.model,
        "prereg": "A2ii-control-declaration-2026-07-14.md (12d, frozen before data). "
                  "Swap-residue vs length-matched neutral-prefix residue, paired per "
                  "continuation, matched-token seed-floor ruler (floor cancels in the "
                  "comparison), own-tail Wilcoxon + BH across the confirmatory cells.",
        "n_continuations": len(keys), "alpha": args.alpha,
        "verdict": verdict,
        "passed_flagged_cells": passed_flagged,
        "cells": results,
        "verdict_note": {
            "UN-EMBARGO": "swap EXCEEDS neutral on a flagged attention cell (BH-sig, +effect) "
                          "→ carriage is instruction-content, not length; localization claim "
                          "un-embargoes, scoped to the passing cells.",
            "STAYS-EMBARGOED": "swap ≈ neutral on the flagged cells → the 0.16–0.30× residue "
                               "is a prefix-length artifact; carriage-localization withdrawn.",
            "INDETERMINATE": "significance off the flagged cells only → report per-cell, no "
                             "blanket un-embargo.",
        }[verdict],
        "law": {"n": len(keys), "M": args.model,
                "law": "matched-token seed-floor residues (12b ruler), paired swap-vs-neutral "
                       "Wilcoxon one-sided + BH; verdict classes frozen 12d (declaration doc)",
                "floor_type": "seed-floor(replay)"},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    p = args.out_dir / f"a2ii_control_{args.model}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"[{args.model}] A2-ii control: n={len(keys)}  VERDICT={verdict}")
    for c in CONFIRMATORY:
        if c in results:
            r = results[c]
            print(f"  {c:28s} swap {r['swap_residue_med']:.4f} vs neutral "
                  f"{r['neutral_residue_med']:.4f}  Δ{r['effect_med_swap_minus_neutral']:+.4f} "
                  f"p={r['p_swap_gt_neutral']:.4f} {'SIG' if r['bh_significant'] else ''}")
    print(f"  → {p}")


if __name__ == "__main__":
    main()
