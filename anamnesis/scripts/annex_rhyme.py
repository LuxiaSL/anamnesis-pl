"""ANNEX — the per-partner rhyme check: is the natural-axes class ARCHITECTURE-level or WEIGHTS-level?

⚠ ANNEX LANE: nothing here is quotable until it graduates through a frozen prereg cell.
Ledger: outputs/battery/annex/ANNEX-LEDGER.md

The kotodama bank pools SIX partner weight-states (same architecture, different weights:
base, instruct, two soups, selfsim, synthloom). Under the per-model guardrail those are six
different models for signature purposes, so pooling raw-centered would make "which weights
wrote this" a dominant axis. Cell-centering removes partner by construction — every
(conv, turn) cell lies within one partner — which is what makes the pooled cell-centered
variant legitimate AT ALL.

This script tests that legitimacy instead of asserting it, and the same compute answers a
real question (outer loop, 2026-07-15):

  cell-centered spectra RHYME across partners  -> the natural-axes class is ARCHITECTURE-level
  cell-centered spectra DIVERGE across partners -> it is WEIGHTS-level

That is the program's per-model guardrail MEASURED rather than assumed, inside one dataset,
at zero marginal cost — and it echoes the M4 base-vs-instruct scoping lesson.

    PYTHONPATH=pipeline python -m anamnesis.scripts.annex_rhyme --k 6
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from anamnesis.scripts.annex_corpus import REPO, load_power, prepare
from anamnesis.scripts.annex_ranked_spectrum import rank_normal, winsorize
from anamnesis.scripts.annex_spectrum import apply_weighting, pca

OUT = REPO / "outputs/battery/annex"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="cell")
    ap.add_argument("--weighting", default="raw")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--transforms", default="raw",
                    help="comma list of feature bases: raw | rank | winsor. 'raw' alone "
                         "reproduces the session-1 rhyme byte-for-byte. Session 4 runs "
                         "raw,rank,winsor: the raw top-6 subspace is measured at cos 0.204 from "
                         "its own outlier-controlled counterpart, so the .747-.971 rhyme was "
                         "computed over axes that mostly do not survive a basis change.")
    args = ap.parse_args()
    transforms = [t.strip() for t in args.transforms.split(",") if t.strip()]

    full = load_power()
    levels = [n for n in json.loads(
        json.dumps([x for x in full.notes if x.startswith("partners")]))]
    # partner level names live in notes; recover them from the factor coding instead
    import re
    m = re.search(r"\[(.*?)\]", next(n for n in full.notes if n.startswith("partners")))
    names = [s.strip().strip("'\"") for s in m.group(1).split(",")]

    # ⚠ The transform is applied PER PARTNER, deliberately. `load_power(partner=nm)` already runs
    # robust_scale on that partner's rows ALONE, so the floor-z is per-partner; a GLOBAL rank would
    # encode partner identity into the ranks — precisely what cell-centering exists to remove, and
    # it would manufacture the agreement this rung is trying to measure.
    BUILD = {"raw": lambda X: X, "rank": rank_normal, "winsor": winsorize}
    specs: dict[str, dict] = {t: {} for t in transforms}
    meta: dict[str, dict] = {t: {} for t in transforms}
    for t in transforms:
        print(f"\n=== BASIS: {t} ===")
        for nm in names:
            c = load_power(partner=nm)
            c = c.model_copy(update={"X": np.ascontiguousarray(BUILD[t](c.X))})
            X, df = prepare(c, args.variant)
            Xw, _ = apply_weighting(X, c, args.weighting)
            sp = pca(Xw, df, k=args.k)
            specs[t][nm] = sp.components[:args.k].copy()
            meta[t][nm] = {"n": c.n, "df": df, "n_cells": int(len(np.unique(c.cell))),
                           "var_top": [round(float(v), 4) for v in sp.var_ratio[:args.k]],
                           "eff_rank_topk": round(sp.effective_rank_pr, 2)}
            print(f"  {nm:26s} n={c.n:6d} df={df:6d} cells={meta[t][nm]['n_cells']:5d} "
                  f"var_top3={meta[t][nm]['var_top'][:3]}", flush=True)
            del c, X, Xw, sp

    def sub_cos(A: np.ndarray, B: np.ndarray) -> float:
        """Mean principal cosine. Read the SUBSPACE, never PC identity — near-degenerate
        eigenvalues rotate freely (session 1's soup-3e4 trap: PC1-vs-PC1 |cos| .040 while the
        subspaces agreed at .907-.962)."""
        return float(np.mean(np.linalg.svd(A @ B.T, compute_uv=False)))

    floor = float(np.sqrt(args.k / 2252))
    all_rows: dict[str, dict] = {}
    for t in transforms:
        print(f"\n=== SUBSPACE RHYME [{t}]: mean principal cos between partners' top-{args.k} "
              f"cell-centered subspaces   (random floor ~ {floor:.3f}) ===")
        rows: dict[str, dict] = {}
        print("  " + " " * 26 + "".join(f"{n.split('-')[0][:8]:>9s}" for n in names))
        for a in names:
            line = f"  {a:26s}"
            rows[a] = {}
            for b in names:
                v = sub_cos(specs[t][a], specs[t][b])
                rows[a][b] = round(v, 3)
                line += f"{v:9.3f}"
            print(line)
        off = [rows[a][b] for a in names for b in names if a != b]
        print(f"  ⇒ off-diagonal: median {np.median(off):.3f}  min {min(off):.3f}  "
              f"max {max(off):.3f}   (floor {floor:.3f})")
        all_rows[t] = rows

    # ── ★ THE DECISIVE PER-PARTNER CHECK: does the basis change move EACH partner's axes? ──
    cross: dict[str, float] = {}
    if "raw" in transforms and len(transforms) > 1:
        print(f"\n=== ★ PER-PARTNER raw ↔ other-basis top-{args.k} subspace cos "
              f"(does the basis change move each partner's OWN axes?) ===")
        print(f"  {'partner':26s}" + "".join(f"{t:>10s}" for t in transforms if t != "raw"))
        for nm in names:
            line = f"  {nm:26s}"
            for t in transforms:
                if t == "raw":
                    continue
                v = sub_cos(specs["raw"][nm], specs[t][nm])
                cross[f"{nm}|raw_vs_{t}"] = round(v, 3)
                line += f"{v:10.3f}"
            print(line)

    print(f"\n=== PC1-vs-PC1 |cos| [raw basis] (the single dominant axis) ===")
    pc1 = {}
    base_t = "raw" if "raw" in transforms else transforms[0]
    for a in names:
        for b in names:
            if a < b:
                pc1[f"{a} | {b}"] = round(abs(float(specs[base_t][a][0] @ specs[base_t][b][0])), 3)
    for k_, v in sorted(pc1.items(), key=lambda x: -x[1]):
        print(f"  {v:5.3f}  {k_}")
    off = list(pc1.values())
    print(f"\n  PC1 agreement: median {np.median(off):.3f}  min {min(off):.3f}  max {max(off):.3f}")
    print(f"  random-direction floor in 2252-d ~ {np.sqrt(1/2252):.4f}")
    print("  ⚠ PC1-vs-PC1 is the TRAP, kept only so its disagreement with the subspace table "
          "stays visible. Do not read it as a result.")

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(transforms) if transforms != ["raw"] else args.weighting
    p = OUT / f"annex_rhyme_power_{args.variant}_{suffix}.json"
    summ = {t: {"median_offdiag": round(float(np.median(
                    [all_rows[t][a][b] for a in names for b in names if a != b])), 3),
                "min_offdiag": round(float(min(
                    [all_rows[t][a][b] for a in names for b in names if a != b])), 3),
                "max_offdiag": round(float(max(
                    [all_rows[t][a][b] for a in names for b in names if a != b])), 3)}
            for t in transforms}
    p.write_text(json.dumps({
        "ANNEX_RULE": "NOT QUOTABLE until graduated via a frozen prereg cell",
        "rung": "rhyme (per-partner gate + architecture-vs-weights) — MULTI-BASIS",
        "variant": args.variant, "weighting": args.weighting, "k": args.k,
        "transforms": transforms,
        "random_floor": round(floor, 4),
        "why_multi_basis": "session 1's rhyme (.747-.971 vs a .052 floor) ran on the RAW top-6 "
                           "cell-centered subspace. The rank basis has since measured the pooled "
                           "raw top-6 at cos 0.204 from its outlier-controlled counterpart, and "
                           "the raw top-2 are degenerate-row shards (kurt 11,861 / 3,281). So the "
                           "rhyme was computed over axes that mostly do not survive a basis "
                           "change. This re-runs it on rank + winsor.",
        "the_fake_rhyme_mechanism": "stated BEFORE running (frozen prediction): each partner has "
                                    "its OWN outlier rows, but a shard's direction is the "
                                    "direction of a few extreme rows inside the 144-feature "
                                    "COLLINEAR attn_res block — so every partner's shard lands in "
                                    "roughly the SAME low-dim subspace. If that is the mechanism, "
                                    "the rhyme measures 'all six weight-states have outliers in "
                                    "the same collinear block' (a FEATURE-CENSUS fact, true of any "
                                    "six partners of this architecture) and NOT 'all six share "
                                    "natural axes'.",
        "transform_scope": "applied PER PARTNER. load_power(partner=) already robust_scales that "
                           "partner's rows alone; a GLOBAL rank would encode partner identity into "
                           "the ranks and manufacture the agreement being measured.",
        "partners": meta,
        "subspace_mean_principal_cos": all_rows,
        "offdiag_summary": summ,
        "per_partner_raw_vs_basis": cross,
        "pc1_pairwise_abscos_RAW_BASIS_TRAP": pc1,
        "pc1_caveat": "PC1-vs-PC1 is the TRAP (session 1: soup-3e4 PC1 |cos| .040 vs subspace "
                      ".907-.962 — near-degenerate eigenvalues rotate freely). Banked only so its "
                      "disagreement with the subspace table stays visible. Not a result.",
        "reading": "rhyme => natural-axes class is ARCHITECTURE-level; diverge => WEIGHTS-level. "
                   "Also the gate on whether the pooled cell-centered variant is legitimate. "
                   "READ THE RANK ROW, not the raw row.",
    }, indent=1))
    print(f"\n  → {p}")


if __name__ == "__main__":
    main()
