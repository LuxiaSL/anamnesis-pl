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
from anamnesis.scripts.annex_spectrum import apply_weighting, pca

OUT = REPO / "outputs/battery/annex"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="cell")
    ap.add_argument("--weighting", default="raw")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    full = load_power()
    levels = [n for n in json.loads(
        json.dumps([x for x in full.notes if x.startswith("partners")]))]
    # partner level names live in notes; recover them from the factor coding instead
    import re
    m = re.search(r"\[(.*?)\]", next(n for n in full.notes if n.startswith("partners")))
    names = [s.strip().strip("'\"") for s in m.group(1).split(",")]

    specs, meta = {}, {}
    for i, nm in enumerate(names):
        c = load_power(partner=nm)
        X, df = prepare(c, args.variant)
        Xw, _ = apply_weighting(X, c, args.weighting)
        sp = pca(Xw, df, k=args.k)
        specs[nm] = sp
        meta[nm] = {"n": c.n, "df": df, "n_cells": int(len(np.unique(c.cell))),
                    "var_top": [round(float(v), 4) for v in sp.var_ratio[:args.k]],
                    "eff_rank_topk": round(sp.effective_rank_pr, 2)}
        print(f"  {nm:26s} n={c.n:6d} df={df:6d} cells={meta[nm]['n_cells']:5d} "
              f"var_top3={meta[nm]['var_top'][:3]}", flush=True)

    print(f"\n=== SUBSPACE RHYME: |cos| principal angles between partners' top-{args.k} "
          f"cell-centered subspaces ===")
    print("  (1.0 = identical subspace; a random {k}-dim subspace pair in 2252-d ~ "
          f"{np.sqrt(args.k/2252):.3f})")
    rows = {}
    hdr = "  " + " " * 26 + "".join(f"{n.split('-')[0][:8]:>9s}" for n in names)
    print(hdr)
    for a in names:
        line = f"  {a:26s}"
        rows[a] = {}
        for b in names:
            # mean of the principal-angle cosines between the two top-k subspaces
            M = specs[a].components @ specs[b].components.T
            sv = np.linalg.svd(M, compute_uv=False)
            v = float(np.mean(sv))
            rows[a][b] = round(v, 3)
            line += f"{v:9.3f}"
        print(line)

    print(f"\n=== PC1-vs-PC1 |cos| (the single dominant axis) ===")
    pc1 = {}
    for a in names:
        for b in names:
            if a < b:
                v = abs(float(specs[a].components[0] @ specs[b].components[0]))
                pc1[f"{a} | {b}"] = round(v, 3)
    for k_, v in sorted(pc1.items(), key=lambda x: -x[1]):
        print(f"  {v:5.3f}  {k_}")
    off = [v for v in pc1.values()]
    print(f"\n  PC1 agreement: median {np.median(off):.3f}  min {min(off):.3f}  max {max(off):.3f}")
    print(f"  random-direction floor in 2252-d ~ {np.sqrt(1/2252):.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"annex_rhyme_power_{args.variant}_{args.weighting}.json"
    p.write_text(json.dumps({
        "ANNEX_RULE": "NOT QUOTABLE until graduated via a frozen prereg cell",
        "rung": "rhyme (per-partner gate + architecture-vs-weights)",
        "variant": args.variant, "weighting": args.weighting, "k": args.k,
        "partners": meta, "subspace_mean_principal_cos": rows, "pc1_pairwise_abscos": pc1,
        "reading": "rhyme => natural-axes class is ARCHITECTURE-level; diverge => WEIGHTS-level. "
                   "Also the gate on whether the pooled cell-centered variant is legitimate.",
    }, indent=1))
    print(f"\n  → {p}")


if __name__ == "__main__":
    main()
