"""ANNEX — roster members: TEXT-LEVEL (expression-grade) consequence readout, per the
PRICING-inshadow-roster certifying-consequence table and the S8-11 standing rule.

Marker machinery is IMPORTED VERBATIM from the banked 14n hedging instrument (HEDGE_RE /
DEF_RE) — its resample-group statistics don't fit n=40 single-seed cells, so the statistical
frame here is the cell-vs-references one every roster text read uses: per-cell rates vs the
pooled alpha=0 riders + the matched Rband envelope (max |null excess|). Declared adaptation,
not a new lexicon.

Per cell: hedge/1k · definitive/1k · net-hedge (H−D) · mean generated length · TTR ·
trigram-rep · n. References: rider pool (3 x _a0.0 cells) + Rband cells at matched dose.

Usage (from pipeline/):
    python -m anamnesis.scripts.annex_roster_text_readout \
        --roster-run ../outputs/battery/annex/vmb_roster_3b \
        --rider-run ../outputs/battery/vmb_a5_3b \
        --rband-run ../outputs/battery/vmb_b7_3b \
        --out ../outputs/battery/annex/roster_text_readout.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anamnesis.scripts.vmb_14n_hedging_index import DEF_RE, HEDGE_RE


def cell_stats(d: Path) -> dict | None:
    md_path = d / "metadata.json"
    if not md_path.exists():
        return None
    md = json.loads(md_path.read_text())
    gens = md["generations"] if "generations" in md else md
    h, dm, words, lens, ttrs, reps = 0, 0, 0, [], [], []
    for g in gens:
        t = g.get("generated_text", "")
        toks = t.split()
        if not toks:
            continue
        h += len(HEDGE_RE.findall(t))
        dm += len(DEF_RE.findall(t))
        words += len(toks)
        lens.append(g.get("num_generated_tokens", len(toks)))
        ttrs.append(len(set(toks)) / len(toks))
        tri = [" ".join(toks[i:i + 3]) for i in range(len(toks) - 2)]
        reps.append(1.0 - len(set(tri)) / max(len(tri), 1))
    if not lens:
        return None
    k = 1000.0 / max(words, 1)
    return {"n": len(lens), "hedge_per_1k": round(h * k, 3), "def_per_1k": round(dm * k, 3),
            "net_hedge": round((h - dm) * k, 3), "mean_len": round(float(np.mean(lens)), 1),
            "frac_at_cap": round(float(np.mean([l >= 512 for l in lens])), 3),
            "mean_ttr": round(float(np.mean(ttrs)), 4),
            "mean_trigram_rep": round(float(np.mean(reps)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster-run", type=Path, required=True)
    ap.add_argument("--rider-run", type=Path, required=True)
    ap.add_argument("--rband-run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = {}
    for d in sorted(args.roster_run.iterdir()):
        s = cell_stats(d)
        if s:
            rows[d.name] = s
    riders = [cell_stats(d) for d in sorted(args.rider_run.iterdir())
              if d.name.endswith("_a0.0")]
    riders = [r for r in riders if r]
    rband = {}
    for d in sorted(args.rband_run.iterdir()):
        if d.name.upper().startswith("RBAND"):
            s = cell_stats(d)
            if s:
                rband[d.name] = s

    def pool(key):
        return round(float(np.mean([r[key] for r in riders])), 3)

    out = {
        "provenance": "roster text-level consequence readout (S8-11 expression standard; "
                      "14n marker lexicon imported verbatim; cell-vs-rider-vs-Rband-envelope frame)",
        "rider_pool": {k: pool(k) for k in
                       ("hedge_per_1k", "def_per_1k", "net_hedge", "mean_len", "frac_at_cap",
                        "mean_ttr", "mean_trigram_rep")},
        "rider_n_cells": len(riders),
        "roster_cells": rows,
        "rband_cells": rband,
    }
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
