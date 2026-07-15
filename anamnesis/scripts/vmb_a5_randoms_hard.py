"""§4.2 randoms-pushed-hard (session-4 A6): does the past-collapse basin capture hard-pushed
RANDOM vectors? If pushing a random direction to α∈{2,3} produces the SAME text-degeneracy
signature as pushing the mode/identity vectors past coherence (V3/V1 @ α=1.0), the collapse
basin is a generic high-dose attractor — not mode-specific. Text-derived readout (TTR,
trigram-rep, length, marker/field rates) over the banked free-gen metadata. Reuses text_rates
(markers+field) and text_stats (len/ttr/rep) verbatim. CPU-only. First-read → outer loop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anamnesis.scripts.vmb_a5_frozen_directional import text_rates
from anamnesis.scripts.vmb_arm_a5_analyze import text_stats

# label → cell-dir name (handles the V3 doubled-tag naming)
CELLS = {
    "coherent_R1_a0.3": "R1_L14_a0.3",
    "collapse_R1_a1.0": "R1_L14_a1.0",
    "collapse_V1_a1.0": "V1_L14_a1.0",
    "collapse_V3_a1.0": "V3_L14_L14_a1.0",
    "hard_R1_a2.0": "R1_L14_a2.0", "hard_R2_a2.0": "R2_L14_a2.0", "hard_R3_a2.0": "R3_L14_a2.0",
    "hard_R1_a3.0": "R1_L14_a3.0", "hard_R2_a3.0": "R2_L14_a3.0", "hard_R3_a3.0": "R3_L14_a3.0",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a5-root", type=Path, required=True, help="vmb_a5_3b run dir")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--model", default="3b")
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, cd in CELLS.items():
        d = args.a5_root / cd
        mp = d / "metadata.json"
        if not mp.exists():
            continue
        ts, tr = text_stats(mp), text_rates(mp)
        rows.append({"label": label, "cell": cd, "class": label.split("_")[0],
                     "n": ts["n"], "mean_len": round(ts["mean_len"], 1),
                     "mean_ttr": round(ts["mean_ttr"], 3),
                     "mean_trigram_rep": round(ts["mean_trigram_rep"], 3),
                     "markers_per_1k": round(tr["markers_per_1k"], 2),
                     "field_mass_per_1k": round(tr["field_mass_per_1k"], 2)})

    def _mean(cls, key):
        v = [r[key] for r in rows if r["class"] == cls]
        return float(np.mean(v)) if v else None

    # basin verdict: does hard-R degeneracy match the α=1.0 collapse anchors, away from coherent?
    verdict = {}
    for key in ("mean_len", "mean_ttr", "mean_trigram_rep", "markers_per_1k", "field_mass_per_1k"):
        verdict[key] = {"coherent": _mean("coherent", key), "collapse_a1.0": _mean("collapse", key),
                        "hard_randoms": _mean("hard", key)}
    # captured = hard randoms land on the collapse side (rep up / ttr or len shifted, markers→0)
    hr, co, cl = verdict["mean_trigram_rep"]["hard_randoms"], verdict["mean_trigram_rep"]["coherent"], verdict["mean_trigram_rep"]["collapse_a1.0"]
    captured = bool(hr is not None and cl is not None and co is not None
                    and abs(hr - cl) < abs(hr - co))

    out = {"model": args.model, "arm": "A5-§4.2-randoms-hard",
           "STATUS": "FIRST_READ_PENDING (C§8 — no stamps ship before outer-loop read)",
           "law": "text-derived degeneracy (TTR/trigram-rep/len) + marker/field rates; "
                  "hard R{1,2,3}@{2,3} vs collapse anchors {V3,V1,R1}@1.0 vs coherent R1@0.3",
           "cells": rows, "basin_comparison": verdict,
           "hard_randoms_in_collapse_basin_by_trigram_rep": captured}
    args.out_json.write_text(json.dumps(out, indent=1))
    print(json.dumps({"rows": [(r["label"], r["mean_len"], r["mean_ttr"], r["mean_trigram_rep"],
                                r["markers_per_1k"], r["field_mass_per_1k"]) for r in rows],
                      "captured_by_trigram_rep": captured}, indent=1))


if __name__ == "__main__":
    main()
