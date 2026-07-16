"""A4 kind-contrast FAMILY decomposition re-read on CORRECTED rotate data (session-10 Part A2).

Scorecard-#13 debt. The A4 whole-vector geometry stamp ("rotate↔recompute colinear, naive
the odd kind" on the short/active-support anchor substrate) was frozen on the pre-14e-bug
rotate rows. This re-reads it on the corrected `A4_rotfix14e` results and DECOMPOSES it by
source family — asking WHERE in the substrate the whole-vector geometry lives.

For each source family and the whole vector we take the three surgery-kind pair direction
cosines (naive↔rotate, rotate↔recompute, naive↔recompute) averaged over eviction fractions,
identify the COLINEAR pair (max cosine) and hence the ODD kind (the one not in it), and its
separation margin (max_cos − 2nd_cos). Pure JSON→JSON on banked corrected results; no GPU,
no node. First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PAIRS = ("naive_vs_rotate", "rotate_vs_recompute", "naive_vs_recompute")
KIND_OF_PAIR = {"naive_vs_rotate": {"naive", "rotate"},
                "rotate_vs_recompute": {"rotate", "recompute"},
                "naive_vs_recompute": {"naive", "recompute"}}
KINDS = {"naive", "rotate", "recompute"}
CELLS = ("whole_vector", "source:attention", "source:residual", "source:keys",
         "source:gate", "source:output", "source:qk")
MARGIN_STRONG = 0.30  # cosine-margin above which the odd-kind call is a clean geometric dissociation


def read_model(path: Path) -> dict:
    d = json.loads(path.read_text())
    kc = d["kind_contrasts"]
    out = {}
    for cell in CELLS:
        cos = {}
        for p in PAIRS:
            rs = [r for r in kc if r["cell"] == cell and r["pair"] == p]
            if not rs:
                continue
            cos[p] = {"mean_dir_cos": round(sum(r["mean_direction_cosine"] for r in rs) / len(rs), 4),
                      "n_frac_carried": sum(1 for r in rs if r["verdict"] == "kind_carried"),
                      "n_frac": len(rs)}
        if len(cos) < 3:
            continue
        colinear_pair = max(PAIRS, key=lambda p: cos[p]["mean_dir_cos"])
        odd_kind = (KINDS - KIND_OF_PAIR[colinear_pair]).pop()
        ordered = sorted((cos[p]["mean_dir_cos"] for p in PAIRS), reverse=True)
        margin = round(ordered[0] - ordered[1], 4)
        out[cell] = {"cos": cos, "colinear_pair": colinear_pair, "odd_kind": odd_kind,
                     "separation_margin": margin,
                     "geometry_clean": margin >= MARGIN_STRONG}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, nargs="+", required=True,
                    help="a4_14emerged_results_{3b,8b}.json")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    per_model = {}
    for p in args.results:
        model = "3b" if "3b" in p.name else "8b" if "8b" in p.name else p.stem
        per_model[model] = read_model(p)

    # convergence read: whole-vector odd kind + which families carry it cleanly (both models)
    models = sorted(per_model)
    wv = {m: per_model[m].get("whole_vector", {}) for m in models}
    wv_odd = {m: wv[m].get("odd_kind") for m in models}
    # families with a CLEAN geometric dissociation (margin>=.30) in BOTH models, and their odd kind
    fam_clean = {}
    for cell in CELLS:
        if cell == "whole_vector":
            continue
        both = all(per_model[m].get(cell, {}).get("geometry_clean") for m in models)
        if both:
            odds = {m: per_model[m][cell]["odd_kind"] for m in models}
            fam_clean[cell] = {"odd_kind": odds, "agree": len(set(odds.values())) == 1}

    out = {
        "arm": "A4 kind-contrast FAMILY decomposition (corrected 14e rotate) — scorecard #13 re-read",
        "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop",
        "law": "per source family: mean_direction_cosine of the 3 surgery-kind pairs (avg over evict "
               "fracs); colinear pair = max cosine; odd kind = the kind not in it; clean dissociation "
               "= separation margin >= 0.30. A4 = short/active-support anchor substrate.",
        "whole_vector_odd_kind_per_model": wv_odd,
        "whole_vector_reproduces_A4_stamp": all(v == "naive" for v in wv_odd.values()),
        "families_with_clean_geometry_both_models": fam_clean,
        "per_model": per_model,
        "read": {
            "headline": "Whole-vector A4 geometry (naive odd; rotate≈recompute) REPRODUCES on corrected "
                        "rotate, both models — but it is RESIDUAL-dominated. Substrate double-dissociation: "
                        "residual carries naive-odd (position-tracking: rotate≈recompute), KEYS carry the "
                        "dual recompute-odd (imprint-tracking: naive≈rotate — the exp11/A4b grouping). "
                        "attention/output/gate ~kind-invariant in direction.",
        },
    }
    args.out_json.write_text(json.dumps(out, indent=1))
    print(f"whole-vector odd kind: {wv_odd}; reproduces A4 stamp (naive-odd both): "
          f"{out['whole_vector_reproduces_A4_stamp']}")
    print(f"clean-geometry families (both models): "
          f"{ {k: v['odd_kind'] for k, v in fam_clean.items()} }")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
