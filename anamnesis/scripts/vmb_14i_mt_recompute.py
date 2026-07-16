"""14i ITEM 1 — MT headline recompute minus chosen_rank (verification leg; P=0.90).

ADDENDUM 14i: `mean_chosen_rank`/`std_chosen_rank` (feat 211/212) measure the FORCING in
matched-token replay (chosen_rank[i] = (x > x[cid]).sum(), state_extractor.py:191), not the
injection→computation map — so they are EXCLUDED from any whole-vector ‖Δs‖ magnitude on MT
cells. This leg recomputes the banked headline MT `ratio_seed_floor` (whole_vector) with and
without the 2 features and confirms no low-α (12b-bar) headline moves materially.

Reproduces the MT block of vmb_arm_a5_analyze.py exactly (median mean|Δz| over the cell mask
÷ stage0 seed-floor), whole_vector only, twice: full mask vs mask-minus-{211,212}.
CPU-only, banked MT sigs. First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from anamnesis.analysis.battery.deltas import ConditionCorpus, load_floor_scale, within_condition_deltas
from anamnesis.analysis.battery.floors import build_cells, load_signature_matrix
from anamnesis.analysis.battery.manifest import MODEL_META

VISIBILITY = 0.1  # 12b bar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--model", default="3b")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    mm = MODEL_META[args.model]

    stage0 = args.battery_root / mm.stage0_dir
    med, scale = load_floor_scale(stage0 / "signatures_v3")
    s0 = ConditionCorpus(stage0 / "signatures_v3", stage0 / "metadata.json", med, scale,
                         f"{args.model}-stage0")
    names = s0.feature_names
    cells = build_cells(names, mm.n_layers)
    wv = cells["whole_vector"].copy()                    # all-True bool mask
    cr_idx = [i for i, n in enumerate(names) if n in ("mean_chosen_rank", "std_chosen_rank")]
    assert len(cr_idx) == 2, f"expected 2 chosen_rank feats, got {cr_idx}"
    wv_minus = wv.copy()
    wv_minus[cr_idx] = False                              # drop the 2 forcing features

    # seed-floor for whole_vector (same as analyze; unaffected by the 2 feats — kept identical
    # so the ONLY change is the MT numerator, isolating the effect)
    sf = {c: max(float(np.median(v)), 1e-12)
          for c, v in within_condition_deltas(s0, {"whole_vector": wv}).items()}["whole_vector"]

    s0X, s0names, s0gids = load_signature_matrix(stage0 / "signatures_v3")
    s0Z = (s0X - med) / scale
    s0map = {g: i for i, g in enumerate(s0gids)}

    mt_root = args.battery_root / f"vmb_a5_mt_{args.model}"
    rows = []
    for d in sorted(mt_root.iterdir()):
        sd = d / "signatures_v3"
        if not sd.exists():
            continue
        X, nms, gids_ = load_signature_matrix(sd)
        if list(nms) != list(names):
            continue
        Z = (X - med) / scale
        m_ = re.match(r"^(?P<key>.+)_a(?P<a>[\d.]+)$", d.name)
        key, af = m_.group("key"), float(m_.group("a"))
        vec = key.split("_")[0]
        D = np.stack([Z[i] - s0Z[s0map[g]] for i, g in enumerate(gids_) if g in s0map])
        r_full = float(np.median(np.abs(D[:, wv]).mean(axis=1)) / sf)
        r_minus = float(np.median(np.abs(D[:, wv_minus]).mean(axis=1)) / sf)
        rows.append({"cell": d.name, "vector": vec, "alpha_frac": af, "n": int(D.shape[0]),
                     "ratio_seed_floor_full": round(r_full, 5),
                     "ratio_seed_floor_minus_chosenrank": round(r_minus, 5),
                     "abs_delta": round(r_minus - r_full, 5),
                     "rel_change_pct": round(100.0 * (r_minus - r_full) / max(r_full, 1e-9), 3),
                     "visible_012b_full": bool(r_full >= VISIBILITY),
                     "visible_012b_minus": bool(r_minus >= VISIBILITY),
                     "visibility_flips": bool((r_full >= VISIBILITY) != (r_minus >= VISIBILITY)),
                     "is_null": vec.upper().startswith("R")})

    # R-relative (whole-vector headline): trait ratio ÷ mean R ratio, per alpha — BOTH variants
    def r_mean(af, field):
        v = [r[field] for r in rows if r["is_null"] and r["alpha_frac"] == af]
        return float(np.mean(v)) if v else None
    for r in rows:
        if not r["is_null"]:
            for field, out in (("ratio_seed_floor_full", "ratio_over_r_full"),
                               ("ratio_seed_floor_minus_chosenrank", "ratio_over_r_minus")):
                base = r_mean(r["alpha_frac"], field)
                r[out] = round(float(r[field] / max(base, 1e-9)), 4) if base else None

    # verdict: material low-α headline change? "low-α" = 12b-relevant doses ≤ .1; "material" =
    # any visibility flip OR |rel change| ≥ 5% on a non-null whole-vector cell
    low = [r for r in rows if r["alpha_frac"] <= 0.1 and not r["is_null"]]
    flips = [r["cell"] for r in low if r["visibility_flips"]]
    big = [(r["cell"], r["rel_change_pct"]) for r in low if abs(r["rel_change_pct"]) >= 5.0]
    inside = (not flips) and (not big)
    max_abs_rel = max((abs(r["rel_change_pct"]) for r in low), default=0.0)

    out = {"arm": "14i item 1 — MT headline recompute minus chosen_rank (whole_vector)",
           "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop",
           "model": args.model,
           "excluded_features": {211: "mean_chosen_rank", 212: "std_chosen_rank"},
           "law": "median(mean|Δz| over mask)/stage0-seed-floor; whole_vector mask ∓ {211,212}; "
                  "seed-floor held identical (isolates numerator); R-relative per alpha",
           "P_filed": 0.90,
           "verdict": {"test": "no low-α (≤.1, 12b-bar) whole-vector headline moves materially "
                               "(no visibility flip; |rel change|<5%)",
                       "result": "INSIDE" if inside else "MISS",
                       "max_abs_rel_change_pct_low_alpha": round(max_abs_rel, 3),
                       "visibility_flips_low_alpha": flips,
                       "material_changes_low_alpha": big},
           "rows": rows}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["verdict"], indent=1))
    for r in sorted((x for x in rows if not x["is_null"]), key=lambda x: (x["vector"], x["alpha_frac"])):
        print(f"  {r['cell']:16} full={r['ratio_seed_floor_full']:.4f} "
              f"minus={r['ratio_seed_floor_minus_chosenrank']:.4f} Δrel={r['rel_change_pct']:+.2f}%")


if __name__ == "__main__":
    main()
