"""Inward→outward bridge rank correlation across coordinates (session-11 Part A.2; 14q item 7).

Closes the A5 first-read §7b deferral: "Proper rank correlation needs more traits —
replication-era." The bridge prediction (A5-DESIGN-CONSTRAINTS §5): low-α state deformation
magnitude predicts the α at which the behavioral metric shifts. At k=4 it was qualitative
("rank-consistent at the extremes; V1-vs-V3 inversion within metric-comparability noise").
Post-14k the roster is 5 coordinates with banked rows: {V1 (formality; V1b = family member,
cos .988, no behavioral leg — sensitivity row only), V3 (mode/dir0), Ksoclin (soclin),
V_temp (temperature, data route), V7 (temperature, formula route)}.

⛔ ROSTER RULE (session-11 UPDATE block item 3): Vconf/Vrep have NO A5-inv rows — they are
NOT included and the correlation must not be extended to them this session.

Design, stated before computing:
- INWARD = whole-vector off-axis free-gen state deformation at α=.03, floor-z, vs pooled
  α=0 riders (the uniform gg/C§6 frame). Sourced from gg_partA_selectivity rows
  (off_target = target/selectivity; verified identical to the lever artifacts'
  total_deformation: V7 20.60↔20.63, Vtemp 8.87↔8.88). Ksoclin sourced from the 14k
  soclin lever (soclin_off_target; same floor-z matched-R convention, DIFFERENT removed
  axis + reference set — declared, not hidden; its cell sigs are node-side only).
  NOTE: §7b used MATCHED-TOKEN deformation; MT rows exist only for V1/V3 of this roster,
  so the free-gen variant is the common banked instrument (declared departure).
- OUTWARD = behavioral-threshold α: lowest α ∈ {.03,.1,.3} at which the coordinate's own
  behavioral/certifying instrument clears its null envelope, TEXT-LEVEL or judge or
  certified-consequence instruments only (α≤.1 injected-frac re-scope honored — no
  injected-sig frac is used anywhere here). Two readings where instruments disagree
  (the 14r named tension), correlation reported under BOTH:
  * strict: V1 .03 (2AFC .875) · V3 .3 (text-only frac .281; .1 reads floor .025) ·
    Ksoclin .3 (soc marker excess 4.2 ≈ 2× baseline) · Vtemp .1 (likelihood-AUC .807/.722
    vs .570/.541@.03; diversity 1.057/1.088) · V7 .03 (entropy +.039 vs 6 Rband nulls ≈0)
  * marker-lenient: V3 .1 (markers 4× baseline, n=160) · Ksoclin .1 (excess +29%; no
    banked marker null envelope — stated) · others unchanged.
- Spearman ρ + EXACT permutation p (5! = 120, two-sided) — descriptive; n=5; no filed P.
- Sensitivity rows: swap V1→V1b inward (no V1b behavioral leg exists, threshold held at
  V1's — labeled as such) · drop one of the temperature pair (V7/Vtemp same coordinate,
  two routes) · drop Ksoclin (the one convention-mixed inward row).

CPU-only, banked rows only. First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ARMS = Path(__file__).resolve()  # placeholder; real default set in main()


def exact_perm_p(x: list[float], y: list[float]) -> tuple[float, float]:
    """Spearman rho + exact two-sided permutation p over all n! label permutations."""
    rho = float(spearmanr(x, y).statistic)
    n = len(x)
    count = 0
    total = 0
    yy = np.asarray(y, dtype=float)
    for perm in itertools.permutations(range(n)):
        r = float(spearmanr(x, yy[list(perm)]).statistic)
        total += 1
        if abs(r) >= abs(rho) - 1e-12:
            count += 1
    return rho, count / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms-dir", type=Path, required=True, help="outputs/battery/arms")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    gg = json.loads((args.arms_dir / "A5/gg_partA_selectivity_3b.json").read_text())
    soc = json.loads((args.arms_dir / "A5/14k_soclin_lever_3b.json").read_text())

    def gg_off_target(vector: str, alpha: float) -> tuple[float, int]:
        for r in gg["rows"]:
            if r["vector"] == vector and r["alpha_frac"] == alpha:
                sel = r["selectivity_effect_per_offtarget"]
                if sel == 0:
                    raise SystemExit(f"zero selectivity for {vector}@{alpha}")
                return r["efficacy_raw_target_movement"] / sel, int(r["n"])
        raise SystemExit(f"no gg row for {vector}@{alpha}")

    def soclin_off_target(alpha: float) -> float:
        for r in soc["rows"]:
            if r["vector"] == "Ksoclin" and r["alpha_frac"] == alpha:
                return float(r["soclin_off_target"])
        raise SystemExit(f"no soclin row @{alpha}")

    LOW_ALPHA = 0.03
    inward: dict[str, dict] = {}
    for v in ("V1", "V1b", "V3", "V7", "Vtemp"):
        off, n = gg_off_target(v, LOW_ALPHA)
        inward[v] = {"off_axis_deformation_a003": round(off, 3), "n": n,
                     "source": "gg_partA_selectivity_3b.json (uniform C§6 frame)"}
    inward["Ksoclin"] = {"off_axis_deformation_a003": round(soclin_off_target(LOW_ALPHA), 3),
                         "n": 160,
                         "source": "14k_soclin_lever_3b.json (soclin axis removed; own matched-R "
                                   "reference — convention-mixed row, declared)"}

    # behavioral thresholds: banked-instrument citations in-line (per-row provenance)
    thresholds = {
        "strict": {
            "V1": (0.03, "2AFC judge .875@.03 (A5 first-read §7; arms/A5/judge/)"),
            "V3": (0.3, "text-only frac .013/.025/.281 (annex 14r_v3_textonly_gg.json); "
                        ".1 reads at the injected-null floor once injection leaves the read"),
            "Ksoclin": (0.3, "socratic_marker_excess 4.2@.3 ≈ 2× baseline 4.22/1k "
                             "(14k_soclin_lever_3b.json)"),
            "Vtemp": (0.1, "likelihood-AUC .807/.722@.1 vs .570/.541@.03 (<.65 = weak per 14f); "
                           "resample-diversity 1.057/1.088@.1 (arms/C2/c3_certifying_*)"),
            "V7": (0.03, "entropy rise +.039@.03 vs 6 Rband nulls ≈0, certified both doses "
                         "(14j_leg2_entropy_V7_3b.json, Pj2)"),
        },
        "marker_lenient": {
            "V1": (0.03, "unchanged"),
            "V3": (0.1, "analogy markers 1.07/1k = 4× baseline .26 @.1, n=160 (14r §1; the named "
                        "low-dose marker-vs-frac tension read on the marker side)"),
            "Ksoclin": (0.1, "marker excess +1.24 = +29% over baseline @.1 — NO banked marker "
                             "null envelope; lenient by construction (stated)"),
            "Vtemp": (0.1, "unchanged"),
            "V7": (0.03, "unchanged"),
        },
    }

    ROSTER = ["V1", "V3", "Ksoclin", "Vtemp", "V7"]

    def corr(members: list[str], reading: str, inward_map: dict[str, str] | None = None) -> dict:
        imap = inward_map or {}
        x = [inward[imap.get(m, m)]["off_axis_deformation_a003"] for m in members]
        y = [thresholds[reading][m][0] for m in members]
        rho, p = exact_perm_p(x, y)
        return {"members": members, "reading": reading,
                "inward_swaps": imap or None,
                "deformation_a003": x, "threshold_alpha": y,
                "spearman_rho": round(rho, 4), "exact_perm_p_two_sided": round(p, 4)}

    results = {
        "primary_strict": corr(ROSTER, "strict"),
        "primary_marker_lenient": corr(ROSTER, "marker_lenient"),
        "sensitivity": [
            corr(ROSTER, "strict", inward_map={"V1": "V1b"}) |
            {"note": "V1b inward row swapped in; threshold held at V1's (V1b has no behavioral "
                     "leg — 2AFC deferred); family-member sensitivity only"},
            corr([m for m in ROSTER if m != "V7"], "strict") |
            {"note": "temperature pair reduced to data route (V7 dropped)"},
            corr([m for m in ROSTER if m != "Vtemp"], "strict") |
            {"note": "temperature pair reduced to formula route (Vtemp dropped)"},
            corr([m for m in ROSTER if m != "Ksoclin"], "strict") |
            {"note": "convention-mixed inward row dropped (gg-frame-only roster)"},
        ],
    }

    out = {
        "arm": "inward→outward bridge rank correlation (A5 §7b deferral closed; 14q item 7)",
        "STATUS": "FIRST_READ_PENDING (C§8)",
        "model": "3b",
        "law": "Spearman(off-axis free-gen deformation @α=.03 floor-z, behavioral-threshold α), "
               "exact permutation p (5!=120, two-sided); text-level/judge/certified instruments "
               "only for thresholds (α≤.1 injected-frac re-scope honored); DESCRIPTIVE, no filed P",
        "roster_rule": "Vconf/Vrep EXCLUDED per session-11 UPDATE block item 3 (no A5-inv rows); "
                       "do not extend without their rows",
        "bridge_prediction_direction": "negative rho (more low-α state movement → behavioral "
                                       "effect at lower α) — A5-DESIGN-CONSTRAINTS §5",
        "inward_rows": inward,
        "threshold_citations": {k: {m: v[1] for m, v in d.items()} for k, d in thresholds.items()},
        "results": results,
        "scope_notes": [
            "n=5 coordinates; exact-p floor at n=5 is 2/120 ≈ .0167 — treat magnitudes as "
            "descriptive ordering evidence, not certification",
            "inward instrument is free-gen (§7b used matched-token; MT rows absent for 3/5) — "
            "declared departure",
            "V7/Vtemp = one coordinate, two routes (roster adjudication); both kept because the "
            "bridge question is about LEVERS; single-route sensitivity rows included",
            "V7 rows are n=40 (others n=160)",
        ],
    }
    args.out_json.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: results[k] for k in ("primary_strict", "primary_marker_lenient")}, indent=1))
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
