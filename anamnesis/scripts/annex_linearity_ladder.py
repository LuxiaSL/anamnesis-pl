"""ANNEX A6′ — THE DOSE LADDER: does linearity break because of MAGNITUDE, or because of V4?

⚠ ANNEX LANE: nothing here is quotable until it graduates through a frozen prereg cell.
Ledger: outputs/battery/annex/ANNEX-LEDGER.md

A6 (2 doses, α .03→.1) found deviation-from-linearity tracks RESPONSE MAGNITUDE rather than vector
identity (spearman +.86/+.93, surviving V4's removal). But magnitude and identity were CONFOUNDED
BY CONSTRUCTION: no banked vector reached V4's response range, so no magnitude-matched contrast
existed. This rung breaks that confound with the ladder the on-policy sweep authorised:

    V1/V2/V3/R1/R2/R3 — α ∈ {.03, .1, .15, .2, .3, .5}   (6 doses, 16.7x range, ALL on-policy)
    V4                — α ∈ {.03, .1, .15}                (on-policy ENDS at .15: agreement .834
                                                           at .2, .101 at 1.0 — beyond that,
                                                           matched-token deltas measure an
                                                           incoherent counterfactual and were
                                                           NOT run. That is the gate working.)

★ THE DECISIVE TEST — LOCAL SLOPE vs RESPONSE MAGNITUDE, pooled across vectors.
For a linear map ‖Δs(α)‖ ∝ α, so log‖Δs‖ = b·log α + c with **b = 1 exactly**. Fit b per vector
(global) and locally (per interior dose, via central difference). Then plot b_local against the
RESPONSE MAGNITUDE at that dose, for every vector at once:

  · if all vectors fall on ONE curve  ⇒ **magnitude explains it**: the map is linear for small
    responses and stiffens as you push, regardless of WHICH direction you push. V4 was never
    special — it just pushes hardest at equal α. A6's confound reading is CONFIRMED.
  · if V4's points sit OFF the curve traced by the others ⇒ **V4 is special**: something about the
    gradient direction breaks linearity beyond mere size. A6's confound reading is REFUTED.

V3@0.5 should reach a response near V4@0.1's (.754 measured), so the two overlap in magnitude
while differing in identity — exactly the contrast that was impossible with the banked doses.

Reads: per-gen matched-token Δ vs the stage0 twin (bitwise-deterministic replay ⇒ Δ is pure
intervention response, NO noise floor). Reported on the full 3358-d vector, C2's non-trivial
1282-d subvector (the C§1-honest space), and the trivial complement (the POSITIVE CONTROL: the
linear image of αv must read b ≈ 1 at every magnitude).

    PYTHONPATH=pipeline python -m anamnesis.scripts.annex_linearity_ladder
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from anamnesis.analysis.battery.deltas import load_floor_scale
from anamnesis.scripts.annex_corpus import REPO, VENUE_DIR, load_venue
from anamnesis.scripts.annex_linearity import MT_PAT, OUT, load_dose_cells

logger = logging.getLogger(__name__)
F32 = NDArray[np.float32]
C2_AXIS = REPO / "outputs/battery/arms/C2/c2_orphaned_axis_3b.npz"


def loglog_slope(alphas: list[float], mags: list[float]) -> float:
    """b in ‖Δs‖ ∝ α^b. Linear map ⇒ b = 1 exactly."""
    if len(alphas) < 2:
        return float("nan")
    return float(np.polyfit(np.log(alphas), np.log(mags), 1)[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT / "annex_linearity_ladder.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    med, scale = load_floor_scale(VENUE_DIR / "signatures_v3")
    c = load_venue(capped_only=True)
    names = c.feature_names

    print("=== MATCHED-TOKEN LADDER (per-gen Δ vs stage0 twins) ===")
    cells = load_dose_cells(med, scale, names)

    # group by vector -> {alpha: cell}
    by_vec: dict[str, dict[float, object]] = {}
    for name, cell in cells.items():
        m = MT_PAT.match(name)
        key = m.group("vec") if not m.group("site") else f"{m.group('vec')}_L{m.group('site')}"
        by_vec.setdefault(key, {})[cell.alpha] = cell

    z2 = np.load(C2_AXIS)
    nti = np.array([names.index(str(x)) for x in z2["feature_names"]], dtype=np.int64)
    triv = np.array([i for i in range(len(names)) if i not in set(nti.tolist())], dtype=np.int64)
    spaces = {"full_3358": np.arange(len(names), dtype=np.int64),
              "nontrivial_1282": nti,
              "trivial_2076": triv}

    results: dict[str, dict] = {}
    for space, cols in spaces.items():
        print(f"\n{'='*78}\n=== SPACE: {space} (d={len(cols)})  — linear map ⇒ log-log slope b = 1\n{'='*78}")
        rows: dict[str, dict] = {}
        pooled = []          # (vector, alpha, response_magnitude, b_local)
        for vec in sorted(by_vec):
            doses = sorted(by_vec[vec])
            mags, meandirs = [], {}
            for a in doses:
                cell = by_vec[vec][a]
                gids = sorted(cell.deltas)
                D = np.stack([cell.deltas[g][cols] for g in gids])
                mags.append(float(np.linalg.norm(D, axis=1).mean()))
                meandirs[a] = D.mean(0)
            b_global = loglog_slope(doses, mags)

            # local slope by central difference in log-log space
            b_local: dict[float, float] = {}
            for i in range(1, len(doses) - 1):
                b_local[doses[i]] = float(
                    (np.log(mags[i + 1]) - np.log(mags[i - 1]))
                    / (np.log(doses[i + 1]) - np.log(doses[i - 1])))
                pooled.append((vec, doses[i], mags[i], b_local[doses[i]]))

            # direction rotation across the ladder, vs the smallest dose
            base = meandirs[doses[0]]
            rot = {a: float(base @ meandirs[a]
                            / max(np.linalg.norm(base) * np.linalg.norm(meandirs[a]), 1e-12))
                   for a in doses}
            rows[vec] = {
                "alphas": doses, "response_magnitude": [round(m, 4) for m in mags],
                "loglog_slope_global": round(b_global, 4),
                "loglog_slope_local": {str(k): round(v, 4) for k, v in b_local.items()},
                "direction_cos_vs_lowest_dose": {str(k): round(v, 4) for k, v in rot.items()},
            }
            print(f"\n  {vec}   b_global = {b_global:.3f}"
                  f"{'   <-- LINEAR' if abs(b_global-1) < .1 else ''}")
            print(f"    {'α':>6s} {'‖Δs‖':>9s} {'b_local':>9s} {'dir cos vs α_min':>17s}")
            for i, a in enumerate(doses):
                bl = b_local.get(a)
                print(f"    {a:6.2f} {mags[i]:9.4f} "
                      f"{(f'{bl:9.3f}' if bl is not None else '        —')} {rot[a]:17.4f}")

        # ★ the decisive contrast: does b_local depend on MAGNITUDE alone, across vectors?
        pooled.sort(key=lambda t: t[2])
        v4 = [p for p in pooled if p[0].startswith("V4")]
        oth = [p for p in pooled if not p[0].startswith("V4")]
        print(f"\n  ★ POOLED b_local vs RESPONSE MAGNITUDE (all vectors, sorted by magnitude)")
        print(f"    {'vector':10s} {'α':>6s} {'‖Δs‖':>9s} {'b_local':>9s}")
        for vec, a, mg, bl in pooled:
            print(f"    {vec:10s} {a:6.2f} {mg:9.4f} {bl:9.3f}"
                  f"{'   <-- V4' if vec.startswith('V4') else ''}")
        # is V4 off the curve traced by the others? compare at matched magnitude via interpolation
        verdict = None
        if v4 and len(oth) >= 2:
            om = np.array([p[2] for p in oth]); ob = np.array([p[3] for p in oth])
            o = np.argsort(om)
            resid = []
            for vec, a, mg, bl in v4:
                if om[o].min() <= mg <= om[o].max():
                    pred = float(np.interp(mg, om[o], ob[o]))
                    resid.append(bl - pred)
                    print(f"    V4 @ α={a} (‖Δs‖={mg:.4f}): b_local={bl:.3f} vs "
                          f"others-at-same-magnitude {pred:.3f}  → residual {bl-pred:+.3f}")
            if resid:
                verdict = ("MAGNITUDE explains it (V4 on the others' curve)"
                           if max(abs(r) for r in resid) < 0.25
                           else "V4 is OFF the curve — identity matters beyond magnitude")
                print(f"    ⇒ {verdict}")
            else:
                verdict = "V4's magnitude range does not overlap the others — NOT decidable here"
                print(f"    ⇒ {verdict}")
        results[space] = {"vectors": rows,
                          "pooled_b_local_by_magnitude": [
                              {"vector": v, "alpha": a, "magnitude": round(m, 4),
                               "b_local": round(b, 4)} for v, a, m, b in pooled],
                          "magnitude_matched_verdict": verdict}

    OUT.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "ANNEX_RULE": "NOT QUOTABLE until graduated via a frozen prereg cell",
        "rung": "A6′ — the dose ladder (magnitude vs identity, confound broken)",
        "ladder": "V1/V2/V3/R1/R2/R3 at α ∈ {.03,.1,.15,.2,.3,.5}; V4 at {.03,.1,.15} ONLY — "
                  "V4's on-policy regime ENDS at .15 (agreement .834 at .2, .101 at 1.0). Cells "
                  "beyond it were NOT run: forced tokens there measure an incoherent "
                  "counterfactual (vmb_a5_onpolicy_gate.py:1). The gate defined this ladder.",
        "statistic": "‖Δs‖ ∝ α^b; a LINEAR map gives b = 1 exactly. b_local by central difference "
                     "in log-log space. Direction rotation = cos(mean Δ(α), mean Δ(α_min)).",
        "decisive_test": "pool b_local across ALL vectors against RESPONSE MAGNITUDE. One curve ⇒ "
                         "magnitude explains it and V4 was never special. V4 off the curve ⇒ "
                         "identity matters beyond size.",
        "no_noise_floor": "bitwise-deterministic replay + matched tokens ⇒ Δ is pure response.",
        "positive_control": "trivial_2076 = the linear image of αv ⇒ MUST read b ≈ 1 at every "
                            "magnitude. If it does not, the assay is broken and nothing else here "
                            "may be read.",
        "spaces": results,
    }, indent=1))
    print(f"\n  → {args.out}")


if __name__ == "__main__":
    main()
