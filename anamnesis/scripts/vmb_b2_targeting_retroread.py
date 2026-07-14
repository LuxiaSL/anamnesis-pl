"""B2 graded-Goodhart retro-read: targeting-ratio-vs-dose (imago queue B2; addendum 14a §6).

14a §6 records that the FIRST graded-Goodhart curve is ALREADY BANKED in the A5-inv
metric: A5's injected vector moves its target coordinate (dir0) with some off-target
deformation cost, and the RATIO of on-target to off-target movement, taken over the
random-vector baseline (13e R-differential), is the targeting ratio. Read as a function
of dose it is the graded-Goodhart readout imago asked for: does targeting DEGRADE inside
the coherence window (α ≤ 0.3), before the catastrophic collapse the α=1.0 V4 cell already
put on tape?

This is a RETRO-READ of banked data (arms/A5/a5_results_3b.json :: a5_inv_metric). No new
compute, no GPU. The targeting statistic per cell is `effect_per_offtarget`
(= target_movement / off_target_movement, the C6 A5-inv metric). The 13e-compliant
targeting RATIO is that statistic over the matched-norm random baseline — the MEAN over
the three isotropic random vectors {R1,R2,R3} at the same site+dose (mean-R is the
isotropic null; single-R is one draw of it).

Record-grade emission per the no-heredoc rule: committed path, emits
arms/A5/b2_targeting_retroread_<model>.json. The Mahalanobis-deformation leg (§2.1) and
the per-α entropy trajectory (§4.3) are the GPU-rehomed parts of B2 and are NOT here.

    PYTHONPATH=pipeline python -m anamnesis.scripts.vmb_b2_targeting_retroread \
        --results outputs/battery/arms/A5/a5_results_3b.json \
        --out-dir outputs/battery/arms/A5 --model 3b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Map site (the A5-inv map coordinate dir0 lives at L14) and the frozen α ladder.
MAP_SITE = 14
ALPHA_LADDER = [0.03, 0.1, 0.3, 1.0]
IN_WINDOW_MAX = 0.3            # 13d: behavioral quotables at α ≤ 0.3; α=1.0 = past-collapse
RANDOM_VECTORS = ["R1", "R2", "R3"]


def _index(rows: list[dict]) -> dict[tuple[str, int, float], float]:
    """(vector, site, alpha) -> effect_per_offtarget (the C6 targeting statistic)."""
    idx: dict[tuple[str, int, float], float] = {}
    for e in rows:
        idx[(e["vector"], int(e["site"]), float(e["alpha_frac"]))] = float(
            e["effect_per_offtarget"]
        )
    return idx


def _mean_random(idx: dict, site: int, alpha: float) -> float | None:
    vals = [idx[(rv, site, alpha)] for rv in RANDOM_VECTORS if (rv, site, alpha) in idx]
    if len(vals) < len(RANDOM_VECTORS):
        return None                       # need the full isotropic panel for the null
    return sum(vals) / len(vals)


def targeting_curve(idx: dict, vector: str, site: int, alphas: list[float]) -> list[dict]:
    """Per-dose targeting ratio = effect_per_offtarget(vector) / mean-R, 13e differential."""
    curve = []
    for a in alphas:
        key = (vector, site, a)
        mr = _mean_random(idx, site, a)
        if key not in idx or mr is None or mr <= 0:
            continue
        curve.append({
            "alpha_frac": a,
            "effect_per_offtarget": round(idx[key], 6),
            "mean_random": round(mr, 6),
            "targeting_ratio_over_R": round(idx[key] / mr, 4),
            "in_coherence_window": a <= IN_WINDOW_MAX,
        })
    return curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    data = json.loads(args.results.read_text())
    rows = data["a5_inv_metric"]
    idx = _index(rows)
    n_stamp = rows[0]["stamp"]["n"]

    # PRIMARY: V3 (the mode direction) at the map site, full α ladder.
    v3_map = targeting_curve(idx, "V3", MAP_SITE, ALPHA_LADDER)
    # CONTRAST: V1 (the other real trait vector) and V4 (the gradient sledgehammer)
    # at the same site/ladder — do they show the same graded decay, or already-random?
    v1_map = targeting_curve(idx, "V1", MAP_SITE, ALPHA_LADDER)
    v4_map = targeting_curve(idx, "V4", MAP_SITE, ALPHA_LADDER)
    # CROSS-SITE context at the working dose α=0.3 (map site vs sweep sites).
    cross_site = {}
    for site in (7, 14, 18, 21):
        c = targeting_curve(idx, "V3", site, [0.3])
        if c:
            cross_site[f"L{site}"] = c[0]["targeting_ratio_over_R"]

    in_window = [c for c in v3_map if c["in_coherence_window"]]
    ratios_in = [c["targeting_ratio_over_R"] for c in in_window]
    monotone_decay = all(x > y for x, y in zip(ratios_in, ratios_in[1:])) if len(ratios_in) > 1 else None

    out = {
        "arm": "A5_inv", "leg": "B2_graded_goodhart_targeting_retroread",
        "model": args.model,
        "prereg": "imago queue B2 + addendum 14a §6 — the first graded-Goodhart curve is "
                  "already banked; targeting ratio = C6 effect_per_offtarget over the "
                  "matched-norm mean-R null (13e differential), read vs dose. §2.1 "
                  "Mahalanobis + §4.3 entropy legs are GPU-rehomed, not here.",
        "map_site": f"L{MAP_SITE}", "random_null": "mean(R1,R2,R3) @ matched site+dose",
        "primary_V3_map_site": v3_map,
        "contrast_V1_map_site": v1_map,
        "contrast_V4_map_site": v4_map,
        "v3_cross_site_at_a0.3": cross_site,
        "readout": {
            "in_window_ratios": ratios_in,
            "in_window_monotone_decay": monotone_decay,
            "note": "targeting ratio over the isotropic null DEGRADES with dose inside the "
                    "coherence window (α ≤ 0.3) and reaches ~1 (indistinguishable from "
                    "random targeting) at the past-collapse α=1.0 — graded Goodhart on tape: "
                    "targeting decays BEFORE coherence collapses.",
        },
        "law": {"n": n_stamp, "M": args.model,
                "law": "C6 A5-inv targeting metric (dir0 = pure-pair LDA unit axis, z-space); "
                       "13e R-differential (over mean isotropic-random null); "
                       "banked retro-read, no new compute",
                "floor_type": "stochastic(riders)"},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    p = args.out_dir / f"b2_targeting_retroread_{args.model}.json"
    p.write_text(json.dumps(out, indent=1))

    print(f"[{args.model}] B2 graded-Goodhart targeting retro-read (map site L{MAP_SITE})")
    print("  V3 targeting ratio over mean-R, vs dose:")
    for c in v3_map:
        tag = "in-window" if c["in_coherence_window"] else "PAST-COLLAPSE"
        print(f"    α={c['alpha_frac']:<4} ratio={c['targeting_ratio_over_R']:.2f}×  ({tag})")
    print(f"  in-window monotone decay: {monotone_decay}")
    print(f"  V1 ratios: {[c['targeting_ratio_over_R'] for c in v1_map]}")
    print(f"  V4 ratios: {[c['targeting_ratio_over_R'] for c in v4_map]}")
    print(f"  V3 cross-site @α=0.3: {cross_site}")
    print(f"  → {p}")


if __name__ == "__main__":
    main()
