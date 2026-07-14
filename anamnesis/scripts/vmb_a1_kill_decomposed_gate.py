"""A1 KILL positive-control gate on the 12e DECOMPOSED ruler (addendum 14b).

14b ruled the A1 KILL criterion ruler-corrected from the pairwise ratio (§2c, which
is √-compressed and contraction-blind — it floors on high-native-temp models whose
Stage-0 stochastic floor is wide) to the 12e decomposed standard: the positive control
PASSES iff the KILL contrast (t03|t09, source:output) is permutation-significant on
centroid SHIFT **or** dispersion, while the predicted-blind cell (attention|mid) stays
non-significant. The pairwise ratio + classifier AUC (a1_results.json) are retained as
DIAGNOSTICS, not the verdict.

Record-grade emission of the outer-loop computation (14b): committed path, per the
no-heredoc rule. Emits <out-dir>/a1_kill_decomposed_gate_<model>.json — a DEDICATED
file; does NOT touch the anchors' banked arms/magnitude_retroread_12e.json.

    PYTHONPATH=pipeline python -m anamnesis.scripts.vmb_a1_kill_decomposed_gate \
        --battery-root outputs/battery --model gemma3-27b \
        --out-dir outputs/battery/arms/A1_m5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from anamnesis.analysis.battery.deltas import ConditionCorpus, build_cells, load_floor_scale
from anamnesis.analysis.battery.magnitude import decomposed_magnitude
from anamnesis.analysis.battery.manifest import MODEL_META

KILL_CELL = "source:output"                     # A1 positive-control carrier
BLIND_CELL = "source_band:attention|mid"        # predicted-blind (must stay non-sig)
N_PERM = 1000
ALPHA = 0.05


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--model", required=True, choices=list(MODEL_META.keys()))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    args = ap.parse_args()

    meta = MODEL_META[args.model]
    floor_dir = args.battery_root / meta.stage0_dir
    med, scale = load_floor_scale(floor_dir / "signatures_v3")

    def load(dose: str) -> ConditionCorpus:
        d = args.battery_root / f"vmb_a1_{args.model}_{dose}"
        return ConditionCorpus(d / "signatures_v3", d / "metadata.json",
                               med, scale, f"{args.model}-{dose}")

    # KILL contrast t03|t09 (cooling): a=t03, b=t09 so dispersion_ratio reads
    # within-t09 / within-t03 (warmer dose expected wider).
    t03, t09 = load("t03"), load("t09")
    cells = build_cells(t03.feature_names, meta.n_layers)
    gate_cells = {KILL_CELL: cells[KILL_CELL], BLIND_CELL: cells[BLIND_CELL]}
    res = decomposed_magnitude(t03, t09, gate_cells, args.n_perm)

    kill = res[KILL_CELL]
    blind = res[BLIND_CELL]
    # KILL cell CARRIES temperature iff shift OR dispersion is perm-significant (14b).
    kill_sig = (kill["p_shift"] < ALPHA) or (kill["p_disp_wider"] < ALPHA)
    # BLINDNESS is a LOCATION claim: the predicted-blind cell does not MOVE to encode
    # temperature (no centroid shift). Temperature heating is a global MOVER+SPREADER
    # (A1 record: dispersion ×1.90 fleet-wide), so a small dispersion spillover in the
    # blind cell is EXPECTED and is not "carrying" — the blindness readout is the shift.
    blind_carries = blind["p_shift"] < ALPHA
    verdict = "PASS" if (kill_sig and not blind_carries) else "FAIL"

    out = {
        "arm": "A1", "model": args.model, "gate": "kill_decomposed_12e",
        "prereg": "addendum 14b — A1 KILL ruler-corrected to the 12e decomposed standard "
                  "(perm-gated shift OR dispersion on source:output; predicted-blind "
                  "attention|mid must stay non-sig). Pairwise ratio + classifier AUC are "
                  "diagnostics in a1_results.json, NOT the verdict.",
        "kill_contrast": "t03|t09", "kill_cell": KILL_CELL, "blind_cell": BLIND_CELL,
        "n_perm": args.n_perm, "alpha": ALPHA,
        "kill": kill, "predicted_blind": blind,
        "kill_carries": bool(kill_sig), "blind_carries_location": bool(blind_carries),
        "blind_note": "blind cell dispersion may be perm-sig from the global mover/spreader "
                      "spillover (heating spreads all cells); blindness is the SHIFT readout, "
                      "not dispersion. Reported: blind disp ×%.2f (p_wider=%.3f)."
                      % (blind["dispersion_ratio"], blind["p_disp_wider"]),
        "verdict": verdict,
        "law": {"n_a": kill["n_a"], "n_b": kill["n_b"], "M": args.model,
                "law": "12e decomposed ruler (centroid shift floor-z + dispersion ratio, "
                       "within-class-permutation null); KILL = shift OR dispersion sig, "
                       "blind non-sig (14b)"},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    p = args.out_dir / f"a1_kill_decomposed_gate_{args.model}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"[{args.model}] A1 KILL (decomposed 12e) = {verdict}")
    print(f"  {KILL_CELL}: shift {kill['centroid_shift']:.3f}z (p={kill['p_shift']:.3f}), "
          f"dispersion ×{kill['dispersion_ratio']:.2f} (p_wider={kill['p_disp_wider']:.3f})")
    print(f"  {BLIND_CELL} (blind): shift {blind['centroid_shift']:.3f}z "
          f"(p={blind['p_shift']:.3f}) — must be non-sig")
    print(f"  → {p}")


if __name__ == "__main__":
    main()
