"""14m item 4 — housekeeping on the two leg-2 artifacts (ADDENDUM 14m).

The leg-2 (b)/(c) JSONs were produced by the generalized C3 scripts, so their `arm:`/`law:`
strings still say "V_temp"/"Rc" and the entropy ratio keys are `_over_Rc` (the C3 null family)
when the actual null family for the V7 cells is Rband. Also the α.03 `entropy_rise_over_Rc`
= 1.08e18 is a §B.5 near-zero-denominator artifact (Rband α.03 mean rise ≈ 0). This rewrites:
  - arm/law strings V_temp→V7, Rc→Rband
  - key rename `_over_Rc` → `_over_Rband`
  - zero-guard: any ratio whose null denominator |mean| < --eps is set null + a note
Corrected artifacts are written ALONGSIDE the originals (`*_corrected.json`); originals untouched.
CPU. First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _fix_entropy(d: dict, eps: float) -> dict:
    d = json.loads(json.dumps(d))  # deep copy
    d["arm"] = "14j LEG 2 (b) — per-token entropy under V7 (formula) vs Rband matched-support nulls"
    d["law"] = d["law"].replace("V_temp", "V7").replace("mean(Rc)", "mean(Rband)")
    d["housekeeping_14m"] = ("arm/law V_temp→V7; keys _over_Rc→_over_Rband; near-zero null "
                             f"denominators (|mean(Rband)|<{eps}) zero-guarded to null")
    # recompute null (Rband) means per (site, alpha) to zero-guard
    def rband_mean(site, af, key):
        vals = [r[key] for r in d["rows"] if r.get("is_null") and r["site"] == site
                and r["alpha_frac"] == af]
        return float(np.mean(vals)) if vals else None
    for r in d["rows"]:
        for old in [k for k in list(r) if k.endswith("_over_Rc")]:
            base_key = old[:-len("_over_Rc")]
            new = f"{base_key}_over_Rband"
            denom = rband_mean(r["site"], r["alpha_frac"], base_key)
            if r.get("is_null") or denom is None:
                r[new] = r.pop(old)
            elif abs(denom) < eps:
                r.pop(old)
                r[new] = None
                r[f"{new}_note"] = f"null denominator mean(Rband {base_key})={denom:.2e} < eps ({eps}) — ratio unstable; read raw {base_key}"
            else:
                r[new] = r.pop(old)
    return d


def _fix_content(d: dict) -> dict:
    d = json.loads(json.dumps(d))
    d["arm"] = "14j LEG 2 (c) — content rung, V7-steered vs rider (and vs Rband-steered)"
    d["law"] = d["law"].replace("V_temp-steered", "V7-steered").replace("Rc-steered", "Rband-steered")
    d["housekeeping_14m"] = "arm/law V_temp→V7, Rc→Rband (keys already generalized to _vs_null)"
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entropy-json", type=Path, required=True)
    ap.add_argument("--content-json", type=Path, required=True)
    ap.add_argument("--eps", type=float, default=0.005, help="near-zero null-denominator threshold")
    args = ap.parse_args()
    for path, fixer in ((args.entropy_json, lambda d: _fix_entropy(d, args.eps)),
                        (args.content_json, _fix_content)):
        d = json.loads(path.read_text())
        out = path.with_name(path.stem + "_corrected.json")
        out.write_text(json.dumps(fixer(d), indent=1))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
