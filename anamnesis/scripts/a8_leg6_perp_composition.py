"""A8 Leg-6 / Item 3 — orthogonalized field-basis composition (P8-PERP .50).

The L4-b successor. L4-b's Reading A summed cos(u, dial_i) x dial_i-effect over a dial set
that is badly non-orthogonal (cos(Vconf, V7) ~ -.91 in both dense targets), so Reading A
double-counts the shared V7/Vconf component by construction and Reading B (Gram-corrected
least squares) was filed beside it. add-4 asks the obvious repair: orthogonalize the dial
basis against V7 first, then re-predict.

CONSTRUCTION IS STAMPED BEFORE ANY RE-PREDICTION (add-4's explicit requirement). It is
written to the readout's `construction` block and logged before a single ratio is computed.

TWO CONSTRUCTIONS, BOTH REPORTED — add-4 says "sequentially orthogonalized against V7",
which admits two readings, and this arm's standing practice (the Reading A/B precedent,
and the rules the previous three ambiguities bought) is to report every defensible one
rather than silently pick:

  P  "perp-to-V7"      : every non-V7 dial is Gram-Schmidt'd against V7 ALONE. The dials
                         stay mutually oblique; only the V7 double-count is removed.
  Q  "full sequential" : ordered Gram-Schmidt over [V7, Vrep_perp, Veos_perp, Vconf,
                         Vtemp] — a genuinely orthonormal basis. "Sequential Gram-Schmidt"
                         as a term of art.

BANKED PERP FORMS ARE USED WHERE THEY EXIST, AND NAMED: Vrep_perp and Veos_perp are
already banked Gram-Schmidt-against-V7 objects (the 8B forms carry cos-to-V7 ~ 4e-17 per
the fire log). Under construction P they are therefore passed through UNCHANGED, and the
readout says so per dial. Only Vconf and Vtemp are actually re-orthogonalized under P.

EFFECTS ARE CONVENTION-CARRIED, NOT RE-MEASURED (add-4's conditional: "P8-PERP inherits
the L4-b fork — convention-carried terms named, never freshly measured mid-session"). The
banked dose-ladder of each dial is attached to that dial's ORTHOGONALIZED form. This is a
substitution and it is named: the ladders were measured on the original vectors. For
Vrep_perp/Veos_perp under P the substitution is vacuous (vector unchanged); for Vconf,
Vtemp, and everything under Q it is real.

THE NEAR-ZERO ROW, reported beside as add-4 requires: 8b->qwen Vrep_perp at +0.3, whose
Leg-4F Reading-A prediction was -.0359 against obs -.0529 (ratio 1.472, within x2). A
near-zero prediction landing inside x2 is fragile — it can flip sign under any basis
change — so its behaviour under P and Q is reported explicitly rather than folded into
the count.

BAR (add-4, symmetric): Qwen >=4/5 within x2 AND 8B >=4/5 within x2, aggregation >=4/5
per sheet. NOT self-scored — the desk scores P8-PERP.

UNSTAMPED (C section 8).
Run: PYTHONPATH=pipeline python -m anamnesis.scripts.a8_leg6_perp_composition
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from anamnesis.scripts.a8_fit_g import load_transport_map
from anamnesis.scripts.a8_leg4_composition import (
    DOSES, PAIRS, PANEL, SCORE_DOSE, _observed, _sheet_rows, _target_dials)
from anamnesis.scripts.a8_rosetta import _unit, cos, load_axes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_leg6_perp_composition")

OUT = Path("outputs/battery/arms/A8_conjugation/leg6/readouts_cpu")
BASIS_ORDER = ("V7", "Vrep_perp", "Veos_perp", "Vconf", "Vtemp")
ORTHO_TOL = 1e-6                 # |cos| below this counts as "already orthogonal"
NEAR_ZERO_ROW = {"pair": "8b->qwen", "vector": "Vrep_perp", "dose": "+0.3",
                 "leg4f_reading_A_pred": -0.0359, "leg4f_obs": -0.0529,
                 "leg4f_ratio": 1.472, "leg4f_within_x2": True}


def _gs(v: np.ndarray, against: list[np.ndarray]) -> np.ndarray:
    """Gram-Schmidt v against an already-orthonormal list."""
    w = v.astype(np.float64).copy()
    for a in against:
        w = w - (w @ a) * a
    n = float(np.linalg.norm(w))
    if n < 1e-12:
        raise RuntimeError("dial collapsed to zero under Gram-Schmidt")
    return w / n


def build_bases(dials: dict[str, np.ndarray]) -> tuple[dict, dict, dict]:
    """Returns (basis_P, basis_Q, construction_record) — STAMPED BEFORE PREDICTION."""
    names = [n for n in BASIS_ORDER if n in dials]
    v7 = _unit(dials["V7"])
    rec: dict[str, dict] = {}

    basis_P: dict[str, np.ndarray] = {"V7": v7}
    for n in names:
        if n == "V7":
            rec[n] = {"action": "basis anchor, unchanged", "cos_to_V7_before": 1.0}
            continue
        c_before = cos(dials[n], v7)
        if abs(c_before) <= ORTHO_TOL:
            basis_P[n] = _unit(dials[n])
            rec[n] = {"action": "BANKED PERP FORM — passed through unchanged",
                      "cos_to_V7_before": round(c_before, 12),
                      "cos_to_V7_after": round(c_before, 12)}
        else:
            basis_P[n] = _gs(dials[n], [v7])
            rec[n] = {"action": "Gram-Schmidt against V7",
                      "cos_to_V7_before": round(c_before, 4),
                      "cos_to_V7_after": round(cos(basis_P[n], v7), 12)}

    basis_Q: dict[str, np.ndarray] = {}
    done: list[np.ndarray] = []
    for n in names:
        w = _unit(dials[n]) if not done else _gs(dials[n], done)
        basis_Q[n] = w
        done.append(w)
        rec[n]["full_sequential_cos_to_prior"] = [
            round(cos(w, d), 12) for d in done[:-1]]
    return basis_P, basis_Q, rec


def predict(cfg: dict, basis: dict[str, np.ndarray], effects: dict) -> dict:
    tm = load_transport_map(cfg["fit"])
    src_reads, src_extras, _ = load_axes(cfg["src"])
    src = {k: (src_reads[k].vec if k in src_reads else src_extras[k].vec) for k in PANEL}
    names = list(basis)
    D = np.stack([basis[n] for n in names])
    G = D @ D.T
    Ginv = np.linalg.pinv(G, rcond=1e-6)
    sheet, obs = _sheet_rows(cfg), _observed(cfg)

    rows = []
    for vname, v in src.items():
        u = _unit(tm.transport(v))
        cosv = D @ u
        coeff = Ginv @ cosv
        per_dose = {}
        for dose in DOSES:
            a = float(dose)
            pa = float(np.nansum([cosv[i] * effects[n].get(a, np.nan)
                                  for i, n in enumerate(names)]))
            pb = float(np.nansum([coeff[i] * effects[n].get(a, np.nan)
                                  for i, n in enumerate(names)]))
            o = obs.get(vname, {}).get(dose)
            per_dose[dose] = {
                "pred_A_letter_cos_sum": round(pa, 4),
                "pred_B_gram_corrected": round(pb, 4),
                "obs": o,
                "ratio_obs_over_A": (round(o / pa, 3) if o is not None and abs(pa) > 1e-9
                                     else None),
                "ratio_obs_over_B": (round(o / pb, 3) if o is not None and abs(pb) > 1e-9
                                     else None),
            }
        rows.append({"vector": vname,
                     "cos_transported_vs_dials": {n: round(float(cosv[i]), 4)
                                                  for i, n in enumerate(names)},
                     "residual_out_of_dial_span": round(
                         float(np.linalg.norm(u - (coeff @ D))), 4),
                     "per_dose": per_dose})

    def _within(key: str) -> dict:
        hits = {r["vector"]: {
            "ratio": r["per_dose"][SCORE_DOSE][key],
            "within_x2": (r["per_dose"][SCORE_DOSE][key] is not None
                          and 0.5 <= abs(r["per_dose"][SCORE_DOSE][key]) <= 2.0
                          and r["per_dose"][SCORE_DOSE][key] > 0)} for r in rows}
        return {"per_vector": hits,
                "n_within_x2": f"{sum(1 for v in hits.values() if v['within_x2'])}/{len(hits)}"}

    return {"gram_condition_number": round(float(np.linalg.cond(G)), 4),
            "rows": rows,
            f"score_dose_{SCORE_DOSE}": {"reading_A_letter": _within("ratio_obs_over_A"),
                                         "reading_B_gram": _within("ratio_obs_over_B")}}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    out: dict = {
        "STATUS": "UNSTAMPED (C section 8) — no self-scored P; the desk scores P8-PERP (.50)",
        "leg": "A8 Leg-6 / Item 3 — orthogonalized field-basis composition",
        "bar_as_frozen": "Qwen >=4/5 within x2 AND 8B >=4/5 within x2 (symmetric; "
                         "aggregation >=4/5 per sheet)",
        "effects_convention":
            "banked dose-ladders are CONVENTION-CARRIED onto the orthogonalized dial of "
            "the same name; nothing re-measured mid-session (add-4 conditional, inheriting "
            "the L4-b fork). Vacuous for banked perp forms under construction P; real for "
            "Vconf, Vtemp, and all dials under Q.",
        "near_zero_row_watch": NEAR_ZERO_ROW,
        "pairs": {},
    }

    # ---- CONSTRUCTION FIRST, for every pair, before any prediction is computed --------
    prepared = {}
    for pair, cfg in PAIRS.items():
        dials, effects, notes = _target_dials(cfg)
        bP, bQ, rec = build_bases(dials)
        prepared[pair] = (cfg, effects, bP, bQ)
        out["pairs"][pair] = {"dial_provenance": notes, "construction": rec}
        logger.info("[%s] CONSTRUCTION STAMPED", pair)
        for n, r in rec.items():
            logger.info("    %-12s %s (cos->V7 %s -> %s)", n, r["action"],
                        r["cos_to_V7_before"], r.get("cos_to_V7_after", "-"))
    (OUT / "perp_composition_construction.json").write_text(
        json.dumps({k: v["construction"] for k, v in out["pairs"].items()}, indent=1))
    logger.info("construction stamped -> %s", OUT / "perp_composition_construction.json")

    # ---- only now: re-prediction ------------------------------------------------------
    for pair, (cfg, effects, bP, bQ) in prepared.items():
        out["pairs"][pair]["P_perp_to_V7"] = predict(cfg, bP, effects)
        out["pairs"][pair]["Q_full_sequential"] = predict(cfg, bQ, effects)
        nz = NEAR_ZERO_ROW
        if pair == nz["pair"]:
            for ck, cname in (("P_perp_to_V7", "P"), ("Q_full_sequential", "Q")):
                row = next(r for r in out["pairs"][pair][ck]["rows"]
                           if r["vector"] == nz["vector"])
                out["pairs"][pair].setdefault("near_zero_row_result", {})[cname] = \
                    row["per_dose"][nz["dose"]]

    (OUT / "perp_composition.json").write_text(json.dumps(out, indent=1))
    for pair, blk in out["pairs"].items():
        for ck in ("P_perp_to_V7", "Q_full_sequential"):
            s = blk[ck][f"score_dose_{SCORE_DOSE}"]
            logger.info("[%s] %-18s A: %s   B: %s", pair, ck,
                        s["reading_A_letter"]["n_within_x2"],
                        s["reading_B_gram"]["n_within_x2"])
    logger.info("wrote %s", OUT / "perp_composition.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
