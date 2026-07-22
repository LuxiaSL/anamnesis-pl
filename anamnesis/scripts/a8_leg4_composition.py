"""A8 Leg-4 / L4-b — field-basis composition re-prediction (CPU; frozen bar P8-L4b .55).

The Leg-3 desk pass (§9 ruling 4) on the §2b over-performance pattern: the F-ii sheets
predicted each transported vector's entropy consequence from the V7 row ALONE
(pred = cos(g·v, V7_tgt) x V7's own dose law), and every magnitude excursion was an
OVER-performance.  The hypothesis: the transported vectors carry material cos on the
TARGET'S OTHER field dials, so the single-dial model structurally under-calls.  Re-predict
under the target's whole banked field basis and see whether the ratios close toward 1.

Two readings are computed and BOTH filed (Add-1.3 discipline — the letter's Sigma-form is
degenerate on a non-orthogonal basis, so the desk picks the scoring reading):

  READING A (letter-literal):  pred = SUM_i cos(u, dial_i) x effect_i(dose)
      Exactly as written in add-2.  NOTE the basis is NOT orthogonal — cos(Vconf, V7)
      ~ -0.91 in both targets — so the V7 and Vconf terms overlap and A double-counts
      the shared component by construction.  Filed because it is the letter.

  READING B (Gram-corrected):  c = pinv(G) @ [cos(u, dial_i)],  pred = SUM_i c_i x effect_i
      where G is the dial Gram.  This is the least-squares decomposition of u onto the
      dial span — the non-degenerate form of "the vector's consequence is its projection
      onto the target's dial frame".  Gram condition number filed beside.

Basis (inventoried, per target):
  * V7, Vrep_perp, Veos_perp, Vconf — all four have BANKED entropy ladders
    (V7: arms/A5_matrix/{8b,qwen}/entropy_*.json, the same law file the original sheet
     used, so the V7 term is unchanged and the delta is purely the added dials;
     field triple: arms/A5_matrix/field{,_qwen}/entropy_field*.json, doses +-0.1/+-0.3).
  * Vtemp — NO banked entropy ladder exists (checked: only V7 + the field triple are
    banked at these sites).  Per the pre-ruled fork, its term uses the F-ii-filed
    prediction convention (effect := cos(Vtemp_tgt, V7_tgt) x V7's law) and the
    substitution is NAMED in the output.  It is not a new measurement.

Everything UNSTAMPED (C§8).  Mechanics only — the desk scores P8-L4b.

Run (repo root): PYTHONPATH=pipeline python -m anamnesis.scripts.a8_leg4_composition
                 [--selftest]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from anamnesis.scripts.a8_fit_g import load_transport_map
from anamnesis.scripts.a8_rosetta import _unit, cos, load_axes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_leg4_composition")

ARM = Path("outputs/battery/arms/A8_conjugation")
OUT = ARM / "leg4" / "readouts_cpu"
A5 = Path("outputs/battery/arms/A5_matrix")

PAIRS = {
    "3b->8b": {
        "src": "3b", "tgt": "8b",
        "fit": ARM / "fits/fit_3bL14__8bL16_native_proc_k512.npz",
        "sheet": ARM / "readouts/f2_predictions.json",
        "anchor": "3bL14->8bL16",
        "scoring": ARM / "leg3/readouts_final/scoring_8b.json",
        "v7_law": A5 / "8b/entropy_8b.json",
        "field_law": A5 / "field/entropy_field.json",
        "site": 16,
    },
    "8b->qwen": {
        "src": "8b", "tgt": "qwen-7b",
        "fit": ARM / "leg1/fits/fit_8bL16__qwen-7bL21_native_proc_k512.npz",
        "sheet": ARM / "leg1/readouts/f2_predictions.json",
        "anchor": "8bL16->qwen-7bL21",
        "scoring": ARM / "leg3/readouts_final/scoring_qwen-7b.json",
        "v7_law": A5 / "qwen/entropy_qwen.json",
        "field_law": A5 / "field_qwen/entropy_field_qwen.json",
        "site": 21,
    },
}

# source-side vectors of the Leg-3 injection panel (the sheet's rows)
PANEL = ("V7", "Vrep_perp", "Vconf", "Vtemp", "oblique")
# target dials: banked-law name in the field file -> axis key
FIELD_DIALS = {"Vrep": "Vrep_perp", "Veos": "Veos_perp", "Vconf": "Vconf"}
SCORE_DOSE = "+0.3"
DOSES = ("-0.3", "-0.1", "+0.1", "+0.3")


def _law(path: Path, vector: str) -> dict[float, float]:
    d = json.loads(path.read_text())
    return {float(r["alpha_frac"]): float(r["entropy_rise"])
            for r in d["rows"] if r.get("vector") == vector}


def _target_dials(cfg: dict) -> tuple[dict[str, np.ndarray], dict[str, dict], dict]:
    """(unit dial vectors, dial dose-effects, provenance notes)."""
    reads, extras, _ = load_axes(cfg["tgt"])
    vecs = {"V7": reads["V7"].vec}
    effects = {"V7": _law(cfg["v7_law"], "V7")}
    notes = {"V7": f"{cfg['v7_law']}::V7 (the SAME law file the original sheet used)"}
    for law_name, axis in FIELD_DIALS.items():
        vec = reads[axis].vec if axis in reads else extras[axis].vec
        eff = _law(cfg["field_law"], law_name)
        if not eff:
            raise RuntimeError(f"{cfg['field_law']}: no rows for {law_name}")
        vecs[axis] = vec
        effects[axis] = eff
        notes[axis] = f"{cfg['field_law']}::{law_name} (banked ladder, doses {sorted(eff)})"
    # Vtemp: no banked ladder -> F-ii convention substitution (named)
    vt = reads["Vtemp"].vec
    c_vt = cos(vt, vecs["V7"])
    vecs["Vtemp"] = vt
    effects["Vtemp"] = {a: c_vt * r for a, r in effects["V7"].items()}
    notes["Vtemp"] = ("NO BANKED LADDER — pre-ruled fork: effect := cos(Vtemp_tgt, "
                      f"V7_tgt)={c_vt:+.4f} x V7's own law (the F-ii-filed prediction "
                      "convention). Substitution NAMED, not a new measurement.")
    return vecs, effects, notes


def _sheet_rows(cfg: dict) -> dict[str, dict]:
    d = json.loads(Path(cfg["sheet"]).read_text())["predictions"]
    row = next(r for r in d if r["site_pair"] == cfg["anchor"] and r["arm"] == "native"
               and r["family"] == "proc_k512")
    return {r["vector"]: r for r in row["rows"]}


def _observed(cfg: dict) -> dict[str, dict[str, float]]:
    d = json.loads(Path(cfg["scoring"]).read_text())["p8_3ii"]["vectors"]
    return {v: {dose: cell["obs"] for dose, cell in blk["per_dose"].items()}
            for v, blk in d.items()}


def run_pair(pair: str, cfg: dict, basis_keys: tuple[str, ...] | None = None) -> dict:
    tm = load_transport_map(cfg["fit"])
    src_reads, src_extras, _ = load_axes(cfg["src"])
    src = {k: (src_reads[k].vec if k in src_reads else src_extras[k].vec) for k in PANEL}
    dials, effects, notes = _target_dials(cfg)
    if basis_keys:
        dials = {k: v for k, v in dials.items() if k in basis_keys}
        effects = {k: v for k, v in effects.items() if k in basis_keys}
        notes = {k: v for k, v in notes.items() if k in basis_keys}
    names = list(dials)
    D = np.stack([dials[n] for n in names])            # (k, d) unit rows
    G = D @ D.T
    cond = float(np.linalg.cond(G))
    Ginv = np.linalg.pinv(G, rcond=1e-6)
    sheet = _sheet_rows(cfg)
    obs = _observed(cfg)

    rows = []
    for vname, v in src.items():
        u = _unit(tm.transport(v))
        cosv = D @ u                                    # cos(u, dial_i) (unit rows)
        coeff = Ginv @ cosv
        per_dose = {}
        for dose in DOSES:
            a = float(dose)
            terms_a = {n: float(cosv[i] * effects[n].get(a, np.nan))
                       for i, n in enumerate(names)}
            terms_b = {n: float(coeff[i] * effects[n].get(a, np.nan))
                       for i, n in enumerate(names)}
            pa = float(np.nansum(list(terms_a.values())))
            pb = float(np.nansum(list(terms_b.values())))
            o = obs.get(vname, {}).get(dose)
            single = sheet[vname]["predicted_entropy_rise_per_alpha"].get(
                dose.lstrip("+"), None) if vname in sheet else None
            per_dose[dose] = {
                "pred_A_letter_cos_sum": round(pa, 4),
                "pred_B_gram_corrected": round(pb, 4),
                "pred_single_dial_original_sheet": single,
                "obs": o,
                "ratio_obs_over_A": round(o / pa, 3) if o is not None and abs(pa) > 1e-9 else None,
                "ratio_obs_over_B": round(o / pb, 3) if o is not None and abs(pb) > 1e-9 else None,
                "ratio_obs_over_single": (round(o / single, 3) if o is not None
                                          and single not in (None, 0) else None),
                "terms_A": {k: round(t, 4) for k, t in terms_a.items()},
            }
        rows.append({
            "vector": vname,
            "cos_transported_vs_dials": {n: round(float(cosv[i]), 4)
                                         for i, n in enumerate(names)},
            "gram_coefficients": {n: round(float(coeff[i]), 4) for i, n in enumerate(names)},
            "residual_out_of_dial_span": round(
                float(np.linalg.norm(u - (coeff @ D))), 4),
            "per_dose": per_dose,
        })

    def _within(key: str) -> dict:
        hits = {}
        for r in rows:
            c = r["per_dose"][SCORE_DOSE]
            ratio = c[key]
            hits[r["vector"]] = {"ratio": ratio,
                                 "within_x2": (ratio is not None
                                               and 0.5 <= abs(ratio) <= 2.0
                                               and ratio > 0)}
        n = sum(1 for v in hits.values() if v["within_x2"])
        return {"per_vector": hits, "n_within_x2": f"{n}/{len(hits)}"}

    return {
        "pair": pair,
        "basis": names,
        "dial_provenance": notes,
        "dial_gram": {f"{a}|{b}": round(float(G[i, j]), 4)
                      for i, a in enumerate(names) for j, b in enumerate(names) if i < j},
        "gram_condition_number": round(cond, 1),
        "non_orthogonality_note": (
            "cos(Vconf, V7) ~ -0.91 in both dense targets: Reading A double-counts the "
            "shared V7/Vconf component by construction. Reading B is the non-degenerate "
            "least-squares form. BOTH filed; desk picks (Add-1.3)."),
        "rows": rows,
        f"score_dose_{SCORE_DOSE}": {
            "reading_A_letter": _within("ratio_obs_over_A"),
            "reading_B_gram": _within("ratio_obs_over_B"),
            "original_single_dial": _within("ratio_obs_over_single"),
        },
    }


def _md(res: dict) -> str:
    lines = ["# L4-b — field-basis composition re-prediction (UNSTAMPED, C§8)", "",
             "Bar (add-2): obs/(field-basis prediction) within x2, >=4/5 vectors; "
             "reference = the re-predicted sheet. Both readings filed (Add-1.3).", ""]
    for pr in res["pairs"]:
        lines += [f"## {pr['pair']}  (basis {', '.join(pr['basis'])}; "
                  f"Gram cond {pr['gram_condition_number']})", "",
                  "| vector | obs@+.3 | single-dial pred | A (cos-sum) | B (Gram) | "
                  "obs/single | obs/A | obs/B |", "|---|---|---|---|---|---|---|---|"]
        for r in pr["rows"]:
            c = r["per_dose"][SCORE_DOSE]
            lines.append(
                f"| {r['vector']} | {c['obs']} | {c['pred_single_dial_original_sheet']} | "
                f"{c['pred_A_letter_cos_sum']} | {c['pred_B_gram_corrected']} | "
                f"{c['ratio_obs_over_single']} | {c['ratio_obs_over_A']} | "
                f"{c['ratio_obs_over_B']} |")
        s = pr[f"score_dose_{SCORE_DOSE}"]
        lines += ["", f"within x2 at {SCORE_DOSE}: original single-dial "
                      f"**{s['original_single_dial']['n_within_x2']}** · "
                      f"Reading A **{s['reading_A_letter']['n_within_x2']}** · "
                      f"Reading B **{s['reading_B_gram']['n_within_x2']}**", ""]
    return "\n".join(lines) + "\n"


def selftest() -> int:
    ok = True
    # planted world: orthonormal dials, known effects, a vector with known coefficients
    d = 32
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.normal(size=(d, 3)))
    D = Q.T
    eff = np.array([1.0, -0.5, 0.25])
    c_true = np.array([0.6, 0.3, -0.2])
    u = _unit(c_true @ D)
    cosv = D @ u
    G = D @ D.T
    coeff = np.linalg.pinv(G, rcond=1e-6) @ cosv
    pa = float(cosv @ eff)
    pb = float(coeff @ eff)
    ok &= abs(pa - pb) < 1e-9
    print(f"[{'OK' if abs(pa - pb) < 1e-9 else 'BAD'}] orthonormal basis: A == B "
          f"({pa:.6f} vs {pb:.6f})")
    exp = float((c_true / np.linalg.norm(c_true)) @ eff)
    ok &= abs(pb - exp) < 1e-9
    print(f"[{'OK' if abs(pb - exp) < 1e-9 else 'BAD'}] B recovers planted coefficients "
          f"({pb:.6f} vs {exp:.6f})")
    # collinear basis: A double-counts, B does not
    D2 = np.stack([D[0], _unit(D[0] + 0.05 * D[1])])
    eff2 = np.array([1.0, 1.0])
    u2 = D[0]
    cos2 = D2 @ u2
    a2 = float(cos2 @ eff2)
    b2 = float(np.linalg.pinv(D2 @ D2.T, rcond=1e-6) @ cos2 @ eff2)
    ok &= a2 > 1.5 and abs(b2 - 1.0) < 0.2
    print(f"[{'OK' if a2 > 1.5 and abs(b2 - 1.0) < 0.2 else 'BAD'}] near-collinear basis: "
          f"A double-counts ({a2:.3f}) while B stays ~1 ({b2:.3f})")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"STATUS": "UNSTAMPED (C§8) — mechanics only, desk scores P8-L4b",
           "leg": "A8 Leg-4 / L4-b", "prereg": "A8-add-2 P8-L4b",
           "score_dose": SCORE_DOSE,
           "pairs": [run_pair(p, cfg) for p, cfg in PAIRS.items()],
           "variant_banked_ladders_only": {
               "what": "same computation with the Vtemp term REMOVED — basis = the four "
                       "dials that own banked entropy ladders. Isolates the contribution "
                       "of the pre-ruled Vtemp convention substitution.",
               "pairs": [{"pair": p, **{k: v for k, v in
                                        run_pair(p, cfg, ("V7", "Vrep_perp", "Veos_perp",
                                                          "Vconf")).items()
                                        if k in (f"score_dose_{SCORE_DOSE}",
                                                 "gram_condition_number")}}
                         for p, cfg in PAIRS.items()]}}
    (OUT / "l4b_field_basis_composition.json").write_text(json.dumps(res, indent=1))
    (OUT / "l4b_field_basis_composition.md").write_text(_md(res))
    logger.info("wrote %s", OUT / "l4b_field_basis_composition.json")
    print(_md(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
