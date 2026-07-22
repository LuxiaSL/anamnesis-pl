"""A8 Leg-3 / P8-2 scoring — merge entropy-readout shards, score against the FROZEN letters.

Mechanics only; the desk decides P's (C§8, no self-scored P's). Frozen criteria applied:
  P8-4  (per dense target): transported V7 writes the target's entropy DOSE-ORDERED and
        OUTSIDE the transported-R band (controls = source Rbands through the SAME g).
  P8-3ii (per pair, 5 test vectors {V7,Vrep_perp,Vconf,Vtemp,oblique}): sign correct on
        >=4/5 AND magnitude within x2 of the target's own law. Scoring dose = +/-0.3
        (largest frozen signal); ALL doses filed. NEAR-ZERO SEMANTICS FLAGGED, not
        legislated: for a near-zero predicted rise the ratio-based x2 band punishes a
        correct ~nothing (obs .001 vs pred .018 -> ratio .06 "fails"); both the ratio
        and the absolute deviation are filed and the desk reads the row.
  P8-2  (dsv2): raw u=unit(g.V7) landing FAILS and whitened w=unit(Sigma^-1 u) PASSES —
        each arm read against ITS OWN matched null family (gRband vs wRband; the merged
        add_null_ratios pooling would mix families, so bands here are computed per
        family explicitly).
  P8-2d: the alignment diagnostic (cos(g.V7, V7_tgt), transported) must ORDER landing
        success across >=3 targets.

Inputs: leg3/readouts/entropy_{model}_shard*.json (node-side shards, rsynced local).
Outputs: leg3/readouts_final/entropy_{model}_merged.json (rows + guarded null columns)
         + scoring_{model}.json + scoring_summary.json (+ .md table).

Run (repo root): PYTHONPATH=pipeline python -m anamnesis.scripts.a8_leg3_score
"""
from __future__ import annotations

import json
import logging
import sys
from glob import glob
from pathlib import Path

import numpy as np

from anamnesis.scripts.vmb_c3_entropy_replay import add_null_ratios

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_leg3_score")

ARM = Path("outputs/battery/arms/A8_conjugation")
LEG3 = ARM / "leg3"
DOSES = (-0.3, -0.1, -0.03, 0.03, 0.1, 0.3)
TEST_VECTORS = ("V7", "Vrep_perp", "Vconf", "Vtemp", "oblique")
CELL_VEC = {"V7": "gV7", "Vrep_perp": "gVrep_perp", "Vconf": "gVconf",
            "Vtemp": "gVtemp", "oblique": "goblique"}

TARGETS = {
    "8b": {"site": 16, "nulls": {"raw": ("gRband1", "gRband2", "gRband3")},
           "answer_key": ARM / "readouts/f2_predictions.json"},
    "qwen-7b": {"site": 21, "nulls": {"raw": ("gRband1", "gRband2", "gRband3")},
                "answer_key": ARM / "leg1/readouts/f2_predictions.json"},
    "dsv2-lite": {"site": 22, "nulls": {"raw": ("gRband1", "gRband2", "gRband3"),
                                        "whitened": ("wRband1", "wRband2", "wRband3")},
                  "answer_key": None},
}


def parse_cell(cell: str) -> tuple[str, int, float]:
    """'gVrep_perp_L21_a+0.03' -> ('gVrep_perp', 21, 0.03)"""
    stem, frac = cell.rsplit("_a", 1)
    vname, site = stem.rsplit("_L", 1)
    return vname, int(site), float(frac)


def merge_shards(model: str) -> list[dict]:
    paths = sorted(glob(str(LEG3 / "readouts" / f"entropy_{model}_shard*.json")))
    if not paths:
        raise FileNotFoundError(f"no shards for {model} under {LEG3/'readouts'}")
    rows, seen = [], set()
    for p in paths:
        for r in json.loads(Path(p).read_text())["rows"]:
            if r["cell"] in seen:
                raise ValueError(f"duplicate cell {r['cell']} across shards ({p})")
            seen.add(r["cell"])
            for k in list(r):          # strip per-shard partial null columns
                if k.endswith("_over_Rc") or k.endswith("_vs_Rc_band"):
                    del r[k]
            vname, site, frac = parse_cell(r["cell"])
            r["vector"] = vname       # fix the split('_')[0] truncation for _perp names
            r["alpha_frac"] = frac
            rows.append(r)
    logger.info("%s: merged %d rows from %d shards", model, len(rows), len(paths))
    return rows


def rises_by_vector(rows: list[dict]) -> dict[str, dict[float, dict]]:
    out: dict[str, dict[float, dict]] = {}
    for r in rows:
        out.setdefault(r["vector"], {})[r["alpha_frac"]] = r
    return out


def band(rows_by_vec: dict, null_names: tuple, frac: float) -> dict:
    vals = [rows_by_vec[n][frac]["entropy_rise"] for n in null_names
            if n in rows_by_vec and frac in rows_by_vec[n]]
    return {"n": len(vals), "min": round(min(vals), 4), "max": round(max(vals), 4),
            "mean": round(float(np.mean(vals)), 4)} if vals else {"n": 0}


def landing_read(rows_by_vec: dict, vec: str, null_names: tuple) -> dict:
    """Dose-ordering + per-dose outside-band for one vector vs one null family."""
    doses = [d for d in DOSES if vec in rows_by_vec and d in rows_by_vec[vec]]
    rises = [rows_by_vec[vec][d]["entropy_rise"] for d in doses]
    from scipy.stats import spearmanr
    rho = float(spearmanr(doses, rises).statistic) if len(doses) >= 3 else None
    per_dose = {}
    outside = 0
    for d, rise in zip(doses, rises):
        b = band(rows_by_vec, null_names, d)
        is_out = bool(b["n"] and (rise < b["min"] or rise > b["max"]))
        outside += is_out
        per_dose[f"{d:+g}"] = {"rise": rise, "band": b, "outside_band": is_out}
    return {"doses_present": len(doses), "spearman_rho_dose_vs_rise": rho,
            "monotone_nondecreasing": bool(all(b >= a for a, b in zip(rises, rises[1:]))),
            "outside_band_doses": f"{outside}/{len(doses)}", "per_dose": per_dose}


def f2_scoring(rows_by_vec: dict, key_path: Path) -> dict:
    """P8-3ii mechanics vs the frozen native proc_k512 prediction row."""
    frozen = json.loads(key_path.read_text())["predictions"]
    fr = next(r for r in frozen if r["arm"] == "native" and r["family"] == "proc_k512")
    out = {"answer_key": str(key_path), "frozen_row": "native/proc_k512", "vectors": {}}
    signs_ok = 0
    for tv in TEST_VECTORS:
        pred_row = next(r for r in fr["rows"] if r["vector"] == tv)
        cell_v = CELL_VEC[tv]
        per_dose = {}
        for d in DOSES:
            k = f"{d:g}"
            pred = pred_row["predicted_entropy_rise_per_alpha"].get(k)
            obs = rows_by_vec.get(cell_v, {}).get(d, {}).get("entropy_rise")
            if pred is None or obs is None:
                continue
            per_dose[f"{d:+g}"] = {"pred": pred, "obs": obs,
                                   "abs_dev": round(abs(obs - pred), 4),
                                   "ratio_obs_over_pred": (round(obs / pred, 3)
                                                           if abs(pred) > 1e-9 else None)}
        top = per_dose.get("+0.3", {})
        sign_match = (top and np.sign(top["obs"]) == np.sign(top["pred"])
                      and abs(top["pred"]) > 1e-9)
        ratio = top.get("ratio_obs_over_pred")
        within_x2 = bool(ratio is not None and 0.5 <= abs(ratio) <= 2.0)
        near_zero = bool(abs(top.get("pred", 0)) < 0.05)
        signs_ok += bool(sign_match)
        out["vectors"][tv] = {
            "cos_g_v__V7_tgt_frozen": pred_row["cos_g_v__V7_tgt"],
            "per_dose": per_dose,
            "sign_match_at_+0.3": bool(sign_match),
            "magnitude_within_x2_at_+0.3": within_x2,
            "near_zero_prediction_row": near_zero,
            "near_zero_note": ("x2-ratio semantics ambiguous for near-zero predictions; "
                               "abs_dev filed beside — desk reads" if near_zero else None),
        }
    out["sign_correct_count"] = f"{signs_ok}/{len(TEST_VECTORS)}"
    out["p8_3ii_sign_leg"] = bool(signs_ok >= 4)
    return out


def main() -> int:
    out_dir = LEG3 / "readouts_final"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"grade": "UNSTAMPED (C§8) — mechanical scoring vs frozen letters; "
                        "desk decides P's", "prereg_tag": "prereg-arm8-v1",
               "builder": "a8_leg3_score.py", "targets": {}}
    landing_metrics = {}

    for model, cfg in TARGETS.items():
        try:
            rows = merge_shards(model)
        except FileNotFoundError as e:
            logger.warning("skip %s: %s", model, e)
            summary["targets"][model] = {"status": f"SHARDS MISSING: {e}"}
            continue
        add_null_ratios(rows, ("GRBAND",) if "whitened" not in cfg["nulls"]
                        else ("GRBAND", "WRBAND"))
        merged = {"model": model, "n_rows": len(rows),
                  "null_pooling_note": ("dsv2: add_null_ratios pools g+w null families "
                                        "by (site,alpha); scoring bands below are "
                                        "per-family explicit" if model == "dsv2-lite"
                                        else None),
                  "rows": rows}
        (out_dir / f"entropy_{model}_merged.json").write_text(json.dumps(merged, indent=1))

        expected = 60 if model == "dsv2-lite" else 48
        by_vec = rises_by_vector(rows)
        rec: dict = {"n_rows": len(rows), "expected_rows": expected,
                     "complete": len(rows) == expected}
        if len(rows) != expected:
            rec["status"] = (f"PARTIAL ({len(rows)}/{expected} cells) — shards still "
                             "landing; numbers below are NOT the read of record")
        if model != "dsv2-lite":
            rec["p8_4_landing_V7"] = landing_read(by_vec, "gV7", cfg["nulls"]["raw"])
            rec["p8_3ii"] = f2_scoring(by_vec, cfg["answer_key"])
            top = rec["p8_4_landing_V7"]["per_dose"].get("+0.3", {})
            landing_metrics[model] = {
                "diagnostic_cos": rec["p8_3ii"]["vectors"]["V7"]["cos_g_v__V7_tgt_frozen"],
                "rise_at_+0.3": top.get("rise"),
                "outside_band_doses": rec["p8_4_landing_V7"]["outside_band_doses"],
                "spearman": rec["p8_4_landing_V7"]["spearman_rho_dose_vs_rise"]}
        else:
            rec["p8_2_raw_landing_gV7"] = landing_read(by_vec, "gV7", cfg["nulls"]["raw"])
            rec["p8_2_whitened_landing_wV7"] = landing_read(by_vec, "wV7",
                                                            cfg["nulls"]["whitened"])
            rec["exploratory_beside_Vconf"] = {
                "raw": landing_read(by_vec, "gVconf", cfg["nulls"]["raw"]),
                "whitened": landing_read(by_vec, "wVconf", cfg["nulls"]["whitened"]),
                "note": "NOT letter-scoring (bank stamp: exploratory-beside)"}
            top = rec["p8_2_raw_landing_gV7"]["per_dose"].get("+0.3", {})
            landing_metrics[model] = {
                "diagnostic_cos": 0.2659,   # identity-checked vs prep diagnostics
                "rise_at_+0.3": top.get("rise"),
                "outside_band_doses": rec["p8_2_raw_landing_gV7"]["outside_band_doses"],
                "spearman": rec["p8_2_raw_landing_gV7"]["spearman_rho_dose_vs_rise"]}
        (out_dir / f"scoring_{model}.json").write_text(json.dumps(rec, indent=1))
        summary["targets"][model] = rec

    if len(landing_metrics) >= 3:
        from scipy.stats import spearmanr
        ms = sorted(landing_metrics.items(), key=lambda kv: -kv[1]["diagnostic_cos"])
        diag = [kv[1]["diagnostic_cos"] for kv in ms]
        rises = [kv[1]["rise_at_+0.3"] for kv in ms]
        if all(r is not None for r in rises):
            summary["p8_2d_ordering"] = {
                "targets_by_diagnostic": [kv[0] for kv in ms],
                "diagnostic_cos": diag, "rise_at_+0.3_raw_transported_V7": rises,
                "spearman_diag_vs_rise": round(float(spearmanr(diag, rises).statistic), 3),
                "note": "n=3 ordering; landing-success currency = raw transported-V7 "
                        "rise@+0.3 (dsv2 raw arm — the diagnostic predicts RAW landing)"}
    (out_dir / "scoring_summary.json").write_text(json.dumps(summary, indent=1))
    logger.info("scoring filed -> %s", out_dir / "scoring_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
