"""A8 extension-pairs smalls — difficulty-curve rows + the P8-XG / P8-XO clause tables.

Three jobs, all CPU on banked artifacts:

  1. GEMMA FIELDS ROW — the fields series gains a fourth dense rung and the arm's THIRD
     architecture family as a target. Rake-14 discipline: this row and the needle row
     are separate lines, and the needle is NEVER divided by a_hat.
  2. GEMMA NEEDLE ROW — read ONLY against the PAIR-MATCHED L36 vintage. Gemma banks
     dir0 twice under DIFFERENT mode pairs (rake 33, applied prospectively for the
     first time in this arm):
         a5_vectors_gemma3_27b      L23/L35/L41  [socratic, contrastive]  <- WRONG PAIR
         a5_vectors_gemma3_27b_L36  L36          [analogical, contrastive] <- the arm's
     L35 sits ONE LAYER from L36 carrying the wrong contrast. Had the site been picked
     by proximity rather than by the stamp, this row would have been the Leg-6 DSV2
     mistake repeated. Both stamps were read before the registry block was written.
     Primary g and the mode-free g (Add-1.2) both reported; Delta = primary - modefree
     is the watch-constant's next data point.
  3. OLMO FIT-GATE TABLE — OLMo-2-1124-7B is a BASE model: no chat template, so RAW ARM
     ONLY, and NO banked vectors of any kind, so NO a_hat and NO star node. What it can
     answer is P8-XO: does generic-text pairing find real structure between an RLHF'd
     instruct model and an RLHF-free base model? That is a fit-validity claim, and the
     alignment curve (held-out R^2 / CKA by site) is also how OLMo's site of record gets
     picked — from a curve, never by fiat, since no banked site curve exists for it.

P8-XG's letter named the panel {V7, Vrep_perp, Vconf, Vtemp, Veos_raw}. Veos_raw is an
`extras` object, not one of the five standard axis reads, so the standard Rosetta table
does not carry it. Rather than amend the letter (park-don't-amend), this script measures
Veos_raw on the same instrument and scores the clause under BOTH readings — the literal
panel as written, and the standard five-axis panel the instrument actually emits.

UNSTAMPED (C section 8). No P self-scored — the desk scores P8-XG / P8-X1 / P8-XO.

Run (repo root): PYTHONPATH=pipeline python -m anamnesis.scripts.a8_smalls_curverows
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from anamnesis.scripts.a8_fit_g import (
    PCABank, load_labels, load_state_bank, load_transport_map, make_split)
from anamnesis.scripts.a8_rosetta import (
    N_RANDOM_NULLS, TOP_PC_J, _unit, cos, load_axes)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_smalls_curverows")

SMALLS = Path("outputs/battery/arms/A8_conjugation/smalls")
OUT = SMALLS / "readouts_cpu"
SEED = 80

GEMMA = "gemma3-27b"
ANCHOR = {"3b": 14, "8b": 16, GEMMA: 36}
PRIMARY_FAMILY = "proc_k128"        # rank-guarded at n_train 600 (k <= 500)
FAMILIES = ("proc_k32", "proc_k128", "proc_k512", "ridge")
PANEL_LETTER = ("V7", "Vrep_perp", "Vconf", "Vtemp", "Veos_raw")   # as P8-XG wrote it
PANEL_INSTRUMENT = ("V7", "Vrep_perp", "Vconf", "Vtemp", "dir0")   # what Rosetta emits


def _read_block(arm_root: Path, fits_dirname: str, src: str, tgt: str, arm: str,
                family: str, fit_strata: Optional[list[str]] = None) -> Optional[dict]:
    """Axis reads + envelope + top-PC control for one (pair, arm, family)."""
    s_site, t_site = ANCHOR[src], ANCHOR[tgt]
    pair = f"{src}L{s_site}->{tgt}L{t_site}"
    p = arm_root / fits_dirname / f"fit_{src}L{s_site}__{tgt}L{t_site}_{arm}_{family}.npz"
    if not p.exists():
        return None
    tm = load_transport_map(p)
    src_axes, src_extras, src_pool = load_axes(src)
    tgt_axes, tgt_extras, _ = load_axes(tgt)
    src_all = {**src_axes, **src_extras}
    tgt_all = {**tgt_axes, **tgt_extras}

    rng = np.random.default_rng(SEED)
    d_src = src_axes["V7"].vec.shape[0]
    randoms = rng.standard_normal((N_RANDOM_NULLS, d_src))
    randoms /= np.linalg.norm(randoms, axis=1, keepdims=True)
    nulls = ([_unit(tm.transport(r)) for r in randoms]
             + [_unit(tm.transport(a.vec)) for a in src_pool])

    # top-PC control on the SAME split, strata and code path as the fit
    states = arm_root / "states"
    sb, tb = load_state_bank(states, src, arm), load_state_bank(states, tgt, arm)
    labels = load_labels(arm_root / "corpus" / "corpus_manifest.json", sb.text_ids)
    train, _, _ = make_split(labels)
    mask = (np.isin(labels.stratum, fit_strata) if fit_strata
            else np.ones(len(sb.text_ids), bool))
    pcs_s = PCABank.fit(sb.matrix(s_site)[train & mask],
                        TOP_PC_J).components * sb.median_norms[s_site]

    rows: dict[str, dict] = {}
    for name in set(PANEL_LETTER) | set(PANEL_INSTRUMENT):
        if name not in src_all or name not in tgt_all:
            continue
        tgt_vec = tgt_all[name].vec
        c = cos(_unit(tm.transport(src_all[name].vec)), tgt_vec)
        q95 = float(np.quantile([abs(cos(n, tgt_vec)) for n in nulls], 0.95))
        toppc = max(abs(cos(tm.transport(pcs_s[j]), tgt_vec))
                    for j in range(pcs_s.shape[0]))
        rows[name] = {
            "cos": round(c, 4), "null_q95_abs": round(q95, 4),
            "exceeds_envelope": bool(abs(c) > q95),
            "top_pc_max": round(toppc, 4),
            "axis_specific": bool(c > toppc),
            "PASSES_CLAUSE": bool(abs(c) > q95 and c > toppc),
            "target_source": tgt_all[name].source,
        }
    return {"site_pair": pair, "arm": arm, "family": family,
            "fit_strata": fit_strata, "reads": rows}


# The Delta-sharpening series, RE-DERIVED AT ONE FAMILY. The banked Delta values
# (.242 / .236 / .118) are proc_k512 reads; the Gemma mode-free fit has n_train ~370, so
# the rank guard forbids k512 there and the new rung can only be read at k128. Comparing
# a k128 Delta against k512 Deltas would be rake-40 family-mixing inside the very series
# the desk is using to judge a constant. So every rung is re-read at k128 here, and the
# banked k512 column rides beside it. Nobody had made this series family-consistent.
DELTA_RUNGS = {
    "3b->8b": {
        "fits": Path("outputs/battery/arms/A8_conjugation/fits"),
        "fits_mf": Path("outputs/battery/arms/A8_conjugation/fits_modefree"),
        "src": "3b", "s_site": 14, "tgt_tag": "8bL16",
        "tgt": ("a5_vectors_8b/a5_vectors.npz", "V3_L16"),
        "banked_k512_delta": 0.242, "pair_matched": True},
    "8b->qwen-7b": {
        "fits": Path("outputs/battery/arms/A8_conjugation/leg1/fits"),
        "fits_mf": Path("outputs/battery/arms/A8_conjugation/leg1/fits_modefree"),
        "src": "8b", "s_site": 16, "tgt_tag": "qwen-7bL21",
        "tgt": ("a5_vectors_qwen_7b/a5_vectors.npz", "V3_L21"),
        "banked_k512_delta": 0.236, "pair_matched": True},
    "8b->dsv2-lite (PAIR-MATCHED whitened)": {
        "fits": Path("outputs/battery/arms/A8_conjugation/leg2/fits"),
        "fits_mf": Path("outputs/battery/arms/A8_conjugation/leg2/fits_modefree"),
        "src": "8b", "s_site": 16, "tgt_tag": "dsv2-liteL18",
        "tgt": ("a5_vectors_dsv2_lite_v3whiten/a5_vectors.npz", "V3w_L18"),
        "banked_k512_delta": 0.118, "pair_matched": True},
}


def delta_series(family: str = PRIMARY_FAMILY) -> dict:
    """dir0 primary vs mode-free at ONE family, every rung, for comparability."""
    from anamnesis.scripts.a8_rosetta import _load_key
    out: dict = {}
    for label, cfg in DELTA_RUNGS.items():
        src_axes, _, _ = load_axes(cfg["src"])
        dir0 = src_axes["dir0"].vec
        tgt = _unit(_load_key(*cfg["tgt"]))
        name = f"fit_{cfg['src']}L{cfg['s_site']}__{cfg['tgt_tag']}_native_{family}.npz"
        row: dict = {"family": family, "pair_matched": cfg["pair_matched"],
                     "banked_delta_at_k512": cfg["banked_k512_delta"]}
        for tag, d in (("primary", cfg["fits"]), ("modefree", cfg["fits_mf"])):
            p = d / name
            row[tag] = (round(cos(_unit(load_transport_map(p).transport(dir0)), tgt), 4)
                        if p.exists() else None)
        if row.get("primary") is not None and row.get("modefree") is not None:
            row["delta"] = round(row["primary"] - row["modefree"], 4)
        out[label] = row
    return out


def _fit_table(fits_dir: Path) -> dict:
    cp2 = json.loads((fits_dir / "cp2_summary.json").read_text())
    by_pair: dict[str, dict] = {}
    for r in cp2["records"]:
        by_pair.setdefault(f"{r['site_pair']}::{r['arm']}", {})[r["family"]] = {
            "r2": r["r2"], "valid": r["valid"], "carried": r["strata_carried"],
            "cka_before": r["cka_before"], "cka_after": r["cka_after"],
            "null_q95_shuffled": r["r2_null_shuffled_q95"],
            "null_q95_stratum": r["r2_null_stratum_q95"]}
    return {"pair": cp2["pair"], "arms": cp2.get("arms"),
            "n_train": cp2["split"]["n_train"], "n_test": cp2["split"]["n_test"],
            "n_valid": cp2["n_valid"], "n_fits": cp2["n_fits"], "by_pair": by_pair}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    doc: dict = {
        "STATUS": "UNSTAMPED (C section 8) — no P self-scored; the desk scores "
                  "P8-XG (.70), P8-X1 (.55), P8-XO (.65)",
        "leg": "A8 extension-pairs smalls — curve rows + clause tables",
        "prereg": "A8-add-7",
        "rank_guard": "n_train 600 on the hub/second pairs -> k <= 500; proc_k512 is "
                      "RANK-FORBIDDEN and appears only as a flagged beside. Primary "
                      f"family = {PRIMARY_FAMILY} (stated before fitting, rake 40).",
        "rake_14_discipline": "fields series and needle series are SEPARATE lines; the "
                              "needle is never divided by a_hat",
    }

    # ---- 1/2. Gemma: fields + needle, primary and mode-free -------------------
    gemma: dict = {}
    for label, (fits, strata) in {
        "primary_g (S1+S2+S3)": ("fits_gemma", None),
        "mode_free_g (S1+S2; never saw a mode contrast)":
            ("fits_gemma_modefree", ["S1", "S2"]),
    }.items():
        blk: dict = {}
        for family in FAMILIES:
            r = _read_block(SMALLS, fits, "8b", GEMMA, "native", family, strata)
            if r:
                blk[family] = r
        if blk:
            gemma[label] = blk
    doc["gemma_8b_hub_native"] = gemma

    prim = gemma.get("primary_g (S1+S2+S3)", {}).get(PRIMARY_FAMILY, {}).get("reads", {})
    mf = next((v.get(PRIMARY_FAMILY, {}).get("reads", {})
               for k, v in gemma.items() if k.startswith("mode_free")), {})

    if prim:
        doc["P8_XG_CLAUSE_TABLE"] = {
            "clause": "at least 3 of 5 field objects above envelope AND top-PC below "
                      "the read; native arm, forward, proc_k128, 8bL16->gemmaL36",
            "reading_LITERAL (panel as the letter wrote it, incl. Veos_raw)": {
                "panel": list(PANEL_LETTER),
                "per_axis": {a: prim.get(a, {}).get("PASSES_CLAUSE")
                             for a in PANEL_LETTER},
                "n_passing": sum(bool(prim.get(a, {}).get("PASSES_CLAUSE"))
                                 for a in PANEL_LETTER),
            },
            "reading_INSTRUMENT (the five axes Rosetta actually emits)": {
                "panel": list(PANEL_INSTRUMENT),
                "per_axis": {a: prim.get(a, {}).get("PASSES_CLAUSE")
                             for a in PANEL_INSTRUMENT},
                "n_passing": sum(bool(prim.get(a, {}).get("PASSES_CLAUSE"))
                                 for a in PANEL_INSTRUMENT),
            },
            "why_two_readings": "P8-XG's panel named Veos_raw, which is an `extras` "
                                "object the standard axis table does not carry. Both "
                                "readings are reported rather than the letter amended "
                                "(park-don't-amend); the desk picks.",
            "fit_clause": "72/72 fits VALID on this pair (both arms, all site pairs) — "
                          "see olmo_and_gemma_fit_tables",
            "SCORING": "reported, NOT self-scored",
        }
        doc["GEMMA_NEEDLE_ROW"] = {
            "pair_matching": "PAIR-MATCHED. 8B dir0 = [analogical, contrastive]; "
                             "gemma L36 V3 = [analogical, contrastive] (stamp-verified "
                             "BEFORE the read, rake 33). The other gemma dir0 vintage "
                             "(L23/L35/L41) is [socratic, contrastive] and would have "
                             "been the WRONG contrast — L35 is one layer from L36.",
            "dir0_primary": prim.get("dir0", {}).get("cos"),
            "dir0_modefree": mf.get("dir0", {}).get("cos"),
            "delta_sharpening": (
                round(prim["dir0"]["cos"] - mf["dir0"]["cos"], 4)
                if "dir0" in prim and "dir0" in mf else None),
            "watch_constant_context": {
                "3b->8b": 0.242, "8b->qwen": 0.236, "8b->dsv2 (pair-matched)": 0.118,
                "desk_form_of_record": "DEMOTED, not retired (DESK-RULINGS-LEG6 §4): "
                                       "sharpening EXISTS at every rung but is not a "
                                       "constant — it attenuates with rung difficulty.",
            },
        }

    # ---- 2b. the Delta series, made family-consistent -------------------------
    series = delta_series(PRIMARY_FAMILY)
    if prim and "dir0" in prim and "dir0" in mf:
        series["8b->gemma3-27b (NEW, PAIR-MATCHED)"] = {
            "family": PRIMARY_FAMILY, "pair_matched": True,
            "banked_delta_at_k512": None,
            "primary": prim["dir0"]["cos"], "modefree": mf["dir0"]["cos"],
            "delta": round(prim["dir0"]["cos"] - mf["dir0"]["cos"], 4)}
    doc["DELTA_SHARPENING_SERIES_at_one_family"] = {
        "why": "the banked Delta values (.242/.236/.118) are proc_k512 reads; Gemma's "
               "mode-free fit has n_train ~370, where the add-3 rank guard forbids "
               "k512. Comparing a k128 Delta to k512 Deltas would be rake-40 "
               "family-mixing inside the very series the desk uses to judge the "
               f"constant. Every rung is therefore re-read at {PRIMARY_FAMILY}.",
        "rows": series,
        "desk_form_of_record_being_tested":
            "DESK-RULINGS-LEG6 §4: 'contrast-data sharpening EXISTS at every rung but "
            "is NOT a constant — it attenuates with rung difficulty (dense .242/.236 "
            "-> MoE ~.11)'. The Gemma rung is the first new data point since that "
            "demotion. Reported, not adjudicated.",
    }

    # ---- 3. OLMo + Gemma fit tables ------------------------------------------
    tables = {}
    for tag, d in (("gemma", SMALLS / "fits_gemma"), ("olmo", SMALLS / "fits_olmo"),
                   ("gemma_3b_second_pair", SMALLS / "fits_gemma_3b")):
        if (d / "cp2_summary.json").exists():
            tables[tag] = _fit_table(d)
    doc["olmo_and_gemma_fit_tables"] = tables

    if "olmo" in tables:
        t = tables["olmo"]
        curve = {k: v.get(PRIMARY_FAMILY, {}).get("r2") for k, v in t["by_pair"].items()}
        best = max((v, k) for k, v in curve.items() if v is not None)
        doc["P8_XO_CLAUSE_TABLE"] = {
            "clause": "the 8B<->OLMo fit passes the frozen validity gates (both nulls "
                      "+ >=2 strata carried) on the RAW arm at >=1 site pair",
            "arm": "raw ONLY — OLMo-2-1124-7B is a base model with no chat template, so "
                   "its native arm does not exist. A8-add-7.1 binds: any constant "
                   "derived here would live in the raw-arm system, never the native one.",
            "n_valid_of_n_fits": f"{t['n_valid']}/{t['n_fits']}",
            "site_pairs_all_valid": bool(t["n_valid"] == t["n_fits"]),
            "alignment_curve_r2_primary_family": curve,
            "site_of_record_FROM_THE_CURVE": {"site_pair": best[1], "r2": best[0]},
            "NO_A_HAT": "a_hat(8B->OLMo) is NOT MEASURABLE: OLMo has no banked V7 (no "
                        "a5 section-B.7 build was ever run for it) and no banked vector "
                        "of any kind. No star node, no c_OLMo, in either arm. Declared "
                        "in advance in A8-add-7, not discovered here.",
            "SCORING": "reported, NOT self-scored",
        }

    (OUT / "smalls_curverows.json").write_text(json.dumps(doc, indent=1))

    if prim:
        logger.info("GEMMA fields (native, %s, 8bL16->gemmaL36):", PRIMARY_FAMILY)
        for a in ("V7", "Vrep_perp", "Vconf", "Vtemp", "Veos_raw", "dir0"):
            r = prim.get(a)
            if r:
                logger.info("  %-11s cos %+.4f  q95 %.4f  topPC %.4f  env=%s axspec=%s",
                            a, r["cos"], r["null_q95_abs"], r["top_pc_max"],
                            r["exceeds_envelope"], r["axis_specific"])
        n = doc["GEMMA_NEEDLE_ROW"]
        logger.info("NEEDLE primary %s / modefree %s / delta %s",
                    n["dir0_primary"], n["dir0_modefree"], n["delta_sharpening"])
    for tag, t in tables.items():
        logger.info("FITS %-22s %s  valid %d/%d  n_train %d", tag, t["pair"],
                    t["n_valid"], t["n_fits"], t["n_train"])
    logger.info("wrote %s", OUT / "smalls_curverows.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
