"""A8 extension-pairs smalls — hub constants, star PREDICTIONS, and the verify pass.

PHASE-ORDERED BY CONSTRUCTION (rake 29's lesson, applied here as a CLI gate rather than
as an intention): the star prediction for a pair must be written to disk BEFORE that
pair is fitted. This script refuses to do both in one invocation.

  --phase predict   read a_hat(8B -> M) from the HUB fit only; derive c_M inside its own
                    (arm x family) system per A8-add-7.1; emit the +-.05 band for every
                    other pair involving M. REFUSES to run if the second pair's fits
                    already exist — a prediction filed after the fact is not a prediction.
  --phase verify    re-read the frozen prediction file and add the observed a_hat for the
                    second pair. Never recomputes the prediction; if the prediction file
                    is absent it refuses rather than back-filling one.

The arm-consistency rule (A8-add-7.1): c_M lives in the system its a_hat was measured
in. Gemma has a chat template, so it gets both arms and its constant of record is the
NATIVE one. A raw-arm-only model's constant may only ever be combined with raw-arm
constants — and a lower raw-arm a_hat is ARM-CONFOUNDED before it is a transport fact.

UNSTAMPED (C section 8). No P self-scored — the desk scores P8-XG / P8-X1 / P8-XO.

Run (repo root):
  PYTHONPATH=pipeline python -m anamnesis.scripts.a8_smalls_extension_rows --phase predict
  PYTHONPATH=pipeline python -m anamnesis.scripts.a8_smalls_extension_rows --phase verify
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from anamnesis.scripts.a8_fit_g import load_transport_map
from anamnesis.scripts.a8_rosetta import _unit, cos
from anamnesis.scripts.a8_smalls_star_systems import (
    ARMS, FAMILIES, RANK_GUARD_DIVISOR, V7, _family_k, _load_v7, _n_train)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_smalls_extension_rows")

ARM = Path("outputs/battery/arms/A8_conjugation")
SMALLS = ARM / "smalls"
OUT = SMALLS / "readouts_cpu"
SYSTEMS = OUT / "star_systems.json"
PREDICTION_FILE = OUT / "star_predictions_A8-add-7.json"

NEW_MODEL = "gemma3-27b"
HUB = "8b"
SECOND = "3b"                      # the free out-of-sample node
FITS_HUB = SMALLS / "fits_gemma"           # 8b -> gemma
FITS_SECOND = SMALLS / "fits_gemma_3b"     # 3b -> gemma (must not exist at predict time)
BAND_HALFWIDTH = 0.05              # add-4 rule
SYSTEM_OF_RECORD = ("native", "proc_k128")
PARALLEL_SYSTEM = ("raw", "proc_k128")


def _a_hat(fits_dir: Path, src: str, tgt: str, arm: str, family: str
           ) -> Optional[dict]:
    s_site, t_site = V7[src][2], V7[tgt][2]
    p = fits_dir / f"fit_{src}L{s_site}__{tgt}L{t_site}_{arm}_{family}.npz"
    if not p.exists():
        return None
    n_train = _n_train(fits_dir)
    max_k = (n_train / RANK_GUARD_DIVISOR) if n_train else None
    k = _family_k(family)
    tm = load_transport_map(p)
    return {
        "site_pair": f"{src}L{s_site}->{tgt}L{t_site}",
        "value": round(cos(_unit(tm.transport(_load_v7(src))), _load_v7(tgt)), 4),
        "n_train": n_train,
        "rank_forbidden": bool(max_k is not None and k is not None and k > max_k),
    }


def _usable_systems() -> dict:
    if not SYSTEMS.exists():
        raise SystemExit(f"run a8_smalls_star_systems first: {SYSTEMS} absent")
    return json.loads(SYSTEMS.read_text())["systems"]


def phase_predict() -> int:
    if FITS_SECOND.exists() and any(FITS_SECOND.glob("fit_*.npz")):
        raise SystemExit(
            f"REFUSING: {FITS_SECOND} already holds fits. A star prediction filed "
            "after its own test has been fitted is not a prediction (A8-add-7.3).")
    systems = _usable_systems()
    OUT.mkdir(parents=True, exist_ok=True)

    hub: dict = {}
    for arm in ARMS:
        for family in FAMILIES:
            cell = _a_hat(FITS_HUB, HUB, NEW_MODEL, arm, family)
            if cell:
                hub.setdefault(arm, {})[family] = cell

    derived: dict = {}
    predictions: dict = {}
    for arm, family in (SYSTEM_OF_RECORD, PARALLEL_SYSTEM):
        key = f"{arm}::{family}"
        sysblk = systems.get(key)
        cell = hub.get(arm, {}).get(family)
        if sysblk is None or cell is None:
            continue
        chk = sysblk.get("out_of_sample_check", {})
        if not chk.get("within_pm_0.05", False):
            derived[key] = {"REFUSED": "system fails its own out-of-sample check "
                                       "(A8-add-7.2) — no constant may hang on it"}
            continue
        c_hub = sysblk["constants"]["c_8B"]
        c_new = cell["value"] / c_hub
        derived[key] = {
            "a_hat_hub": cell["value"], "site_pair": cell["site_pair"],
            "c_8B_of_this_system": c_hub,
            f"c_{NEW_MODEL}": round(c_new, 4),
            "rank_forbidden": cell["rank_forbidden"],
        }
        c_second = sysblk["constants"]["c_3B"]
        pred = c_second * c_new
        predictions[key] = {
            "pair": f"{SECOND}->{NEW_MODEL}",
            "formula": f"c_3B ({c_second}) x c_{NEW_MODEL} ({round(c_new, 4)})",
            "predicted": round(pred, 4),
            "band": [round(pred - BAND_HALFWIDTH, 4), round(pred + BAND_HALFWIDTH, 4)],
            "SCOPE": "forward fit direction, single a_hat number, no panel; V7 anchor "
                     "sites (3bL14 -> gemma3-27bL36); this arm and this family only",
        }

    doc = {
        "STATUS": "UNSTAMPED (C section 8) — FROZEN PREDICTION, filed before the "
                  f"{SECOND}->{NEW_MODEL} fit was submitted. No P self-scored.",
        "prereg": "A8-add-7 (P8-X1 .55)",
        "phase": "predict",
        "arm_consistency_rule": "A8-add-7.1 — c_M belongs to the (arm x family) system "
                                "its a_hat was measured in; never mixed across systems.",
        "hub_a_hats_all_cells": hub,
        "derived_constants": derived,
        "PREDICTIONS": predictions,
        "verification_hook": "re-run with --phase verify AFTER the second pair is "
                             "fitted; this file is read, never rewritten.",
    }
    PREDICTION_FILE.write_text(json.dumps(doc, indent=1))
    for k, v in derived.items():
        logger.info("DERIVED %-18s %s", k, v)
    for k, v in predictions.items():
        logger.info("PREDICT %-18s %s -> %.4f band %s", k, v["pair"], v["predicted"],
                    v["band"])
    logger.info("FROZEN -> %s", PREDICTION_FILE)
    return 0


def phase_verify() -> int:
    if not PREDICTION_FILE.exists():
        raise SystemExit(
            f"REFUSING: {PREDICTION_FILE} absent. The prediction must have been filed "
            "before the fit; back-filling one now would be fabrication.")
    doc = json.loads(PREDICTION_FILE.read_text())
    verdicts: dict = {}
    for key, pred in doc["PREDICTIONS"].items():
        arm, family = key.split("::")
        cell = _a_hat(FITS_SECOND, SECOND, NEW_MODEL, arm, family)
        if cell is None:
            verdicts[key] = {"observed": None, "note": "second-pair fit absent"}
            continue
        lo, hi = pred["band"]
        verdicts[key] = {
            "predicted": pred["predicted"], "band": pred["band"],
            "observed": cell["value"], "site_pair": cell["site_pair"],
            "abs_error": round(abs(cell["value"] - pred["predicted"]), 4),
            "IN_BAND": bool(lo <= cell["value"] <= hi),
            "rank_forbidden": cell["rank_forbidden"],
            "SCORING": "reported, NOT self-scored — the desk rules P8-X1",
        }

    # full-family beside, so the family choice is visible rather than asserted
    beside: dict = {}
    for arm in ARMS:
        for family in FAMILIES:
            cell = _a_hat(FITS_SECOND, SECOND, NEW_MODEL, arm, family)
            if cell:
                beside.setdefault(arm, {})[family] = cell

    out = {
        "STATUS": "UNSTAMPED (C section 8) — verify pass; the prediction file was "
                  "read, never rewritten",
        "prereg": "A8-add-7 (P8-X1 .55)",
        "phase": "verify",
        "frozen_prediction_file": str(PREDICTION_FILE),
        "VERDICTS": verdicts,
        "observed_all_cells_beside": beside,
    }
    (OUT / "star_fifth_node_verify.json").write_text(json.dumps(out, indent=1))
    for k, v in verdicts.items():
        logger.info("VERIFY %-18s pred %s band %s obs %s IN_BAND=%s", k,
                    v.get("predicted"), v.get("band"), v.get("observed"),
                    v.get("IN_BAND"))
    logger.info("wrote %s", OUT / "star_fifth_node_verify.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=("predict", "verify"), required=True)
    args = ap.parse_args()
    return phase_predict() if args.phase == "predict" else phase_verify()


if __name__ == "__main__":
    raise SystemExit(main())
