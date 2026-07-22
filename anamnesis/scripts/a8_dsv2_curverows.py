"""A8 — dsv2 difficulty-curve rows + the Δ-watch third instance (CPU, banked inputs).

Fills the 8B↔DSV2 row of DIFFICULTY-CURVE-provisional (rake-14 discipline: fields and
needle series stay separate; needle never divided by â) and measures the third
instance of the Δ≈.24 sharpening watch-constant (Add-1.2 standing instrument):

  fields row  — â(g) = cos(g·V7_8B, V7_dsv2@L22) via the anchor fit (primary);
                Vtemp beside via the L18 fit (Vtemp_L18 is the banked dsv2 analog;
                Vrep⊥/Vconf have NO banked dsv2 targets — cells stay empty).
  needle row  — dir0: cos(g·dir0_8B, V3_L18_dsv2) under the PRIMARY g (S1+S2+S3 fit)
                vs the MODE-FREE g (S1+S2 only), same site pair, native proc_k512.
                Δ = primary − mode-free.

Output: leg3/readouts_cpu/dsv2_curverows.json (rows appended to the curve doc at
close-out by hand, with this file as the raw artifact).

Run (repo root): PYTHONPATH=pipeline python -m anamnesis.scripts.a8_dsv2_curverows
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from anamnesis.scripts.a8_fit_g import load_transport_map
from anamnesis.scripts.a8_rosetta import _load_key, _unit, cos, load_axes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_dsv2_curverows")

ARM = Path("outputs/battery/arms/A8_conjugation")
OUT = ARM / "leg3" / "readouts_cpu"
FITS = ARM / "leg2/fits"
FITS_MF = ARM / "leg2/fits_modefree"
TGT = {
    "V7_L22": ("a5_vectors_dsv2_lite_b7_L22/a5_vectors.npz", "V7_L22"),
    "dir0_L18": ("a5_vectors_dsv2_lite/a5_vectors.npz", "V3_L18"),
    "Vtemp_L18": ("a5_vectors_dsv2_lite_vtemp/a5_vectors.npz", "Vtemp_L18"),
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    axes, extras, _ = load_axes("8b")
    src = {**axes, **extras}
    tgt = {k: _unit(_load_key(rel, key)) for k, (rel, key) in TGT.items()}

    def read(fits_dir: Path, fit_name: str, src_name: str, tgt_name: str) -> float:
        tm = load_transport_map(fits_dir / fit_name)
        return round(cos(_unit(tm.transport(src[src_name].vec)), tgt[tgt_name]), 4)

    fit22 = "fit_8bL16__dsv2-liteL22_native_proc_k512.npz"
    fit18 = "fit_8bL16__dsv2-liteL18_native_proc_k512.npz"

    fields = {
        "a_hat_V7_L22_primary": read(FITS, fit22, "V7", "V7_L22"),
        "a_hat_V7_L22_modefree": read(FITS_MF, fit22, "V7", "V7_L22"),
        "Vtemp_L18_primary": read(FITS, fit18, "Vtemp", "Vtemp_L18"),
        "Vrep_perp": "NO BANKED dsv2 TARGET — cell empty",
        "Vconf": "NO BANKED dsv2 TARGET — cell empty (raw read vs V7_L22 lives in "
                 "leg2_diagnostics: -.246, collapse-structure-visible-raw)",
        "fit_R2_anchor_native": 0.5393,
    }
    needle_primary = read(FITS, fit18, "dir0", "dir0_L18")
    needle_modefree = read(FITS_MF, fit18, "dir0", "dir0_L18")
    needle = {
        "site_pair": "8bL16->dsv2-liteL18 (dir0 banked at L18 = V3_L18)",
        "dir0_primary": needle_primary,
        "dir0_modefree": needle_modefree,
        "delta_sharpening": round(needle_primary - needle_modefree, 4),
        "watch_constant_context": {"3b->8b": 0.242, "8b->qwen": 0.236,
                                   "rule": "desk-filed n=2; third instance ~.24 = "
                                           "something structural conserved"},
        "consistency": "matches leg2 prep diagnostics exactly (-.0865 same fit; "
                       "flat across source sites L14/L16/L18: -.081/-.087/-.090)",
        "v3w_beside_read": "PARKED — V3w (whitened dir0, the known DSV2 CAA rescue) "
                           "has NO banked vector npz (stamps only, whiten_dir0_stamps"
                           ".json); raw V3_L18 may itself be the mean-diff object the "
                           "2026-07-19 anomaly showed is ⊥ discriminative on DSV2. "
                           "The needle-cliff read is therefore RAW-TARGET-scoped; "
                           "whitened-target needle read = baton item for the "
                           "whitened-landing design.",
    }
    doc = {
        "grade": "UNSTAMPED (C§8) — difficulty-curve raw artifact, rake-14 discipline "
                 "(fields/needle series separate; needle never divided by a_hat)",
        "prereg_tag": "prereg-arm8-v1", "builder": "a8_dsv2_curverows.py",
        "date": "2026-07-22", "pair": "8b->dsv2-lite (dense->MoE, regime rung)",
        "fields_series": fields, "needle_series": needle,
    }
    (OUT / "dsv2_curverows.json").write_text(json.dumps(doc, indent=1))
    logger.info("curve rows filed -> %s", OUT / "dsv2_curverows.json")
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
