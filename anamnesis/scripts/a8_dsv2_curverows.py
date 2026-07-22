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

Output: leg3/readouts_cpu/dsv2_curverows_allfam.json — one block per fit family
(proc_k32/proc_k128/proc_k512/ridge). The proc_k512 block reproduces the banked
single-family dsv2_curverows.json bit-for-bit; the other blocks bank the same rows
under the alternate fit families (e.g. proc_k128 for the P8-STAR rider). Rows are
appended to the curve doc at close-out by hand, with this file as the raw artifact.

Run (repo root): PYTHONPATH=pipeline python -m anamnesis.scripts.a8_dsv2_curverows
                 [--families proc_k32,proc_k128,proc_k512,ridge]
"""
from __future__ import annotations

import argparse
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
DEFAULT_FAMILIES = ("proc_k32", "proc_k128", "proc_k512", "ridge")


def fit_name(tlevel: int, family: str) -> str:
    """Anchor-family fit filename for the 8bL16 -> dsv2-lite native site pairs."""
    return f"fit_8bL16__dsv2-liteL{tlevel}_native_{family}.npz"


def anchor_r2(family: str) -> float | None:
    """Native L22 anchor-fit R2 for `family`, from the leg-2 grid summary (or None)."""
    summ = FITS / "cp2_summary.json"
    if not summ.exists():
        return None
    for rec in json.loads(summ.read_text()).get("records", []):
        if (rec.get("site_pair") == "8bL16->dsv2-liteL22"
                and rec.get("arm") == "native" and rec.get("family") == family):
            return rec.get("r2")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--families", default=",".join(DEFAULT_FAMILIES),
                    help="comma-separated fit families (default: %(default)s)")
    args = ap.parse_args()
    families = [f.strip() for f in args.families.split(",") if f.strip()]

    OUT.mkdir(parents=True, exist_ok=True)
    axes, extras, _ = load_axes("8b")
    src = {**axes, **extras}
    tgt = {k: _unit(_load_key(rel, key)) for k, (rel, key) in TGT.items()}

    def read(fits_dir: Path, fname: str, src_name: str, tgt_name: str) -> float:
        tm = load_transport_map(fits_dir / fname)
        return round(cos(_unit(tm.transport(src[src_name].vec)), tgt[tgt_name]), 4)

    def family_block(family: str) -> dict:
        fit22 = fit_name(22, family)
        fit18 = fit_name(18, family)
        fields = {
            "a_hat_V7_L22_primary": read(FITS, fit22, "V7", "V7_L22"),
            "a_hat_V7_L22_modefree": read(FITS_MF, fit22, "V7", "V7_L22"),
            "Vtemp_L18_primary": read(FITS, fit18, "Vtemp", "Vtemp_L18"),
            "Vrep_perp": "NO BANKED dsv2 TARGET — cell empty",
            "Vconf": "NO BANKED dsv2 TARGET — cell empty (raw read vs V7_L22 lives in "
                     "leg2_diagnostics: -.246, collapse-structure-visible-raw)",
            "fit_R2_anchor_native": anchor_r2(family),
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
        return {"fields_series": fields, "needle_series": needle}

    doc = {
        "grade": "UNSTAMPED (C§8) — difficulty-curve raw artifact, rake-14 discipline "
                 "(fields/needle series separate; needle never divided by a_hat)",
        "prereg_tag": "prereg-arm8-v1", "builder": "a8_dsv2_curverows.py",
        "date": "2026-07-22", "pair": "8b->dsv2-lite (dense->MoE, regime rung)",
        "families": {fam: family_block(fam) for fam in families},
    }
    out_path = OUT / "dsv2_curverows_allfam.json"
    out_path.write_text(json.dumps(doc, indent=1))
    logger.info("curve rows (per family: %s) filed -> %s", ",".join(families), out_path)
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
