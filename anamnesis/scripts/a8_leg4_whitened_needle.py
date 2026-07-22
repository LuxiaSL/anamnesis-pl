"""A8 Leg-4F / L4-c — whitened-target needle read: PARK RECORD + labelled exploratory beside.

The letter (A8-add-2 P8-L4c) asks for V3w(DSV2-L22) = unit(Sigma^-1 Delta-mu) rebuilt
FROM THE STAMPED RECIPE (banked Sigma npz + the banked class means), identity-checked
against the whiten-stamps' recorded diagnostics, then cos(g.dir0_8B, V3w_dsv2) vs a fresh
transported-null envelope.

INVENTORY RESULT — the letter's object cannot be built from banked inputs:
  1. No banked V3raw at DSV2 L22 (a5_vectors_dsv2_lite carries V3_L9/L11/L15/L18 only).
  2. No banked class means: the 2026-07-19 whiten build wrote STAMPS ONLY (no vectors npz)
     — the same fact the Leg-3 curve rows already flagged.
  3. Sigma is banked at L22 only; Sigma@L18 (where V3raw IS banked) is NOT banked, and the
     pre-ruled fork forbids re-deriving Sigma.
  4. Two whiten vintages carry the same recipe name with different diagnostics:
     whiten/v3whiten_stamps.json (lw_shrinkage .2576) cos_delta_whitened L22 = .1273
     (the baton's ~.06-.13 reproduction target) vs whiten_dir0_stamps.json
     (lw_shrinkage .0611) = .3257. Recorded as a rake.
  => L4-c PARKS at the letter (park-don't-amend). The missing input is named precisely:
     either Sigma@L18, or the L22 class means / V3raw_L22.

EXPLORATORY BESIDE (this file's numbers; NOT the letter's object, scores NOTHING):
a whitened dsv2 needle target built from the A8 arm's OWN banked material —
Delta-mu = mean(S3 analogical) - mean(S3 contrastive) over the Leg-2 dsv2 state bank at
L22 (own-voice primary, pooled beside), whitened with the BANKED Sigma. Different capture
(per-text mean states, A8 corpus, n=30/pole) from the stamped recipe (per-token positions,
pole runs, n=160/class), so its diagnostics are reported next to both stamp vintages and
the reader can see the distance. The needle read is then run against it exactly as the
letter would have: transported-null envelope, sign-anchored, both target variants.

UNSTAMPED (C§8). Run: PYTHONPATH=pipeline python -m anamnesis.scripts.a8_leg4_whitened_needle
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from anamnesis.scripts.a8_fit_g import load_transport_map
from anamnesis.scripts.a8_rosetta import _unit, cos, load_axes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_leg4_whitened_needle")

ARM = Path("outputs/battery/arms/A8_conjugation")
OUT = ARM / "leg4" / "readouts_cpu"
SIGMA = Path("outputs/battery/arms/A5_dsv2/a5_sigma_L22_dsv2-lite.npz")
STAMPS = {"v3whiten (05:23, lw .2576)":
          Path("outputs/battery/arms/A5_dsv2/whiten/v3whiten_stamps.json"),
          "whiten_dir0 (later, lw .0611)":
          Path("outputs/battery/arms/A5_dsv2/whiten_dir0_stamps.json")}
FIT = ARM / "leg2/fits/fit_8bL16__dsv2-liteL22_native_proc_k512.npz"
N_RANDOM = 100
SEED = 80


def sigma_inv(v: np.ndarray, z) -> np.ndarray:
    """Banked Sigma convention: eigendecomposition {evals, evecs, ridge};
    Sigma^-1 v = V ((V^T v) / (lam + ridge))."""
    evals, evecs, ridge = z["evals"], z["evecs"], float(z["ridge"])
    return evecs @ ((evecs.T @ v) / (evals + ridge))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(SIGMA)
    entries = json.loads((ARM / "leg2/corpus/corpus_manifest.json").read_text())["entries"]
    bank = np.load(ARM / "leg2/states/states_dsv2-lite_native.npz")
    ids = list(bank["text_ids"])
    S = bank["L22"].astype(np.float64)
    pos = {e["text_id"] for e in entries if e["stratum"] == "S3" and e["mode"] == "analogical"}
    neg = {e["text_id"] for e in entries if e["stratum"] == "S3" and e["mode"] == "contrastive"}

    def delta(voice_filter) -> tuple[np.ndarray, int, int]:
        keep = {e["text_id"] for e in entries if voice_filter(e)}
        ip = [i for i, t in enumerate(ids) if t in pos & keep]
        ineg = [i for i, t in enumerate(ids) if t in neg & keep]
        return S[ip].mean(0) - S[ineg].mean(0), len(ip), len(ineg)

    variants = {
        "own_voice_dsv2": lambda e: e["voice"] == "dsv2-lite",
        "pooled_both_voices": lambda e: True,
    }
    src_axes, _, src_pool = load_axes("8b")
    dir0 = src_axes["dir0"].vec
    rng = np.random.default_rng(SEED)
    randoms = rng.standard_normal((N_RANDOM, dir0.shape[0]))
    randoms /= np.linalg.norm(randoms, axis=1, keepdims=True)

    maps = {"primary_g (S1+S2+S3 fit)": FIT,
            "mode_free_g (S1+S2 fit — never saw S3)": Path(
                str(FIT).replace("/fits/", "/fits_modefree/"))}
    gmaps, unull = {}, {}
    for label, path in maps.items():
        if not path.exists():
            continue
        tm_i = load_transport_map(path)
        gmaps[label] = (_unit(tm_i.transport(dir0)),
                        [_unit(tm_i.transport(r)) for r in randoms]
                        + [_unit(tm_i.transport(a.vec)) for a in src_pool])
    tm = load_transport_map(FIT)
    u, nulls = gmaps["primary_g (S1+S2+S3 fit)"]

    v3_l18 = _unit(np.load("outputs/battery/a5_vectors_dsv2_lite/a5_vectors.npz")["V3_L18"])
    out_variants = {}
    for name, filt in variants.items():
        dmu, npos, nneg = delta(filt)
        raw = _unit(dmu)
        w = _unit(sigma_inv(dmu, z))
        m_d = float(np.sqrt(max(dmu @ sigma_inv(dmu, z), 0.0)))
        reads = {}
        for tgt_name, tgt in (("raw_deltamu_L22", raw), ("whitened_V3w_A8_L22", w)):
            reads[tgt_name] = {}
            for glabel, (ug, nullsg) in gmaps.items():
                null_cos = np.array([abs(cos(n, tgt)) for n in nullsg])
                q95 = float(np.quantile(null_cos, 0.95))
                reads[tgt_name][glabel] = {
                    "cos_transported_dir0_vs_target": round(cos(ug, tgt), 4),
                    "null_q95_abs": round(q95, 4),
                    "exceeds_envelope": bool(abs(cos(ug, tgt)) > q95),
                    "n_nulls": len(nullsg)}
        out_variants[name] = {
            "n_analogical": npos, "n_contrastive": nneg,
            "cos_raw_vs_whitened": round(cos(raw, w), 4),
            "mahalanobis_d_with_magnitude": round(m_d, 4),
            "mahalanobis_d_unit_vector": round(float(np.sqrt(max(
                raw @ sigma_inv(raw, z), 0.0))), 4),
            "cos_raw_L22_vs_banked_V3_L18": round(cos(raw, v3_l18), 4)
            if raw.shape == v3_l18.shape else None,
            "needle_reads": reads,
        }

    res = {
        "STATUS": "UNSTAMPED (C§8)",
        "leg": "A8 Leg-4F / L4-c",
        "letter_status": "PARKED — the letter's object is not buildable from banked inputs",
        "park_record": {
            "missing_inputs": [
                "no banked V3raw at DSV2 L22 (a5_vectors_dsv2_lite: V3_L9/L11/L15/L18 only)",
                "no banked class means for the whitened build (2026-07-19 build wrote "
                "stamps only, no vectors npz)",
                "Sigma banked at L22 only; Sigma@L18 (where V3raw IS banked) not banked, "
                "and the pre-ruled fork forbids re-deriving Sigma",
            ],
            "what_would_unpark_it": "either Sigma@L18 captured (vmb_a5_covariance_screen "
                                    "--save-sigma-site, ~60 gens) so the banked V3raw_L18 / "
                                    "V3w_L18 pair can be used at L18, or the L22 class "
                                    "means re-banked from the stamped pole runs",
            "stamp_vintages": {k: json.loads(v.read_text())["diagnostics"]
                               .get("L22", {}).get("cos_delta_whitened")
                               for k, v in STAMPS.items()},
            "vintage_note": "two builds carry the same recipe name with different "
                            "Ledoit-Wolf shrinkage and different cos_delta_whitened at L22 "
                            "(.1273 vs .3257). The baton's ~.06-.13 reproduction target "
                            "matches the EARLIER vintage. Rake item.",
        },
        "exploratory_beside": {
            "grade": "EXPLORATORY — NOT the letter's object; scores nothing",
            "construction": "Delta-mu = mean(S3 analogical) - mean(S3 contrastive) over the "
                            "A8 Leg-2 dsv2-lite native state bank at L22 (per-text mean "
                            "states), whitened with the BANKED Sigma (eigendecomposition "
                            "convention, ridge from the npz). Sign anchor: + = analogical "
                            "- contrastive, matching dir0's banked recipe order.",
            "TARGET_CONSTRUCTION_CAVEAT": (
                "the target is built from S3 rows that were IN the primary g's fit corpus, "
                "so under the primary map part of any correspondence is what g was fit to "
                "produce. The mode-free g (S1+S2 only — never saw S3) is the leak-free "
                "column; read it, not the primary, when asking whether the needle "
                "transports. The banked, independently-constructed target (V3_L18 from the "
                "pole runs) remains the object of record: -.0865."),
            "difference_from_stamped_recipe": "stamped: per-token positions over pole RUNS, "
                                              "n=160/class, its own LW shrinkage. Here: "
                                              "per-text mean states, A8 corpus, n=30/pole, "
                                              "banked Sigma's ridge.",
            "transport_map": str(FIT),
            "variants": out_variants,
        },
        "reading_aid": "The banked needle-cliff of record stays the raw-target L18 read "
                       "(-.0865, Leg-3 curve rows). Nothing here replaces it.",
    }
    (OUT / "l4c_whitened_needle_park.json").write_text(json.dumps(res, indent=1))
    logger.info("wrote %s", OUT / "l4c_whitened_needle_park.json")
    for name, v in out_variants.items():
        for tgt, blk in v["needle_reads"].items():
            for glabel, r in blk.items():
                logger.info("%s | %s | %s : cos %.4f (q95 %.4f, exceeds %s)", name, tgt,
                            glabel, r["cos_transported_dir0_vs_target"],
                            r["null_q95_abs"], r["exceeds_envelope"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
