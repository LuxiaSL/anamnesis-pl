"""A8 Leg-4F — injection banks for the two appended GPU legs (L4-g needle, L4-f/Leg-5 owl).

Subcommands
-----------
needle : L4-g (P8-JX). Transported dir0 into Qwen at L21 — g.dir0_8B through the banked
         Leg-1 Procrustes (native proc_k512), sign-anchored via a8_rosetta.load_axes
         (recipe order "+ = analogical - contrastive", identical in both models).  Writes
         the injection npz + the multicell cells json: dose ladder +-{.03,.1,.3} plus
         dose-matched transported-R band cells plus an alpha=0 baseline.

owl    : L4-f / Leg-5 (P8-5). The owl install vector carried the OTHER way — Qwen -> 8B via
         the REVERSE Procrustes map (transpose; add-3 names it, rake-15 backs it).  Vdiverge
         lives at QWEN L18, which is NOT one of the Leg-1 site grids, so this leg uses a
         dedicated 8B-L16 <-> Qwen-L18 fit (leg4f_owl/) rather than applying an L18 vector
         through an L19-domain map.  Writes:
           * vec npz  {Valign_L16, Vdiverge_L16}  = unit(g_rev . the A6 unit vectors)
           * ar  npz  {AR1..3_L16}                = transported Leg-1 R-band members
           * stamps json (median_resid_norms.L16, a5 per-token convention)
         and, as the docket's "RAW unconjugated Vdiverge (the should-fail null)", a SECOND
         vec npz built by naive coordinate identification (zero-pad 3584 -> 4096, unit).
         THAT OPERATIONALISATION IS THE ENACTOR'S — the raw Qwen vector has no dimensional
         image in 8B, so "unconjugated" has to be given a concrete meaning; zero-padding is
         the literal "coordinates transfer" null that X-5 denies. Flagged for the desk.

Identity checks run before anything is written (the docket's requirement: "if the vector
cannot be identity-verified, PARK the leg"):
  * cos(Valign, Vdiverge) must reproduce the banked construction.json value (0.316)
  * both A6 vectors must be unit (they are banked unit; raw norms live in the stamp)
  * the fit npz sha256 is recorded in the output stamp

UNSTAMPED (C§8).  Run (repo root):
  PYTHONPATH=pipeline python -m anamnesis.scripts.a8_leg4f_banks needle
  PYTHONPATH=pipeline python -m anamnesis.scripts.a8_leg4f_banks owl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from anamnesis.scripts.a8_fit_g import load_transport_map
from anamnesis.scripts.a8_rosetta import _unit, cos, load_axes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_leg4f_banks")

ARM = Path("outputs/battery/arms/A8_conjugation")
LEG4 = ARM / "leg4"
NODE_ROOT = "/models/anamnesis-extract/battery/arms/A8_conjugation/leg4"
DOSES = (0.03, 0.1, 0.3, -0.03, -0.1, -0.3)
N_PER_CELL_SEEDS = 1          # 20 topics x 4 strata x 1 seed = 80 gens/cell

A6_OWL = Path("outputs/battery/a6_2b_vectors_qwen_owl")
QWEN_NORMS = Path("outputs/battery/a5_vectors_qwen_7b/a5_vectors_stamps.json")
EIGHTB_NORMS = Path("outputs/battery/a5_vectors_8b/a5_vectors_stamps.json")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _norm(stamps: Path, site: int) -> float:
    return float(json.loads(stamps.read_text())["median_resid_norms"][f"L{site}"])


# ------------------------------------------------------------------- L4-g needle
def build_needle() -> int:
    fit = ARM / "leg1/fits/fit_8bL16__qwen-7bL21_native_proc_k512.npz"
    tm = load_transport_map(fit)
    axes, _, pool = load_axes("8b")
    site = 21
    vecs = {f"gdir0_L{site}": _unit(tm.transport(axes["dir0"].vec))}
    for a in pool[:3]:                      # Rband1-3, the banked R-band members
        vecs[f"g{a.name}_L{site}"] = _unit(tm.transport(a.vec))
    tgt_axes, _, _ = load_axes("qwen-7b")
    identity = {"cos_gdir0_vs_qwen_dir0": round(cos(vecs[f"gdir0_L{site}"],
                                                    tgt_axes["dir0"].vec), 4),
                "frozen_leg1_value": 0.9176}
    out_dir = LEG4 / "vectors"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / "a8_leg4g_vectors_qwen-7b.npz"
    np.savez(npz, **{k: v.astype(np.float32) for k, v in vecs.items()})
    norm = _norm(QWEN_NORMS, site)
    stamp = {
        "grade": "UNSTAMPED (C§8) — staging for L4-g (P8-JX judged needle expression)",
        "prereg": "A8-add-3 P8-JX", "builder": "a8_leg4f_banks.py needle",
        "target": "qwen-7b", "inject_site": site,
        "fit_file": str(fit), "fit_sha256": _sha(fit),
        "sign_anchor": "dir0 recipe order '+ = analogical - contrastive' (identical both "
                       "models); orientation from a8_rosetta.load_axes",
        "identity_check": identity,
        "vectors_stored": "UNIT (the write hook re-normalises; only ORIENTATION is "
                          "load-bearing)",
        "norm_convention_for_alpha": {"convention": "PER-TOKEN median residual norm "
                                                    "(a5 stamps)",
                                      "file": str(QWEN_NORMS), f"L{site}": norm},
        "doses_alpha_frac": list(DOSES),
        "n_per_cell": 80,
        "cells": "gdir0 x 6 doses + gRband1 x 6 doses (dose-matched control) + baseline",
    }
    (out_dir / "a8_leg4g_vectors_qwen-7b_stamps.json").write_text(json.dumps(stamp, indent=1))

    cells = [{"out_run_dir": f"{NODE_ROOT}/l4g_gen/qwen-7b/baseline_L{site}_a0.00",
              "seed_namespace": f"A8L4G-qwen-baseline_L{site}_a0.00",
              "inject_key": None, "inject_layer": None, "inject_alpha_frac": None}]
    for key in (f"gdir0_L{site}", f"gRband1_L{site}"):
        for d in DOSES:
            cid = f"{key}_a{d:+.2f}"
            cells.append({"out_run_dir": f"{NODE_ROOT}/l4g_gen/qwen-7b/{cid}",
                          "seed_namespace": f"A8L4G-qwen-{cid}",
                          "inject_key": key, "inject_layer": site,
                          "inject_alpha_frac": d})
    (LEG4 / "cells").mkdir(parents=True, exist_ok=True)
    (LEG4 / "cells/l4g_cells_qwen-7b.json").write_text(
        json.dumps({"cells": cells, "_stamp": stamp}, indent=1))
    logger.info("needle bank: %s (%d cells); identity cos(g·dir0, dir0_qwen)=%.4f "
                "(frozen %.4f)", npz, len(cells),
                identity["cos_gdir0_vs_qwen_dir0"], identity["frozen_leg1_value"])
    return 0


# --------------------------------------------------------------------- L4-f owl
def build_owl() -> int:
    fit = ARM / "leg4f_owl/fits/fit_8bL16__qwen-7bL18_native_proc_k512.npz"
    if not fit.exists():
        raise SystemExit(f"owl fit not ready: {fit}")
    tm = load_transport_map(fit)
    a6 = dict(np.load(A6_OWL / "vectors.npz"))
    con = json.loads((A6_OWL / "construction.json").read_text())
    v_align, v_div = a6["Valign_L18"].astype(np.float64), a6["Vdiverge_L18"].astype(np.float64)
    ident = {
        "cos_Valign_Vdiverge_recomputed": round(cos(v_align, v_div), 4),
        "cos_Valign_Vdiverge_banked": con["cos_Valign_Vdiverge"],
        "norms_recomputed": {k: round(float(np.linalg.norm(a6[k])), 6) for k in a6},
        "raw_delta_norms_banked": con["vector_norms_raw"],
        "inject_site_banked": con["inject_site"],
        "adapter_path_banked": con["adapter_path"],
    }
    if abs(ident["cos_Valign_Vdiverge_recomputed"] - con["cos_Valign_Vdiverge"]) > 0.002:
        raise SystemExit("IDENTITY CHECK FAILED — parking the owl leg per the docket "
                         f"({ident})")
    site = 16
    rev = {"Valign_L16": _unit(tm.transport(v_align, direction="rev")),
           "Vdiverge_L16": _unit(tm.transport(v_div, direction="rev"))}
    # transported R-band members (Qwen's own banked R band, carried back the same way)
    _, _, qpool = load_axes("qwen-7b")
    ar = {f"AR{i+1}_L16": _unit(tm.transport(qpool[i].vec, direction="rev"))
          for i in range(3)}
    # the "RAW unconjugated" null: naive coordinate identification, zero-padded
    d_tgt = rev["Vdiverge_L16"].shape[0]
    pad = np.zeros(d_tgt)
    pad[:min(d_tgt, v_div.shape[0])] = v_div[:min(d_tgt, v_div.shape[0])]
    raw_null = {"Valign_L16": _unit(np.pad(v_align, (0, max(0, d_tgt - v_align.shape[0])))[:d_tgt]),
                "Vdiverge_L16": _unit(pad)}

    out_dir = LEG4 / "vectors"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "a8_leg5_owl_transported_8b.npz",
             **{k: v.astype(np.float32) for k, v in rev.items()})
    np.savez(out_dir / "a8_leg5_owl_rawnull_8b.npz",
             **{k: v.astype(np.float32) for k, v in raw_null.items()})
    np.savez(out_dir / "a8_leg5_owl_ar_8b.npz",
             **{k: v.astype(np.float32) for k, v in ar.items()})
    norm16 = _norm(EIGHTB_NORMS, site)
    (out_dir / "a8_leg5_owl_norms_8b.json").write_text(
        json.dumps({"median_resid_norms": {f"L{site}": norm16}}, indent=1))
    stamp = {
        "grade": "UNSTAMPED (C§8) — staging for L4-f / Leg-5 (P8-5, the owl)",
        "prereg": "A8-add-3 P8-5 execution clauses", "builder": "a8_leg4f_banks.py owl",
        "direction": "Qwen -> 8B via the REVERSE Procrustes map (transpose), sign-anchored "
                     "before use (add-3; rake-15: Procrustes is the family of record for "
                     "reverse-direction claims)",
        "site_note": "Vdiverge is banked at QWEN L18, which is NOT in the Leg-1 site grid "
                     "({19,21,23}). Rather than push an L18 vector through an L19-domain "
                     "map, this leg fits a dedicated 8B-L16 <-> Qwen-L18 Procrustes on the "
                     "Leg-1 corpus (leg4f_owl/) and uses its reverse. Named, not improvised.",
        "fit_file": str(fit), "fit_sha256": _sha(fit),
        "source_vectors": str(A6_OWL / "vectors.npz"),
        "source_construction": str(A6_OWL / "construction.json"),
        "identity_check": ident,
        "raw_null_operationalisation": (
            "ENACTOR'S OPERATIONALISATION, flagged for the desk: the docket's 'RAW "
            "unconjugated Vdiverge (should-fail null)' has no dimensional image in 8B "
            "(3584 vs 4096). Implemented as naive coordinate identification — zero-pad to "
            "4096, unit-normalise — i.e. the literal 'coordinates transfer' null that X-5 "
            "denies. Any other reading of 'unconjugated' is the desk's to name."),
        "norm_convention_for_alpha": {"convention": "PER-TOKEN median residual norm "
                                                    "(a5 stamps)",
                                      "file": str(EIGHTB_NORMS), f"L{site}": norm16},
        "controls": {"transported_R_band": "AR1-3 = Qwen's banked R-band members carried "
                                           "back through the SAME reverse map",
                     "raw_unconjugated": "a8_leg5_owl_rawnull_8b.npz (above)",
                     "placebo + AR floor": "the probe's own alpha=0 disjoint-split placebo "
                                           "and AR ladders (vmb_a6_2b_probe machinery)"},
        "dose_ladder": [0.0, 0.03, 0.1, 0.3, -0.03, -0.1, -0.3],
        "cos_transported_vs_raw_null": round(cos(rev["Vdiverge_L16"],
                                                 raw_null["Vdiverge_L16"]), 4),
    }
    (out_dir / "a8_leg5_owl_stamps.json").write_text(json.dumps(stamp, indent=1))
    logger.info("owl banks written; identity cos(Valign,Vdiverge)=%.4f (banked %.4f); "
                "cos(transported, raw-null)=%.4f",
                ident["cos_Valign_Vdiverge_recomputed"], con["cos_Valign_Vdiverge"],
                stamp["cos_transported_vs_raw_null"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["needle", "owl"])
    args = ap.parse_args()
    return build_needle() if args.mode == "needle" else build_owl()


if __name__ == "__main__":
    raise SystemExit(main())
