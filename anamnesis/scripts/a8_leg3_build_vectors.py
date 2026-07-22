"""A8 Leg-3 — build transported-vector injection banks + multicell cell specs.

The entropy cell that decides P8-J (baton §1) + the P8-2 DSV2 landing (§2): transported
source axes injected into each TARGET's entropy ladder, scored next against the FROZEN
F-ii predictions (8B, Qwen answer keys) and the frozen P8-2 letter (DSV2: raw FAILS ∧
whitened PASSES). This script is CPU-side staging only — no generation here.

Per target it emits, under leg3/:
  vectors/a8_leg3_vectors_{target}.npz     keys "{name}_L{site}" (unit float32; the
                                           write hook re-normalizes at attach, so only
                                           ORIENTATION is load-bearing — orientation
                                           comes from a8_rosetta.load_axes, which
                                           applies the recipe-level sign anchors)
  vectors/a8_leg3_vectors_{target}_stamps.json   fit sha + both norm conventions
                                           (rake item 3) + transported magnitudes
  cells/cells_{target}.json                vmb_a5_gen_multicell --cells-json input

Panels:
  8b / qwen-7b (Leg-3 proper): transported {V7, Vrep_perp, Vconf, Vtemp, oblique}
      + transported Rband1-3 (same g, the envelope controls) + baseline cell.
  dsv2-lite (P8-2): raw u=unit(g·v) vs whitened w=unit(Σ⁻¹u) for {V7, Vconf}
      (Vconf = exploratory-beside, NOT letter-scoring) + raw/whitened Rband1-3
      + baseline cell.

Doses alpha_frac ±{.03,.1,.3}; alpha resolves at launch as frac × the target-site
PER-TOKEN median residual norm (banked a5_vectors_stamps.json — the A5 injection
convention). The a8 mean-STATE median (different convention) is recorded beside it
in the stamps, never used for alphas.

Run (repo root):  PYTHONPATH=pipeline python -m anamnesis.scripts.a8_leg3_build_vectors
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

from anamnesis.scripts.a8_fit_g import load_transport_map
from anamnesis.scripts.a8_leg2_diagnostics import SigmaInv
from anamnesis.scripts.a8_rosetta import _unit, load_axes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_leg3_build_vectors")

ARM = Path("outputs/battery/arms/A8_conjugation")
LEG3 = ARM / "leg3"
DOSES = (-0.3, -0.1, -0.03, 0.03, 0.1, 0.3)
PANEL = ("V7", "Vrep_perp", "Vconf", "Vtemp", "oblique")   # the F-ii predicted set

TARGETS: dict[str, dict] = {
    "8b": {
        "fit": ARM / "fits/fit_3bL14__8bL16_native_proc_k512.npz",
        "source": "3b", "site": 16,
        "a5_norms": Path("outputs/battery/a5_vectors_8b/a5_vectors_stamps.json"),
        "a8_norms": ARM / "states/norms_8b_native.json",
        "answer_key": str(ARM / "readouts/f2_predictions.json"),
        "sigma": None,
    },
    "qwen-7b": {
        "fit": ARM / "leg1/fits/fit_8bL16__qwen-7bL21_native_proc_k512.npz",
        "source": "8b", "site": 21,
        "a5_norms": Path("outputs/battery/a5_vectors_qwen_7b/a5_vectors_stamps.json"),
        "a8_norms": ARM / "leg1/states/norms_qwen-7b_native.json",
        "answer_key": str(ARM / "leg1/readouts/f2_predictions.json"),
        "sigma": None,
    },
    "dsv2-lite": {
        "fit": ARM / "leg2/fits/fit_8bL16__dsv2-liteL22_native_proc_k512.npz",
        "source": "8b", "site": 22,
        "a5_norms": Path("outputs/battery/a5_vectors_dsv2_lite_b7_L22/a5_vectors_stamps.json"),
        "a8_norms": ARM / "leg2/states/norms_dsv2-lite_native.json",
        "answer_key": "P8-2 frozen letter (prereg §P8-2: raw FAILS AND whitened PASSES, "
                      "filed P .55); Vconf rows = exploratory-beside, NOT letter-scoring",
        "sigma": Path("outputs/battery/arms/A5_dsv2/a5_sigma_L22_dsv2-lite.npz"),
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_target(target: str, cfg: dict) -> None:
    site = cfg["site"]
    tm = load_transport_map(cfg["fit"])
    axes, extras, null_axes = load_axes(cfg["source"])
    src = {**{k: axes[k] for k in axes}, **extras}
    rbands = {ax.name: ax for ax in null_axes if ax.name.startswith("Rband")}
    if len(rbands) != 3:
        raise ValueError(f"{target}: expected 3 Rband nulls, got {list(rbands)}")

    vectors: dict[str, np.ndarray] = {}
    magnitudes: dict[str, float] = {}     # ||g·v|| pre-unit (attenuation-informative)
    provenance: dict[str, str] = {}

    def add(key: str, raw_vec: np.ndarray, source_desc: str) -> None:
        t = tm.transport(raw_vec)
        n = float(np.linalg.norm(t))
        if not np.isfinite(n) or n == 0.0:
            raise ValueError(f"{target}/{key}: degenerate transported vector (norm={n})")
        vectors[f"{key}_L{site}"] = _unit(t).astype(np.float32)
        magnitudes[f"{key}_L{site}"] = round(n, 6)
        provenance[f"{key}_L{site}"] = source_desc

    if cfg["sigma"] is None:
        for name in PANEL:
            add(f"g{name}", src[name].vec, f"{cfg['source']}::{src[name].source}")
        for rb_name, ax in sorted(rbands.items()):
            add(f"g{rb_name}", ax.vec, f"{cfg['source']}::{ax.source}")
    else:
        sinv = SigmaInv(cfg["sigma"])

        def add_whitened(key: str, raw_key: str, source_desc: str) -> None:
            u = vectors[f"{raw_key}_L{site}"].astype(np.float64)
            w = sinv(u)
            vectors[f"{key}_L{site}"] = _unit(w).astype(np.float32)
            magnitudes[f"{key}_L{site}"] = round(float(np.linalg.norm(w)), 6)
            provenance[f"{key}_L{site}"] = f"unit(Sigma^-1 · {raw_key}) | {source_desc}"

        for name in ("V7", "Vconf"):
            add(f"g{name}", src[name].vec, f"{cfg['source']}::{src[name].source}")
            add_whitened(f"w{name}", f"g{name}", f"sigma={cfg['sigma']}")
        for rb_name, ax in sorted(rbands.items()):
            add(f"g{rb_name}", ax.vec, f"{cfg['source']}::{ax.source}")
            add_whitened(f"w{rb_name}", f"g{rb_name}", f"sigma={cfg['sigma']}")

    a5_norms = json.loads(cfg["a5_norms"].read_text())["median_resid_norms"]
    if f"L{site}" not in a5_norms:
        raise KeyError(f"{target}: L{site} missing from {cfg['a5_norms']}")
    a8_norms = json.loads(cfg["a8_norms"].read_text())

    vec_dir = LEG3 / "vectors"
    cell_dir = LEG3 / "cells"
    vec_dir.mkdir(parents=True, exist_ok=True)
    cell_dir.mkdir(parents=True, exist_ok=True)
    npz_path = vec_dir / f"a8_leg3_vectors_{target}.npz"
    np.savez(npz_path, **vectors)

    cells = []
    for key in vectors:
        for frac in DOSES:
            cell_name = f"{key}_a{frac:+.2f}"
            cells.append({
                "out_run_dir": f"{LEG3}/runs/{target}/{cell_name}",
                "seed_namespace": f"A8L3-{target}-{cell_name}",
                "inject_key": key, "inject_layer": site, "inject_alpha_frac": frac,
            })
    cells.append({
        "out_run_dir": f"{LEG3}/runs/{target}/baseline",
        "seed_namespace": f"A8L3-{target}-baseline",
        "inject_key": None, "inject_layer": None, "inject_alpha_frac": None,
    })
    cells_path = cell_dir / f"cells_{target}.json"
    cells_path.write_text(json.dumps({"cells": cells}, indent=1))

    stamp = {
        "grade": "UNSTAMPED (C§8) — staging for the Leg-3 entropy cell / P8-2 landing",
        "prereg_tag": "prereg-arm8-v1", "builder": "a8_leg3_build_vectors.py",
        "date": "2026-07-22", "target": target, "inject_site": site,
        "fit_file": str(cfg["fit"]), "fit_sha256": _sha(cfg["fit"]),
        "source_registry": f"a8_rosetta.load_axes('{cfg['source']}') — recipe-level sign "
                           "anchors applied; orientation matches the frozen F-ii rows",
        "answer_key": cfg["answer_key"],
        "sigma_file": str(cfg["sigma"]) if cfg["sigma"] else None,
        "sigma_sha256": _sha(cfg["sigma"]) if cfg["sigma"] else None,
        "norm_conventions": {
            "used_for_alpha": {"convention": "PER-TOKEN median residual norm (a5 stamps)",
                               "file": str(cfg["a5_norms"]),
                               f"L{site}": a5_norms[f"L{site}"]},
            "beside_only": {"convention": "median of MEAN-STATE norms (a8 leg states)",
                            "file": str(cfg["a8_norms"]),
                            f"L{site}": a8_norms.get(f"L{site}")},
        },
        "doses_alpha_frac": list(DOSES), "n_per_cell": 80, "seeds_per_class": 1,
        "n_cells": len(cells),
        "transported_magnitudes_pre_unit": magnitudes,
        "vector_provenance": provenance,
        "note": "vectors stored UNIT; ResidualWriteSpec re-normalizes at attach — only "
                "orientation is load-bearing. Controls = source Rbands through the SAME g.",
    }
    (vec_dir / f"a8_leg3_vectors_{target}_stamps.json").write_text(
        json.dumps(stamp, indent=1))
    logger.info("%s: %d vectors, %d cells (%d gens) -> %s",
                target, len(vectors), len(cells), 80 * len(cells), npz_path)


def main() -> int:
    for target, cfg in TARGETS.items():
        build_target(target, cfg)
    logger.info("leg3 staging complete under %s", LEG3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
