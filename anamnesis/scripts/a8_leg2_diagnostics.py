"""A8 Leg-2 PREP — transported-vector alignment diagnostics into DSV2 (CPU, local).

THE ANALYSIS HALF ONLY (desk parked Leg-2 scoring for a fresh session; Luxia's
park-and-prep ruling 2026-07-22). This files the GEOMETRIC precursors of the frozen
regime rows — nothing here scores P8-2/P8-2d (those score on INJECTION landing
outcomes, next session):

  - transported source axes u = unit(g·v) for v in {V7, Vrep⊥, Vconf, Vtemp, dir0}
    (source registry = a8_rosetta.load_axes)
  - the alignment diagnostic in the TARGET frame at L22 (Σ banked:
    arms/A5_dsv2/a5_sigma_L22_dsv2-lite.npz): c_align = cos(u, Σ⁻¹u) — the banked
    decision rule reads cos ≳.3 raw-ok / ≲.2 whiten (2-D caveat stands) —
    plus the whitened landing object w = unit(Σ⁻¹u)
  - raw vs whitened target-frame reads where a banked analog exists:
    L22: cos(u, V7_L22) and cos(w, V7_L22), envelope from transported nulls
    L18: cos(u, {V3_L18 (dir0), Vtemp_L18}) — Σ_L18 NOT banked (baton item)
  - Mahalanobis-form m = sqrt(uᵀΣ⁻¹u) reported with the CONVENTION CAVEAT flagged in
    the koto ferry (koto d 1.5–2.5 vs dense 60–160 — suspected convention mismatch,
    unresolved; number filed, not interpreted)

PREP-DIAGNOSTIC GRADE: envelope-only (no top-PC control here — the scoring session
runs full Rosetta discipline). Output readouts_prep/leg2_diagnostics.{json,md}.

Run (from repo root):
  PYTHONPATH=pipeline python -m anamnesis.scripts.a8_leg2_diagnostics \
      --arm-root outputs/battery/arms/A8_conjugation/leg2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from anamnesis.scripts.a8_fit_g import load_transport_map
from anamnesis.scripts.a8_rosetta import Axis, _load_key, _unit, cos, load_axes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_leg2_diagnostics")

SIGMA_L22 = Path("outputs/battery/arms/A5_dsv2/a5_sigma_L22_dsv2-lite.npz")
TGT_BANK = {
    22: [("V7_tgt", "a5_vectors_dsv2_lite_b7_L22/a5_vectors.npz", "V7_L22")],
    18: [("dir0_tgt", "a5_vectors_dsv2_lite/a5_vectors.npz", "V3_L18"),
         ("Vtemp_tgt", "a5_vectors_dsv2_lite_vtemp/a5_vectors.npz", "Vtemp_L18")],
}
TGT_NULLS_L22 = [("Rband%d" % i, "a5_vectors_dsv2_lite_b7_L22/a5_vectors.npz",
                  f"Rband{i}_L22") for i in (1, 2, 3)]
N_RANDOM = 100
SEED = 80


class SigmaInv:
    """Banked-Σ inverse action, exactly the a5 convention (covariance_screen):
    stored as eigendecomposition {evals, evecs, ridge}; Σ⁻¹v = V ((Vᵀv)/(λ+ridge))."""

    def __init__(self, path: Path):
        z = np.load(path)
        if "evals" not in z.files:
            raise KeyError(f"{path}: expected eigendecomposition, got {z.files}")
        self.evals = z["evals"].astype(np.float64)
        self.evecs = z["evecs"].astype(np.float64)
        self.ridge = float(z["ridge"])

    def __call__(self, v: np.ndarray) -> np.ndarray:
        coeff = self.evecs.T @ v.astype(np.float64)
        return self.evecs @ (coeff / (self.evals + self.ridge))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm-root", type=Path,
                    default=Path("outputs/battery/arms/A8_conjugation/leg2"))
    ap.add_argument("--source-model", default="8b")
    ap.add_argument("--target-model", default="dsv2-lite")
    ap.add_argument("--fits-dirname", default="fits")
    args = ap.parse_args()

    fits_dir = args.arm_root / args.fits_dirname
    out_dir = args.arm_root / "readouts_prep"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(fits_dir / "cp2_summary.json") as f:
        cp2 = json.load(f)

    src_axes, src_extras, _ = load_axes(args.source_model)
    vecs = {k: src_axes[k] for k in ("V7", "Vrep_perp", "Vconf", "Vtemp", "dir0")}
    if "oblique" in src_extras:
        vecs["oblique"] = src_extras["oblique"]

    sinv = SigmaInv(SIGMA_L22)
    tgt_axes = {s: {name: _unit(_load_key(rel, key))
                    for name, rel, key in TGT_BANK[s]} for s in TGT_BANK}
    tgt_nulls_l22 = [_unit(_load_key(rel, key)) for _, rel, key in TGT_NULLS_L22]

    rng = np.random.default_rng(SEED)
    rows, envelopes = [], []
    for rec in cp2["records"]:
        if not rec["valid"] or rec["family"] not in ("proc_k512", "ridge"):
            continue
        pair, arm, fam = rec["site_pair"], rec["arm"], rec["family"]
        t_site = int(pair.split("->")[1].split("L")[1])
        tm = load_transport_map(
            fits_dir / f"fit_{pair.replace('->', '__')}_{arm}_{fam}.npz")
        d_src = next(iter(vecs.values())).vec.shape[0]
        randoms = rng.standard_normal((N_RANDOM, d_src))
        randoms /= np.linalg.norm(randoms, axis=1, keepdims=True)

        # envelope of raw target-frame reads at this fit (vs V7_L22 when t_site==22)
        env = {}
        if t_site == 22:
            null_cos = [cos(tm.transport(r), tgt_axes[22]["V7_tgt"]) for r in randoms]
            env["q95_raw_V7_tgt"] = round(float(np.quantile(null_cos, 0.95)), 4)
            null_al = []
            for r in randoms[:30]:            # alignment diagnostic null (30 suffice)
                u = _unit(tm.transport(r))
                null_al.append(cos(u, sinv(u)))
            env["align_null_mean"] = round(float(np.mean(null_al)), 4)
            env["align_null_q95"] = round(float(np.quantile(null_al, 0.95)), 4)
            envelopes.append({"site_pair": pair, "arm": arm, "family": fam, **env})

        for name, ax in vecs.items():
            u = _unit(tm.transport(ax.vec))
            row = {"site_pair": pair, "arm": arm, "family": fam, "vector": name}
            if t_site == 22:
                siu = sinv(u)
                w = _unit(siu)
                row.update({
                    "align_cos_u_Sinvu": round(cos(u, siu), 4),
                    "mahalanobis_form_sqrt_uSinvu":
                        round(float(np.sqrt(u @ siu)), 4),
                    "raw_cos_vs_V7_tgt": round(cos(u, tgt_axes[22]["V7_tgt"]), 4),
                    "whitened_cos_vs_V7_tgt":
                        round(cos(w, tgt_axes[22]["V7_tgt"]), 4),
                    "raw_cos_vs_Rband_max": round(max(
                        abs(cos(u, rb)) for rb in tgt_nulls_l22), 4),
                    **env})
            if t_site == 18:
                row.update({
                    "raw_cos_vs_dir0_tgt": round(cos(u, tgt_axes[18]["dir0_tgt"]), 4),
                    "raw_cos_vs_Vtemp_tgt":
                        round(cos(u, tgt_axes[18]["Vtemp_tgt"]), 4),
                    "sigma_L18": "NOT BANKED — baton item"})
            rows.append(row)

    doc = {
        "grade": "PREP-DIAGNOSTIC (envelope-only; no top-PC control; nothing here "
                 "scores P8-2/P8-2d — landing outcomes do, next session)",
        "prereg_tag": "prereg-arm8-v1", "builder": "a8_leg2_diagnostics.py",
        "pair": f"{args.source_model}->{args.target_model}",
        "sigma_source": str(SIGMA_L22),
        "mahalanobis_convention_caveat":
            "ferry 2026-07-18ff: koto d 1.5-2.5 vs dense 60-160 — suspected "
            "convention mismatch, unresolved; filed not interpreted",
        "decision_rule_context": "banked: cos>~.3 raw-ok / <~.2 whiten (2-D caveat)",
        "rows": rows, "envelopes": envelopes,
    }
    with open(out_dir / "leg2_diagnostics.json", "w") as f:
        json.dump(doc, f, indent=1)

    cols = ["site_pair", "arm", "family", "vector", "align_cos_u_Sinvu",
            "raw_cos_vs_V7_tgt", "whitened_cos_vs_V7_tgt", "q95_raw_V7_tgt",
            "raw_cos_vs_dir0_tgt", "raw_cos_vs_Vtemp_tgt"]
    lines = ["# A8 Leg-2 PREP diagnostics (UNSTAMPED, PREP-GRADE — see json header)\n",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    (out_dir / "leg2_diagnostics.md").write_text("\n".join(lines) + "\n")
    logger.info("filed %d rows -> %s", len(rows), out_dir / "leg2_diagnostics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
