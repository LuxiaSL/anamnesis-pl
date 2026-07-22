"""A8 Leg-6 / Item 1 BESIDE — the L4-c letter re-read on an independently captured Sigma.

The letter itself (a8_leg6_l4c_letter.py) reads on the BANKED V3w_L18, which is the
stamped recipe's own output: unit(Sigma_LW^-1 . Delta-mu), Ledoit-Wolf shrinkage .2577
(pair-matched vintage), captured 2026-07-19 over that build's own pole runs.

This is the robustness column. It rebuilds the same direction from the SAME banked
V3raw_L18 but with a Sigma captured fresh in this session, from a different corpus
(vmb_stage0_dsv2_lite, 60 gens) and — importantly — under a DIFFERENT ESTIMATOR:

    banked recipe : Sigma = Ledoit-Wolf shrunk covariance
    fresh capture : Sigma = sample covariance + ridge (1e-3 x mean eigenvalue)
                    (this is what vmb_a5_covariance_screen computes; it is NOT LW)

So agreement between the two columns is evidence that the letter's read does not depend
on one particular Sigma estimate, one corpus, or one shrinkage convention. Disagreement
localises the result to the estimator. Either way the BANKED column stays the letter of
record — this one scores nothing and replaces nothing.

Positive-scalar invariance is what makes this legitimate without class means:
unit(Sigma^-1 Delta-mu) == unit(Sigma^-1 . V3raw) exactly, since V3raw is banked as
unit(mu+ - mu-) with raw_norm recorded (see the letter's unpark basis).

The mode-pair labelling from the letter carries over: the PAIR-MATCHED vintage is the one
whose contrast matches the arm's dir0 (analogical/contrastive). Both are read.

UNSTAMPED (C section 8). No P self-scored.
Run: PYTHONPATH=pipeline python -m anamnesis.scripts.a8_leg6_l4c_sigma_beside
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from anamnesis.scripts.a8_fit_g import load_transport_map
from anamnesis.scripts.a8_leg6_l4c_letter import (
    FAMILIES, N_RANDOM, PRIMARY_FAMILY, SEED, SRC_SITE, TGT_SITE, VINTAGES, _fit_path)
from anamnesis.scripts.a8_rosetta import _unit, cos, load_axes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_leg6_l4c_sigma_beside")

ARM = Path("outputs/battery/arms/A8_conjugation")
OUT = ARM / "leg6" / "readouts_cpu"
SIGMA_FRESH = ARM / "leg6" / "sigma" / f"a5_sigma_L{TGT_SITE}_dsv2-lite.npz"


def sigma_inv(v: np.ndarray, z) -> np.ndarray:
    """Banked Sigma convention: eigendecomposition {evals, evecs, ridge};
    Sigma^-1 v = V ((V^T v) / (lam + ridge))."""
    evals, evecs, ridge = z["evals"], z["evecs"], float(z["ridge"])
    return evecs @ ((evecs.T @ v) / (evals + ridge))


def main() -> int:
    if not SIGMA_FRESH.exists():
        raise SystemExit(f"fresh Sigma not present yet: {SIGMA_FRESH}")
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(SIGMA_FRESH)

    src_axes, _, src_pool = load_axes("8b")
    dir0 = src_axes["dir0"].vec
    rng = np.random.default_rng(SEED)
    randoms = rng.standard_normal((N_RANDOM, dir0.shape[0]))
    randoms /= np.linalg.norm(randoms, axis=1, keepdims=True)

    targets = {}
    for label, path in VINTAGES.items():
        bank = np.load(path, allow_pickle=True)
        raw = _unit(bank[f"V3raw_L{TGT_SITE}"].astype(np.float64))
        targets[label] = {
            "banked_whitened_LW": _unit(bank[f"V3w_L{TGT_SITE}"].astype(np.float64)),
            "fresh_whitened_ridge": _unit(sigma_inv(raw, z)),
            "raw": raw,
        }

    reads: dict[str, dict] = {}
    for modefree in (False, True):
        gtag = ("mode_free_g (S1+S2 fit; never saw S3)" if modefree
                else "primary_g (S1+S2+S3)")
        reads[gtag] = {}
        for family in FAMILIES:
            p = _fit_path(family, modefree)
            if not p.exists():
                continue
            tm = load_transport_map(p)
            ug = _unit(tm.transport(dir0))
            nulls = ([_unit(tm.transport(r)) for r in randoms]
                     + [_unit(tm.transport(a.vec)) for a in src_pool])

            def read(t: np.ndarray) -> dict:
                c = cos(ug, t)
                q95 = float(np.quantile([abs(cos(n, t)) for n in nulls], 0.95))
                return {"cos": round(c, 4), "null_q95_abs": round(q95, 4),
                        "exceeds_envelope": bool(abs(c) > q95)}

            blk = {}
            for label, t in targets.items():
                blk[f"{label} :: banked whitened (LW) — the letter"] = \
                    read(t["banked_whitened_LW"])
                blk[f"{label} :: fresh whitened (ridge) — this beside"] = \
                    read(t["fresh_whitened_ridge"])
                blk[f"{label} :: cos(banked_LW, fresh_ridge)"] = round(
                    cos(t["banked_whitened_LW"], t["fresh_whitened_ridge"]), 4)
            reads[gtag][family] = blk

    res = {
        "STATUS": "UNSTAMPED (C section 8) — BESIDE column; scores nothing, replaces nothing",
        "leg": "A8 Leg-6 / Item 1 beside — independently captured Sigma",
        "what_differs_from_the_letter": {
            "estimator": "fresh = sample covariance + ridge (1e-3 x mean eigenvalue); "
                         "banked = Ledoit-Wolf shrinkage. DIFFERENT ESTIMATORS.",
            "corpus": "fresh = vmb_stage0_dsv2_lite, 60 gens, this session; "
                      "banked = the 2026-07-19 build's own pole runs, n=160/class",
            "site": f"both at DSV2 L{TGT_SITE}",
            "shared": "the same banked V3raw_L{} is whitened by both".format(TGT_SITE),
        },
        "sigma_fresh": {"file": str(SIGMA_FRESH),
                        "ridge": round(float(z["ridge"]), 8),
                        "n_positions": int(z["n_positions"]),
                        "mean_eigenvalue": round(float(np.mean(z["evals"])), 6)},
        "reads": reads,
        "reading_aid": "agreement across the two columns => the letter's read does not "
                       "depend on one Sigma estimate, corpus, or shrinkage convention. "
                       "Disagreement => it localises to the estimator. The BANKED column "
                       "remains the letter of record either way.",
    }
    (OUT / "l4c_sigma_beside.json").write_text(json.dumps(res, indent=1))
    for gtag, fams in reads.items():
        for family, blk in fams.items():
            if family != PRIMARY_FAMILY:
                continue
            for k, v in blk.items():
                if isinstance(v, dict):
                    logger.info("%s | %s | %s: cos %+.4f (q95 %.4f, exceeds %s)",
                                gtag, family, k, v["cos"], v["null_q95_abs"],
                                v["exceeds_envelope"])
                else:
                    logger.info("%s | %s | %s = %s", gtag, family, k, v)
    logger.info("wrote %s", OUT / "l4c_sigma_beside.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
