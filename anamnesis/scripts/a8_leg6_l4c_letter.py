"""A8 Leg-6 / Item 1 — the L4-c letter, UN-PARKED.

THE UNPARK (CP-0 finding, 2026-07-23). Leg-4F parked this letter because the whitened
target could not be built from banked inputs. Two of its three named blockers dissolve:

  1. "No banked class means." The letter's object needs unit(Sigma^-1 . Delta-mu), and
     Delta-mu enters ONLY through a positive scalar: V3raw is banked as
     unit(mu_analogical - mu_contrastive) with raw_norm recorded, so
     unit(Sigma^-1 Delta-mu) == unit(Sigma^-1 . V3raw) IDENTICALLY, sign included.
     Class means were never actually required.
  2. "The 2026-07-19 whiten build wrote stamps only, no vectors npz." The npz exists —
     node-side, under battery/a5_vectors_dsv2_lite_v3whiten{,_dir0}/, not under
     arms/A5_dsv2/whiten*/ where Leg-4F looked. It carries **V3w_L18** built to exactly
     the stamped recipe: unit(Sigma^-1(mu_analogical - mu_contrastive)), Sigma =
     Ledoit-Wolf, shrinkage logged, n_pos/n_neg = 160/160, on the dir0 mode pair.

So the letter reads on the BANKED object, at its own recipe. No Sigma re-derivation, no
substitution, no amendment. The fresh Sigma@L18 capture the baton specifies runs BESIDE
this (a8_leg6_l4c_sigma_beside.py) as the corpus- and convention-independent replication.

CONVENTION TRAP, NAMED (rake): vmb_a5_covariance_screen computes a RIDGE-regularized
sample Sigma (ridge = 1e-3 * mean eigenvalue), NOT Ledoit-Wolf. A fresh capture therefore
builds a DIFFERENT-convention object than the recipe this letter freezes. That is why the
banked LW object is primary and the fresh capture is a beside, not a replacement.

VINTAGE AMBIGUITY, CARRIED (the Leg-4F rake, now with its L18 face): two builds share the
recipe string and the mode pair but differ in shrinkage —
  a5_vectors_dsv2_lite_v3whiten       L18: lw_shrinkage .2577, cos_delta_whitened .1044
  a5_vectors_dsv2_lite_v3whiten_dir0  L18: lw_shrinkage .1205, cos_delta_whitened .3933
Both are read. Neither is silently preferred; the desk rules which is the object.

IDENTITY CHECK FIRST (no-peeking discipline): the same g and same source vector against
the RAW target must reproduce the banked number of record, cos = -.0865 (Leg-3 curve
rows, dsv2 dir0 cliff). Only if that reproduces does the whitened swap mean anything —
the raw->whitened contrast is then the ONLY thing that changed.

UNSTAMPED (C section 8). No P is self-scored here; the desk scores P8-L4c (.35).
Run: PYTHONPATH=pipeline python -m anamnesis.scripts.a8_leg6_l4c_letter
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from anamnesis.scripts.a8_fit_g import load_transport_map
from anamnesis.scripts.a8_rosetta import _unit, cos, load_axes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_leg6_l4c_letter")

ARM = Path("outputs/battery/arms/A8_conjugation")
OUT = ARM / "leg6" / "readouts_cpu"
BANK = Path("outputs/battery")

# The target site. V3raw/V3w are banked at DSV2 L18; the leg-2 fit grid covers
# 8bL16 -> dsv2-liteL18, so the letter reads at its own site with no substitution.
SRC_SITE, TGT_SITE = 16, 18
FAMILIES = ("proc_k32", "proc_k128", "proc_k512", "ridge")
PRIMARY_FAMILY = "proc_k512"      # leg-2 family of record (the -.0865 number's family)
RECORD_RAW_COS = -0.0865          # Leg-3 curve rows; the identity check's target
RECORD_TOL = 0.02

VINTAGES = {
    # PAIR-MATCHED to dir0_8B (analogical vs contrastive) — see MODE_PAIR_AUDIT below
    "v3whiten [analogical/contrastive — PAIR-MATCHED] (lw .2577)":
        BANK / "a5_vectors_dsv2_lite_v3whiten" / "a5_vectors.npz",
    # NOT pair-matched: its raw sits at cos .921 with the plain bank's linear/socratic V3_L18
    "v3whiten_dir0 [linear/socratic — PAIR-MISMATCHED] (lw .1205)":
        BANK / "a5_vectors_dsv2_lite_v3whiten_dir0" / "a5_vectors.npz",
}
N_RANDOM = 100
SEED = 80


def _fit_path(family: str, modefree: bool) -> Path:
    sub = "fits_modefree" if modefree else "fits"
    return (ARM / "leg2" / sub
            / f"fit_8bL{SRC_SITE}__dsv2-liteL{TGT_SITE}_native_{family}.npz")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    src_axes, _, src_pool = load_axes("8b")
    dir0 = src_axes["dir0"].vec
    rng = np.random.default_rng(SEED)
    randoms = rng.standard_normal((N_RANDOM, dir0.shape[0]))
    randoms /= np.linalg.norm(randoms, axis=1, keepdims=True)

    # ---- targets: banked, at their own site, no reconstruction --------------------
    targets: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, dict] = {}
    for label, path in VINTAGES.items():
        z = np.load(path, allow_pickle=True)
        key_w, key_r = f"V3w_L{TGT_SITE}", f"V3raw_L{TGT_SITE}"
        if key_w not in z:
            logger.warning("%s: no %s, skipping", label, key_w)
            continue
        targets[label] = {"whitened": _unit(z[key_w]), "raw": _unit(z[key_r])}
        st = json.loads((path.parent / "a5_vectors_stamps.json").read_text())
        diagnostics[label] = st["diagnostics"].get(f"L{TGT_SITE}", {})

    # The independently-banked raw needle of record (plain bank, linear/socratic pair)
    plain = np.load(BANK / "a5_vectors_dsv2_lite" / "a5_vectors.npz", allow_pickle=True)
    v3_plain = _unit(plain[f"V3_L{TGT_SITE}"])

    # ---- cross-vintage diagnostic -------------------------------------------------
    # The two vintages claim the SAME construction (unit(mu_analogical - mu_contrastive)
    # at L18) and differ only in the shrinkage used for the whitening. Their RAW vectors
    # must therefore agree up to capture noise. If they do NOT, the vintage problem is
    # deeper than a shrinkage choice and the desk needs to see that before ruling.
    labels = list(targets)
    cross: dict[str, float | None] = {}
    if len(labels) == 2:
        a, b = targets[labels[0]], targets[labels[1]]
        cross = {
            f"cos(RAW[{labels[0]}], RAW[{labels[1]}])": round(cos(a["raw"], b["raw"]), 4),
            f"cos(WHITENED[{labels[0]}], WHITENED[{labels[1]}])":
                round(cos(a["whitened"], b["whitened"]), 4),
            f"cos(RAW[{labels[0]}], plain_bank_V3_L18)": round(cos(a["raw"], v3_plain), 4),
            f"cos(RAW[{labels[1]}], plain_bank_V3_L18)": round(cos(b["raw"], v3_plain), 4),
            "note": "plain_bank_V3_L18 is the linear/socratic pair; the vintages are "
                    "analogical/contrastive — they are NOT expected to align with it. "
                    "The two vintages ARE expected to align with each other.",
        }

    results: dict[str, dict] = {}
    for modefree in (False, True):
        gtag = "mode_free_g (S1+S2 fit; never saw S3)" if modefree else "primary_g (S1+S2+S3)"
        results[gtag] = {}
        for family in FAMILIES:
            p = _fit_path(family, modefree)
            if not p.exists():
                continue
            tm = load_transport_map(p)
            ug = _unit(tm.transport(dir0))
            nulls = ([_unit(tm.transport(r)) for r in randoms]
                     + [_unit(tm.transport(a.vec)) for a in src_pool])

            def read(tgt: np.ndarray) -> dict:
                c = cos(ug, tgt)
                q95 = float(np.quantile([abs(cos(n, tgt)) for n in nulls], 0.95))
                return {"cos": round(c, 4), "null_q95_abs": round(q95, 4),
                        "exceeds_envelope": bool(abs(c) > q95), "n_nulls": len(nulls)}

            block = {"plain_bank_V3_L18_raw (linear/socratic pair)": read(v3_plain)}
            for label, t in targets.items():
                block[f"{label} :: RAW target"] = read(t["raw"])
                block[f"{label} :: WHITENED target (the letter's object)"] = read(t["whitened"])
                block[f"{label} :: cos(raw, whitened)"] = round(cos(t["raw"], t["whitened"]), 4)
            results[gtag][family] = block

    # ---- identity check ----------------------------------------------------------
    got = results["primary_g (S1+S2+S3)"][PRIMARY_FAMILY][
        "plain_bank_V3_L18_raw (linear/socratic pair)"]["cos"]
    identity_ok = abs(got - RECORD_RAW_COS) <= RECORD_TOL

    res = {
        "STATUS": "UNSTAMPED (C section 8) — no self-scored P; the desk scores P8-L4c (.35)",
        "leg": "A8 Leg-6 / Item 1 — L4-c letter, UN-PARKED",
        "unpark_basis": {
            "class_means_not_required":
                "V3raw is banked as unit(mu_pos - mu_neg) with raw_norm recorded, so "
                "unit(Sigma^-1 Delta-mu) == unit(Sigma^-1 . V3raw) identically, sign included",
            "vectors_npz_located":
                "battery/a5_vectors_dsv2_lite_v3whiten{,_dir0}/a5_vectors.npz (node-side); "
                "Leg-4F searched arms/A5_dsv2/whiten*/ and found stamps only",
            "no_sigma_rederivation": "the banked V3w_L18 IS the stamped recipe's output; "
                                     "nothing is re-derived here",
        },
        "convention_trap_named":
            "vmb_a5_covariance_screen builds a RIDGE-regularized sample Sigma "
            "(ridge = 1e-3 * mean eigenvalue), NOT Ledoit-Wolf. The fresh Sigma@L18 capture "
            "is therefore a different-convention object and runs BESIDE this letter, not "
            "as its input. See a8_leg6_l4c_sigma_beside.py.",
        "vintage_ambiguity_carried": diagnostics,
        "cross_vintage_diagnostic": cross,
        "MODE_PAIR_AUDIT": {
            "finding": "the arm's dir0 is NOT the same mode contrast on both sides of the "
                       "8B->DSV2 read. Banked stamps: 3B V3_* and 8B V3_* are "
                       "pair=[analogical, contrastive]; DSV2 plain-bank V3_* is "
                       "pair=[linear, socratic].",
            "consequence_for_the_number_of_record":
                "the banked MoE needle cliff (cos = -.0865) was computed as "
                "cos(g.dir0_8B[analogical-contrastive], V3_L18_dsv2[LINEAR-SOCRATIC]) — "
                "source and target are different contrasts. The cliff therefore conflates "
                "(i) a mode-pair mismatch with (ii) raw-vs-whitened target construction. "
                "It is not, on this evidence alone, a transport failure.",
            "which_vintage_is_which":
                "cos(v3whiten.raw, plain V3_L18) = .1025 and cos(v3whiten_dir0.raw, "
                "plain V3_L18) = .9212 at L18 (and the two vintages' raws are mutually "
                "near-orthogonal: +.084 at L18, -.027 at L22). So v3whiten carries the "
                "analogical/contrastive contrast its provenance claims, while "
                "v3whiten_dir0 reproduces the plain bank's linear/socratic direction "
                "despite carrying the SAME provenance string. One of the two stamps is "
                "stale — a banked-artifact provenance defect, filed as a rake.",
            "how_this_readout_handles_it":
                "both vintages are read and labelled by their MEASURED pair, not their "
                "stamp. The pair-MATCHED row is the one that asks the letter's question. "
                "No vintage is silently preferred and nothing is re-banked here; the desk "
                "rules which object is of record.",
            "scope": "read-only; touches P8-L4c directly and the difficulty curve's needle "
                     "series by implication. NOT self-scored.",
        },
        "identity_check": {
            "purpose": "same g, same source vector, RAW target must reproduce the banked "
                       "cliff of record before the whitened swap can mean anything",
            "record_cos": RECORD_RAW_COS, "observed_cos": got,
            "tolerance": RECORD_TOL, "PASS": identity_ok,
            "family": PRIMARY_FAMILY, "map": "primary_g",
        },
        "site": {"source": f"8b L{SRC_SITE}", "target": f"dsv2-lite L{TGT_SITE}",
                 "note": "the letter reads at the site where V3raw/V3w are banked; "
                         "no site substitution"},
        "sign_convention": "+ = analogical - contrastive (dir0's banked recipe order), "
                           "matched on both sides",
        "reads": results,
    }
    (OUT / "l4c_letter_unparked.json").write_text(json.dumps(res, indent=1))
    logger.info("identity check: observed %.4f vs record %.4f -> %s",
                got, RECORD_RAW_COS, "PASS" if identity_ok else "FAIL")
    for gtag, fams in results.items():
        for family, block in fams.items():
            for k, v in block.items():
                if isinstance(v, dict):
                    logger.info("%s | %s | %s: cos %+.4f (q95 %.4f, exceeds %s)",
                                gtag, family, k, v["cos"], v["null_q95_abs"],
                                v["exceeds_envelope"])
    logger.info("wrote %s", OUT / "l4c_letter_unparked.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
