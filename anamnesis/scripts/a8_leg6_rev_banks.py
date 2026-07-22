"""A8 Leg-6 / Item 5 — reverse behavioral certification (P8-REV .75): injection bank.

The arm has never steered the SOURCE-side model. Every behavioral leg so far pushed a
vector FORWARD (3B->8B, 8B->Qwen, 8B->DSV2, and the owl's Qwen->8B which is reverse but
into a target that had also been a forward target). This leg carries 8B's V7 BACK into
3B through the transpose of the banked Leg-0 anchor Procrustes and asks whether it
writes 3B's entropy on the standard frame.

CONSTRUCTION (add-5, frozen):
  * map        = REVERSE of the Leg-0 anchor fit fit_3bL14__8bL16_native_proc_k512
                 (TransportMap.transport(..., direction="rev") — the adjoint; for
                 Procrustes this is the exact inverse of the orthogonal part).
                 Procrustes is the family of record for reverse-direction claims (rake-15).
  * vector     = 8B V7_L16 (a5_vectors_8b_b7), sign-anchored through a8_rosetta.load_axes
                 BEFORE transport (standing rule: sign-anchor before ANY banked vector use).
  * controls   = the REVERSE-TRANSPORTED R band: 8B's own banked Rband1-3 carried back
                 through the SAME map. Native 3B R cells would not be the right control —
                 the question is whether the TRANSPORT carries signal, so the control must
                 travel the same road. add-5 pre-authorises fresh cells for exactly this.
  * doses      = +-{.03, .1, .3} x 3B's per-token median residual norm at L14, plus an
                 alpha=0 baseline. Full signed ladder (the SCOPE clause: monotone across
                 the whole signed ladder, sign flipping through zero).
  * n          = 80/cell (20 topics x 4 stage-0 strata x 1 seed), max_new_tokens 512,
                 attn eager, bare system prompt — the standard entropy frame.

NORM CONVENTION, NAMED: alpha uses the a5 stamps' per-token median residual norm
(L14 = 12.2391), which is the convention every prior behavioral leg used. The A8 leg-0
collection stamp carries a slightly different number for the same site (12.1125) because
it is measured over the A8 corpus rather than the a5 stage-0 pool. The a5 number is used;
the other is recorded beside so the difference is visible rather than silent.

IDENTITY CHECK before anything is written: cos(g_rev . V7_8B, V7_3B) must reproduce the
banked Leg-0 REVERSE read (+.531). If it does not, the leg parks with the trace.

UNSTAMPED (C section 8). No P self-scored. Run (repo root):
  PYTHONPATH=pipeline python -m anamnesis.scripts.a8_leg6_rev_banks
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from anamnesis.scripts.a8_fit_g import load_transport_map
from anamnesis.scripts.a8_rosetta import _unit, cos, load_axes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_leg6_rev_banks")

ARM = Path("outputs/battery/arms/A8_conjugation")
LEG6 = ARM / "leg6"
NODE_ROOT = "/models/anamnesis-extract/battery/arms/A8_conjugation/leg6"
FIT = ARM / "fits" / "fit_3bL14__8bL16_native_proc_k512.npz"
THREEB_STAMPS = Path("outputs/battery/a5_vectors_3b/a5_vectors_stamps.json")
SITE = 14
DOSES = (0.03, 0.1, 0.3, -0.03, -0.1, -0.3)
LEG0_REVERSE_READ = 0.531          # banked Leg-0 reverse V7 read; the identity target
IDENT_TOL = 0.02
A8_CORPUS_NORM_BESIDE = 12.1125    # A8 leg-0 collection stamp, same site, other pool


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if not FIT.exists():
        raise SystemExit(f"leg-0 anchor fit missing: {FIT}")
    tm = load_transport_map(FIT)

    src_axes, _, src_pool = load_axes("8b")        # 8B: the vector's home
    tgt_axes, _, _ = load_axes("3b")               # 3B: where it is being carried

    v7_8b = src_axes["V7"].vec                     # sign-anchored by load_axes
    rev_v7 = _unit(tm.transport(v7_8b, direction="rev"))

    identity = {
        "cos_grev_V7_8b_vs_V7_3b": round(cos(rev_v7, tgt_axes["V7"].vec), 4),
        "banked_leg0_reverse_read": LEG0_REVERSE_READ,
        "tolerance": IDENT_TOL,
    }
    identity["PASS"] = bool(abs(identity["cos_grev_V7_8b_vs_V7_3b"]
                                - LEG0_REVERSE_READ) <= IDENT_TOL)
    if not identity["PASS"]:
        raise SystemExit(f"IDENTITY CHECK FAILED — parking Item 5 per standing rule "
                         f"({identity})")

    vecs = {f"gRV7_L{SITE}": rev_v7}
    for i, a in enumerate(src_pool[:3]):           # 8B's banked Rband1-3
        vecs[f"gRband{i+1}_L{SITE}"] = _unit(tm.transport(a.vec, direction="rev"))

    norm = float(json.loads(THREEB_STAMPS.read_text())["median_resid_norms"][f"L{SITE}"])

    out_dir = LEG6 / "vectors"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / "a8_leg6_rev_vectors_3b.npz"
    np.savez(npz, **{k: v.astype(np.float32) for k, v in vecs.items()})
    (out_dir / "a8_leg6_rev_norms_3b.json").write_text(
        json.dumps({"median_resid_norms": {f"L{SITE}": norm}}, indent=1))

    stamp = {
        "grade": "UNSTAMPED (C section 8) — staging for Leg-6 Item 5 (P8-REV .75)",
        "prereg": "A8-add-5 P8-REV",
        "builder": "a8_leg6_rev_banks.py",
        "direction": "8B -> 3B via the REVERSE (transpose/adjoint) of the banked Leg-0 "
                     "anchor Procrustes; Procrustes = family of record for "
                     "reverse-direction claims (rake-15)",
        "closes": "the 'nothing has ever steered the SOURCE-side model' gap in the arm record",
        "target": "3b", "inject_site": SITE,
        "fit_file": str(FIT), "fit_sha256": _sha(FIT),
        "sign_anchor": "V7 orientation taken from a8_rosetta.load_axes('8b') BEFORE "
                       "transport (standing rule: sign-anchor before any banked vector use)",
        "identity_check": identity,
        "vectors_stored": "UNIT (the write hook re-normalises; only ORIENTATION is "
                          "load-bearing)",
        "norm_convention_for_alpha": {
            "convention": "PER-TOKEN median residual norm (a5 stamps) — the convention "
                          "every prior behavioural leg used",
            "file": str(THREEB_STAMPS), f"L{SITE}": norm,
            "beside_A8_corpus_value": A8_CORPUS_NORM_BESIDE,
            "note": "the A8 leg-0 collection stamp measures the same site over the A8 "
                    "corpus and gets 12.1125; the a5 number is used, the difference is "
                    "recorded rather than silently absorbed",
        },
        "controls": {
            "reverse_transported_R_band": "gRband1-3 = 8B's banked R-band members carried "
                                          "back through the SAME reverse map — the control "
                                          "travels the same road as the signal",
            "why_not_native_3B_R": "the question is whether the TRANSPORT carries signal; "
                                   "a native 3B R band would not be dose- or "
                                   "construction-matched to the transported vector",
            "baseline": "alpha=0 cell",
        },
        "doses_alpha_frac": list(DOSES),
        "n_per_cell": 80,
        "readout": "standard entropy frame (vmb_c3_entropy_replay), n=80/cell",
        "scope_clause": "P8-REV reads dose-ordering across the FULL SIGNED ladder "
                        "+-{.03,.1,.3}, sign flipping through zero; band comparison at "
                        ">=5/6 doses (add-5 as frozen)",
    }
    (out_dir / "a8_leg6_rev_stamps.json").write_text(json.dumps(stamp, indent=1))

    cells = [{"out_run_dir": f"{NODE_ROOT}/runs/3b/baseline_L{SITE}_a0.00",
              "seed_namespace": f"A8L6REV-3b-baseline_L{SITE}_a0.00",
              "inject_key": None, "inject_layer": None, "inject_alpha_frac": None}]
    for key in vecs:
        for d in DOSES:
            cid = f"{key}_a{d:+.2f}"
            cells.append({"out_run_dir": f"{NODE_ROOT}/runs/3b/{cid}",
                          "seed_namespace": f"A8L6REV-3b-{cid}",
                          "inject_key": key, "inject_layer": SITE,
                          "inject_alpha_frac": d})
    (LEG6 / "cells").mkdir(parents=True, exist_ok=True)
    (LEG6 / "cells/l6rev_cells_3b.json").write_text(
        json.dumps({"cells": cells, "_stamp": stamp}, indent=1))

    logger.info("identity: cos(g_rev.V7_8B, V7_3B) = %+.4f (banked leg-0 reverse %.3f) -> %s",
                identity["cos_grev_V7_8b_vs_V7_3b"], LEG0_REVERSE_READ,
                "PASS" if identity["PASS"] else "FAIL")
    logger.info("bank: %s (%d vectors); %d cells x 80 gens; alpha norm L%d = %.4f",
                npz, len(vecs), len(cells), SITE, norm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
