"""A8 Leg-6 / Item 2 — the star factorization's FOURTH NODE (P8-STAR .50).

The star model says every pair's attenuation factors through per-model constants:
    a_hat(A->B) = c_A . c_B
Three banked pairs (3B<->8B .514, 8B<->Qwen .332, 8B<->DSV2 .266) leave the system
underdetermined; it closes on a_hat_direct(3B->Qwen) from L4-e. The k128 closure
reproduces add-4's filed constants exactly:
    c_3B = .6838, c_8B = .7517, c_Qwen = .4417, c_DSV2 = .3539
and predicts the one pair never fitted:  a_hat(3B->DSV2) = .2420, band [.19, .29].

This reads that pair for the first time. 3B was replayed over the LEG-2 corpus (the same
texts the DSV2 bank was built from, sha-verified byte-identical), giving paired states on
shared text; the fit grid then runs exactly as every other leg's.

RANK GUARD: n_train = 600, so k <= 600/1.2 = 500 and **k512 is rank-forbidden** on this
pair. The letter anticipated this ("proc at the rank-guarded k, k128 expected"), so k128
is the primary family here. Full panel reported beside, because the three INPUT a_hat's
are k512 reads — that is the filed family-mixing caveat, and it deserves to be visible
rather than asserted.

PANEL BESIDE (reported, unscored): the field axes and dir0. The dir0 row is the free
context add-4 asked for — does the MoE needle cliff reproduce from a DIFFERENT source
family? It is read three ways, because Leg-6 Item 1 established that the arm's dir0 is
not the same mode contrast on both sides:
  * vs the plain-bank V3_L18   — PAIR-MISMATCHED (3B dir0 = analogical/contrastive,
                                 DSV2 plain V3 = linear/socratic). This is the
                                 construction the banked cliff used.
  * vs v3whiten RAW L18        — PAIR-MATCHED, unwhitened.
  * vs v3whiten WHITENED L18   — PAIR-MATCHED and whitened (the L4-c letter's object).

UNSTAMPED (C section 8). No P self-scored — the desk scores P8-STAR.
Run: PYTHONPATH=pipeline python -m anamnesis.scripts.a8_leg6_star
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from anamnesis.scripts.a8_fit_g import load_transport_map
from anamnesis.scripts.a8_rosetta import _unit, cos, load_axes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_leg6_star")

ARM = Path("outputs/battery/arms/A8_conjugation")
LEG6 = ARM / "leg6"
OUT = LEG6 / "readouts_cpu"
BANK = Path("outputs/battery")

SRC_SITE = 14                      # 3B anchor site
FAMILIES = ("proc_k32", "proc_k128", "proc_k512", "ridge")
PRIMARY_FAMILY = "proc_k128"       # rank-guarded (n_train 600 -> k <= 500)
N_TRAIN = 600
RANK_GUARD_K = N_TRAIN / 1.2
BAND = (0.19, 0.29)
PREDICTED = 0.2420
N_RANDOM = 100
SEED = 80

TARGETS = {
    22: [("V7_tgt", "a5_vectors_dsv2_lite_b7_L22/a5_vectors.npz", "V7_L22")],
    18: [("dir0_tgt_PAIR_MISMATCHED", "a5_vectors_dsv2_lite/a5_vectors.npz", "V3_L18"),
         ("dir0_tgt_PAIR_MATCHED_raw",
          "a5_vectors_dsv2_lite_v3whiten/a5_vectors.npz", "V3raw_L18"),
         ("dir0_tgt_PAIR_MATCHED_whitened",
          "a5_vectors_dsv2_lite_v3whiten/a5_vectors.npz", "V3w_L18"),
         ("Vtemp_tgt", "a5_vectors_dsv2_lite_vtemp/a5_vectors.npz", "Vtemp_L18")],
}
SRC_FOR_TARGET = {
    "V7_tgt": "V7",
    "dir0_tgt_PAIR_MISMATCHED": "dir0",
    "dir0_tgt_PAIR_MATCHED_raw": "dir0",
    "dir0_tgt_PAIR_MATCHED_whitened": "dir0",
    "Vtemp_tgt": "Vtemp",
}


def _load(rel: str, key: str) -> np.ndarray:
    return _unit(np.load(BANK / rel, allow_pickle=True)[key].astype(np.float64))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    axes, _, pool = load_axes("3b")
    rng = np.random.default_rng(SEED)
    d_src = axes["V7"].vec.shape[0]
    randoms = rng.standard_normal((N_RANDOM, d_src))
    randoms /= np.linalg.norm(randoms, axis=1, keepdims=True)

    reads: dict[str, dict] = {}
    for t_site, entries in TARGETS.items():
        for family in FAMILIES:
            path = (LEG6 / "fits"
                    / f"fit_3bL{SRC_SITE}__dsv2-liteL{t_site}_native_{family}.npz")
            if not path.exists():
                continue
            tm = load_transport_map(path)
            nulls = ([_unit(tm.transport(r)) for r in randoms]
                     + [_unit(tm.transport(a.vec)) for a in pool])
            for name, rel, key in entries:
                try:
                    tgt = _load(rel, key)
                except (FileNotFoundError, KeyError) as exc:
                    reads.setdefault(name, {})[family] = {"error": str(exc)}
                    continue
                src = axes[SRC_FOR_TARGET[name]].vec
                u = _unit(tm.transport(src))
                c = cos(u, tgt)
                q95 = float(np.quantile([abs(cos(n, tgt)) for n in nulls], 0.95))
                reads.setdefault(name, {})[family] = {
                    "site_pair": f"3bL{SRC_SITE}->dsv2-liteL{t_site}",
                    "cos": round(c, 4), "null_q95_abs": round(q95, 4),
                    "exceeds_envelope": bool(abs(c) > q95),
                    "rank_forbidden": bool(family == "proc_k512"),
                }

    a_hat = {f: reads["V7_tgt"][f]["cos"] for f in FAMILIES if f in reads.get("V7_tgt", {})}
    primary = a_hat.get(PRIMARY_FAMILY)

    res = {
        "STATUS": "UNSTAMPED (C section 8) — no self-scored P; the desk scores P8-STAR (.50)",
        "leg": "A8 Leg-6 / Item 2 — the star's fourth node",
        "pairing": {
            "method": "3B replayed over the LEG-2 corpus (same texts the DSV2 bank was "
                      "built from), giving paired states on shared text",
            "corpus_manifest_sha256":
                "a6712ca0f3ce0e663dd8ad2ee156859552ee54c2acbfbdc88b5b896c1bc2aa76",
            "sha_verified_identical_to": "the DSV2 leg-2 collection stamp",
            "n_texts": 780, "n_train": N_TRAIN, "n_test": 180,
        },
        "rank_guard": {
            "n_train": N_TRAIN, "max_k": RANK_GUARD_K,
            "k512_forbidden": True,
            "primary_family": PRIMARY_FAMILY,
            "note": "the letter's 'k128 expected' anticipates this. The three INPUT "
                    "a_hat's that fix the star constants are k512 reads, so the "
                    "comparison mixes families — the filed caveat, made visible here "
                    "rather than asserted.",
        },
        "THE_SCORE_ROW": {
            "estimand": "a_hat(3B->DSV2) = cos(g . V7_3B, V7_dsv2_L22), native arm, "
                        "L22-side anchor, forward fit direction, single number (add-4 SCOPE)",
            "predicted_by_star": PREDICTED, "frozen_band": list(BAND),
            "observed_primary": primary,
            "observed_all_families": a_hat,
            "in_band_primary": (None if primary is None
                                else bool(BAND[0] <= primary <= BAND[1])),
            "in_band_by_family": {f: bool(BAND[0] <= v <= BAND[1])
                                  for f, v in a_hat.items()},
            "SCORING": "reported, NOT self-scored — the desk rules the family and the row",
        },
        "panel_beside_unscored": reads,
        "dir0_row_reading_note":
            "add-4 asked for the dir0 row as free context on the needle-cliff question. "
            "Leg-6 Item 1 found the arm's dir0 is analogical/contrastive on 3B and 8B but "
            "linear/socratic in the DSV2 plain bank, so the row is given three ways: the "
            "PAIR-MISMATCHED target (the construction the banked cliff used), and the "
            "PAIR-MATCHED raw and whitened targets. Read them together — a cliff that "
            "survives pair-matching means something different from one that does not.",
    }
    (OUT / "star_fourth_node.json").write_text(json.dumps(res, indent=1))

    logger.info("a_hat(3B->DSV2) by family: %s", a_hat)
    logger.info("primary (%s) = %s | star predicts %.4f, band %s",
                PRIMARY_FAMILY, primary, PREDICTED, BAND)
    for name, fams in reads.items():
        for f, r in fams.items():
            if "cos" in r:
                logger.info("  %-34s %-10s cos %+.4f (q95 %.4f, exceeds %s)%s",
                            name, f, r["cos"], r["null_q95_abs"], r["exceeds_envelope"],
                            "  [rank-forbidden]" if r["rank_forbidden"] else "")
    logger.info("wrote %s", OUT / "star_fourth_node.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
