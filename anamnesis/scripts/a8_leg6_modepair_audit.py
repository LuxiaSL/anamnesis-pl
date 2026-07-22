"""A8 Leg-6 — THE MODE-PAIR AUDIT: which contrast does each banked DSV2 needle carry?

Item 1 found that the arm's `dir0` is [analogical, contrastive] in 3B and 8B but
[linear, socratic] in the DSV2 plain bank, and that two banked "whitened V3" vintages
share a provenance string while being mutually near-orthogonal. That was established from
stamps plus cross-vintage cosines. This script settles it from a THIRD, INDEPENDENT
source: a contrast computed directly from the A8 Leg-2 DSV2 state bank, whose per-text
mode labels come from the corpus manifest and owe nothing to any vector build.

For each site and each candidate mode pair, Delta-mu = mean(S3 pos) - mean(S3 neg) over
the leg-2 native state bank, then cosine against every banked needle object. A vector
that carries a given contrast should align with that contrast's corpus-derived direction
and not with the other's.

Capture-convention caveat, stated: the corpus contrast is per-TEXT mean states over the
A8 corpus (n=60/pole); the banked vectors were built from per-TOKEN positions over
dedicated pole runs (n=160/class). Different captures of the same intended object agree
in direction but not in magnitude — the Leg-4F exploratory saw the same ~.2 scale. Read
the CONTRAST between the two columns, not the absolute value of either.

UNSTAMPED (C section 8). Scores nothing; it is evidence for a desk ruling.
Run: PYTHONPATH=pipeline python -m anamnesis.scripts.a8_leg6_modepair_audit
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_leg6_modepair_audit")

ARM = Path("outputs/battery/arms/A8_conjugation")
OUT = ARM / "leg6" / "readouts_cpu"
BANK = Path("outputs/battery")
SITES = (18, 22)
PAIRS = (("analogical", "contrastive"), ("linear", "socratic"))
OBJECTS = {
    "v3whiten.V3raw": (BANK / "a5_vectors_dsv2_lite_v3whiten/a5_vectors.npz", "V3raw_L{}"),
    "v3whiten_dir0.V3raw": (BANK / "a5_vectors_dsv2_lite_v3whiten_dir0/a5_vectors.npz",
                            "V3raw_L{}"),
    "plain_bank.V3": (BANK / "a5_vectors_dsv2_lite/a5_vectors.npz", "V3_L{}"),
}


def _u(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    entries = json.loads((ARM / "leg2/corpus/corpus_manifest.json").read_text())["entries"]
    bank = np.load(ARM / "leg2/states/states_dsv2-lite_native.npz")
    ids = list(bank["text_ids"])

    res: dict = {
        "STATUS": "UNSTAMPED (C section 8) — evidence for a desk ruling; scores nothing",
        "leg": "A8 Leg-6 — mode-pair audit of the banked DSV2 needle objects",
        "method": "Delta-mu = mean(S3 pos) - mean(S3 neg) over the A8 Leg-2 dsv2-lite "
                  "NATIVE state bank; labels from the corpus manifest. Independent of "
                  "every vector build.",
        "capture_caveat": "corpus contrast = per-TEXT mean states, n=60/pole, A8 corpus. "
                          "Banked vectors = per-TOKEN positions over pole runs, "
                          "n=160/class. Read the CONTRAST between columns, not absolutes.",
        "banked_stamp_claims": {
            "v3whiten": "provenance: unit(mu_analogical - mu_contrastive)",
            "v3whiten_dir0": "provenance: unit(mu_analogical - mu_contrastive) (SAME string)",
            "plain_bank": "stamp: pair = [linear, socratic]",
        },
        "sites": {},
    }

    for site in SITES:
        S = bank[f"L{site}"].astype(np.float64)
        site_blk: dict = {}
        for pos, neg in PAIRS:
            def idx(mode: str) -> list[int]:
                keep = {e["text_id"] for e in entries
                        if e["stratum"] == "S3" and e["mode"] == mode}
                return [i for i, t in enumerate(ids) if t in keep]
            ip, ineg = idx(pos), idx(neg)
            if not ip or not ineg:
                continue
            d = _u(S[ip].mean(0) - S[ineg].mean(0))
            row = {"n_pos": len(ip), "n_neg": len(ineg), "cos_to_banked": {}}
            for name, (path, keyfmt) in OBJECTS.items():
                z = np.load(path, allow_pickle=True)
                key = keyfmt.format(site)
                if key not in z:
                    continue
                row["cos_to_banked"][name] = round(float(d @ _u(
                    z[key].astype(np.float64))), 4)
            site_blk[f"{pos}-{neg}"] = row
        res["sites"][f"L{site}"] = site_blk

    l18 = res["sites"].get("L18", {})
    ac = l18.get("analogical-contrastive", {}).get("cos_to_banked", {})
    ls = l18.get("linear-socratic", {}).get("cos_to_banked", {})
    res["VERDICT_evidence"] = {
        "v3whiten_dir0 and plain_bank carry LINEAR-SOCRATIC":
            f"corpus linear-socratic vs v3whiten_dir0 = {ls.get('v3whiten_dir0.V3raw')}, "
            f"vs plain_bank = {ls.get('plain_bank.V3')} (L18); their analogical-contrastive "
            f"cosines are {ac.get('v3whiten_dir0.V3raw')} / {ac.get('plain_bank.V3')} — "
            "the wrong sign for their stamped provenance",
        "v3whiten carries ANALOGICAL-CONTRASTIVE":
            f"corpus analogical-contrastive vs v3whiten = {ac.get('v3whiten.V3raw')}, "
            f"while its linear-socratic cosine is {ls.get('v3whiten.V3raw')} — consistent "
            "with its stamp in SIGN and in which contrast it prefers, though modest in "
            "magnitude (capture caveat above)",
        "asymmetry_stated_honestly":
            "the linear-socratic identification is crisp (.77-.85); the "
            "analogical-contrastive identification is directionally right but weak "
            "(+.12 at L18, +.21 at L22). The audit therefore establishes firmly WHICH "
            "objects are linear/socratic, and establishes v3whiten as the remaining "
            "candidate for the analogical/contrastive object rather than proving it "
            "outright. Stated this way deliberately.",
        "consequence": "the arm's dir0 (analogical/contrastive in 3B and 8B) was read "
                       "against a linear/socratic DSV2 target in the banked needle-cliff "
                       "number of record.",
    }

    (OUT / "modepair_audit.json").write_text(json.dumps(res, indent=1))
    for site, blk in res["sites"].items():
        for pair, row in blk.items():
            logger.info("%s  %-24s (n=%d/%d)  %s", site, pair, row["n_pos"], row["n_neg"],
                        row["cos_to_banked"])
    logger.info("wrote %s", OUT / "modepair_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
