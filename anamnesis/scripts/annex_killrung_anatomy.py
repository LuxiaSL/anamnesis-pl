"""ANNEX Tenant-A — killrung-winner anatomy: what does a searched gauge-aimer look like,
materially? (Predictions KA-1..3 frozen in the ledger S8-4 BEFORE this ran.)

Decomposes route5_killrung/best_candidate.npy (direction only; banked norm echoed as a
constructional fact) in the banked Sigma_L14 eigenbasis with the SAME conventions as the
banked covariance screen (ridge = 1e-3 x mean eigenvalue; bottom/top-768 eigenmass) and the
b7 band profile (descending order, band [16:256]); cosine table against every banked L14
unit vector. The 8 shuffled-null AXES are feature-space (3,358-d) — no cosine to a
residual-space vector exists; the summary's own-draw/other-draw ACC table is the aim record.

CPU-only, banked inputs only. Annex write-surface.

Usage (from pipeline/):
    python -m anamnesis.scripts.annex_killrung_anatomy \
        --winner ../outputs/battery/annex/route5_killrung/best_candidate.npy \
        --sigma ../outputs/battery/arms/A5/a5_sigma_L14_3b.npz \
        --screen ../outputs/battery/arms/A5/a5_covariance_screen_3b.json \
        --vectors ../outputs/battery/a5_vectors_3b/a5_vectors.npz \
        --b7-vectors ../outputs/battery/a5_vectors_3b_b7/a5_vectors.npz \
        --ra-vectors ../outputs/battery/annex/a5_vectors_3b_14r/a5_vectors.npz \
        --out ../outputs/battery/annex/annex_killrung_winner_anatomy.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BAND = (16, 256)
BOTTOM_K = 768


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--winner", type=Path, required=True)
    ap.add_argument("--sigma", type=Path, required=True)
    ap.add_argument("--screen", type=Path, required=True)
    ap.add_argument("--vectors", type=Path, required=True)
    ap.add_argument("--b7-vectors", type=Path, required=True)
    ap.add_argument("--ra-vectors", type=Path, required=True)
    ap.add_argument("--ridge-rel", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    w_raw = np.load(args.winner).astype(np.float64)
    banked_norm = float(np.linalg.norm(w_raw))
    w = w_raw / banked_norm

    S = np.load(args.sigma)
    evals = S["evals"].astype(np.float64)
    evecs = S["evecs"].astype(np.float64)
    ridge = args.ridge_rel * float(evals.mean())
    banked = json.loads(args.screen.read_text())["sites"]["14"]
    if not np.isclose(ridge, banked["ridge"], rtol=1e-6):
        raise SystemExit(f"ridge mismatch vs banked screen: {ridge} vs {banked['ridge']}")

    coeff = evecs.T @ w
    maha = float((coeff ** 2 / (evals + ridge)).sum())
    asc = np.argsort(evals)
    m = coeff ** 2
    bottom = float(m[asc[:BOTTOM_K]].sum())
    top = float(m[asc[-BOTTOM_K:]].sum())
    order = np.argsort(evals)[::-1]
    md = (evecs[:, order].T @ w) ** 2
    # cumulative mass curve (descending): rank needed to hold 50/90% of the vector
    cum = np.cumsum(md)
    rank50 = int(np.searchsorted(cum, 0.5) + 1)
    rank90 = int(np.searchsorted(cum, 0.9) + 1)

    cos_table = {}
    for bank_path in (args.vectors, args.b7_vectors, args.ra_vectors):
        bank = np.load(bank_path)
        for name in bank.files:
            if name.endswith("_L14") or name in ("R1", "R2", "R3"):
                v = bank[name].astype(np.float64)
                cos_table[name] = float(w @ (v / np.linalg.norm(v)))

    out = {
        "provenance": "annex Tenant-A killrung-winner anatomy (ledger S8-4 predictions frozen "
                      "first); conventions = banked covariance screen + b7 band profile",
        "banked_norm_constructional": banked_norm,
        "site": 14, "ridge": ridge,
        "mahalanobis": maha,
        f"bottom_{BOTTOM_K}_eigenmass": bottom,
        f"top_{BOTTOM_K}_eigenmass": top,
        "tail_over_top": bottom / max(top, 1e-12),
        "mass_top16": float(md[:BAND[0]].sum()),
        "mass_band16_256": float(md[BAND[0]:BAND[1]].sum()),
        "mass_tail256plus": float(md[BAND[1]:].sum()),
        "rank_for_50pct_mass_desc": rank50,
        "rank_for_90pct_mass_desc": rank90,
        "cos_to_banked_L14": cos_table,
        "max_abs_cos": float(max(abs(c) for c in cos_table.values())),
        "comparators_echo": {k: {"mahalanobis": v["mahalanobis"],
                                 "tail_over_top": v["tail_over_top"]}
                             for k, v in banked["vectors"].items()},
        "summary_echo": {"acc_sel_on_target": 0.23376540244988547,
                         "acc_sel_on_gauge_FULL3358": 0.0450638380449184,
                         "cos_to_V3_L14": 0.01974791350314789,
                         "cos_to_V4_L14": -0.004200324674913734},
    }
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
