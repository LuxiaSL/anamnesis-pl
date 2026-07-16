"""14k assay — Kreg/Ktopic shape null + cos-family distinctness (session-10 Part A3).

Closes the provenance gap and the AMENDMENT-1 concern on the register/topic write-anatomy
shape calls. Two CPU legs on banked Σ_L14 (no GPU):

  (1) cos-family distinctness: cos(Kreg/Ktopic, {V3, Ksoclin, V1b}) — is the register axis a
      distinct member of the needle census, or a rotated copy of an existing needle?

  (2) shape null: AMENDMENT-1 warned that a topic Δμ lives in Σ's top eigenspace by
      construction and Σ-weighted noise is top-heavy by expectation, so a low-mahal / no-tail
      shape is uninformative UNNULLED. We build the null shape-stat distribution from random
      vectors under two models — ISOTROPIC (u~N(0,I)) and Σ-SHAPED (g~N(0,Σ), "Σ-weighted
      noise") — and place V3 (needle anchor), V7 (band-field anchor), Kreg, Ktopic as
      percentiles. A vector whose (mahal, tail, band) sits INSIDE the Σ-shaped null envelope
      is shape-uninformative → per the assay law, its shape call is OUT-OF-VOCABULARY.

NOTE this is the CONSTRUCTION-AGNOSTIC null (random vectors through the same Σ geometry),
which is what AMENDMENT-1 literally names ("Σ-weighted noise top-heavy by expectation"). The
LITERAL shuffled-pole Δμ null requires the bare-sort→pole-decile construction (per-gen sort
data) and rides with the owed GPU bare-sort rebuild — flagged, not silently substituted.
CPU. First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def shape_stats(v: np.ndarray, evals: np.ndarray, evecs: np.ndarray, ridge: float) -> dict:
    """mahalanobis vᵀΣ⁻¹v + top/tail/band eigenmass, on a UNIT vector (evecs desc by eval)."""
    v = v / np.linalg.norm(v)
    c = evecs.T @ v            # coordinates in eigenbasis
    c2 = c ** 2
    tot = c2.sum()
    mahal = float((c2 / (evals + ridge)).sum())
    return {"mahalanobis": mahal,
            "top768_eigenmass": float(c2[:768].sum() / tot),
            "tail768_eigenmass": float(c2[-768:].sum() / tot),
            "band_16_256_eigenmass": float(c2[16:256].sum() / tot)}


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-npz", type=Path, required=True)
    ap.add_argument("--assay-npz", type=Path, required=True, help="assay_cd_vectors.npz (Kreg/Ktopic)")
    ap.add_argument("--v3-npz", type=Path, required=True)
    ap.add_argument("--soclin-npz", type=Path, required=True)
    ap.add_argument("--v1b-npz", type=Path, required=True)
    ap.add_argument("--v7-npz", type=Path, default=None, help="optional band-field anchor (b7)")
    ap.add_argument("--site", default="L14")
    ap.add_argument("--k", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260716)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    S = np.load(args.sigma_npz)
    evals, evecs, ridge = S["evals"], S["evecs"], float(S["ridge"])
    order = np.argsort(evals)[::-1]           # ensure descending
    evals, evecs = evals[order], evecs[:, order]

    vecs = {}
    vecs["Kreg"] = np.load(args.assay_npz)[f"Kreg_{args.site}"]
    vecs["Ktopic"] = np.load(args.assay_npz)[f"Ktopic_{args.site}"]
    vecs["V3"] = np.load(args.v3_npz)[f"V3_{args.site}"]
    vecs["Ksoclin"] = np.load(args.soclin_npz)[f"Ksoclin_{args.site}"]
    vecs["V1b"] = np.load(args.v1b_npz)[f"V1b_{args.site}"]
    if args.v7_npz and args.v7_npz.exists():
        d7 = np.load(args.v7_npz)
        k7 = next((k for k in d7 if k.startswith("V7")), None)
        if k7:
            vecs["V7"] = d7[k7]

    # (1) cos-family distinctness
    census = ["V3", "Ksoclin", "V1b"]
    cos_family = {}
    for probe in ("Kreg", "Ktopic"):
        cos_family[probe] = {ref: round(cos(vecs[probe], vecs[ref]), 4) for ref in census}
    cos_family["Kreg_vs_Ktopic"] = round(cos(vecs["Kreg"], vecs["Ktopic"]), 4)

    # observed shape stats
    obs = {name: shape_stats(v, evals, evecs, ridge) for name, v in vecs.items()}

    # (2) shape null: isotropic + Σ-shaped random vectors
    rng = np.random.default_rng(args.seed)
    d = evecs.shape[0]
    sqrt_evals = np.sqrt(np.clip(evals, 0, None))
    null = {"isotropic": {"mahalanobis": [], "tail768_eigenmass": [], "band_16_256_eigenmass": []},
            "sigma_shaped": {"mahalanobis": [], "tail768_eigenmass": [], "band_16_256_eigenmass": []}}
    for _ in range(args.k):
        u = rng.standard_normal(d)
        si = shape_stats(u, evals, evecs, ridge)
        g = evecs @ (sqrt_evals * rng.standard_normal(d))   # ~ N(0, Σ)
        ss = shape_stats(g, evals, evecs, ridge)
        for m in null["isotropic"]:
            null["isotropic"][m].append(si[m]); null["sigma_shaped"][m].append(ss[m])

    def pct(model, metric, val):
        arr = np.asarray(null[model][metric])
        return round(float((arr < val).mean()) * 100, 1)  # percentile of val within null

    null_summary = {}
    for model in null:
        null_summary[model] = {m: {"mean": round(float(np.mean(null[model][m])), 4),
                                   "p5": round(float(np.percentile(null[model][m], 5)), 4),
                                   "p95": round(float(np.percentile(null[model][m], 95)), 4)}
                               for m in null[model]}
    placements = {}
    for name in obs:
        placements[name] = {}
        for model in null:
            placements[name][model] = {m: {"value": round(obs[name][m], 4),
                                           "null_percentile": pct(model, m, obs[name][m])}
                                       for m in null[model]}

    # per-assay-law shape calls (re-emit): NEEDLE = high-mahal + tail-conc, clearly outside Σ-null;
    # FIELD = band-conc + low-mahal like Σ-noise; AMBIGUOUS (inside Σ-null) = OUT-OF-VOCABULARY.
    def call(name):
        mp = placements[name]["sigma_shaped"]["mahalanobis"]["null_percentile"]
        tp = placements[name]["sigma_shaped"]["tail768_eigenmass"]["null_percentile"]
        # outside Σ-null on the high side of mahal AND tail ⇒ NEEDLE; indistinguishable ⇒ OOV
        if mp >= 95 and tp >= 95:
            return "NEEDLE (mahal & tail both above the Σ-shaped null 95th pct)"
        if mp <= 50 and tp <= 50:
            return "SHAPE INDISTINGUISHABLE FROM Σ-WEIGHTED NOISE → OUT-OF-VOCABULARY"
        return "INTERMEDIATE (mixed vs Σ-null) → OUT-OF-VOCABULARY per assay law"

    out = {
        "arm": "14k assay — Kreg/Ktopic shape null + cos-family distinctness",
        "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop",
        "site": args.site, "k_null": args.k,
        "law": "shape stats on banked Σ (unit vector): mahal vᵀΣ⁻¹v; top/tail/band eigenmass. Null = "
               "random vectors, isotropic + Σ-shaped (Σ-weighted noise). Inside the Σ-shaped null "
               "envelope ⇒ shape-uninformative ⇒ shape call OUT-OF-VOCABULARY.",
        "note_construction": "CAA-built Kreg/Ktopic (session-9 first-pass); null is CONSTRUCTION-AGNOSTIC "
                             "(random-vector, the AMENDMENT-1 'Σ-weighted noise' model). The LITERAL "
                             "shuffled-pole Δμ null rides with the owed GPU bare-sort rebuild.",
        "cos_family_distinctness": cos_family,
        "observed_shape": {k: {m: round(v[m], 4) for m in v} for k, v in obs.items()},
        "null_summary": null_summary,
        "placements_null_percentile": placements,
        "shape_calls_vs_null": {name: call(name) for name in ("V3", "V7", "Kreg", "Ktopic") if name in obs},
    }
    args.out_json.write_text(json.dumps(out, indent=1))
    print("cos-family:", json.dumps(cos_family))
    print("shape calls vs Σ-null:", json.dumps(out["shape_calls_vs_null"]))
    for name in ("V3", "V7", "Kreg", "Ktopic"):
        if name in placements:
            s = placements[name]["sigma_shaped"]
            print(f"  {name}: mahal={s['mahalanobis']['value']} (Σ-null pct {s['mahalanobis']['null_percentile']}) "
                  f"tail={s['tail768_eigenmass']['value']} (pct {s['tail768_eigenmass']['null_percentile']})")
    print("wrote", args.out_json)


if __name__ == "__main__":
    main()
