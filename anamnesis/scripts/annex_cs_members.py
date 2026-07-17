"""ANNEX control-surface tenancy — member builder (CPU; probe-members pattern).

From banked raw CS gradients (annex_cs_pulses.py, 6 keys G*_L14) build:
  - 6 band members     V{name}_L14        = unit(P_band · G) through Σ_L14
                       (band [16:256] of the DESCENDING eigenorder — b7 conventions)
  - 6 ⊥ members        V{name}_perp_L14   = Gram-Schmidt vs V7_L14 (standing ⊥ rule)
  - Rband1/2/3_L14 COPIED from the b7 bank (so in-run null cells inject from the same npz)

Anatomy (cs_anatomy.json) carries the freeze-time inputs:
  - cos_to_V7 per band member  → THE LEAK PREDICTION input (THE ENTROPY-LEAK LAW,
    STANDING: predicted rise = cos(v, V7) × V7-effect at matched dose, sign included;
    CS-5 is the law's first PROSPECTIVE test). If --v7-effects is given
    ({"0.1": rise, "0.3": rise} from the banked b7/14j entropy of record), numeric
    predictions are emitted for both the band member and its ⊥ (⊥ ≈ 0 by construction —
    silence is the prediction, scored against the null band).
  - cross-cos vs the banked reference members (V7, Veos⊥, Vrep⊥, Vconf⊥, raw Veos/Vrep)
    — CS-4 scores |cos(Vwrap⊥, Veos⊥)| < .5 from here; CS-2 scores |cos(Vtail band, V7)|.
  - pairwise cos among the six ⊥ members (dissociation geometry for CS-3).

RIDER 1 note: the lexrarity member is V_lex (rarity/diction) — NEVER "register";
Kreg keeps that name and its standing WAIT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BAND = (16, 256)
NAMES = {"Glex": "Vlex", "Gcopy": "Vcopy", "Gselfrep": "Vselfrep",
         "Gtail": "Vtail", "Gwrap": "Vwrap", "Gfreqrep": "Vfreqrep"}
RBAND_KEYS = ("Rband1_L14", "Rband2_L14", "Rband3_L14")


def band_basis(sigma_npz: Path) -> np.ndarray:
    S = np.load(sigma_npz)
    evals = S["evals"].astype(np.float64)
    evecs = S["evecs"].astype(np.float64)
    order = np.argsort(evals)[::-1]
    return evecs[:, order[BAND[0]:BAND[1]]]


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-30:
        raise SystemExit("zero vector in unit()")
    return v / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gradients", type=Path, required=True, help="cs_gradients.npz")
    ap.add_argument("--sigma-l14", type=Path, required=True)
    ap.add_argument("--b7-npz", type=Path, required=True,
                    help="b7 bank: V7_L14 + Rband1/2/3_L14")
    ap.add_argument("--eosrep-npz", type=Path, required=True,
                    help="eosrep bank: Veos_L14/Veos_perp_L14/Vrep_L14/Vrep_perp_L14")
    ap.add_argument("--vconf-npz", type=Path, required=True,
                    help="vconf bank: Vconf_perp_L14")
    ap.add_argument("--norms-json", type=Path, required=True,
                    help="a5_vectors_stamps.json of record (median_resid_norms)")
    ap.add_argument("--v7-effects", type=Path, default=None,
                    help='optional json {"0.1": entropy_rise, "0.3": ...} of V7_L14 at '
                         'matched dose (banked numbers of record) -> numeric leak preds')
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    G = dict(np.load(args.gradients))
    b7 = dict(np.load(args.b7_npz))
    v7 = unit(b7["V7_L14"].astype(np.float64))
    refs = {"V7_L14": v7}
    eosrep = dict(np.load(args.eosrep_npz))
    for k in ("Veos_L14", "Veos_perp_L14", "Vrep_L14", "Vrep_perp_L14"):
        refs[k] = unit(eosrep[k].astype(np.float64))
    refs["Vconf_perp_L14"] = unit(np.load(args.vconf_npz)["Vconf_perp_L14"].astype(np.float64))
    norms_src = json.loads(args.norms_json.read_text())["median_resid_norms"]
    v7_effects = (json.loads(args.v7_effects.read_text())
                  if args.v7_effects and args.v7_effects.exists() else None)

    Ub = band_basis(args.sigma_l14)
    vectors: dict[str, np.ndarray] = {}
    anatomy: dict[str, dict] = {}

    for gkey, mname in NAMES.items():
        full = f"{gkey}_L14"
        if full not in G:
            raise SystemExit(f"missing raw gradient {full} in {args.gradients}")
        g = G[full].astype(np.float64)
        band = Ub @ (Ub.T @ g)
        band_norm = float(np.linalg.norm(band))
        if band_norm < 1e-12:
            raise SystemExit(f"{full}: zero band projection — pulse degenerate")
        m = band / band_norm
        c = float(m @ v7)
        perp = m - c * v7
        pn = float(np.linalg.norm(perp))
        row = {
            "raw_key": full,
            "band_norm_fraction": band_norm / max(float(np.linalg.norm(g)), 1e-30),
            "cos_to_V7": c,
            "perp_residual_fraction": pn,
            "cos_refs_band": {rk: float(m @ rv) for rk, rv in refs.items()},
        }
        vectors[f"{mname}_L14"] = m.astype(np.float32)
        if pn < 1e-6:
            row["perp"] = "COLLAPSED — no ⊥ member exists (fully V7-aligned)"
        else:
            p = perp / pn
            vectors[f"{mname}_perp_L14"] = p.astype(np.float32)
            row["cos_refs_perp"] = {rk: float(p @ rv) for rk, rv in refs.items()}
        if v7_effects:
            row["leak_prediction"] = {
                f"band@{d}": c * e for d, e in v7_effects.items()
            } | {f"perp@{d}": 0.0 for d in v7_effects}
        anatomy[f"{mname}_L14"] = row

    # pairwise cos among ⊥ members (CS-3 dissociation geometry)
    perp_keys = sorted(k for k in vectors if k.endswith("_perp_L14"))
    pair = {}
    for i, a in enumerate(perp_keys):
        for b in perp_keys[i + 1:]:
            pair[f"{a}·{b}"] = float(vectors[a].astype(np.float64)
                                     @ vectors[b].astype(np.float64))
    anatomy["_pairwise_perp_cos"] = pair

    # passthroughs from the b7 bank of record: in-run nulls + the V7 reference cells
    # (in-run V7 at matched dose = the CS-5 leak multiplier, same regime same night —
    # the RA-8B arithmetic precedent)
    for rk in RBAND_KEYS + ("V7_L14",):
        vectors[rk] = b7[rk].astype(np.float32)

    stamps = {
        "median_resid_norms": {"L14": float(norms_src["L14"])},
        "band": list(BAND),
        "provenance": "control-surface tenancy members: Σ_L14 band-pass (b7 conventions); "
                      "⊥ vs V7_L14 (standing rule); Rband1/2/3 copied from the b7 bank of "
                      "record for in-run null cells; leak predictions = cos_to_V7 × banked "
                      "V7-effect (THE ENTROPY-LEAK LAW, prospective — CS-5). RIDER 1: the "
                      "lexrarity member is V_lex, never 'register'.",
        "v7_effects_source": str(args.v7_effects) if v7_effects else None,
    }
    np.savez(args.out_dir / "a5_vectors.npz", **vectors)
    (args.out_dir / "a5_vectors_stamps.json").write_text(json.dumps(stamps, indent=1))
    (args.out_dir / "cs_anatomy.json").write_text(json.dumps(anatomy, indent=1))
    print(json.dumps({"built": sorted(vectors.keys())}, indent=1))
    print(json.dumps({k: {"cos_to_V7": round(v["cos_to_V7"], 4),
                          "perp_residual_fraction": round(v["perp_residual_fraction"], 4)}
                      for k, v in anatomy.items() if not k.startswith("_")}, indent=1))


if __name__ == "__main__":
    main()
