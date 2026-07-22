"""A8 — Veos anatomy-matched read (CPU, ungated; baton §4). NUMBERS TO THE DESK, NO VERDICT.

Desk mechanism hypothesis for the Leg-1 Veos commutation break (cross-family .566,
converging with the X-4 behavioral flag): Qwen possesses NOTHING in the geometric
relation Veos⊥ holds to V7/Vrep on the dense (Llama) side — no landing site for
commutation. This files the anatomy:

  1. RELATION PROFILES — per model, cos of Veos_raw and Veos_perp against the anchor
     panel {V7, Vrep_raw, Vrep_perp, Vconf, Vtemp, dir0}. Dense-side conservation
     (3B vs 8B) is the baseline the Qwen profile is read against. (3B Veos_perp is
     not banked — constructed at use as unit(GS(Veos_raw, V7)), the spec-verbatim
     recipe, flagged in output.)
  2. COMMUTATION REDERIVED, both rungs — dense→dense (leg0 g, 3B→8B) beside
     dense→Qwen (leg1 g, 8B→Qwen), for Veos_raw and Veos_perp, each with a
     100-seeded-random transported-null q95 envelope through the SAME g.
  3. TRANSPORTED PROFILE — does unit(g·Veos⊥_8B) carry the 8B relation profile into
     the Qwen frame even where it misses Qwen's own Veos⊥?
  4. CONSTRAINED-DIRECTION DATUM — among ALL unit directions in Qwen space holding
     exactly the 8B Veos⊥ relation profile to Qwen's anchors, (a) does one exist
     (least-norm feasibility), (b) how close can any get to Qwen's actual Veos⊥
     (closed-form max cos subject to the linear constraints)? If (b) is low, the
     relation profile and Qwen's own Veos⊥ are mutually exclusive to that degree —
     the "nothing in the right relation" shape, quantified.

Output: leg3/readouts_cpu/veos_anatomy.{json,md}. UNSTAMPED (C§8).

Run (repo root): PYTHONPATH=pipeline python -m anamnesis.scripts.a8_veos_anatomy
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from anamnesis.scripts.a8_fit_g import load_transport_map
from anamnesis.scripts.a8_rosetta import _unit, cos, load_axes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_veos_anatomy")

ARM = Path("outputs/battery/arms/A8_conjugation")
OUT = ARM / "leg3" / "readouts_cpu"
FITS = {"3b->8b": ARM / "fits/fit_3bL14__8bL16_native_proc_k512.npz",
        "8b->qwen-7b": ARM / "leg1/fits/fit_8bL16__qwen-7bL21_native_proc_k512.npz"}
ANCHOR_PANEL = ("V7", "Vrep_raw", "Vrep_perp", "Vconf", "Vtemp", "dir0")
N_RANDOM = 100
SEED = 80


def gs_perp(v: np.ndarray, against: np.ndarray) -> np.ndarray:
    a = _unit(against)
    return _unit(v - (v @ a) * a)


def get_family(model: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    """(anchors, veos {raw, perp}, notes) — all unit vectors."""
    axes, extras, _ = load_axes(model)
    pool = {**axes, **extras}
    anchors = {n: _unit(pool[n].vec) for n in ANCHOR_PANEL}
    notes = []
    veos = {"Veos_raw": _unit(pool["Veos_raw"].vec)}
    if "Veos_perp" in pool:
        veos["Veos_perp"] = _unit(pool["Veos_perp"].vec)
    else:
        veos["Veos_perp"] = gs_perp(pool["Veos_raw"].vec, pool["V7"].vec)
        notes.append(f"{model}: Veos_perp NOT banked — constructed unit(GS(Veos_raw, V7))")
    return anchors, veos, notes


def profile(v: np.ndarray, anchors: dict[str, np.ndarray]) -> dict[str, float]:
    return {n: round(cos(v, a), 4) for n, a in anchors.items()}


def profile_l2(p: dict[str, float], q: dict[str, float]) -> float:
    return round(float(np.sqrt(sum((p[k] - q[k]) ** 2 for k in p))), 4)


def constrained_direction_datum(anchors: dict[str, np.ndarray], target_profile: dict[str, float],
                                e: np.ndarray) -> dict:
    """Directions v (unit) with anchors^T v = target_profile: least-norm feasibility +
    max cos(v, e) in closed form. v = v* + w with w ⊥ span(A), ||w||=sqrt(1-||v*||^2);
    max_w cos = e·v* + ||P_perp e|| * sqrt(1-||v*||^2)."""
    A = np.stack([anchors[n] for n in ANCHOR_PANEL], axis=1)          # [d, k]
    c = np.array([target_profile[n] for n in ANCHOR_PANEL])
    gram = A.T @ A
    # The anchor panel is deliberately anatomical, not orthogonal (V7/Vconf cos ~ -.84
    # in every frame) — an exact Gram solve explodes on the near-dependency. Pseudo-
    # inverse least-squares is the honest object: minimal-norm v whose profile is as
    # close as the panel's conditioning allows, with the residual filed beside it.
    gram_pinv = np.linalg.pinv(gram, rcond=1e-6)
    coef = gram_pinv @ c
    v_star = A @ coef
    nv = float(np.linalg.norm(v_star))
    residual = float(np.linalg.norm(A.T @ v_star - c))
    feasible = nv <= 1.0
    out = {"least_norm": round(nv, 4),
           "profile_residual_l2": round(residual, 4),
           "gram_condition_number": round(float(np.linalg.cond(gram)), 1),
           "feasible_unit_direction_exists": bool(feasible)}
    if feasible:
        e_par = A @ (gram_pinv @ (A.T @ e))
        e_perp_norm = float(np.linalg.norm(e - e_par))
        max_cos = float(e @ v_star + e_perp_norm * np.sqrt(max(0.0, 1.0 - nv ** 2)))
        out["max_cos_to_target_Veos_perp_subject_to_profile"] = round(max_cos, 4)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fam = {m: get_family(m) for m in ("3b", "8b", "qwen-7b")}
    notes = [n for _, _, ns in fam.values() for n in ns]

    profiles = {m: {vn: profile(vv, fam[m][0]) for vn, vv in fam[m][1].items()} for m in fam}
    conservation = {
        "dense_dense_3b_vs_8b": {vn: profile_l2(profiles["3b"][vn], profiles["8b"][vn])
                                 for vn in ("Veos_raw", "Veos_perp")},
        "qwen_vs_8b": {vn: profile_l2(profiles["qwen-7b"][vn], profiles["8b"][vn])
                       for vn in ("Veos_raw", "Veos_perp")},
        "qwen_vs_3b": {vn: profile_l2(profiles["qwen-7b"][vn], profiles["3b"][vn])
                       for vn in ("Veos_raw", "Veos_perp")},
    }

    rng = np.random.default_rng(SEED)
    commutation = {}
    transported_profile = {}
    for pair, fit in FITS.items():
        tm = load_transport_map(fit)
        src, tgt = pair.split("->")
        d_src = fam[src][1]["Veos_raw"].shape[0]
        randoms = rng.standard_normal((N_RANDOM, d_src))
        randoms /= np.linalg.norm(randoms, axis=1, keepdims=True)
        rows = {}
        for vn in ("Veos_raw", "Veos_perp"):
            t = _unit(tm.transport(fam[src][1][vn]))
            null_cos = np.abs([cos(_unit(tm.transport(r)), fam[tgt][1][vn]) for r in randoms])
            rows[vn] = {"cos_gv_vs_target_own": round(cos(t, fam[tgt][1][vn]), 4),
                        "null_q95_abs": round(float(np.quantile(null_cos, 0.95)), 4)}
        commutation[pair] = rows
        t_perp = _unit(tm.transport(fam[src][1]["Veos_perp"]))
        tp = profile(t_perp, fam[tgt][0])
        transported_profile[pair] = {
            "profile_of_g_Veos_perp_in_target_frame": tp,
            "l2_vs_source_profile": profile_l2(tp, profiles[src]["Veos_perp"]),
            "l2_vs_target_own_profile": profile_l2(tp, profiles[tgt]["Veos_perp"]),
        }

    datum = constrained_direction_datum(fam["qwen-7b"][0], profiles["8b"]["Veos_perp"],
                                        fam["qwen-7b"][1]["Veos_perp"])
    baseline_datum = constrained_direction_datum(fam["8b"][0], profiles["3b"]["Veos_perp"],
                                                 fam["8b"][1]["Veos_perp"])

    doc = {
        "grade": "UNSTAMPED (C§8) — CPU anatomy read, baton §4; NUMBERS ONLY, no verdict",
        "prereg_tag": "prereg-arm8-v1", "builder": "a8_veos_anatomy.py", "date": "2026-07-22",
        "anchor_panel": list(ANCHOR_PANEL), "construction_notes": notes,
        "relation_profiles": profiles,
        "profile_conservation_l2": conservation,
        "commutation_rederived": commutation,
        "transported_profile": transported_profile,
        "constrained_direction_datum_qwen": datum,
        "constrained_direction_datum_dense_baseline_8b_from_3b_profile": baseline_datum,
        "reading_aid": "datum: among unit directions in the target space holding the "
                       "source Veos_perp relation profile exactly, max achievable cos to "
                       "the target's OWN Veos_perp. Low = profile and native Veos are "
                       "mutually exclusive shapes; compare qwen datum to dense baseline.",
    }
    (OUT / "veos_anatomy.json").write_text(json.dumps(doc, indent=1))

    lines = ["# A8 Veos anatomy read (UNSTAMPED, numbers-to-desk)\n",
             "## Relation profiles (cos vs anchor panel)\n",
             "| model | vector | " + " | ".join(ANCHOR_PANEL) + " |",
             "|---|---|" + "---|" * len(ANCHOR_PANEL)]
    for m in profiles:
        for vn, p in profiles[m].items():
            lines.append(f"| {m} | {vn} | " + " | ".join(f"{p[a]:+.3f}" for a in ANCHOR_PANEL) + " |")
    lines += ["\n## Profile conservation (L2 over panel)\n", "```",
              json.dumps(conservation, indent=1), "```",
              "\n## Commutation (cos(g·v, target's own) vs |null| q95)\n", "```",
              json.dumps(commutation, indent=1), "```",
              "\n## Transported Veos_perp profile\n", "```",
              json.dumps(transported_profile, indent=1), "```",
              "\n## Constrained-direction datum\n", "```",
              json.dumps({"qwen": datum, "dense_baseline": baseline_datum}, indent=1), "```"]
    (OUT / "veos_anatomy.md").write_text("\n".join(lines) + "\n")
    logger.info("filed -> %s", OUT / "veos_anatomy.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
