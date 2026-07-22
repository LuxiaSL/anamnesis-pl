"""A8 Leg-0 — T3: Phase C Rosetta + algebra suite (CPU, local). Session spec Phase C.

Runs on every VALID g from a8_fit_g (cp2_summary.json). Instruments:
  1. Transported-null envelope — 100 seeded unit randoms + all banked R-band members
     (+ iso R members, labeled) through the SAME g; per-axis signed-cos q95, both
     directions.
  2. Axis reads — cos(g·v_src, v_tgt) and cos(g_rev·v_tgt, v_src), sign-anchored, for
     {V7, Vrep⊥, Vconf, V_temp, dir0}. 3B Vrep⊥ is BUILT AT USE = unit(GS(Grep_L14,
     V7_L14)) per the ferry note of 2026-07-22 (banked 3B object is raw, 45%
     V7-aligned); the band-passed variant GS(Vrep_L14, V7_L14) rides as a robustness
     row (the 8B side was band-passed before GS — asymmetry noted, not amended).
  3. Top-PC control — max_{j<=5} |cos(g·PC_j^src, v_tgt)| beside every axis read
     (source PCs recomputed from the state banks with the fit's own split — same code
     path as the fit, deterministic).
  4. F-i   cos(g·Vconf_src, V7_tgt)  (target-frame collapse identity; reverse beside).
  5. F-iii cos( g·(v ⊥ V7_src), (g·v) ⊥ V7_tgt ) for v in {Grep_raw, Geos_raw,
     oblique}; oblique := unit(unit(V7_src) + unit(Grep_src)) (stamped formula).
  6. F-iv  mode-simplex — 5 mode centroid OFFSETS (from the S3 grand mean) per model,
     own-voice primary / pooled + held-out-rows robustness; identity assignment score
     vs the exact 5! = 120 permutation null.
  7. F-ii  PREDICTIONS ONLY — predicted target entropy rise per dose =
     cos(g·v, V7_tgt) x target's banked V7 entropy law (arms/A5_matrix/<m>/entropy_*.json,
     itself C§8-unstamped); filed to readouts/f2_predictions.json for Leg 3.

Sign anchors (recipe-level; stamps carry no explicit sign field — rake note): all
gradient-family vectors point toward INCREASING functional (entropy / margin-confidence
/ repetition-mass); dir0 = mean(analogical) - mean(contrastive), same order both
models; V7's + direction behaviorally confirmed by the target entropy law.

Readouts: numbers and mechanical booleans only — no verdict language on frozen P's
(the desk scores). Artifacts under --arm-root/readouts/.

Run (from pipeline/):
  python -m anamnesis.scripts.a8_rosetta --selftest     # synthetic mechanics
  python -m anamnesis.scripts.a8_rosetta --inventory    # axis registry load check
  python -m anamnesis.scripts.a8_rosetta                 # real suite (post CP-2)
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np

from anamnesis.scripts.a8_fit_g import (
    A8_SEED, ARMS, DEFAULT_ARM_ROOT, PCABank, SITES, StateBank, TransportMap,
    load_labels, load_state_bank, load_transport_map, make_split)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_rosetta")

N_RANDOM_NULLS = 100
TOP_PC_J = 5
MODES = ("linear", "socratic", "contrastive", "dialectical", "analogical")
ANCHOR_SITE = {"3b": 14, "8b": 16, "qwen-7b": 21,
               # extension pair (A8-add-7): L36 is where the whole gemma field
               # roster is banked, so it is the anchor by construction, not choice.
               "gemma3-27b": 36}
BANK = Path("outputs/battery")

SIGN_ANCHORS = {
    "V7": "+ = entropy-increasing (entropy-gradient recipe; behaviorally confirmed "
          "by the target's banked V7 entropy law)",
    "Vrep_perp": "+ = repetition-mass-increasing component ⊥ V7 (Grep gradient)",
    "Vrep_perp_band": "+ = repetition-mass-increasing, band-passed variant ⊥ V7",
    "Vconf": "+ = margin/confidence-increasing (Gmargin gradient, band-passed)",
    "Vtemp": "recipe order per a5 ctemp/vtemp contrast build (same both models)",
    "dir0": "+ = analogical - contrastive (V3 route-1 recipe order, both models)",
}


@dataclass
class Axis:
    name: str
    vec: np.ndarray                 # unit float64
    source: str                     # "path::key" or derivation formula
    sign_anchor: str


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero vector")
    return v.astype(np.float64) / n


def _gs(v: np.ndarray, against: np.ndarray) -> np.ndarray:
    """unit( v - (v·u)u ), u = unit(against)."""
    u = _unit(against)
    return _unit(v.astype(np.float64) - float(v @ u) * u)


def _load_key(rel: str, key: str) -> np.ndarray:
    z = np.load(BANK / rel)
    if key not in z.files:
        raise KeyError(f"{rel}: key {key} not found (has {sorted(z.files)[:8]}…)")
    return z[key].astype(np.float64)


# ---------------------------------------------------------------- axis registry
def load_axes(model: str) -> tuple[dict[str, Axis], dict[str, Axis], list[Axis]]:
    """Returns (read_axes, extra_vectors, null_pool) for one model.

    read_axes: the 5 per-axis transport reads (+ the 3B band variant).
    extra_vectors: F-iii inputs (raw functional gradients + oblique), source side.
    null_pool: banked R-band members + iso-R members (labeled), unit.
    """
    if model == "3b":
        v7 = _unit(_load_key("a5_vectors_3b_b7/a5_vectors.npz", "V7_L14"))
        grep = _load_key("annex/roster_vectors_3b/roster_gradients.npz", "Grep_L14")
        geos = _load_key("annex/roster_vectors_3b/roster_gradients.npz", "Geos_L14")
        reads = {
            "V7": Axis("V7", v7, "a5_vectors_3b_b7::V7_L14", SIGN_ANCHORS["V7"]),
            "Vrep_perp": Axis(
                "Vrep_perp", _gs(grep, v7),
                "DERIVED unit(GS(roster_gradients::Grep_L14, V7_L14)) "
                "[ferry 2026-07-22: banked object raw, 45% V7-aligned]",
                SIGN_ANCHORS["Vrep_perp"]),
            "Vrep_perp_band": Axis(
                "Vrep_perp_band",
                _gs(_load_key("annex/roster_vectors_3b/a5_vectors.npz", "Vrep_L14"), v7),
                "DERIVED unit(GS(annex a5_vectors::Vrep_L14 (band-passed), V7_L14)) "
                "[robustness row: band-parity with the 8B construction]",
                SIGN_ANCHORS["Vrep_perp_band"]),
            "Vconf": Axis(
                "Vconf",
                _unit(_load_key("annex/roster_vectors_3b/a5_vectors.npz", "Vconf_L14")),
                "annex/roster_vectors_3b::Vconf_L14", SIGN_ANCHORS["Vconf"]),
            "Vtemp": Axis(
                "Vtemp",
                _unit(_load_key("a5_vectors_3b_ctemp/a5_vectors.npz", "Vtemp_L14")),
                "a5_vectors_3b_ctemp::Vtemp_L14", SIGN_ANCHORS["Vtemp"]),
            "dir0": Axis(
                "dir0", _unit(_load_key("a5_vectors_3b/a5_vectors.npz", "V3_L14")),
                "a5_vectors_3b::V3_L14", SIGN_ANCHORS["dir0"]),
        }
        extras = {
            "Vrep_raw": Axis("Vrep_raw", _unit(grep),
                             "roster_gradients::Grep_L14 (raw)", SIGN_ANCHORS["Vrep_perp"]),
            "Veos_raw": Axis("Veos_raw", _unit(geos),
                             "roster_gradients::Geos_L14 (raw)",
                             "+ = eos/stopping-functional-increasing"),
            "oblique": Axis("oblique", _unit(v7 + _unit(grep)),
                            "DERIVED unit(unit(V7_L14) + unit(Grep_L14))",
                            "oblique test vector (stamped formula)"),
        }
        pool = ([Axis(f"Rband{i}", _unit(_load_key("a5_vectors_3b_b7/a5_vectors.npz",
                                                   f"Rband{i}_L14")),
                      f"a5_vectors_3b_b7::Rband{i}_L14", "banked R-band member")
                 for i in (1, 2, 3)]
                + [Axis(f"Riso{i}", _unit(_load_key("a5_vectors_3b/a5_vectors.npz",
                                                    f"R{i}")),
                        f"a5_vectors_3b::R{i}", "banked iso-R member")
                   for i in (1, 2, 3)])
        return reads, extras, pool

    if model == "8b":
        v7 = _unit(_load_key("a5_vectors_8b_b7/a5_vectors.npz", "V7_L16"))
        reads = {
            "V7": Axis("V7", v7, "a5_vectors_8b_b7::V7_L16", SIGN_ANCHORS["V7"]),
            "Vrep_perp": Axis(
                "Vrep_perp",
                _unit(_load_key("a5_field_8b/perp/a5_vectors.npz", "Vrep_perp_L16")),
                "a5_field_8b/perp::Vrep_perp_L16 (banked GS, cos-to-V7 4e-17 per "
                "fire-log)", SIGN_ANCHORS["Vrep_perp"]),
            "Vconf": Axis(
                "Vconf",
                _unit(_load_key("a5_field_8b/members/a5_vectors.npz", "Vconf_L16")),
                "a5_field_8b/members::Vconf_L16", SIGN_ANCHORS["Vconf"]),
            "Vtemp": Axis(
                "Vtemp",
                _unit(_load_key("a5_vectors_8b_vtemp/a5_vectors.npz", "Vtemp_L16")),
                "a5_vectors_8b_vtemp::Vtemp_L16", SIGN_ANCHORS["Vtemp"]),
            "dir0": Axis(
                "dir0", _unit(_load_key("a5_vectors_8b/a5_vectors.npz", "V3_L16")),
                "a5_vectors_8b::V3_L16", SIGN_ANCHORS["dir0"]),
        }
        extras = {
            "Vrep_raw": Axis("Vrep_raw",
                             _unit(_load_key("a5_field_8b/members/a5_vectors.npz",
                                             "Vrep_L16")),
                             "a5_field_8b/members::Vrep_L16 (band-passed, pre-GS)",
                             SIGN_ANCHORS["Vrep_perp"]),
            "Veos_raw": Axis("Veos_raw",
                             _unit(_load_key("a5_field_8b/members/a5_vectors.npz",
                                             "Veos_L16")),
                             "a5_field_8b/members::Veos_L16 (band-passed, pre-GS)",
                             "+ = eos/stopping-functional-increasing"),
            "Veos_perp": Axis("Veos_perp",
                              _unit(_load_key("a5_field_8b/perp/a5_vectors.npz",
                                              "Veos_perp_L16")),
                              "a5_field_8b/perp::Veos_perp_L16", "banked GS"),
        }
        extras["oblique"] = Axis(
            "oblique", _unit(v7 + extras["Vrep_raw"].vec),
            "DERIVED unit(unit(V7_L16) + unit(members::Vrep_L16))",
            "oblique test vector (stamped formula)")
        pool = ([Axis(f"Rband{i}", _unit(_load_key("a5_vectors_8b_b7/a5_vectors.npz",
                                                   f"Rband{i}_L16")),
                      f"a5_vectors_8b_b7::Rband{i}_L16", "banked R-band member")
                 for i in (1, 2, 3)]
                + [Axis(f"Riso{i}", _unit(_load_key("a5_vectors_8b/a5_vectors.npz",
                                                    f"R{i}")),
                        f"a5_vectors_8b::R{i}", "banked iso-R member")
                   for i in (1, 2, 3)])
        return reads, extras, pool

    if model == "qwen-7b":
        v7 = _unit(_load_key("a5_vectors_qwen-7b_b7/a5_vectors.npz", "V7_L21"))
        reads = {
            "V7": Axis("V7", v7, "a5_vectors_qwen-7b_b7::V7_L21", SIGN_ANCHORS["V7"]),
            "Vrep_perp": Axis(
                "Vrep_perp",
                _unit(_load_key("a5_field_qwen-7b/perp/a5_vectors.npz",
                                "Vrep_perp_L21")),
                "a5_field_qwen-7b/perp::Vrep_perp_L21 (banked GS)",
                SIGN_ANCHORS["Vrep_perp"]),
            "Vconf": Axis(
                "Vconf",
                _unit(_load_key("a5_field_qwen-7b/members/a5_vectors.npz",
                                "Vconf_L21")),
                "a5_field_qwen-7b/members::Vconf_L21", SIGN_ANCHORS["Vconf"]),
            "Vtemp": Axis(
                "Vtemp",
                _unit(_load_key("a5_vectors_qwen-7b_vtemp/a5_vectors.npz",
                                "Vtemp_L21")),
                "a5_vectors_qwen-7b_vtemp::Vtemp_L21", SIGN_ANCHORS["Vtemp"]),
            "dir0": Axis(
                "dir0", _unit(_load_key("a5_vectors_qwen_7b/a5_vectors.npz",
                                        "V3_L21")),
                "a5_vectors_qwen_7b::V3_L21", SIGN_ANCHORS["dir0"]),
        }
        extras = {
            "Vrep_raw": Axis("Vrep_raw",
                             _unit(_load_key("a5_field_qwen-7b/members/a5_vectors.npz",
                                             "Vrep_L21")),
                             "a5_field_qwen-7b/members::Vrep_L21 (band-passed, pre-GS)",
                             SIGN_ANCHORS["Vrep_perp"]),
            "Veos_raw": Axis("Veos_raw",
                             _unit(_load_key("a5_field_qwen-7b/members/a5_vectors.npz",
                                             "Veos_L21")),
                             "a5_field_qwen-7b/members::Veos_L21 (band-passed, pre-GS)",
                             "+ = eos/stopping-functional-increasing"),
            "Veos_perp": Axis("Veos_perp",
                              _unit(_load_key("a5_field_qwen-7b/perp/a5_vectors.npz",
                                              "Veos_perp_L21")),
                              "a5_field_qwen-7b/perp::Veos_perp_L21", "banked GS"),
        }
        extras["oblique"] = Axis(
            "oblique", _unit(v7 + extras["Vrep_raw"].vec),
            "DERIVED unit(unit(V7_L21) + unit(members::Vrep_L21))",
            "oblique test vector (stamped formula)")
        pool = ([Axis(f"Rband{i}",
                      _unit(_load_key("a5_vectors_qwen-7b_b7/a5_vectors.npz",
                                      f"Rband{i}_L21")),
                      f"a5_vectors_qwen-7b_b7::Rband{i}_L21", "banked R-band member")
                 for i in (1, 2, 3)]
                + [Axis(f"Riso{i}", _unit(_load_key("a5_vectors_qwen_7b/a5_vectors.npz",
                                                    f"R{i}")),
                        f"a5_vectors_qwen_7b::R{i}", "banked iso-R member")
                   for i in (1, 2, 3)])
        return reads, extras, pool

    if model == "gemma3-27b":
        # EXTENSION PAIR (A8-add-7). Everything lives at L36 — the site is not a
        # choice, it is where the whole banked field roster was built (rake 26:
        # banked-vector site MUST be inside the fit grid; the smalls grid is
        # {34,36,38} for exactly this reason).
        #
        # ⚠ TWO dir0 VINTAGES EXIST FOR THIS MODEL, UNDER DIFFERENT MODE PAIRS
        # (rake 33, the Leg-6 lesson applied prospectively rather than
        # retrospectively):
        #   a5_vectors_gemma3_27b      L23/L35/L41  pair = [socratic, contrastive]
        #   a5_vectors_gemma3_27b_L36  L36          pair = [analogical, contrastive]
        # The arm's dir0 is [analogical, contrastive] on 3B and 8B, so the L36
        # vintage is the PAIR-MATCHED one and the only admissible needle target.
        # Both stamps were read before this block was written; the near-miss is
        # that L35 sits one layer from L36 with the WRONG contrast.
        v7 = _unit(_load_key("a5_vectors_gemma3-27b_b7/a5_vectors.npz", "V7_L36"))
        reads = {
            "V7": Axis("V7", v7, "a5_vectors_gemma3-27b_b7::V7_L36", SIGN_ANCHORS["V7"]),
            "Vrep_perp": Axis(
                "Vrep_perp",
                _unit(_load_key("a5_field_gemma3-27b/perp/a5_vectors.npz",
                                "Vrep_perp_L36")),
                "a5_field_gemma3-27b/perp::Vrep_perp_L36 (banked GS; perp_anatomy "
                "cos_perp_V7 7.5e-17 — construction identity CHECKED, rake 8)",
                SIGN_ANCHORS["Vrep_perp"]),
            "Vconf": Axis(
                "Vconf",
                _unit(_load_key("a5_field_gemma3-27b/members/a5_vectors.npz",
                                "Vconf_L36")),
                "a5_field_gemma3-27b/members::Vconf_L36 (band-passed)",
                SIGN_ANCHORS["Vconf"]),
            "Vtemp": Axis(
                "Vtemp",
                _unit(_load_key("a5_vectors_gemma3-27b_vtemp/a5_vectors.npz",
                                "Vtemp_L36")),
                "a5_vectors_gemma3-27b_vtemp::Vtemp_L36", SIGN_ANCHORS["Vtemp"]),
            "dir0": Axis(
                "dir0",
                _unit(_load_key("a5_vectors_gemma3_27b_L36/a5_vectors.npz", "V3_L36")),
                "a5_vectors_gemma3_27b_L36::V3_L36 — PAIR-MATCHED "
                "[analogical, contrastive], stamp-verified against 3B/8B dir0",
                SIGN_ANCHORS["dir0"]),
        }
        extras = {
            "Vrep_raw": Axis("Vrep_raw",
                             _unit(_load_key("a5_field_gemma3-27b/members/a5_vectors.npz",
                                             "Vrep_L36")),
                             "a5_field_gemma3-27b/members::Vrep_L36 (band-passed, pre-GS)",
                             SIGN_ANCHORS["Vrep_perp"]),
            "Veos_raw": Axis("Veos_raw",
                             _unit(_load_key("a5_field_gemma3-27b/members/a5_vectors.npz",
                                             "Veos_L36")),
                             "a5_field_gemma3-27b/members::Veos_L36 (band-passed, pre-GS)",
                             "+ = eos/stopping-functional-increasing"),
            "Veos_perp": Axis("Veos_perp",
                              _unit(_load_key("a5_field_gemma3-27b/perp/a5_vectors.npz",
                                              "Veos_perp_L36")),
                              "a5_field_gemma3-27b/perp::Veos_perp_L36 (banked GS)",
                              "banked GS"),
        }
        extras["oblique"] = Axis(
            "oblique", _unit(v7 + extras["Vrep_raw"].vec),
            "DERIVED unit(unit(V7_L36) + unit(members::Vrep_L36))",
            "oblique test vector (stamped formula)")
        pool = ([Axis(f"Rband{i}",
                      _unit(_load_key("a5_vectors_gemma3-27b_b7/a5_vectors.npz",
                                      f"Rband{i}_L36")),
                      f"a5_vectors_gemma3-27b_b7::Rband{i}_L36", "banked R-band member")
                 for i in (1, 2, 3)]
                + [Axis(f"Riso{i}",
                        _unit(_load_key("a5_vectors_gemma3_27b_L36/a5_vectors.npz",
                                        f"R{i}")),
                        f"a5_vectors_gemma3_27b_L36::R{i}", "banked iso-R member")
                   for i in (1, 2, 3)])
        return reads, extras, pool

    if model == "olmo2-7b":
        # EXTENSION PAIR (A8-add-7). OLMo has NO banked vectors of any kind — no
        # V7, no field roster, no dir0 (it entered the battery for A1/A4 only).
        # There is therefore nothing to read against and no â to measure; the OLMo
        # leg is a FIT-VALIDITY row (P8-XO) and nothing more. Raised as an explicit
        # error rather than an empty registry so no caller can quietly produce a
        # zero-axis "clean" readout for this model.
        raise ValueError(
            "olmo2-7b has NO banked target vectors (no a5 §B.7 V7, no field roster, "
            "no dir0) — every A8 read is undefined for it, and â(·→olmo) is not a "
            "measurement that can be made. See A8-add-7's declared-in-advance ⚫. "
            "Building an OLMo V7 is an a5 vector-build arc, not a state collection.")

    raise ValueError(f"no axis registry for model {model!r}")


ENTROPY_TAG = {"3b": "3b", "8b": "8b", "qwen-7b": "qwen", "gemma3-27b": "gemma"}


def load_entropy_law(model: str) -> dict:
    """Target model's banked V7 entropy law: {alpha_frac: entropy_rise}."""
    tag = ENTROPY_TAG[model]
    path = Path(f"outputs/battery/arms/A5_matrix/{tag}/entropy_{tag}.json")
    with open(path) as f:
        d = json.load(f)
    law = {float(r["alpha_frac"]): float(r["entropy_rise"])
           for r in d["rows"] if r.get("vector") == "V7"}
    if not law:
        raise RuntimeError(f"{path}: no V7 rows")
    return {"law": law, "source": str(path),
            "source_status": d.get("STATUS", "unknown"),
            "n_per_cell": d["rows"][0].get("n")}


# ---------------------------------------------------------------- instruments
def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def transported_null_cos(tm: TransportMap, nulls: list[Axis], targets: dict[str, Axis],
                         direction: str, rng: np.random.Generator, d_random: int
                         ) -> tuple[dict[str, np.ndarray], list[str]]:
    """cos of every null (banked + N seeded randoms), transported, vs every target
    axis. Returns per-axis arrays + null labels (banked first)."""
    randoms = rng.standard_normal((N_RANDOM_NULLS, d_random))
    randoms /= np.linalg.norm(randoms, axis=1, keepdims=True)
    pool = [(a.name, a.vec) for a in nulls] + [
        (f"rand{i:03d}", randoms[i]) for i in range(N_RANDOM_NULLS)]
    out = {name: np.empty(len(pool)) for name in targets}
    for i, (_, v) in enumerate(pool):
        tv = tm.transport(v, direction=direction)
        for name, ax in targets.items():
            out[name][i] = cos(tv, ax.vec)
    return out, [lab for lab, _ in pool]


def axis_reads(tm: TransportMap, src_axes: dict[str, Axis], tgt_axes: dict[str, Axis],
               null_fwd: dict[str, np.ndarray], null_rev: dict[str, np.ndarray],
               top_pcs_src: np.ndarray, top_pcs_tgt: np.ndarray) -> list[dict]:
    rows = []
    for name, sax in src_axes.items():
        tgt_name = "Vrep_perp" if name == "Vrep_perp_band" else name
        if tgt_name not in tgt_axes:
            continue
        tax = tgt_axes[tgt_name]
        tv = tm.transport(sax.vec, direction="fwd")
        c_fwd = cos(tv, tax.vec)
        q95_fwd = float(np.quantile(null_fwd[tgt_name], 0.95))
        toppc_fwd = [abs(cos(tm.transport(top_pcs_src[j], direction="fwd"), tax.vec))
                     for j in range(top_pcs_src.shape[0])]
        rv = tm.transport(tax.vec, direction="rev")
        c_rev = cos(rv, sax.vec)
        q95_rev = float(np.quantile(null_rev[name] if name in null_rev
                                    else null_rev[tgt_name], 0.95))
        toppc_rev = [abs(cos(tm.transport(top_pcs_tgt[j], direction="rev"), sax.vec))
                     for j in range(top_pcs_tgt.shape[0])]
        rows.append({
            "axis": name, "target_axis": tgt_name,
            "cos_fwd": round(c_fwd, 4), "null_q95_fwd": round(q95_fwd, 4),
            "exceeds_envelope_fwd": bool(c_fwd > q95_fwd),
            "top_pc_max_fwd": round(max(toppc_fwd), 4),
            "top_pc_argmax_fwd": int(np.argmax(toppc_fwd)),
            "axis_specific_fwd": bool(c_fwd > max(toppc_fwd)),
            "cos_rev": round(c_rev, 4), "null_q95_rev": round(q95_rev, 4),
            "exceeds_envelope_rev": bool(c_rev > q95_rev),
            "top_pc_max_rev": round(max(toppc_rev), 4),
            "axis_specific_rev": bool(c_rev > max(toppc_rev)),
            "sign_anchor": sax.sign_anchor, "source": sax.source})
    return rows


def f1_read(tm: TransportMap, src_axes, tgt_axes, null_fwd) -> dict:
    tv = tm.transport(src_axes["Vconf"].vec, direction="fwd")
    c = cos(tv, tgt_axes["V7"].vec)
    rv = tm.transport(tgt_axes["Vconf"].vec, direction="rev")
    return {"cos_g_Vconf_src__V7_tgt": round(c, 4),
            "null_q95_V7_tgt": round(float(np.quantile(null_fwd["V7"], 0.95)), 4),
            "null_q05_V7_tgt": round(float(np.quantile(null_fwd["V7"], 0.05)), 4),
            "cos_grev_Vconf_tgt__V7_src": round(cos(rv, src_axes["V7"].vec), 4)}


def f3_reads(tm: TransportMap, src_axes, src_extras, tgt_axes) -> list[dict]:
    v7s, v7t = src_axes["V7"].vec, tgt_axes["V7"].vec
    rows = []
    for name in ("Vrep_raw", "Veos_raw", "oblique"):
        if name not in src_extras:
            continue
        v = src_extras[name].vec
        lhs = tm.transport(_gs(v, v7s), direction="fwd")
        rhs = _gs(tm.transport(v, direction="fwd"), v7t)
        rows.append({"vector": name, "cos_commutation": round(cos(lhs, rhs), 4),
                     "cos_v_to_V7_src": round(cos(v, v7s), 4),
                     "cos_gv_to_V7_tgt": round(cos(tm.transport(v, direction="fwd"),
                                                   v7t), 4),
                     "source": src_extras[name].source})
    return rows


def f4_mode_simplex(tm: TransportMap, src_bank: StateBank, tgt_bank: StateBank,
                    s_site: int, t_site: int, labels_src, labels_tgt,
                    rows_mask_src: np.ndarray, rows_mask_tgt: np.ndarray,
                    voice_src: Optional[str], voice_tgt: Optional[str],
                    manifest: Path) -> Optional[dict]:
    """Centroid OFFSETS (from S3 grand mean) per mode, transported, identity
    assignment score vs the exact 120-permutation null."""
    with open(manifest) as f:
        entries = {e["text_id"]: e for e in json.load(f)["entries"]}

    def offsets(bank, site, labels, mask, voice):
        x = bank.matrix(site)
        sel_all = (labels.stratum == "S3") & mask
        if voice is not None:
            v = np.array([entries[t]["voice"] for t in bank.text_ids])
            sel_all &= (v == voice)
        if not sel_all.any():
            return None
        grand = x[sel_all].mean(axis=0)
        m_lab = np.array([entries[t]["mode"] for t in bank.text_ids])
        cents, counts = [], []
        for m in MODES:
            sel = sel_all & (m_lab == m)
            if sel.sum() < 3:
                return None
            cents.append(x[sel].mean(axis=0) - grand)
            counts.append(int(sel.sum()))
        return np.stack(cents), counts

    src = offsets(src_bank, s_site, labels_src, rows_mask_src, voice_src)
    tgt = offsets(tgt_bank, t_site, labels_tgt, rows_mask_tgt, voice_tgt)
    if src is None or tgt is None:
        return None
    src_off, src_n = src
    tgt_off, tgt_n = tgt
    # transport offsets in RAW hidden units (offsets of normalized states x norm)
    trans = np.stack([tm.transport(src_off[i] * tm.src_norm, direction="fwd")
                      for i in range(5)])
    cmat = np.array([[cos(trans[i], tgt_off[j]) for j in range(5)] for i in range(5)])
    scores = {perm: float(sum(cmat[i, perm[i]] for i in range(5)))
              for perm in itertools.permutations(range(5))}
    ident = tuple(range(5))
    s_id = scores[ident]
    p = sum(1 for s in scores.values() if s >= s_id) / len(scores)
    best = max(scores, key=scores.get)
    return {"identity_score": round(s_id, 4),
            "p_identity_exact": round(p, 4),
            "best_perm": [int(i) for i in best],
            "best_perm_score": round(scores[best], 4),
            "argmax_assignment": [int(np.argmax(cmat[i])) for i in range(5)],
            "diag_cos": [round(float(cmat[i, i]), 4) for i in range(5)],
            "n_per_mode_src": src_n, "n_per_mode_tgt": tgt_n,
            "modes_order": list(MODES)}


def f2_predictions(tm: TransportMap, src_axes, src_extras, tgt_axes,
                   law_info: dict) -> dict:
    law = law_info["law"]
    vecs = {**{k: src_axes[k] for k in ("V7", "Vrep_perp", "Vconf", "Vtemp")
              if k in src_axes}}
    if "oblique" in src_extras:
        vecs["oblique"] = src_extras["oblique"]
    rows = []
    for name, ax in vecs.items():
        c = cos(tm.transport(ax.vec, direction="fwd"), tgt_axes["V7"].vec)
        rows.append({
            "vector": name, "cos_g_v__V7_tgt": round(c, 4),
            "predicted_sign": int(np.sign(c)) if abs(c) > 1e-6 else 0,
            "predicted_entropy_rise_per_alpha": {
                str(a): round(c * r, 4) for a, r in sorted(law.items())}})
    return {"target_law": {str(a): r for a, r in sorted(law.items())},
            "target_law_source": law_info["source"],
            "target_law_status": law_info["source_status"],
            "note": "PREDICTIONS ONLY (Leg 0) — observed test is Leg 3; magnitude "
                    "bar at scoring: within x2 of target's own law (P8-3ii)",
            "rows": rows}


# ---------------------------------------------------------------- md rendering
def _md_table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_(no rows)_\n"
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = "".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n"
                   for r in rows)
    return head + body


# ---------------------------------------------------------------- the suite
def run_suite(arm_root: Path, src_model: str, tgt_model: str,
              include_invalid: bool = False, fits_dirname: str = "fits",
              readouts_dirname: str = "readouts") -> dict:
    fits_dir, states_dir = arm_root / fits_dirname, arm_root / "states"
    readouts = arm_root / readouts_dirname
    readouts.mkdir(parents=True, exist_ok=True)
    with open(fits_dir / "cp2_summary.json") as f:
        cp2 = json.load(f)
    fit_strata = cp2.get("fit_strata")   # top-PC controls must see what g saw
    records = [r for r in cp2["records"] if r["valid"] or include_invalid]
    if not records:
        logger.warning("no valid fits in cp2_summary — nothing to read")
        return {}

    src_axes, src_extras, src_pool = load_axes(src_model)
    tgt_axes, tgt_extras, tgt_pool = load_axes(tgt_model)
    law_info = load_entropy_law(tgt_model)
    manifest = arm_root / "corpus" / "corpus_manifest.json"
    anchor_pair = f"{src_model}L{ANCHOR_SITE[src_model]}->{tgt_model}L{ANCHOR_SITE[tgt_model]}"

    # primary flag: highest held-out R2 among valid proc fits per (pair, arm)
    primary: dict[tuple[str, str], str] = {}
    for r in records:
        if r["family"].startswith("proc") and r["valid"]:
            key = (r["site_pair"], r["arm"])
            if key not in primary or r["r2"] > next(
                    x["r2"] for x in records
                    if x["site_pair"] == key[0] and x["arm"] == key[1]
                    and x["family"] == primary[key]):
                primary[key] = r["family"]

    out: dict = {"axis": [], "f1": [], "f3": [], "f4": [], "f2": [],
                 "envelope": [], "cp2_recap": records}
    bank_cache: dict[tuple[str, str], StateBank] = {}

    def get_bank(model, arm):
        if (model, arm) not in bank_cache:
            bank_cache[(model, arm)] = load_state_bank(states_dir, model, arm)
        return bank_cache[(model, arm)]

    for rec in records:
        pair, arm, fam = rec["site_pair"], rec["arm"], rec["family"]
        s_site = int(pair.split("->")[0].split("L")[1])
        t_site = int(pair.split("->")[1].split("L")[1])
        tm = load_transport_map(
            fits_dir / f"fit_{pair.replace('->', '__')}_{arm}_{fam}.npz")
        is_primary = primary.get((pair, arm)) == fam
        tag = {"site_pair": pair, "arm": arm, "family": fam, "primary": is_primary,
               "fit_r2": rec["r2"], "fit_valid": rec["valid"]}

        src_bank, tgt_bank = get_bank(src_model, arm), get_bank(tgt_model, arm)
        labels_s = load_labels(manifest, src_bank.text_ids)
        labels_t = load_labels(manifest, tgt_bank.text_ids)
        train_s, test_s, _ = make_split(labels_s)
        all_s = np.ones(len(src_bank.text_ids), bool)
        all_t = np.ones(len(tgt_bank.text_ids), bool)

        # F-iv on every fit (needs no banked vectors)
        for variant, vs, vt, ms, mt in (
                ("own_voice_all", src_model, tgt_model, all_s, all_t),
                ("pooled_all", None, None, all_s, all_t),
                ("own_voice_heldout", src_model, tgt_model, test_s, test_s)):
            f4 = f4_mode_simplex(tm, src_bank, tgt_bank, s_site, t_site,
                                 labels_s, labels_t, ms, mt, vs, vt, manifest)
            if f4 is not None:
                out["f4"].append({**tag, "variant": variant, **f4})

        if pair != anchor_pair:
            continue                      # axis suite lives at the anchor site-pair

        rng = np.random.default_rng(A8_SEED)
        d_src = src_axes["V7"].vec.shape[0]
        d_tgt = tgt_axes["V7"].vec.shape[0]
        null_fwd, labels_fwd = transported_null_cos(
            tm, src_pool, tgt_axes, "fwd", rng, d_src)
        null_rev, labels_rev = transported_null_cos(
            tm, tgt_pool, src_axes, "rev", rng, d_tgt)
        np.savez_compressed(
            readouts / f"envelope_{arm}_{fam}.npz",
            null_labels_fwd=np.array(labels_fwd), null_labels_rev=np.array(labels_rev),
            **{f"fwd_{k}": v for k, v in null_fwd.items()},
            **{f"rev_{k}": v for k, v in null_rev.items()})
        out["envelope"].append({**tag, "n_nulls_fwd": len(labels_fwd),
                                "n_nulls_rev": len(labels_rev),
                                **{f"q95_fwd_{k}": round(float(np.quantile(v, .95)), 4)
                                   for k, v in null_fwd.items()},
                                **{f"q95_rev_{k}": round(float(np.quantile(v, .95)), 4)
                                   for k, v in null_rev.items()}})

        # top-5 source/target train PCs (same split + code path + strata as the fit)
        fit_mask = (np.isin(labels_s.stratum, fit_strata) if fit_strata
                    else np.ones(len(src_bank.text_ids), bool))
        x_tr = src_bank.matrix(s_site)[train_s & fit_mask]
        y_tr = tgt_bank.matrix(t_site)[train_s & fit_mask]
        pcs_s = PCABank.fit(x_tr, TOP_PC_J).components * src_bank.median_norms[s_site]
        pcs_t = PCABank.fit(y_tr, TOP_PC_J).components * tgt_bank.median_norms[t_site]

        for row in axis_reads(tm, src_axes, tgt_axes, null_fwd, null_rev, pcs_s, pcs_t):
            out["axis"].append({**tag, **row})
        out["f1"].append({**tag, **f1_read(tm, src_axes, tgt_axes, null_fwd)})
        for row in f3_reads(tm, src_axes, src_extras, tgt_axes):
            out["f3"].append({**tag, **row})
        if is_primary or fam == "ridge":
            out["f2"].append({**tag, **f2_predictions(tm, src_axes, src_extras,
                                                      tgt_axes, law_info)})

    stamp = {"arm": "A8_conjugation", "leg": 0, "builder": "a8_rosetta.py",
             "prereg_tag": "prereg-arm8-v1", "pair": f"{src_model}->{tgt_model}",
             "fits_dirname": fits_dirname, "fit_strata": fit_strata,
             "anchor_site_pair": anchor_pair, "n_random_nulls": N_RANDOM_NULLS,
             "top_pc_j": TOP_PC_J, "sign_anchor_basis": SIGN_ANCHORS,
             "axis_sources": {m: {k: a.source for k, a in ax.items()}
                              for m, ax in (( src_model, src_axes),
                                            (tgt_model, tgt_axes))}}
    with open(readouts / "rosetta_readout.json", "w") as f:
        json.dump({"stamp": stamp, **out}, f, indent=1)
    with open(readouts / "f2_predictions.json", "w") as f:
        json.dump({"stamp": stamp, "predictions": out["f2"]}, f, indent=1)

    md = [f"# A8 Leg-0 ROSETTA READOUT — {date.today().isoformat()} (UNSTAMPED, C§8)\n",
          f"Pair {src_model}->{tgt_model} · anchor {anchor_pair} · numbers only — "
          f"desk scores all frozen P's.\n",
          "## Fit validity recap (CP-2)\n",
          _md_table(records, ["site_pair", "arm", "family", "r2",
                              "r2_null_shuffled_q95", "r2_null_stratum_q95",
                              "strata_carried", "valid"]),
          "\n## Axis transport reads (anchor pair; sign-anchored cos)\n",
          _md_table(out["axis"], ["arm", "family", "primary", "axis", "cos_fwd",
                                  "null_q95_fwd", "exceeds_envelope_fwd",
                                  "top_pc_max_fwd", "axis_specific_fwd", "cos_rev",
                                  "null_q95_rev", "exceeds_envelope_rev",
                                  "axis_specific_rev"]),
          "\n## F-i — target-frame collapse identity\n",
          _md_table(out["f1"], ["arm", "family", "primary", "cos_g_Vconf_src__V7_tgt",
                                "null_q05_V7_tgt", "cos_grev_Vconf_tgt__V7_src"]),
          "\n## F-iii — orthogonalization commutation\n",
          _md_table(out["f3"], ["arm", "family", "vector", "cos_commutation",
                                "cos_v_to_V7_src", "cos_gv_to_V7_tgt"]),
          "\n## F-iv — mode-simplex assignment (exact 120-perm null)\n",
          _md_table(out["f4"], ["site_pair", "arm", "family", "variant",
                                "identity_score", "p_identity_exact", "best_perm",
                                "argmax_assignment"]),
          "\n## F-ii — filed predictions (observed test = Leg 3)\n",
          _md_table([{**{k: p[k] for k in ("arm", "family", "primary")},
                      **{"vector": r["vector"], "cos": r["cos_g_v__V7_tgt"],
                         "pred_rise_by_alpha": r["predicted_entropy_rise_per_alpha"]}}
                     for p in out["f2"] for r in p["rows"]],
                    ["arm", "family", "vector", "cos", "pred_rise_by_alpha"]),
          "\n## Envelope summary\n",
          _md_table(out["envelope"],
                    ["arm", "family"] + [f"q95_fwd_{k}" for k in tgt_axes] ),
          ]
    md_path = readouts / f"ROSETTA-READOUT-{date.today().isoformat()}.md"
    md_path.write_text("".join(md))
    logger.info("readout: %s (+ rosetta_readout.json, f2_predictions.json)", md_path)
    return out


# ---------------------------------------------------------------- inventory mode
def inventory(src_model: str, tgt_model: str) -> int:
    ok = True
    for model in (src_model, tgt_model):
        try:
            reads, extras, pool = load_axes(model)
            d = reads["V7"].vec.shape[0]
            for group, axes in (("reads", reads), ("extras", extras)):
                for name, ax in axes.items():
                    dim_ok = ax.vec.shape[0] == d
                    unit_ok = abs(np.linalg.norm(ax.vec) - 1) < 1e-9
                    print(f"  [{'OK' if dim_ok and unit_ok else 'BAD'}] {model} "
                          f"{group}/{name} d={ax.vec.shape[0]}  {ax.source[:70]}")
                    ok &= dim_ok and unit_ok
            print(f"  [OK] {model} null pool: {[a.name for a in pool]}")
            if model == "3b":
                c = cos(reads["Vrep_perp"].vec, reads["V7"].vec)
                print(f"  [{'OK' if abs(c) < 1e-9 else 'BAD'}] 3b Vrep_perp ⊥ V7 "
                      f"(cos={c:.2e})")
                raw45 = cos(_unit(_load_key(
                    "annex/roster_vectors_3b/roster_gradients.npz", "Grep_L14")),
                    reads["V7"].vec)
                print(f"  [info] 3b raw Grep vs V7 cos={raw45:.3f} "
                      f"(ferry said ~0.45-aligned)")
        except Exception as e:  # noqa: BLE001 — inventory reports, never crashes
            print(f"  [BAD] {model}: {type(e).__name__}: {e}")
            ok = False
    try:
        law = load_entropy_law(tgt_model)
        print(f"  [OK] target entropy law ({law['source_status']}): "
              f"{ {a: round(r, 3) for a, r in sorted(law['law'].items())} }")
    except Exception as e:  # noqa: BLE001
        print(f"  [BAD] entropy law: {e}")
        ok = False
    print(f"inventory: {'ALL OK' if ok else 'GAPS FOUND'}")
    return 0 if ok else 1


# ---------------------------------------------------------------- selftest
def selftest() -> int:
    from anamnesis.scripts.a8_fit_g import _synthetic_world, _selftest_split, run_pair_arm

    failures: list[str] = []

    def check(cond, msg):
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        if not cond:
            failures.append(msg)

    rng = np.random.default_rng(0)
    (src, tgt), labels, (v_a, v_b) = _synthetic_world(rng, paired=True)
    train, test = _selftest_split(labels, rng)
    _, maps = run_pair_arm(src, tgt, 1, 1, labels, train, test,
                           np.random.default_rng(1), k_grid=(16,), n_null=4)
    tm = maps["proc_k16"]
    d_a, d_b = 64, 96

    print("== rosetta selftest 1: envelope + axis-read mechanics ==")
    pool = [Axis(f"Rband{i}", _unit(np.random.default_rng(100 + i)
                                    .standard_normal(d_a)), "synthetic", "")
            for i in (1, 2, 3)]
    tgt_ax = {"V7": Axis("V7", _unit(v_b), "synthetic", "")}
    nf, lab = transported_null_cos(tm, pool, tgt_ax, "fwd",
                                   np.random.default_rng(5), d_a)
    q95 = float(np.quantile(nf["V7"], 0.95))
    c_planted = cos(tm.transport(_unit(v_a)), v_b)
    check(len(lab) == N_RANDOM_NULLS + 3, f"null pool size {len(lab)}")
    check(-0.5 < q95 < 0.6, f"envelope q95={q95:.3f} sane")
    check(c_planted > q95, f"planted axis cos={c_planted:.3f} > envelope q95={q95:.3f}")
    v_rand = _unit(np.random.default_rng(9).standard_normal(d_a))
    c_rand = cos(tm.transport(v_rand), v_b)
    check(c_rand < q95, f"random axis cos={c_rand:.3f} < q95 (no false exceed)")

    print("== rosetta selftest 2: F-i / F-iii mechanics ==")
    # synthetic "Vconf" = anti-parallel to the planted "V7" axis
    f1c = cos(tm.transport(-_unit(v_a)), _unit(v_b))
    check(f1c < -0.9, f"F-i mechanics: anti-parallel planted cos={f1c:.3f} < -0.9")
    v_obl = _unit(_unit(v_a) + _unit(np.random.default_rng(3).standard_normal(d_a)))
    lhs = tm.transport(_gs(v_obl, v_a))
    rhs = _gs(tm.transport(v_obl), v_b)
    c3 = cos(lhs, rhs)
    check(c3 > 0.5, f"F-iii commutation on near-orthogonal g: cos={c3:.3f} > 0.5")

    print("== rosetta selftest 3: F-iv exact permutation mechanics ==")
    # build mode-offset clouds directly: shared latent simplex, mapped both sides
    rng4 = np.random.default_rng(4)
    lat = rng4.standard_normal((5, 12)) * 2
    lat -= lat.mean(axis=0)
    amap = np.linalg.qr(rng4.standard_normal((d_a, 12)))[0].T
    bmap = np.linalg.qr(rng4.standard_normal((d_b, 12)))[0].T
    src_off, tgt_off = lat @ amap, lat @ bmap
    trans = np.stack([tm.transport(src_off[i] * tm.src_norm) for i in range(5)])
    cmat = np.array([[cos(trans[i], tgt_off[j]) for j in range(5)] for i in range(5)])
    scores = {p: float(sum(cmat[i, p[i]] for i in range(5)))
              for p in itertools.permutations(range(5))}
    # NOTE: tm was fit on the world's cluster structure, not on this fresh simplex —
    # identity need not win; the MACHINERY assertions are exactness + normalization.
    check(len(scores) == 120, "exact 120 permutations enumerated")
    p_id = sum(1 for s in scores.values() if s >= scores[tuple(range(5))]) / 120
    check(0 < p_id <= 1, f"p_identity={p_id:.3f} well-formed")
    # and with a planted-aligned map (transport the SAME latent through both maps),
    # identity must win: emulate by comparing lat@bmap to itself
    cmat2 = np.array([[cos(tgt_off[i], tgt_off[j]) for j in range(5)] for i in range(5)])
    s2 = {p: float(sum(cmat2[i, p[i]] for i in range(5)))
          for p in itertools.permutations(range(5))}
    p2 = sum(1 for s in s2.values() if s >= s2[tuple(range(5))]) / 120
    check(p2 == 1 / 120, f"identity wins on identical simplex (p={p2:.4f} = 1/120)")

    print("== rosetta selftest 4: F-ii prediction mechanics ==")
    law = {"law": {0.1: 0.14, 0.3: 0.94, -0.1: -0.08}, "source": "synthetic",
           "source_status": "synthetic", "n_per_cell": 0}
    fake_src = {"V7": Axis("V7", _unit(v_a), "s", ""),
                "Vrep_perp": Axis("Vrep_perp", _unit(_gs(
                    np.random.default_rng(6).standard_normal(d_a), v_a)), "s", ""),
                "Vconf": Axis("Vconf", -_unit(v_a), "s", ""),
                "Vtemp": Axis("Vtemp", _unit(
                    np.random.default_rng(7).standard_normal(d_a)), "s", "")}
    preds = f2_predictions(tm, fake_src, {}, {"V7": Axis("V7", _unit(v_b), "s", "")},
                           law)
    by_name = {r["vector"]: r for r in preds["rows"]}
    check(by_name["V7"]["predicted_sign"] == 1, "F-ii: V7 predicted sign +")
    check(by_name["Vconf"]["predicted_sign"] == -1, "F-ii: anti-parallel sign -")
    v7row = by_name["V7"]["predicted_entropy_rise_per_alpha"]
    check(abs(float(v7row["0.3"]) - by_name["V7"]["cos_g_v__V7_tgt"] * 0.94) < 5e-4,
          "F-ii: magnitude = cos x law (4-dp rounding tolerance)")

    print(f"\nrosetta selftest: {len(failures)} failure(s)")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm-root", type=Path, default=DEFAULT_ARM_ROOT)
    ap.add_argument("--source-model", default="3b", choices=sorted(SITES))
    ap.add_argument("--target-model", default="8b", choices=sorted(SITES))
    ap.add_argument("--include-invalid", action="store_true",
                    help="diagnostic only: also read invalid fits")
    ap.add_argument("--fits-dirname", default="fits")
    ap.add_argument("--readouts-dirname", default="readouts")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--inventory", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.inventory:
        return inventory(args.source_model, args.target_model)
    run_suite(args.arm_root, args.source_model, args.target_model,
              include_invalid=args.include_invalid, fits_dirname=args.fits_dirname,
              readouts_dirname=args.readouts_dirname)
    return 0


if __name__ == "__main__":
    sys.exit(main())
