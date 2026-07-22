"""A8 extension-pairs smalls — THE STAR SYSTEMS, one per (template-arm x family).

The desk's arm-consistency rule (Luxia, 2026-07-23): a star constant belongs to the
system it was solved in. The native-arm constants of record (c_3B=.6382, c_8B=.7545,
c_Qwen=.4730, c_DSV2=.4415 at k128) are NATIVE-ARM numbers. Any model whose fits are
raw-arm-only — OLMo has no chat template, so its native arm does not exist — must take
its constant from a PARALLEL RAW-ARM SYSTEM, derived from the already-banked raw-arm
a_hat rows. Slotting a raw-arm a_hat into the native-arm system would be the
family-mixing sin one level up, and family-mixing has already authored two false scores
in this arm (rakes 40, 14).

This script builds every system that closes, from banked fits only. ZERO GPU COMPUTE.

The algebra. Star model: a_hat(A->B) = c_A . c_B. With models {3B, 8B, Qwen, DSV2} the
three hub pairs (3B-8B, 8B-Qwen, 8B-DSV2) leave the system underdetermined; the direct
3B->Qwen fit (Leg-4E) closes it:

    c_3B^2  = a(3B->8B) . a(3B->Qwen) / a(8B->Qwen)
    c_8B    = a(3B->8B)  / c_3B
    c_Qwen  = a(8B->Qwen)/ c_8B
    c_DSV2  = a(8B->DSV2)/ c_8B

3B->DSV2 is then a PREDICTION, and Leg 6 measured it — so every system printed here
carries its own out-of-sample check, and a system whose check fails is not a system you
may hang a new constant on. Reported per (arm, family); families are never mixed inside
one equation (rake 40) and the rank guard k <= n_train/1.2 is applied per pair, since
n_train differs across legs (600 hub, 360 for the Leg-4E closer).

UNSTAMPED (C section 8). Scores nothing; this is the substrate for A8-add-7's
predictions, which are filed BEFORE any new pair is fitted.

Run (repo root): PYTHONPATH=pipeline python -m anamnesis.scripts.a8_smalls_star_systems
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

from anamnesis.scripts.a8_fit_g import load_transport_map
from anamnesis.scripts.a8_rosetta import _unit, cos

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_smalls_star_systems")

ARM = Path("outputs/battery/arms/A8_conjugation")
BANK = Path("outputs/battery")
OUT = ARM / "smalls" / "readouts_cpu"

ARMS = ("native", "raw")
FAMILIES = ("proc_k32", "proc_k128", "proc_k512", "ridge")
RANK_GUARD_DIVISOR = 1.2          # add-3: k <= n_train / 1.2

# V7 = the attenuation-class reference axis. a_hat(A->B) := cos(g.V7_A, V7_B).
V7: dict[str, tuple[str, str, int]] = {
    "3b": ("a5_vectors_3b_b7/a5_vectors.npz", "V7_L14", 14),
    "8b": ("a5_vectors_8b_b7/a5_vectors.npz", "V7_L16", 16),
    "qwen-7b": ("a5_vectors_qwen-7b_b7/a5_vectors.npz", "V7_L21", 21),
    "dsv2-lite": ("a5_vectors_dsv2_lite_b7_L22/a5_vectors.npz", "V7_L22", 22),
    # gemma3-27b is added by the smalls collection leg (V7_L36 banked, node-side)
    "gemma3-27b": ("a5_vectors_gemma3-27b_b7/a5_vectors.npz", "V7_L36", 36),
    # olmo2-7b: NO BANKED V7 EXISTS. a_hat is undefined for OLMo until one is built.
}

# Banked pairs: label -> (src, tgt, fits_dir). Sites come from V7's banked site.
PAIRS: dict[str, tuple[str, str, Path]] = {
    "3b->8b": ("3b", "8b", ARM / "fits"),
    "8b->qwen-7b": ("8b", "qwen-7b", ARM / "leg1" / "fits"),
    "8b->dsv2-lite": ("8b", "dsv2-lite", ARM / "leg2" / "fits"),
    "3b->qwen-7b": ("3b", "qwen-7b", ARM / "leg4e_transitivity" / "fits"),   # the closer
    "3b->dsv2-lite": ("3b", "dsv2-lite", ARM / "leg6" / "fits"),             # the check
}

# The system's closing equations, in solve order.
CLOSERS = ("3b->8b", "8b->qwen-7b", "3b->qwen-7b")
DERIVED = {"c_DSV2": "8b->dsv2-lite"}
CHECK_PAIR = "3b->dsv2-lite"
CHECK_FORMULA = ("c_3B", "c_DSV2")


def _load_v7(model: str) -> np.ndarray:
    rel, key, _ = V7[model]
    return _unit(np.load(BANK / rel, allow_pickle=True)[key].astype(np.float64))


def _n_train(fits_dir: Path) -> Optional[int]:
    p = fits_dir / "cp2_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["split"].get("n_train")


def _fit_path(fits_dir: Path, src: str, s_site: int, tgt: str, t_site: int,
              arm: str, family: str) -> Path:
    return fits_dir / f"fit_{src}L{s_site}__{tgt}L{t_site}_{arm}_{family}.npz"


def _family_k(family: str) -> Optional[int]:
    return int(family.split("_k")[1]) if family.startswith("proc_k") else None


def measure_a_hats() -> dict:
    """a_hat for every (pair, arm, family) whose fit exists, with the rank guard applied."""
    out: dict[str, dict] = {}
    for label, (src, tgt, fits_dir) in PAIRS.items():
        if src not in V7 or tgt not in V7:
            continue
        try:
            v_src, v_tgt = _load_v7(src), _load_v7(tgt)
        except (FileNotFoundError, KeyError) as exc:
            out[label] = {"UNAVAILABLE": f"banked V7 missing: {exc}"}
            continue
        s_site, t_site = V7[src][2], V7[tgt][2]
        n_train = _n_train(fits_dir)
        max_k = (n_train / RANK_GUARD_DIVISOR) if n_train else None
        blk: dict = {"src": src, "tgt": tgt, "site_pair": f"{src}L{s_site}->{tgt}L{t_site}",
                     "fits_dir": str(fits_dir), "n_train": n_train,
                     "rank_guard_max_k": max_k, "a_hat": {}}
        for arm in ARMS:
            for family in FAMILIES:
                p = _fit_path(fits_dir, src, s_site, tgt, t_site, arm, family)
                if not p.exists():
                    continue
                k = _family_k(family)
                forbidden = bool(max_k is not None and k is not None and k > max_k)
                tm = load_transport_map(p)
                blk["a_hat"].setdefault(arm, {})[family] = {
                    "value": round(cos(_unit(tm.transport(v_src)), v_tgt), 4),
                    "rank_forbidden": forbidden,
                }
        out[label] = blk
    return out


def solve_system(a: dict[str, float]) -> Optional[dict]:
    """Solve the 4-model star from the three closers. Returns None if ill-posed."""
    try:
        a38, a8q, a3q = a["3b->8b"], a["8b->qwen-7b"], a["3b->qwen-7b"]
    except KeyError:
        return None
    if a8q <= 0 or a38 <= 0 or a3q <= 0:
        return None                      # the star is a positive-factor model
    c3 = math.sqrt(a38 * a3q / a8q)
    if c3 <= 0:
        return None
    c8 = a38 / c3
    cq = a8q / c8
    res = {"c_3B": round(c3, 4), "c_8B": round(c8, 4), "c_Qwen": round(cq, 4)}
    if "8b->dsv2-lite" in a and c8 > 0:
        res["c_DSV2"] = round(a["8b->dsv2-lite"] / c8, 4)
    return res


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    a_hats = measure_a_hats()

    systems: dict[str, dict] = {}
    for arm in ARMS:
        for family in FAMILIES:
            a: dict[str, float] = {}
            forbidden: list[str] = []
            for label, blk in a_hats.items():
                cell = blk.get("a_hat", {}).get(arm, {}).get(family)
                if cell is None:
                    continue
                a[label] = cell["value"]
                if cell["rank_forbidden"]:
                    forbidden.append(label)
            const = solve_system(a)
            if const is None:
                continue
            key = f"{arm}::{family}"
            entry: dict = {
                "arm": arm, "family": family,
                "self_consistent": True,
                "inputs_a_hat": a,
                "rank_forbidden_inputs": forbidden,
                "constants": const,
                "USABLE_FOR_NEW_CONSTANTS": not forbidden,
            }
            if CHECK_PAIR in a and all(k in const for k in CHECK_FORMULA):
                pred = const[CHECK_FORMULA[0]] * const[CHECK_FORMULA[1]]
                obs = a[CHECK_PAIR]
                entry["out_of_sample_check"] = {
                    "pair": CHECK_PAIR,
                    "predicted": round(pred, 4), "observed": round(obs, 4),
                    "abs_error": round(abs(pred - obs), 4),
                    "within_pm_0.05": bool(abs(pred - obs) <= 0.05),
                    "note": "the Leg-6 fourth node, re-derived inside THIS system. A "
                            "system that fails its own check is not a system a new "
                            "constant may be hung on.",
                }
            systems[key] = entry

    res = {
        "STATUS": "UNSTAMPED (C section 8) — scores nothing; substrate for A8-add-7",
        "leg": "A8 extension-pairs smalls — star systems by (template-arm x family)",
        "arm_consistency_rule": (
            "Luxia 2026-07-23: a star constant belongs to the system it was solved in. "
            "The constants of record are NATIVE-arm. A model whose fits are raw-arm-only "
            "(OLMo: no chat template) takes its constant from the RAW-arm system below, "
            "never from the native one. A lower a_hat measured on the raw arm is "
            "ARM-CONFOUNDED before it is a transport fact — the raw arm is known to "
            "stress Procrustes (rake 13: Qwen cross-model CKA-before .47 raw vs .97 "
            "native)."),
        "estimand": "a_hat(A->B) = cos(g . V7_A, V7_B), forward direction, banked V7 "
                    "sites, per (template-arm x family) — never mixed across families",
        "rank_guard": f"add-3: k <= n_train/{RANK_GUARD_DIVISOR}, applied per pair "
                      "(n_train differs by leg: 600 hub, 360 for the Leg-4E closer)",
        "a_hats": a_hats,
        "systems": systems,
        "olmo_note": (
            "olmo2-7b has NO banked V7 (no a5 b7 build was ever run for it — the model "
            "entered the battery for A4/A1 only). a_hat(8B->OLMo) is therefore not "
            "measurable, and c_OLMo cannot be solved in ANY system, native or raw. This "
            "is a missing-target problem, not an arm problem; the raw-arm system below "
            "is where c_OLMo will live once a V7 exists."),
    }
    (OUT / "star_systems.json").write_text(json.dumps(res, indent=1))

    for label, blk in a_hats.items():
        if "UNAVAILABLE" in blk:
            logger.info("%-16s UNAVAILABLE: %s", label, blk["UNAVAILABLE"])
            continue
        for arm, fams in blk.get("a_hat", {}).items():
            logger.info("%-16s %-7s n_train=%s  %s", label, arm, blk["n_train"],
                        {f: v["value"] for f, v in fams.items()})
    for key, e in systems.items():
        chk = e.get("out_of_sample_check", {})
        logger.info("SYSTEM %-18s %s  usable=%s  check pred %s vs obs %s (|e|=%s)",
                    key, e["constants"], e["USABLE_FOR_NEW_CONSTANTS"],
                    chk.get("predicted"), chk.get("observed"), chk.get("abs_error"))
    logger.info("wrote %s", OUT / "star_systems.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
