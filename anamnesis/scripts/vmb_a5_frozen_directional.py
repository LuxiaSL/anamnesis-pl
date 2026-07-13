"""ARM A5 — frozen-text DIRECTIONAL diff + channel decomposition (WAVE2-A5 addendum 13d).

Ports the exp11 matched-text DIRECTIONAL template to A5's banked matched-token cells.
The Wave-1 analyzer (`vmb_arm_a5_analyze` §D) built the isolated frozen-text delta
`Δ = sig(steered-replay) − sig(unsteered stage0 sig, SAME gen id)` but collapsed it to
`abs(Δ).mean()` per family — MAGNITUDE ONLY. Every directional readout there runs on the
token-CONFOUNDED free-gen centroid shift. This script computes the DIRECTION of the
isolated deformation, the exp11 way.

FIRST-READ DISCIPLINE (C§8): emits `a5_frozen_directional_<model>.json` marked
FIRST_READ_PENDING; NO stamps ship before the outer-loop read. Everything is CPU/replay on
ALREADY-BANKED signatures + text — no GPU, run anytime.

Sections (addendum 13d):
  §1.1  delta colinearity — per-gen paired cos(Δ_α1,Δ_α2) across doses; cos(Δ_Vi,Δ_Vj)
        across vectors at matched α; whole-vector + per confirmatory family; R-pair null.
  §1.2  family decomposition of the mean matched-token delta, floor-ruled + bootstrap CI.
  §1.3  dir0-axis projection of the ISOLATED (matched-token) delta.
  §1.4  CHANNEL DECOMPOSITION (prereg §1 mandated 4th readout; channel.py was a stub):
        direct = matched-token delta; token-mediated = free-gen shift − direct.
  §3.1  R1(+V1) site-sweep control (state/semantics) — gates "map site confirmed".
  §3.2  V3 cross-coordinate SELECTIVITY: dir0 movement / mean panel-axis movement, vs R.
  §4.1  lexical-field mass vs α alongside the marker rate — de-confounds the '0.45 sweet spot'.

Usage:
    python -m anamnesis.scripts.vmb_a5_frozen_directional \
        --battery-root ../outputs/battery --out-dir ../outputs/battery/arms/A5 --model 3b
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from anamnesis.analysis.battery.channel import decompose_channel
from anamnesis.analysis.battery.deltas import (
    ConditionCorpus,
    load_floor_scale,
    within_condition_deltas,
)
from anamnesis.analysis.battery.floors import build_cells, load_signature_matrix
from anamnesis.analysis.battery.manifest import MODEL_META

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]

CONFIRMATORY_CELLS = ["whole_vector", "source:attention", "source_band:attention|mid",
                      "source:residual", "source:gate", "source:keys", "source:output"]
MAP_SITE_KEYS = {"V1": 14, "V2": 13, "V3": 14, "V4": 14, "R1": 14, "R2": 14, "R3": 14}
DIR0_PAIR = ("analogical", "contrastive")
# other mode pairs for the selectivity panel (§3.2) — never the same pole as dir0.
PANEL_PAIRS = [("linear", "socratic"), ("dialectical", "analogical"), ("linear", "dialectical")]
ALL_MODES = ["analogical", "contrastive", "dialectical", "linear", "socratic"]
N_RANDOM_PANEL = 8
BOOT = 1000
SEED = 20260713

# §4.1 lexical-field wordlist (sensory/metaphor vehicles). TUNABLE — flagged in output;
# the de-confound reads the SHAPE (monotone vs peaked) not the absolute level, so exact
# membership is second-order. Kept disjoint from the structural analogy MARKERS.
FIELD_WORDS = {
    "fragrance", "scent", "aroma", "perfume", "taste", "flavor", "flavour", "texture",
    "hue", "hues", "color", "colors", "colour", "colours", "shade", "shades", "glow",
    "shimmer", "sound", "sounds", "melody", "symphony", "harmony", "rhythm", "cymbal",
    "whisper", "echo", "warmth", "touch", "tapestry", "weave", "woven", "thread", "threads",
    "fabric", "canvas", "brush", "brushstroke", "palette", "dance", "tide", "tides",
    "current", "currents", "garden", "seed", "seeds", "bloom", "blossom", "mosaic",
    "blend", "blends", "flavors", "flavours", "petal", "petals", "silk", "velvet", "glimmer",
}
MARKERS = [r'\blike a\b', r'\blike an\b', r'\bas if\b', r'\bimagine\b', r'\bthink of\b',
           r'\bmuch like\b', r'\bakin to\b', r'\bmetaphor', r'\banalog', r'\bjust as\b',
           r'\bsimilar to\b', r'\bas though\b']


@dataclass
class MTCell:
    name: str
    vector: str
    site: int
    alpha: float
    gids: list[int]
    Dmap: dict[int, F32]              # gid → signed delta vector (z-space)
    mean_delta: F32 = field(default=None)  # type: ignore[assignment]


def parse_mt(name: str) -> dict | None:
    m = re.match(r"^(?P<vec>V\d|R\d)(?:_L(?P<site>\d+))?_a(?P<a>[\d.]+)$", name)
    if not m:
        return None
    vec = m.group("vec")
    return {"vector": vec, "site": int(m.group("site") or MAP_SITE_KEYS.get(vec, 14)),
            "alpha": float(m.group("a"))}


def parse_freegen(name: str) -> dict | None:
    m = re.match(r"^(?P<vec>V\d|R\d)(?:_L(?P<vl>\d+))?(?:_L\d+)?(?:_at(?P<at>\d+))?_a(?P<a>[\d.]+)$", name)
    if not m:
        return None
    vec = m.group("vec")
    site = int(m.group("at") or m.group("vl") or MAP_SITE_KEYS.get(vec, 14))
    return {"vector": vec, "site": site, "alpha": float(m.group("a"))}


def cos(u: F32, v: F32) -> float:
    d = float(np.linalg.norm(u) * np.linalg.norm(v))
    return float(u @ v / d) if d > 1e-12 else 0.0


def rms(v: F32) -> float:
    return float(np.linalg.norm(v) / np.sqrt(max(len(v), 1)))


def paired_cos(a: MTCell, b: MTCell, mask: NDArray) -> dict:
    """Per-gen paired cosine of the two cells' matched-token deltas over `mask`."""
    shared = sorted(set(a.gids) & set(b.gids))
    if not shared:
        return {"n": 0}
    cs = [cos(a.Dmap[g][mask], b.Dmap[g][mask]) for g in shared]
    return {"n": len(cs), "mean": float(np.mean(cs)), "std": float(np.std(cs)),
            "mean_delta_cos": cos(a.mean_delta[mask], b.mean_delta[mask])}


def load_mt_cells(mt_root: Path, s0Z: F32, s0map: dict[int, int], names: list[str],
                  med: F32, scale: F32) -> dict[str, MTCell]:
    out: dict[str, MTCell] = {}
    for d in sorted(mt_root.iterdir()) if mt_root.exists() else []:
        sd = d / "signatures_v3"
        if not sd.exists():
            continue
        info = parse_mt(d.name)
        if info is None:
            logger.warning(f"unparseable MT cell {d.name} — skipped")
            continue
        X, nms, gids = load_signature_matrix(sd)
        if list(nms) != names:
            logger.warning(f"{d.name}: feature fork — skipped")
            continue
        Z = (X - med) / scale
        Dmap = {g: (Z[i] - s0Z[s0map[g]]).astype(np.float32)
                for i, g in enumerate(gids) if g in s0map}
        if not Dmap:
            logger.warning(f"{d.name}: no gid overlap with stage0 — skipped")
            continue
        cell = MTCell(d.name, info["vector"], info["site"], info["alpha"],
                      list(Dmap.keys()), Dmap)
        cell.mean_delta = np.mean(np.stack(list(Dmap.values())), axis=0).astype(np.float32)
        out[d.name] = cell
    return out


def build_axes(battery_root: Path, model: str, med: F32, scale: F32,
               n_features: int) -> dict[str, F32]:
    """dir0 LDA axis + selectivity panel (other mode-pair LDAs + random z-directions)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    pures: dict[str, ConditionCorpus] = {}
    for m_ in ALL_MODES:
        d = battery_root / f"vmb_a2_{model}_pure_{m_}"
        if (d / "signatures_v3").exists():
            pures[m_] = ConditionCorpus(d / "signatures_v3", d / "metadata.json",
                                        med, scale, f"pure-{m_}")

    def lda_axis(pa: str, pb: str) -> F32 | None:
        if pa not in pures or pb not in pures:
            return None
        X = np.vstack([pures[pa].Z, pures[pb].Z])
        y = np.r_[np.ones(pures[pa].Z.shape[0]), np.zeros(pures[pb].Z.shape[0])]
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)
        a = clf.coef_[0].astype(np.float64)
        n = np.linalg.norm(a)
        return (a / n).astype(np.float32) if n > 1e-12 else None

    axes: dict[str, F32] = {}
    d0 = lda_axis(*DIR0_PAIR)
    if d0 is not None:
        axes["dir0"] = d0
    for pa, pb in PANEL_PAIRS:
        ax = lda_axis(pa, pb)
        if ax is not None:
            axes[f"panel:{pa}_vs_{pb}"] = ax
    rng = np.random.default_rng(SEED)
    for i in range(N_RANDOM_PANEL):
        r = rng.standard_normal(n_features).astype(np.float32)
        axes[f"panel:random{i}"] = (r / np.linalg.norm(r)).astype(np.float32)
    return axes


def project(delta: F32, axis: F32) -> dict:
    tgt = float(abs(delta @ axis))
    total = float(np.linalg.norm(delta))
    off = float(np.sqrt(max(total ** 2 - tgt ** 2, 0.0)))
    return {"target": tgt, "total": total, "off_target": off,
            "effect_per_offtarget": float(tgt / max(off, 1e-9))}


def freegen_shifts(a5_root: Path, names: list[str], med: F32, scale: F32,
                   ) -> tuple[dict[tuple[str, int, float], F32], ConditionCorpus | None, dict]:
    """Signed free-gen centroid shift (steered − pooled rider), per (vector, site, alpha).

    Returns (shift_by_key, pooled_rider_corpus, rider_meta). The pooled rider is the
    reference for the channel decomposition's free-gen half.
    """
    riders: list[ConditionCorpus] = []
    steered: dict[tuple[str, int, float], ConditionCorpus] = {}
    for d in sorted(a5_root.iterdir()) if a5_root.exists() else []:
        if not (d / "signatures_v3").exists():
            continue
        info = parse_freegen(d.name)
        if info is None:
            continue
        X, nms, _ = load_signature_matrix(d / "signatures_v3")
        if list(nms) != names:
            continue
        cc = ConditionCorpus(d / "signatures_v3", d / "metadata.json", med, scale, d.name)
        if info["alpha"] == 0.0:
            riders.append(cc)
        else:
            steered[(info["vector"], info["site"], info["alpha"])] = cc
    if not riders:
        return {}, None, {"n_rider_cells": 0}
    ref = riders[0]
    Z = ref.Z
    for rc in riders[1:]:
        Z = np.vstack([Z, rc.Z])
    ref_centroid = Z.mean(axis=0)
    shifts = {k: (cc.Z.mean(axis=0) - ref_centroid).astype(np.float32)
              for k, cc in steered.items()}
    return shifts, ref, {"n_rider_cells": len(riders), "n_rider_gens": int(Z.shape[0])}


def text_rates(meta_path: Path) -> dict:
    md = json.loads(meta_path.read_text())
    gens = md["generations"] if isinstance(md, dict) and "generations" in md else md
    mk, fld = [], []
    for g in gens:
        t = g.get("generated_text", "").lower()
        toks = t.split()
        w = max(len(toks), 1)
        mk.append(sum(len(re.findall(m, t)) for m in MARKERS) / w * 1000)
        fld.append(sum(1 for tok in toks if tok.strip(".,;:!?\"'()").lower() in FIELD_WORDS) / w * 1000)
    return {"n": len(mk), "markers_per_1k": float(np.mean(mk)) if mk else 0.0,
            "field_mass_per_1k": float(np.mean(fld)) if fld else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default="3b")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = args.model
    mm = MODEL_META[model]
    rng = np.random.default_rng(SEED)

    stage0 = args.battery_root / mm.stage0_dir
    med, scale = load_floor_scale(stage0 / "signatures_v3")
    s0X, names, s0g = load_signature_matrix(stage0 / "signatures_v3")
    names = list(names)
    s0Z = (s0X - med) / scale
    s0map = {g: i for i, g in enumerate(s0g)}
    cells = build_cells(names, mm.n_layers)
    conf = {c: cells[c] for c in CONFIRMATORY_CELLS if c in cells}
    s0c = ConditionCorpus(stage0 / "signatures_v3", stage0 / "metadata.json", med, scale, "s0")
    seed_floor = {c: max(float(np.median(v)), 1e-12)
                  for c, v in within_condition_deltas(s0c, cells).items()}

    mt = load_mt_cells(args.battery_root / f"vmb_a5_mt_{model}", s0Z, s0map, names, med, scale)
    logger.info(f"{len(mt)} matched-token cells loaded")
    if not mt:
        raise SystemExit("no matched-token cells — nothing to do")

    # ── §1.1 delta colinearity ──────────────────────────────────────────────
    colin_dose, colin_vec = [], []
    alphas = sorted({c.alpha for c in mt.values()})
    vectors = sorted({c.vector for c in mt.values()})
    # across doses (same vector): cos(Δ_αi, Δ_αj)
    for vec in vectors:
        by_a = {c.alpha: c for c in mt.values() if c.vector == vec}
        for i in range(len(alphas)):
            for j in range(i + 1, len(alphas)):
                a, b = by_a.get(alphas[i]), by_a.get(alphas[j])
                if a is None or b is None:
                    continue
                row = {"vector": vec, "alpha_lo": alphas[i], "alpha_hi": alphas[j]}
                for cell, mask in conf.items():
                    row[cell] = paired_cos(a, b, mask)
                colin_dose.append(row)
    # across vectors (matched α): trait pairs + R-pair null
    for a_ in alphas:
        by_v = {c.vector: c for c in mt.values() if c.alpha == a_}
        trait_pairs = [("V1", "V3"), ("V3", "V4"), ("V1", "V4"), ("V1", "V2")]
        r_pairs = [("R1", "R2"), ("R1", "R3"), ("R2", "R3")]
        for tag, pairs in (("trait", trait_pairs), ("random_null", r_pairs)):
            for va, vb in pairs:
                if va not in by_v or vb not in by_v:
                    continue
                row = {"alpha": a_, "pair": f"{va}-{vb}", "kind": tag,
                       "sites": [by_v[va].site, by_v[vb].site]}
                for cell, mask in conf.items():
                    row[cell] = paired_cos(by_v[va], by_v[vb], mask)
                colin_vec.append(row)

    # ── §1.2 family decomposition of the mean matched-token delta ────────────
    fam_rows = []
    fam_cells = {c: m for c, m in cells.items()
                 if c.startswith(("source:", "source_band:")) or c == "whole_vector"}
    for name, c in sorted(mt.items()):
        stacked = np.stack([c.Dmap[g] for g in c.gids])
        for cell, mask in fam_cells.items():
            r = rms(c.mean_delta[mask])
            fl = seed_floor.get(cell, 1e-12)
            boot = np.empty(BOOT)
            n = stacked.shape[0]
            for b in range(BOOT):
                idx = rng.integers(0, n, n)
                boot[b] = rms(stacked[idx].mean(axis=0)[mask])
            fam_rows.append({
                "cell_run": name, "vector": c.vector, "site": c.site, "alpha": c.alpha,
                "map_cell": cell, "rms_delta_z": r, "seed_floor": fl,
                "ratio_seed_floor": float(r / fl),
                "boot_ci95": [float(np.quantile(boot, 0.025) / fl),
                              float(np.quantile(boot, 0.975) / fl)],
                "trivial_flag": ("injection_self_read_C1" if cell.startswith(("source:residual", "source_band:residual"))
                                 else "logit_conditioning_C1" if cell.startswith(("source:output", "source_band:output"))
                                 else None),
                "n": n,
            })

    # ── axes for §1.3 / §3.2 ─────────────────────────────────────────────────
    axes = build_axes(args.battery_root, model, med, scale, len(names))
    have_dir0 = "dir0" in axes
    panel_keys = [k for k in axes if k.startswith("panel:")]

    # ── §1.3 dir0 projection of the ISOLATED (matched-token) delta ───────────
    inv_iso = []
    if have_dir0:
        for name, c in sorted(mt.items()):
            p = project(c.mean_delta, axes["dir0"])
            inv_iso.append({"cell_run": name, "vector": c.vector, "site": c.site,
                            "alpha": c.alpha, **p})
        r_eff = {}
        for r in inv_iso:
            if r["vector"].startswith("R"):
                r_eff.setdefault(r["alpha"], []).append(r["effect_per_offtarget"])
        for r in inv_iso:
            base = r_eff.get(r["alpha"])
            if base and not r["vector"].startswith("R"):
                r["effect_over_r_baseline"] = float(r["effect_per_offtarget"] / max(np.mean(base), 1e-9))

    # ── free-gen shifts (for §1.4 channel + §1.3 direct-vs-free comparison) ──
    shifts, rider_ref, rider_meta = freegen_shifts(args.battery_root / f"vmb_a5_{model}",
                                                    names, med, scale)

    # ── §1.4 channel decomposition (direct = MT ; token-mediated = free − direct) ──
    channel_rows = []
    for name, c in sorted(mt.items()):
        key = (c.vector, c.site, c.alpha)
        fg = shifts.get(key)
        if fg is None:
            continue
        for cell, mask in conf.items():
            split = decompose_channel(c.mean_delta, fg, mask, faithfulness_floor=0.0)
            channel_rows.append({
                "cell_run": name, "vector": c.vector, "site": c.site, "alpha": c.alpha,
                "map_cell": cell, **split,
                "caveats": ["MT-vs-stage0 vs free-vs-riders reference mismatch",
                            "direct estimated on DIFFERENT continuations than free-gen",
                            "low-alpha only (MT ran alpha in {.03,.1})"],
            })

    # ── §1.3 direct-vs-free dir0 movement (isolated vs confounded lever) ─────
    lever_compare = []
    if have_dir0:
        for name, c in sorted(mt.items()):
            key = (c.vector, c.site, c.alpha)
            fg = shifts.get(key)
            row = {"cell_run": name, "vector": c.vector, "site": c.site, "alpha": c.alpha,
                   "matched_token": project(c.mean_delta, axes["dir0"])}
            if fg is not None:
                row["free_gen"] = project(fg, axes["dir0"])
            lever_compare.append(row)

    # ── §3.2 V3 cross-coordinate selectivity ─────────────────────────────────
    selectivity = []
    if have_dir0 and panel_keys:
        for name, c in sorted(mt.items()):
            d0 = float(abs(c.mean_delta @ axes["dir0"]))
            panel = [float(abs(c.mean_delta @ axes[k])) for k in panel_keys]
            selectivity.append({
                "cell_run": name, "vector": c.vector, "site": c.site, "alpha": c.alpha,
                "readout": "matched_token",
                "dir0_movement": d0, "panel_mean": float(np.mean(panel)),
                "panel_max": float(np.max(panel)),
                "selectivity_ratio": float(d0 / max(np.mean(panel), 1e-9)),
                "n_panel": len(panel_keys)})
            fg = shifts.get((c.vector, c.site, c.alpha))
            if fg is not None:
                d0f = float(abs(fg @ axes["dir0"]))
                pf = [float(abs(fg @ axes[k])) for k in panel_keys]
                selectivity.append({
                    "cell_run": name, "vector": c.vector, "site": c.site, "alpha": c.alpha,
                    "readout": "free_gen",
                    "dir0_movement": d0f, "panel_mean": float(np.mean(pf)),
                    "panel_max": float(np.max(pf)),
                    "selectivity_ratio": float(d0f / max(np.mean(pf), 1e-9)),
                    "n_panel": len(panel_keys)})

    # ── §3.1 R1(+V1) site-sweep control (state + semantics) ──────────────────
    site_sweep = []
    if have_dir0 and shifts:
        for (vec, site, a_), sv in sorted(shifts.items()):
            if vec not in ("R1", "V1", "V3"):
                continue
            site_sweep.append({
                "vector": vec, "site": site, "alpha": a_,
                "whole_vector_shift_z": rms(sv),
                "dir0_projection": float(abs(sv @ axes["dir0"])),
            })

    # ── §4.1 lexical-field mass vs α (V3 ladder + explore) alongside markers ──
    field_rows = []
    for base in (f"vmb_a5_{model}", "vmb_a5_explore"):
        root = args.battery_root / base
        for d in sorted(root.iterdir()) if root.exists() else []:
            if not (d / "metadata.json").exists() or not d.name.startswith("V3"):
                continue
            info = parse_freegen(d.name) or {}
            field_rows.append({"cell_run": d.name, "alpha": info.get("alpha"),
                               **text_rates(d / "metadata.json")})

    out = {
        "model": model,
        "STATUS": "FIRST_READ_PENDING (C§8 — no stamps ship before outer-loop read)",
        "provenance": "WAVE2-A5-extensions ADDENDUM 13d §1/§3.1/§3.2/§4.1; CPU on banked "
                      "vmb_a5_mt_/vmb_a5_/vmb_a5_explore signatures+text",
        "law": {"colinearity": "per-gen paired cos of matched-token deltas (exp11 template); "
                               "R-pair cos = the isotropic null",
                "family_decomp": "RMS of mean matched-token delta / seed-floor median (12b); "
                                 "bootstrap CI over gens",
                "channel": "direct=matched-token delta; token_mediated=free-gen shift − direct "
                           "(prereg §1 mandated 4th readout; channel.py implemented)",
                "selectivity": "|Δ·dir0| / mean_k|Δ·panel_k|; panel = other mode-pair LDAs + "
                               f"{N_RANDOM_PANEL} random z-directions",
                "field": "lexical-field mass (TUNABLE wordlist) vs structural marker rate"},
        "axes_available": {"dir0": have_dir0, "panel_size": len(panel_keys)},
        "rider_reference": rider_meta,
        "colinearity_across_doses": colin_dose,
        "colinearity_across_vectors": colin_vec,
        "family_decomp_of_delta": fam_rows,
        "dir0_projection_isolated": inv_iso,
        "lever_isolated_vs_freegen": lever_compare,
        "channel_decomposition": channel_rows,
        "selectivity": selectivity,
        "site_sweep_control": site_sweep,
        "lexical_field_vs_alpha": field_rows,
        "notes": [
            "§4.1 FIELD_WORDS is tunable; the de-confound reads SHAPE (monotone field mass vs "
            "peaked marker rate), not absolute level.",
            "channel decomposition caveats are per-row; low-alpha only + reference mismatch.",
            "colinearity/selectivity use MEAN matched-token delta directions (exp11 uses the "
            "per-cell mean delta); per-gen paired cosine reported alongside for the dose pairs.",
        ],
    }
    p = args.out_dir / f"a5_frozen_directional_{model}.json"
    p.write_text(json.dumps(out, indent=1))
    logger.info(f"banked (unstamped, first-read pending) -> {p}")


if __name__ == "__main__":
    main()
