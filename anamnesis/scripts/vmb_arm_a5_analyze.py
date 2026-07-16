"""ARM A5 analyzer (activation-write; ratified 13a; constraints C-doc BINDING).

FIRST-READ DISCIPLINE (C§8): this analyzer emits result JSONs but NO record stamps
ship until the outer loop reads them — rows carry stamp dicts for n/M/law hygiene,
and the emitted file is marked FIRST_READ_PENDING.

Sections:
  A. Free-gen deformation per cell (vector x site x alpha) vs the pooled alpha=0
     rider corpus, 12e decomposed ruler (centroid shift floor-z + dispersion ratio,
     permutation-gated) per feature_map cell; every effect also RELATIVE to the
     R1-R3 isotropic baseline at the same (site, alpha) (C§2).
  B. Dose-response: shift vs alpha Spearman per vector at the map site.
  C. Identity-specificity (C§4): V1-vs-V3 deformation-direction angle vs the
     random-pair angle distribution + held-out LDA (GroupKFold by topic), per alpha.
  D. Matched-token cells: delta vs banked Stage-0 signature of the SAME gen,
     12b seed-floor units; trivial channels flagged (C§1: injection-site self-read
     = residual bands at/after the site; logit conditioning = source:output).
  E. Coherence panel per cell (N5-adjacent, C§3): mean length, TTR, repetition.
  F. Semantics-match (block §3): analogical-vs-contrastive LDA (trained on banked
     pure-mode corpora) applied to steered gens -> fraction-analogical per alpha.
     Pre-stated: positive alpha on V3/V4 -> analogical pole; V1/R flat.
  G. A5-inv readout (C§6): dir0-axis target movement / off-target movement,
     relative to the R baseline, per (route, site, alpha); no-lever verdict is
     SCOPED to the searched grid and authored outer-loop, never here.

Usage (local, after syncing runs):
    python -m anamnesis.scripts.vmb_arm_a5_analyze --battery-root ../outputs/battery \
        --out-dir ../outputs/battery/arms/A5 --model 3b
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

from anamnesis.analysis.battery.deltas import ConditionCorpus, load_floor_scale, within_condition_deltas
from anamnesis.analysis.battery.floors import build_cells, load_signature_matrix
from anamnesis.analysis.battery.magnitude import decomposed_magnitude
from anamnesis.analysis.battery.manifest import MODEL_META
from anamnesis.analysis.battery.stats import bh_fdr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]

CONFIRMATORY_CELLS = ["whole_vector", "source:attention", "source_band:attention|mid",
                      "source:residual", "source:gate", "source:keys", "source:output"]
MAP_SITE_KEYS = {"V1": 14, "V2": 13, "V3": 14, "V4": 14, "R1": 14, "R2": 14, "R3": 14}
DIR0_PAIR = ("analogical", "contrastive")
VISIBILITY = 0.1


def parse_cell_dir(name: str) -> dict | None:
    """vmb_a5 cell dir name → {vector, site, alpha_frac} (riders included).

    Handles the three naming generations: V1_L14_a0.3 · V1_L14_L14_a0.0 (rider,
    doubled tag) · V4_L14_at7_a0.3 (cross-site injection)."""
    # vec is non-greedy so the trailing _L<site>/_at<site>/_a<frac> tags bind first — accepts
    # V1/V3/V4/R1-3 AND the newer names (V3selbare, V7, V3top/V3tail, Rband/Rtop/Rtail).
    m = re.match(r"^(?P<vec>[A-Za-z][A-Za-z0-9]*?)(?:_L(?P<vl>\d+))?(?:_L\d+)?(?:_at(?P<at>\d+))?_a(?P<a>[\d.]+)$", name)
    if not m:
        return None
    vec = m.group("vec")
    site = int(m.group("at") or m.group("vl") or MAP_SITE_KEYS.get(vec, 14))
    return {"vector": vec, "site": site, "alpha_frac": float(m.group("a"))}


def trivial_flags_a5(cell: str, site: int, n_layers: int) -> list[str]:
    flags = []
    if cell == "source:output" or cell.startswith("source_band:output"):
        flags.append("logit_conditioning_C1")
    if cell.startswith("source:residual") or cell.startswith("source_band:residual"):
        flags.append("injection_self_read_candidate_C1")
    return flags


def rekey_topic(cc: ConditionCorpus) -> ConditionCorpus:
    rk: dict[tuple[int, str], list[int]] = {}
    for (t, _m), rows in cc.rows_by_class.items():
        rk.setdefault((t, "t"), []).extend(rows)
    cc.rows_by_class = rk
    return cc


def text_stats(meta_path: Path) -> dict:
    md = json.loads(meta_path.read_text())
    gens = md["generations"] if "generations" in md else md
    lens, ttrs, reps = [], [], []
    for g in gens:
        txt = g.get("generated_text", "")
        toks = txt.split()
        if not toks:
            continue
        lens.append(g.get("num_generated_tokens", len(toks)))
        ttrs.append(len(set(toks)) / len(toks))
        tri = [" ".join(toks[i:i + 3]) for i in range(len(toks) - 2)]
        reps.append(1.0 - (len(set(tri)) / max(len(tri), 1)))
    return {"n": len(lens), "mean_len": float(np.mean(lens)) if lens else 0.0,
            "mean_ttr": float(np.mean(ttrs)) if ttrs else 0.0,
            "mean_trigram_rep": float(np.mean(reps)) if reps else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default="3b")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--map-site", type=int, default=None,
                    help="per-model map site (8B=16, Qwen=18); overrides the 3B "
                         "MAP_SITE_KEYS defaults for the R-relative + dose-response logic")
    args = ap.parse_args()
    if args.map_site is not None:
        for _k in MAP_SITE_KEYS:
            MAP_SITE_KEYS[_k] = args.map_site
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = args.model
    mm = MODEL_META[model]

    stage0 = args.battery_root / mm.stage0_dir
    med, scale = load_floor_scale(stage0 / "signatures_v3")
    s0 = ConditionCorpus(stage0 / "signatures_v3", stage0 / "metadata.json",
                         med, scale, f"{model}-stage0")
    names = s0.feature_names
    cells = build_cells(names, mm.n_layers)
    conf_cells = {c: cells[c] for c in CONFIRMATORY_CELLS if c in cells}
    seed_floor = {c: max(float(np.median(v)), 1e-12)
                  for c, v in within_condition_deltas(s0, {c: cells[c] for c in CONFIRMATORY_CELLS if c in cells}).items()}

    # ── discover free-gen cells ──
    a5_root = args.battery_root / f"vmb_a5_{model}"
    cell_dirs = {}
    for d in sorted(a5_root.iterdir()) if a5_root.exists() else []:
        if not (d / "signatures_v3").exists():
            continue
        info = parse_cell_dir(d.name)
        if info is None:
            logger.warning(f"unparseable cell dir {d.name} — skipped")
            continue
        cell_dirs[d.name] = info | {"dir": d}
    logger.info(f"{len(cell_dirs)} free-gen cells discovered")

    def corpus(d: Path, label: str) -> ConditionCorpus:
        return rekey_topic(ConditionCorpus(d / "signatures_v3", d / "metadata.json",
                                           med, scale, label))

    riders = [n for n, i in cell_dirs.items() if i["alpha_frac"] == 0.0]
    steered = {n: i for n, i in cell_dirs.items() if i["alpha_frac"] > 0.0}
    if not riders:
        raise SystemExit("no alpha=0 rider cells found — reference corpus missing")
    rider_ccs = [corpus(cell_dirs[r]["dir"], f"rider-{r}") for r in riders]
    # pooled rider reference: concatenate corpora (same distribution by construction)
    ref = rider_ccs[0]
    for rc in rider_ccs[1:]:
        off = ref.Z.shape[0]
        ref.Z = np.vstack([ref.Z, rc.Z])
        for cls, rows in rc.rows_by_class.items():
            ref.rows_by_class.setdefault(cls, []).extend([r + off for r in rows])
    logger.info(f"rider reference: {ref.Z.shape[0]} gens from {len(riders)} rider cells")
    ref_centroid = ref.Z.mean(axis=0)

    # ── A: deformation per steered cell ──
    deform_rows = []
    shift_vectors: dict[str, F32] = {}      # cellname → whole-vector centroid shift
    cc_cache: dict[str, ConditionCorpus] = {}
    for cname, info in sorted(steered.items()):
        cc = corpus(info["dir"], cname)
        cc_cache[cname] = cc
        mag = decomposed_magnitude(ref, cc, conf_cells, n_perm=args.n_perm)
        shift_vectors[cname] = cc.Z.mean(axis=0) - ref_centroid
        for cell, mrow in mag.items():
            deform_rows.append({
                "model": model, "cell_run": cname, **{k: info[k] for k in ("vector", "site", "alpha_frac")},
                "map_cell": cell,
                "centroid_shift_floorz": mrow["centroid_shift"], "p_shift": mrow["p_shift"],
                "dispersion_ratio": mrow["dispersion_ratio"],
                "p_disp_wider": mrow["p_disp_wider"], "p_disp_narrower": mrow["p_disp_narrower"],
                "trivial_flags": trivial_flags_a5(cell, info["site"], mm.n_layers),
                "confirmatory": True,
                "stamp": {"n": int(mrow["n_b"]), "M": model,
                          "law": "12e decomposed ruler vs pooled alpha=0 riders",
                          "floor_type": "stochastic(riders)"},
            })
        logger.info(f"[A] {cname}: wv shift {mag['whole_vector']['centroid_shift']:.3f}z")

    # R-relative ratios (C§2): per (site, alpha), trait shift / mean R shift
    r_shift: dict[tuple[int, float], list[float]] = {}
    for r in deform_rows:
        if r["vector"].startswith("R") and r["map_cell"] == "whole_vector":
            r_shift.setdefault((r["site"], r["alpha_frac"]), []).append(r["centroid_shift_floorz"])
    for r in deform_rows:
        base = r_shift.get((r["site"], r["alpha_frac"]))
        r["r_baseline_shift"] = float(np.mean(base)) if base else None
        if base and r["map_cell"] == "whole_vector" and not r["vector"].startswith("R"):
            r["shift_over_r_baseline"] = float(r["centroid_shift_floorz"] / max(np.mean(base), 1e-9))

    # ── B: dose response at map site ──
    dose_rows = []
    for vec in sorted({i["vector"] for i in steered.values()}):
        site = MAP_SITE_KEYS.get(vec, 14)
        pts = sorted([(r["alpha_frac"], r["centroid_shift_floorz"]) for r in deform_rows
                      if r["vector"] == vec and r["site"] == site and r["map_cell"] == "whole_vector"])
        if len(pts) >= 3:
            rho, p = spearmanr([p_[0] for p_ in pts], [p_[1] for p_ in pts])
            dose_rows.append({"model": model, "vector": vec, "site": site,
                              "points": pts, "spearman_rho": float(rho), "spearman_p": float(p),
                              "stamp": {"n": len(pts), "M": model, "law": "dose ladder 13a",
                                        "floor_type": "stochastic(riders)"}})

    # ── C: identity-specificity (C§4) ──
    def angle(u: F32, v: F32) -> float:
        c = float(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12))
        return float(np.degrees(np.arccos(np.clip(c, -1, 1))))

    ident_rows = []
    for af in sorted({i["alpha_frac"] for i in steered.values() if i["alpha_frac"] > 0}):
        def sv(vec):  # shift vector at map site & this alpha
            for cname, info in steered.items():
                if info["vector"] == vec and info["alpha_frac"] == af and info["site"] == MAP_SITE_KEYS.get(vec, 14):
                    return shift_vectors.get(cname)
            return None
        v1, v3 = sv("V1"), sv("V3")
        rs = [sv(f"R{i}") for i in (1, 2, 3)]
        rs = [r for r in rs if r is not None]
        if v1 is None or v3 is None or len(rs) < 2:
            continue
        rand_angles = ([angle(rs[i], rs[j]) for i in range(len(rs)) for j in range(i + 1, len(rs))]
                       + [angle(t, r) for t in (v1, v3) for r in rs])
        row = {"model": model, "alpha_frac": af,
               "angle_V1_V3_deg": angle(v1, v3),
               "random_pair_angles_deg": [round(a, 1) for a in rand_angles],
               "random_pair_median_deg": float(np.median(rand_angles))}
        # held-out classifier: V1-cell vs V3-cell gens (GroupKFold by topic)
        c1 = next((cc_cache[c] for c, i in steered.items()
                   if i["vector"] == "V1" and i["alpha_frac"] == af and i["site"] == 14), None)
        c3 = next((cc_cache[c] for c, i in steered.items()
                   if i["vector"] == "V3" and i["alpha_frac"] == af and i["site"] == 14), None)
        if c1 is not None and c3 is not None:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.metrics import roc_auc_score
            from sklearn.model_selection import GroupKFold

            X = np.vstack([c1.Z, c3.Z])
            y = np.r_[np.zeros(c1.Z.shape[0]), np.ones(c3.Z.shape[0])]
            g1 = np.concatenate([[t] * len(rows) for (t, _), rows in sorted(c1.rows_by_class.items())])
            g3 = np.concatenate([[t] * len(rows) for (t, _), rows in sorted(c3.rows_by_class.items())])
            # rows_by_class indexes rows — rebuild aligned group labels
            grp = np.zeros(len(y))
            off = 0
            for cc_ in (c1, c3):
                for (t, _), rows in cc_.rows_by_class.items():
                    for rr in rows:
                        grp[off + rr] = t
                off += cc_.Z.shape[0]
            scores = np.zeros(len(y))
            for tr, te in GroupKFold(n_splits=5).split(X, y, grp):
                m_ = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X[tr], y[tr])
                scores[te] = m_.decision_function(X[te])
            row["lda_auc_V1_vs_V3"] = float(roc_auc_score(y, scores))
        row["stamp"] = {"n": int(sum(len(r_) for r_ in [v1, v3])), "M": model,
                        "law": "C4 identity-specificity; angles in floor-z space",
                        "floor_type": "stochastic(riders)"}
        ident_rows.append(row)

    # ── D: matched-token cells ──
    mt_rows = []
    mt_root = args.battery_root / f"vmb_a5_mt_{model}"
    s0X, s0names, s0gids = load_signature_matrix(stage0 / "signatures_v3")
    s0Z = (s0X - med) / scale
    s0map = {g: i for i, g in enumerate(s0gids)}
    if mt_root.exists():
        for d in sorted(mt_root.iterdir()):
            sd = d / "signatures_v3"
            if not sd.exists():
                continue
            X, nms, gids_ = load_signature_matrix(sd)
            if list(nms) != list(names):
                logger.warning(f"{d.name}: feature fork — skipped")
                continue
            Z = (X - med) / scale
            m_ = re.match(r"^(?P<key>.+)_a(?P<a>[\d.]+)$", d.name)
            key, af = m_.group("key"), float(m_.group("a"))
            vec = key.split("_")[0]
            site = int(key.rsplit("_L", 1)[1]) if "_L" in key else 14
            D = np.stack([Z[i] - s0Z[s0map[g]] for i, g in enumerate(gids_) if g in s0map])
            for cell in CONFIRMATORY_CELLS:
                if cell not in cells:
                    continue
                vals = np.abs(D[:, cells[cell]]).mean(axis=1)
                ratio = float(np.median(vals) / seed_floor[cell])
                mt_rows.append({
                    "model": model, "cell_run": d.name, "vector": vec, "site": site,
                    "alpha_frac": af, "map_cell": cell,
                    "ratio_seed_floor": ratio, "visible_012b": bool(ratio >= VISIBILITY),
                    "trivial_flags": trivial_flags_a5(cell, site, mm.n_layers),
                    "stamp": {"n": int(len(vals)), "M": model,
                              "law": "12b seed-floor units; matched-token vs banked stage0 sig",
                              "floor_type": "stochastic(stage0)"},
                })
        # R-relative for MT whole-vector
        r_mt = {}
        for r in mt_rows:
            if r["vector"].startswith("R") and r["map_cell"] == "whole_vector":
                r_mt.setdefault(r["alpha_frac"], []).append(r["ratio_seed_floor"])
        for r in mt_rows:
            base = r_mt.get(r["alpha_frac"])
            if base and r["map_cell"] == "whole_vector" and not r["vector"].startswith("R"):
                r["ratio_over_r_baseline"] = float(r["ratio_seed_floor"] / max(np.mean(base), 1e-9))

    # ── E: coherence panel ──
    coh_rows = []
    for cname, info in sorted(cell_dirs.items()):
        st = text_stats(info["dir"] / "metadata.json")
        coh_rows.append({"model": model, "cell_run": cname,
                         **{k: info[k] for k in ("vector", "site", "alpha_frac")}, **st})

    # ── F: semantics-match (mode classifier on pure corpora → steered cells) ──
    sem_rows = []
    try:
        pures = {}
        for m_ in DIR0_PAIR:
            d = args.battery_root / f"vmb_a2_{model}_pure_{m_}"
            pures[m_] = ConditionCorpus(d / "signatures_v3", d / "metadata.json",
                                        med, scale, f"pure-{m_}")
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        Xp = np.vstack([pures[DIR0_PAIR[0]].Z, pures[DIR0_PAIR[1]].Z])
        yp = np.r_[np.ones(pures[DIR0_PAIR[0]].Z.shape[0]), np.zeros(pures[DIR0_PAIR[1]].Z.shape[0])]
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xp, yp)
        frac_ref = float(clf.predict(ref.Z).mean())
        for cname, info in sorted(steered.items()):
            frac = float(clf.predict(cc_cache[cname].Z).mean())
            sem_rows.append({"model": model, "cell_run": cname,
                             **{k: info[k] for k in ("vector", "site", "alpha_frac")},
                             "frac_analogical": frac, "frac_analogical_riders": frac_ref,
                             "stamp": {"n": int(cc_cache[cname].Z.shape[0]), "M": model,
                                       "law": "LDA(pure analogical vs contrastive) applied to steered gens",
                                       "floor_type": "n/a"}})
        # G: A5-inv target/off-target on the dir0 LDA axis
        axis = clf.coef_[0].astype(np.float64)
        axis /= np.linalg.norm(axis)
        inv_rows = []
        for cname, sv_ in shift_vectors.items():
            info = steered[cname]
            tgt = float(abs(sv_ @ axis))
            total = float(np.linalg.norm(sv_))
            off = float(np.sqrt(max(total ** 2 - tgt ** 2, 0)))
            inv_rows.append({"model": model, "cell_run": cname,
                             **{k: info[k] for k in ("vector", "site", "alpha_frac")},
                             "target_movement": tgt, "off_target_movement": off,
                             "effect_per_offtarget": float(tgt / max(off, 1e-9)),
                             "stamp": {"n": int(cc_cache[cname].Z.shape[0]), "M": model,
                                       "law": "C6 metric; dir0 axis = pure-pair LDA (unit, z-space)",
                                       "floor_type": "stochastic(riders)"}})
        # R-relative
        r_eff = {}
        for r in inv_rows:
            if r["vector"].startswith("R"):
                r_eff.setdefault((r["site"], r["alpha_frac"]), []).append(r["effect_per_offtarget"])
        for r in inv_rows:
            base = r_eff.get((r["site"], r["alpha_frac"]))
            if base and not r["vector"].startswith("R"):
                r["effect_over_r_baseline"] = float(r["effect_per_offtarget"] / max(np.mean(base), 1e-9))
    except FileNotFoundError as e:
        logger.warning(f"semantics-match skipped: {e}")
        inv_rows = []

    out = {"model": model, "STATUS": "FIRST_READ_PENDING (C§8 — no stamps ship before outer-loop read)",
           "riders": riders, "deformation": deform_rows, "dose": dose_rows,
           "identity_specificity": ident_rows, "matched_token": mt_rows,
           "coherence": coh_rows, "semantics_match": sem_rows, "a5_inv_metric": inv_rows}
    p = args.out_dir / f"a5_results_{model}.json"
    p.write_text(json.dumps(out, indent=1))
    logger.info(f"banked (unstamped, first-read pending) -> {p}")


if __name__ == "__main__":
    main()
