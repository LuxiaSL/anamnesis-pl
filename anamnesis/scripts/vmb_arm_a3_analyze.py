"""Arm A3 (processing strategy / modes; ONE arm, not the paper) — prereg-vmb-v1 §2c A3.

Design (frozen §2c + addenda a/b/c/d; this script is the readout, not the hypothesis):
  Conditions: the 5 pure-mode corpora banked by the A2 generation pass
  (vmb_a2_{model}_pure_{mode}; 20 topics × 8 seeds = 160/mode/model; battery vector).
  Confirmatory cells (3): whole_vector, source:attention,
  source_band:attention|mid (the predicted carrier region).
  Confirmatory contrasts: all 10 mode pairs. m = 3 × 10 × 2 models = 60 visibility
  tests, BH-FDR α=0.05 across the grid. Floor = POOLED within-mode floor (matched
  history includes each mode's system prompt — the A2 cell-(i) convention);
  arm-variance vs the bare Stage-0 floor reported as the 12a §2 flag (expected to
  fire here: system prompts widen clouds; informational, not a verdict).

Prediction readouts (each 12d significance-gated at authoring):
  - VISIBLE: pair deltas above pooled floor (Mann-Whitney, BH) + k=2 ruler for
    magnitude verdicts (A1 convention: CARRIES / detected-sub-ruler / at-floor).
  - Carriers: per-source 5-way OOF accuracy (LDA lsqr/shrinkage, GroupKFold-by-topic,
    LENGTH-RESIDUALIZED record + raw alongside + length-only confound row).
    Source-ordering claim ladder (L1/L2/L3): adjacent relations of the predicted
    ordering attention ≫ residual > gate > keys > output gated by one-sided exact
    McNemar, BH across 4 relations × 2 models. On ANCHORS this rung is a
    REPLICATION (v3 precedent); the L1 cross-family test waits for M5/M6 — the
    report states that scope explicitly.
  - Low-rank: LDA-transform d ∈ {1..4} → multinomial logistic on the d dims, same
    folds ("~3 directions" = saturation by d=3).
  - static ≥ dynamic: violation-form test (dynamic > static one-sided McNemar);
    the prediction STANDS unless the violation is significant.
  - Dissociation hierarchy (12c): content (TF-IDF logistic) → likelihood (the 8
    banked surprise features) → internals (whole-vector LDA; RF alongside for the
    judge comparison), same folds; per-mode recall at every rung. The judge rung
    (blind sonnet judge, protocol = run_judge_scoring.py) merges via
    --judge-key/--judge-results-dir; sub-behavioral verdict (RF ≫ judge,
    socratic-class) gated by per-mode exact McNemar on shared items.
  - Location/dispersion (12c item 2): per pair, confirmatory cells (JSON; top rows
    in the report).
  - Mode-direction map artifact (GATES A5-inv): LDA directions named by STRUCTURE
    COEFFICIENTS (corr(feature, discriminant score) — factor-naming memory: never
    raw weights), cell mass of top coefficients, best-separated mode pair per
    direction, per-model mid-band injection sites + the A5 construction recipe.

Usage:
    PYTHONPATH=. python -m anamnesis.scripts.vmb_arm_a3_analyze \
        --battery-root ../outputs/battery --out-dir ../outputs/battery/arms/A3 \
        [--judge-key key.json --judge-results-dir results/]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, mannwhitneyu

from anamnesis.analysis.battery.deltas import (
    ConditionCorpus,
    build_cells,
    cross_condition_deltas,
    load_floor_scale,
    location_dispersion,
    within_condition_deltas,
)
from anamnesis.analysis.battery.manifest import MODEL_META
from anamnesis.analysis.battery.stats import bh_fdr
from anamnesis.analysis.feature_map import FeatureMap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODES = ["linear", "analogical", "socratic", "contrastive", "dialectical"]
PAIRS = [(a, b) for i, a in enumerate(MODES) for b in MODES[i + 1:]]
CONFIRMATORY_CELLS = ["whole_vector", "source:attention", "source_band:attention|mid"]
PREDICTED_ORDERING = ["attention", "residual", "gate", "keys", "output"]
RULER_K = 2.0
N_SPLITS = 5


# ── helpers ──────────────────────────────────────────────────────────────────

def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """One-sided exact McNemar p for H1: classifier A beats B (paired correctness)."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return 1.0
    return float(binomtest(b, b + c, 0.5, alternative="greater").pvalue)


def residualize_length(X: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Global OLS length residualization (the v3 record convention): removes the
    component of every feature linear in generated length — mode signal carried BY
    length is deliberately excluded from the record."""
    L = np.column_stack([np.ones(len(lengths)), lengths.astype(np.float64)])
    beta, *_ = np.linalg.lstsq(L, X.astype(np.float64), rcond=None)
    return (X - L @ beta).astype(np.float32)


def oof_predictions(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    clf_factory, n_splits: int = N_SPLITS) -> np.ndarray:
    """Out-of-fold predictions, GroupKFold by topic (leak discipline)."""
    from sklearn.model_selection import GroupKFold
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        clf = clf_factory()
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return pred


def lda_factory():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def per_mode_recall(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {m: float(np.mean(pred[y == m] == m)) for m in MODES}


# ── per-model analysis ───────────────────────────────────────────────────────

def analyze_model(model: str, n_layers: int, root: Path) -> dict:
    floor_dir = root / MODEL_META[model].stage0_dir
    med, scale = load_floor_scale(floor_dir / "signatures_v3")

    conds: dict[str, ConditionCorpus] = {}
    meta: dict[str, list[dict]] = {}
    for mode in MODES:
        d = root / f"vmb_a2_{model}_pure_{mode}"
        cc = ConditionCorpus(d / "signatures_v3", d / "metadata.json",
                             med, scale, f"{model}-pure_{mode}")
        # cross-mode pairing is by TOPIC (matched-history axis): re-key
        # (topic_idx, mode) → (topic_idx, "t") — the A2 convention.
        rekeyed: dict[tuple[int, str], list[int]] = {}
        for (tidx, _m), rows_ in cc.rows_by_class.items():
            rekeyed.setdefault((tidx, "t"), []).extend(rows_)
        cc.rows_by_class = rekeyed
        conds[mode] = cc
        md = json.loads((d / "metadata.json").read_text())
        meta[mode] = sorted(md["generations"], key=lambda g: g["generation_id"])

    names = conds[MODES[0]].feature_names
    cells = build_cells(names, n_layers)
    fm = FeatureMap(names, n_layers)

    # ── visibility block (floor-ruler + Mann-Whitney, BH in main) ──
    within = {m: within_condition_deltas(conds[m], cells) for m in MODES}
    stage0 = ConditionCorpus(floor_dir / "signatures_v3", floor_dir / "metadata.json",
                             med, scale, f"{model}-stage0")
    stage0_floor = within_condition_deltas(stage0, cells)

    rows = []
    for (ma, mb) in PAIRS:
        cross = cross_condition_deltas(conds[ma], conds[mb], cells, max_pairs_per_class=20)
        for cell in CONFIRMATORY_CELLS:
            pooled = np.concatenate([within[m][cell] for m in MODES])
            fmed = max(float(np.median(pooled)), 1e-12)
            ratio = float(np.median(cross[cell]) / fmed)
            p = float(mannwhitneyu(cross[cell], pooled, alternative="greater").pvalue)
            rows.append({
                "model": model, "pair": f"{ma}|{mb}", "cell": cell,
                "effect_ratio_vs_pooled_floor": ratio, "p_mannwhitney": p,
                "n_cross": int(len(cross[cell])), "n_floor": int(len(pooled)),
                "stamp": {"n": int(len(cross[cell])), "M": model,
                          "law": "pooled within-mode floor; ruler k=2; BH across 60-grid",
                          "floor_type": "stochastic(within-condition)"},
            })

    # 12a §2 arm-variance flag vs the BARE Stage-0 floor (informational here:
    # mode system prompts are expected to widen matched-history clouds)
    variance_flags = []
    for m in MODES:
        for cell in CONFIRMATORY_CELLS:
            r = float(np.median(within[m][cell]) / max(np.median(stage0_floor[cell]), 1e-12))
            if r > 1.5:
                variance_flags.append({"mode": m, "cell": cell,
                                       "within_mode_over_stage0": round(r, 3)})

    # ── classification block ──
    Z = np.vstack([conds[m].Z for m in MODES])
    y = np.array([m for m in MODES for _ in range(conds[m].Z.shape[0])])
    groups = np.concatenate([[g["topic_idx"] for g in meta[m]] for m in MODES]).astype(int)
    lengths = np.concatenate([[g["num_generated_tokens"] for g in meta[m]] for m in MODES])
    Zr = residualize_length(Z, lengths)

    # length-only confound row
    from sklearn.linear_model import LogisticRegression
    len_pred = oof_predictions(lengths.reshape(-1, 1), y, groups,
                               lambda: LogisticRegression(max_iter=2000))
    length_only_acc = float(np.mean(len_pred == y))

    # per-source ordering (record = residualized; raw alongside)
    source_block: dict[str, dict] = {}
    src_pred: dict[str, np.ndarray] = {}
    for src in PREDICTED_ORDERING:
        mask = cells[f"source:{src}"]
        pred_r = oof_predictions(Zr[:, mask], y, groups, lda_factory)
        pred_raw = oof_predictions(Z[:, mask], y, groups, lda_factory)
        src_pred[src] = pred_r
        source_block[src] = {
            "n_features": int(mask.sum()),
            "acc_resid": float(np.mean(pred_r == y)),
            "acc_raw": float(np.mean(pred_raw == y)),
            "per_mode_recall_resid": per_mode_recall(y, pred_r),
        }
    ordering_tests = []
    for hi, lo in zip(PREDICTED_ORDERING[:-1], PREDICTED_ORDERING[1:]):
        ordering_tests.append({
            "model": model, "relation": f"{hi}>{lo}",
            "acc_hi": source_block[hi]["acc_resid"], "acc_lo": source_block[lo]["acc_resid"],
            "point_holds": source_block[hi]["acc_resid"] > source_block[lo]["acc_resid"],
            "p_mcnemar": mcnemar_exact(src_pred[hi] == y, src_pred[lo] == y),
            "stamp": {"n": int(len(y)), "M": model,
                      "law": "OOF GroupKFold-by-topic; exact McNemar one-sided; BH 8-grid",
                      "floor_type": "stochastic"},
        })

    # whole-vector + carrier band table
    whole_pred = oof_predictions(Zr, y, groups, lda_factory)
    whole_pred_raw = oof_predictions(Z, y, groups, lda_factory)
    band_accs = {}
    for cname, mask in cells.items():
        if not cname.startswith("source_band:") or mask.sum() < 10:
            continue
        pred = oof_predictions(Zr[:, mask], y, groups, lda_factory)
        band_accs[cname] = {"acc_resid": float(np.mean(pred == y)),
                            "n_features": int(mask.sum())}
    top_bands = sorted(band_accs.items(), key=lambda kv: -kv[1]["acc_resid"])[:8]

    # static vs dynamic (violation-form test)
    stat_mask = fm.mask(dynamic=False)
    dyn_mask = fm.mask(dynamic=True)
    pred_stat = oof_predictions(Zr[:, stat_mask], y, groups, lda_factory)
    pred_dyn = oof_predictions(Zr[:, dyn_mask], y, groups, lda_factory)
    static_dynamic = {
        "acc_static_resid": float(np.mean(pred_stat == y)),
        "acc_dynamic_resid": float(np.mean(pred_dyn == y)),
        "n_static": int(stat_mask.sum()), "n_dynamic": int(dyn_mask.sum()),
        "p_violation_dynamic_gt_static": mcnemar_exact(pred_dyn == y, pred_stat == y),
    }

    # low-rank: LDA transform d=1..4 → logistic on the d dims, same folds.
    # One 4-component fit per fold, sliced per d — the transform's first d
    # columns ARE the d-component transform (eigvecs are sorted), so this is
    # numerically identical to refitting at each d.
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import GroupKFold
    correct_by_d = {d: np.zeros(len(y), dtype=bool) for d in (1, 2, 3, 4)}
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(Zr, y, groups):
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto",
                                         n_components=4)
        T_tr = lda.fit_transform(Zr[tr], y[tr])
        T_te = lda.transform(Zr[te])
        for d in (1, 2, 3, 4):
            clf = LogisticRegression(max_iter=2000).fit(T_tr[:, :d], y[tr])
            correct_by_d[d][te] = clf.predict(T_te[:, :d]) == y[te]
    lowrank = {d: float(np.mean(c)) for d, c in correct_by_d.items()}

    # ── dissociation hierarchy (12c): content → likelihood → internals ──
    texts = [g["generated_text"] for m in MODES for g in meta[m]]

    def tfidf_factory():
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline
        return make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
                             LogisticRegression(max_iter=2000))

    from sklearn.model_selection import GroupKFold as _GKF
    tfidf_pred = np.empty(len(y), dtype=object)
    for tr, te in _GKF(n_splits=N_SPLITS).split(texts, y, groups):
        clf = tfidf_factory()
        clf.fit([texts[i] for i in tr], y[tr])
        tfidf_pred[te] = clf.predict([texts[i] for i in te])

    lik_idx = [i for i, n in enumerate(names) if "surprise" in n]
    lik_pred = oof_predictions(Zr[:, lik_idx], y, groups,
                               lambda: LogisticRegression(max_iter=2000))

    def rf_factory():
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=0)
    rf_pred = oof_predictions(Zr, y, groups, rf_factory)

    hierarchy = {
        "content_tfidf": {"acc": float(np.mean(tfidf_pred == y)),
                          "per_mode_recall": per_mode_recall(y, tfidf_pred)},
        "likelihood_surprise": {"acc": float(np.mean(lik_pred == y)),
                                "n_features": len(lik_idx),
                                "per_mode_recall": per_mode_recall(y, lik_pred)},
        "internals_lda": {"acc": float(np.mean(whole_pred == y)),
                          "acc_raw_unresidualized": float(np.mean(whole_pred_raw == y)),
                          "per_mode_recall": per_mode_recall(y, whole_pred)},
        "internals_rf": {"acc": float(np.mean(rf_pred == y)),
                         "per_mode_recall": per_mode_recall(y, rf_pred)},
        "length_only": {"acc": length_only_acc},
        "chance": 0.2,
    }

    # ── location/dispersion per pair (confirmatory cells) ──
    conf_cells = {c: cells[c] for c in CONFIRMATORY_CELLS}
    loc_disp = {f"{ma}|{mb}": location_dispersion(conds[ma], conds[mb], conf_cells)
                for (ma, mb) in PAIRS}

    # ── mode-direction map artifact (gates A5-inv) ──
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda_full = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=4)
    scores = lda_full.fit_transform(Zr, y)
    ev = lda_full.explained_variance_ratio_[:4]
    directions = []
    mid_layers = [l for l in range(n_layers) if n_layers / 3 <= l < 2 * n_layers / 3]
    for d in range(scores.shape[1]):
        s = scores[:, d]
        sc = (Zr - Zr.mean(0)).T @ ((s - s.mean()) / (s.std() + 1e-12)) / len(s)
        feat_std = Zr.std(0) + 1e-12
        struct = sc / feat_std                     # corr(feature, discriminant score)
        top = np.argsort(-np.abs(struct))[:100]
        cell_mass: dict[str, int] = {}
        for i in top:
            t = fm.tags[i]
            k = f"{t.source.value}|{t.band.value if t.band else 'noband'}"
            cell_mass[k] = cell_mass.get(k, 0) + 1
        mode_means = {m: float(np.mean(s[y == m])) for m in MODES}
        best_pair = max(PAIRS, key=lambda p: abs(mode_means[p[0]] - mode_means[p[1]]))
        directions.append({
            "direction": d, "explained_variance_ratio": float(ev[d]) if d < len(ev) else None,
            "mode_means_on_direction": mode_means,
            "best_separated_pair": list(best_pair),
            "cell_mass_top100_structure_coeffs": dict(
                sorted(cell_mass.items(), key=lambda kv: -kv[1])),
            "top20_features": [{"name": names[i], "structure_coeff": float(struct[i])}
                               for i in np.argsort(-np.abs(struct))[:20]],
        })
    mode_dir_map = {
        "model": model, "n": int(len(y)),
        "law": "LDA eigen/shrinkage on length-residualized battery vector; "
               "directions named by structure coefficients (factor-naming rule)",
        "directions": directions,
        "recommended_injection_band": {"band": "mid", "layers": mid_layers},
        "a5_construction_recipe": (
            "For the chosen direction's best-separated pair (A, B): replay one banked "
            "gen per topic from each pure corpus with residual-stream capture at the "
            "mid-band layers; mode-dir vector = mean_t(resid_A) - mean_t(resid_B) at "
            "the layer whose source_band carries the largest structure-coefficient "
            "mass; unit-normalize; inject via ResidualWriteSpec at that layer "
            "(dose ladder per §2c A5). Replay is bitwise-deterministic, so the "
            "vector is exactly reproducible from text + manifests."),
    }

    return {"rows": rows, "variance_flags": variance_flags,
            "rf_correct": [bool(v) for v in (rf_pred == y)],
            "sources": source_block, "ordering_tests": ordering_tests,
            "band_accuracy_top8": top_bands, "static_dynamic": static_dynamic,
            "lowrank_acc_by_dim": lowrank, "hierarchy": hierarchy,
            "location_dispersion": loc_disp, "mode_direction_map": mode_dir_map,
            "n_per_mode": {m: int(conds[m].Z.shape[0]) for m in MODES},
            "length_by_mode": {m: float(np.mean([g["num_generated_tokens"] for g in meta[m]]))
                               for m in MODES}}


# ── judge merge (sub-behavioral cell) ────────────────────────────────────────

def merge_judge(results: dict, key_path: Path, results_dir: Path, root: Path) -> None:
    key = json.loads(key_path.read_text())
    judged: dict[tuple[str, str, int], str] = {}
    for f in sorted(results_dir.glob("result_*.json")):
        for e in json.loads(f.read_text()):
            k = key.get(e["uid"])
            if k is None or e["primary_mode"] == "FAILED":
                logger.warning(f"judge uid {e['uid']} unusable "
                               f"({'no key' if k is None else 'FAILED'}) — skipped")
                continue
            judged[(k["model"], k["mode"], k["gen_id"])] = e["primary_mode"]

    for model in results["models"]:
        r = results["models"][model]
        # rebuild y/groups aligned with the analyzer's row order (mode-major, gen-sorted)
        y, gid, topics = [], [], []
        for mode in MODES:
            md = json.loads((root / f"vmb_a2_{model}_pure_{mode}" / "metadata.json").read_text())
            for g in sorted(md["generations"], key=lambda g: g["generation_id"]):
                y.append(mode); gid.append(g["generation_id"]); topics.append(g["topic_idx"])
        y = np.array(y)
        jp = np.array([judged.get((model, m, i), "MISSING") for m, i in zip(y, gid)])
        have = jp != "MISSING"
        # RF OOF predictions recomputed here would drift from the analysis pass;
        # instead the judge block stores its own RF comparison on the shared items
        # using the banked per-item RF correctness saved by analyze (rf_correct).
        rf_correct = np.array(r["rf_correct"], dtype=bool)
        judge_correct = jp == y
        per_mode = {}
        for m in MODES:
            sel = (y == m) & have
            per_mode[m] = {
                "judge_recall": float(np.mean(judge_correct[sel])) if sel.any() else None,
                "rf_recall": float(np.mean(rf_correct[sel])) if sel.any() else None,
                "p_rf_gt_judge_mcnemar": mcnemar_exact(rf_correct[sel], judge_correct[sel])
                                          if sel.any() else None,
                "n": int(sel.sum()),
            }
        r["judge"] = {
            "n_judged": int(have.sum()), "n_total": int(len(y)),
            "judge_acc": float(np.mean(judge_correct[have])),
            "rf_acc_on_judged": float(np.mean(rf_correct[have])),
            "p_rf_gt_judge_overall": mcnemar_exact(rf_correct[have], judge_correct[have]),
            "per_mode": per_mode,
            "protocol": "blind claude-fable-5 judge (claude-opus-4-8 fallback on "
                        "refusal/parse failure; Luxia ruling 2026-07-12), "
                        "run_judge_scoring.py descriptions, text+topic only, "
                        "batch-shuffled",
        }


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--models", default="3b,8b")
    ap.add_argument("--judge-key", type=Path, default=None)
    ap.add_argument("--judge-results-dir", type=Path, default=None)
    args = ap.parse_args()

    selected = [(m.strip(), MODEL_META[m.strip()].n_layers)
                for m in args.models.split(",") if m.strip()]

    results = {"arm": "A3_processing_strategy",
               "prereg": "prereg-vmb-v1 §2c A3 + addenda a/b/c/d",
               "scope_note": ("ANCHOR run: source-ordering here is a REPLICATION rung; "
                              "the L1 cross-family claim waits for M5/M6 onboarding."),
               "models": {}, "models_included": [m for m, _ in selected]}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows, all_ordering = [], []
    for model, n_layers in selected:
        logger.info(f"=== A3 {model} ===")
        r = analyze_model(model, n_layers, args.battery_root)
        results["models"][model] = r
        # crash insurance: ~35 min of fits per model — bank each model's block
        # immediately so a failure in the aggregation tail can't lose it
        (args.out_dir / f"a3_model_{model}_intermediate.json").write_text(
            json.dumps(r, default=float))
        all_rows.extend(r["rows"])
        all_ordering.extend(r["ordering_tests"])

    # BH-FDR: visibility grid (60) and ordering grid (8), each its own family
    reject_v, adj_v = bh_fdr([r["p_mannwhitney"] for r in all_rows], alpha=0.05)
    for r, rej, p in zip(all_rows, reject_v, adj_v):
        r["bh_reject"] = bool(rej)
        r["p_bh_adjusted"] = float(p)
        r["passes_ruler"] = r["effect_ratio_vs_pooled_floor"] >= RULER_K
        r["verdict"] = ("CARRIES" if (rej and r["passes_ruler"])
                        else "detected-sub-ruler" if rej else "at-floor")
    reject_o, adj_o = bh_fdr([t["p_mcnemar"] for t in all_ordering], alpha=0.05)
    for t, rej, p in zip(all_ordering, reject_o, adj_o):
        t["bh_significant"] = bool(rej)
        t["p_bh_adjusted"] = float(p)
    results["m_confirmatory_visibility"] = len(all_rows)
    results["m_confirmatory_ordering"] = len(all_ordering)

    # 12d-gated rung per model: L2 requires attention strictly on top with the
    # attention>residual relation BH-significant; remaining relations by point
    # direction (the prereg's ≫ applies to attention only).
    for model in results["models_included"]:
        ts = [t for t in all_ordering if t["model"] == model]
        att_rel = next(t for t in ts if t["relation"] == "attention>residual")
        point_all = all(t["point_holds"] for t in ts)
        results["models"][model]["ordering_rung"] = {
            "attention_dominant_significant": bool(att_rel["bh_significant"]
                                                   and att_rel["point_holds"]),
            "full_ordering_point_direction": point_all,
            "relations": [{k: t[k] for k in
                           ("relation", "acc_hi", "acc_lo", "point_holds",
                            "p_bh_adjusted", "bh_significant")} for t in ts],
            "rung": ((("L2(anchor-replication)" if model in ("3b", "8b")
                       else "cross-family: attention-dominance GATED"
                            + ("" if point_all else "; tail ordering per-model "
                               "(point-reversed relation present)"))
                      if (att_rel["bh_significant"] and att_rel["point_holds"])
                      else "L3(per-model; attention-dominance not significance-gated)")),
        }

    if args.judge_key and args.judge_results_dir:
        merge_judge(results, args.judge_key, args.judge_results_dir, args.battery_root)

    (args.out_dir / "a3_results.json").write_text(json.dumps(results, indent=1))
    for model in results["models_included"]:
        mdm = results["models"][model]["mode_direction_map"]
        (args.out_dir / f"a3_mode_direction_map_{model}.json").write_text(
            json.dumps(mdm, indent=1))

    # ── report ──
    lines = ["# Arm A3 (processing strategy / modes) — results of record", "",
             results["scope_note"], "",
             f"n = 160/mode/model (20 topics × 8 seeds, battery vector 3,358); "
             f"m_visibility = {len(all_rows)} (BH α=0.05), m_ordering = {len(all_ordering)}; "
             f"record classification numbers are LENGTH-RESIDUALIZED, GroupKFold-by-topic.", ""]
    for model in results["models_included"]:
        r = results["models"][model]
        lines.append(f"## {model}")
        lines.append("")
        vis = [row for row in all_rows if row["model"] == model]
        n_carry = sum(1 for row in vis if row["verdict"] == "CARRIES")
        n_det = sum(1 for row in vis if row["verdict"] == "detected-sub-ruler")
        lines.append(f"**Visibility:** {n_carry}/{len(vis)} CARRIES (k=2 ruler), "
                     f"{n_det} detected-sub-ruler, {len(vis) - n_carry - n_det} at-floor "
                     f"(3 cells × 10 pairs).")
        worst = sorted(vis, key=lambda row: row["effect_ratio_vs_pooled_floor"])[:3]
        lines.append("Weakest pairs: " + "; ".join(
            f"{w['pair']}/{w['cell']} ratio {w['effect_ratio_vs_pooled_floor']:.2f} "
            f"({w['verdict']})" for w in worst))
        lines.append("")
        lines.append("| source | n_feat | acc (resid) | acc (raw) |")
        lines.append("|---|---|---|---|")
        for src in PREDICTED_ORDERING:
            s = r["sources"][src]
            lines.append(f"| {src} | {s['n_features']} | {s['acc_resid']:.3f} | "
                         f"{s['acc_raw']:.3f} |")
        o = r["ordering_rung"]
        lines.append("")
        lines.append(f"**Ordering rung: {o['rung']}** — " + "; ".join(
            f"{t['relation']}: {t['acc_hi']:.3f} vs {t['acc_lo']:.3f} "
            f"(p_BH {t['p_bh_adjusted']:.2e}{', sig' if t['bh_significant'] else ''})"
            for t in o["relations"]))
        sd = r["static_dynamic"]
        lines.append(f"static {sd['acc_static_resid']:.3f} vs dynamic "
                     f"{sd['acc_dynamic_resid']:.3f} — violation test (dyn>stat) "
                     f"p={sd['p_violation_dynamic_gt_static']:.3f} → prediction "
                     f"{'VIOLATED' if sd['p_violation_dynamic_gt_static'] < 0.05 else 'stands'}")
        lines.append(f"Low-rank acc by LDA dims: " + ", ".join(
            f"d={d}: {a:.3f}" for d, a in r["lowrank_acc_by_dim"].items()))
        h = r["hierarchy"]
        lines.append(f"**Hierarchy (12c):** content TF-IDF **{h['content_tfidf']['acc']:.3f}** → "
                     f"likelihood (surprise feats) **{h['likelihood_surprise']['acc']:.3f}** → "
                     f"internals LDA **{h['internals_lda']['acc']:.3f}** / RF "
                     f"**{h['internals_rf']['acc']:.3f}** (raw-unresid LDA "
                     f"{h['internals_lda']['acc_raw_unresidualized']:.3f}; length-only "
                     f"{h['length_only']['acc']:.3f}; chance 0.2)")
        lines.append("Top carrier bands (5-way acc, resid): " + ", ".join(
            f"{k.split(':')[1]} {v['acc_resid']:.3f}" for k, v in r["band_accuracy_top8"][:5]))
        lines.append(f"Mean length by mode: " + ", ".join(
            f"{m}: {v:.0f}" for m, v in r["length_by_mode"].items()))
        if r["variance_flags"]:
            lines.append(f"12a §2 flags vs bare Stage-0 floor (expected — system prompts): "
                         f"{len(r['variance_flags'])} cells, max "
                         f"{max(f_['within_mode_over_stage0'] for f_ in r['variance_flags']):.2f}×")
        if "judge" in r:
            j = r["judge"]
            lines.append(f"**Judge rung (blind Fable, opus-4-8 fallback):** acc {j['judge_acc']:.3f} vs RF "
                         f"{j['rf_acc_on_judged']:.3f} on n={j['n_judged']} "
                         f"(p RF>judge {j['p_rf_gt_judge_overall']:.2e}); per-mode gaps: " +
                         ", ".join(f"{m}: RF {v['rf_recall']:.2f}/judge {v['judge_recall']:.2f}"
                                   f" (p {v['p_rf_gt_judge_mcnemar']:.3f})"
                                   for m, v in j["per_mode"].items() if v["n"]))
        lines.append("")
    (args.out_dir / "a3_report.md").write_text("\n".join(lines))
    logger.info(f"A3 report → {args.out_dir}/a3_report.md")


if __name__ == "__main__":
    main()
