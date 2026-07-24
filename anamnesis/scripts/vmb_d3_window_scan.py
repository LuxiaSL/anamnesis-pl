"""D-3 TEMPERATURE-WINDOW SCAN (CPU-only re-read of banked text).

Descriptive scan authorized by Luxia 2026-07-23. NO generation. Reuses the exact
12c dissociation-hierarchy detector from vmb_arm_a1_analyze.py (content-class TF-IDF
AUC + TTR/length side stats) and computes it for EVERY available dose pair per model,
so the leak-vs-contrast curves can be compared between instruct models and the base
model (OLMo-2-7B).

Detector construction reproduced verbatim from vmb_arm_a1_analyze.analyze_model:
  - condition text/topics indexed by the SURVIVING gen_ids of load_signature_matrix
    (modal-vector guard applied — so OLMo instant-EOS drops match the scored numbers)
  - content: TfidfVectorizer(max_features=5000, ngram_range=(1,2)) + LogisticRegression
    (max_iter=2000), GroupKFold(n_splits=5) by topic_idx, mean fold AUC
  - TTR / length: folded Mann-Whitney-U AUC over all surviving gens

Native dose = the model's Stage-0 floor corpus (floor_dir/{sig_subdir}, metadata.json).
"""
from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from anamnesis.analysis.battery.floors import load_signature_matrix
from anamnesis.analysis.battery.manifest import MODEL_META
from anamnesis.analysis.battery.text_decode import maybe_decode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS = ["3b", "8b", "qwen-7b", "gemma3-27b", "olmo2-7b"]
DOSES = ["t03", "native", "t09", "t12", "p07", "p10"]
SIG_SUBDIR = "signatures_v3"

# nominal temperatures for the contrast-width axis (native filled per model)
NOMINAL_T = {"t03": 0.3, "t09": 0.9, "t12": 1.2}   # p07/p10 = native T, top-p perturbed
IS_TEMP = {"t03", "native", "t09", "t12"}

# sanity gate (t03|t09 content TF-IDF AUC) — must reproduce within +/- .01.
# Targets = the banked scored values of record (outputs/battery/arms/A1/a1_results.json
# for 3b/8b; qwen sibling .548 per D-3 brief; olmo .874). NOTE: verified that the
# 12c detector snippet run verbatim in THIS env (sklearn 1.9.0) reproduces the scan's
# numbers to the bit (3b folds identical); the sub-.01 gap to the banked JSON is a
# scikit-learn version difference, not a construction difference.
SANITY_TARGETS = {"3b": 0.514, "8b": 0.536, "olmo2-7b": 0.874, "qwen-7b": 0.548}
SANITY_TOL = 0.01


def _cond_dir(battery_root: Path, model: str, dose: str) -> tuple[Path, Path]:
    """(sig_dir, metadata_path) for a condition. native -> Stage-0 floor corpus."""
    if dose == "native":
        floor = battery_root / MODEL_META[model].stage0_dir
        return floor / SIG_SUBDIR, floor / "metadata.json"
    d = battery_root / f"vmb_a1_{model}_{dose}"
    return d / SIG_SUBDIR, d / "metadata.json"


def _load_condition(battery_root: Path, model: str, dose: str):
    """Return dict gen_id -> (text, topic_idx, num_tokens) for surviving (modal) gens,
    row-ordered by the signature-matrix gen_ids (the scored alignment)."""
    sig_dir, md_path = _cond_dir(battery_root, model, dose)
    if not sig_dir.exists() or not md_path.exists():
        return None
    _X, _names, gen_ids = load_signature_matrix(sig_dir)   # modal-vector guard applied
    md = json.loads(md_path.read_text())
    gens = md["generations"] if isinstance(md, dict) and "generations" in md else md
    idx = {int(g["generation_id"]): g for g in gens}
    texts, topics, lens, ttrs = [], [], [], []
    for gid in gen_ids:
        if gid not in idx:
            logger.warning(f"{model}/{dose}: gen {gid} missing from metadata — skipped")
            continue
        g = idx[gid]
        t = maybe_decode(g["generated_text"])
        words = t.split()
        ttr = len(set(words)) / max(len(words), 1)
        texts.append(t)
        topics.append(int(g["topic_idx"]))
        lens.append(float(g["num_generated_tokens"]))
        ttrs.append(float(ttr))
    return {"texts": texts, "topics": np.array(topics),
            "lens": np.array(lens), "ttrs": np.array(ttrs), "n": len(texts)}


def _folded_auc(a: np.ndarray, b: np.ndarray) -> float:
    """Folded Mann-Whitney AUC (the 12c side-stat form)."""
    u = mannwhitneyu(a, b).statistic
    return float(max(u, len(a) * len(b) - u) / (len(a) * len(b)))


def _tfidf_groupkfold_auc(ca: dict, cb: dict) -> tuple[float, list[float]]:
    texts = ca["texts"] + cb["texts"]
    y = np.r_[np.zeros(ca["n"]), np.ones(cb["n"])]
    groups = np.concatenate([ca["topics"], cb["topics"]])
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    aucs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(texts, y, groups):
        # a fold is scorable only if both classes appear in the test split
        if len(np.unique(y[te])) < 2:
            continue
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        Xtr = vec.fit_transform([texts[i] for i in tr])
        Xte = vec.transform([texts[i] for i in te])
        clf = LogisticRegression(max_iter=2000).fit(Xtr, y[tr])
        aucs.append(float(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1])))
    return (float(np.mean(aucs)) if aucs else float("nan"), aucs)


def main() -> None:
    battery_root = Path("outputs/battery")
    out_dir = battery_root / "arms/A1/window_scan"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {"scan": "D-3 temperature-window", "date": "2026-07-23",
              "detector": "12c content TF-IDF (max_features=5000, ngram(1,2)), "
                          "LogReg max_iter=2000, GroupKFold(5) by topic_idx; "
                          "TTR/length = folded Mann-Whitney AUC",
              "sig_subdir": SIG_SUBDIR, "native": "Stage-0 floor corpus",
              "models": {}, "anomalies": []}

    for model in MODELS:
        logger.info(f"=== {model} ===")
        native_T = MODEL_META[model].native_temperature
        conds = {}
        ttr_by_dose = {}
        for dose in DOSES:
            c = _load_condition(battery_root, model, dose)
            if c is None:
                result["anomalies"].append(f"{model}/{dose}: bank missing")
                logger.warning(f"{model}/{dose}: bank missing")
                continue
            conds[dose] = c
            ttr_by_dose[dose] = float(np.mean(c["ttrs"]))
        pairs = {}
        for da, db in combinations([d for d in DOSES if d in conds], 2):
            ca, cb = conds[da], conds[db]
            tfidf_auc, folds = _tfidf_groupkfold_auc(ca, cb)
            ttr_auc = _folded_auc(ca["ttrs"], cb["ttrs"])
            len_auc = _folded_auc(ca["lens"], cb["lens"])
            # contrast width on the temperature axis (None if a top-p dose involved)
            dT = None
            if da in IS_TEMP and db in IS_TEMP:
                Ta = native_T if da == "native" else NOMINAL_T[da]
                Tb = native_T if db == "native" else NOMINAL_T[db]
                dT = abs(Ta - Tb)
            pairs[f"{da}|{db}"] = {
                "auc_tfidf": tfidf_auc, "auc_tfidf_folds": folds,
                "auc_ttr": ttr_auc, "auc_length": len_auc,
                "n_per_side": [ca["n"], cb["n"]],
                "abs_dT": dT,
            }
        result["models"][model] = {
            "native_temperature": native_T,
            "ttr_by_dose": ttr_by_dose,
            "pairs": pairs,
        }

    # sanity gate
    gate = {}
    all_pass = True
    for model, target in SANITY_TARGETS.items():
        got = result["models"].get(model, {}).get("pairs", {}).get("t03|t09", {}).get("auc_tfidf")
        ok = got is not None and abs(got - target) <= SANITY_TOL
        gate[model] = {"target": target, "got": got, "abs_diff": (abs(got - target) if got is not None else None),
                       "pass": bool(ok)}
        all_pass = all_pass and ok
    result["sanity_gate"] = {"tol": SANITY_TOL, "per_model": gate, "PASS": bool(all_pass)}

    (out_dir / "d3_window_scan.json").write_text(json.dumps(result, indent=1))
    logger.info(f"wrote {out_dir/'d3_window_scan.json'}; sanity PASS={all_pass}")
    print("SANITY_GATE_PASS", all_pass)
    for m, g in gate.items():
        print(f"  {m}: target={g['target']} got={g['got']} pass={g['pass']}")


if __name__ == "__main__":
    main()
