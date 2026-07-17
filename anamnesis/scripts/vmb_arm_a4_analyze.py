"""ARM A4 analyzer (state surgery; RATIFIED block 12h).

Readouts per model (both anchors):
  1. OCCURRENCE: per (kind x f) x feature-map cell, matched-token delta vs FULL in
     SEED-floor units (12b: cell's Stage-0 stochastic floor median = 1.0; visibility
     bar 0.1x). Positive control: whole-vector visibility at f=0.5, every kind.
  2. DOSE: Spearman monotonicity of the ratio in f, per kind x cell.
  3. KIND (the 2c prediction): length-matched contrasts naive-vs-rotate (PRIMARY,
     ratified) and rotate-vs-recompute (the exp11 imprint pair) + naive-vs-recompute.
     Per cell: exact SIGN-FLIP inference on per-gen kind-difference vectors (exp11
     idiom) + mean-direction cosine; per confirmatory cell: LDA AUC (GroupKFold by
     prompt class). Verdict "kind_carried" is 12d own-tail BH-gated; point direction
     never ships as a verdict.
  4. DISSOCIATION (12c three rungs, P3): content rung structural 0.5 (identical
     text by construction); likelihood rung = AUC on teacher-forced NLL; token-KL
     rung = AUC on token-KL-vs-FULL; internals = whole-vector LDA AUC.
  5. Trivially-expected channels (12d rule b, enumerated in the block PRE-run):
     keys/qk rows flagged surgery-read; attention static rows flagged
     renormalization under naive/rotate; ALL occurrence (vs-FULL) rows flagged
     cardinality-crossing. First-read wording note: exp11 REC = shortened fresh
     prefill (length-matched to naive/rotate) — verified empirically from banked
     a4_cache_len_after per condition; the 12h channel-3 'full-length restore'
     wording maps onto vs-FULL contrasts, not rec contrasts.

Run locally from pipeline/ after syncing the A4 runs into outputs/battery/:
    python -m anamnesis.scripts.vmb_arm_a4_analyze --battery-root ../outputs/battery \
        --out-dir ../outputs/battery/arms/A4 --models 3b,8b
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.stats import mannwhitneyu, spearmanr

from anamnesis.analysis.battery.deltas import ConditionCorpus, load_floor_scale, within_condition_deltas
from anamnesis.analysis.battery.floors import build_cells, load_class_labels, load_signature_matrix
from anamnesis.analysis.battery.gates import require_stamp
from anamnesis.analysis.battery.manifest import MODEL_META
from anamnesis.analysis.battery.stats import bh_fdr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]

FRACTIONS = [0.0625, 0.125, 0.25, 0.5]
KINDS = ["naive", "rotate", "recompute"]
KIND_PAIRS = [("naive", "rotate", "PRIMARY_length_matched"),
              ("rotate", "recompute", "imprint_pair_length_matched"),
              ("naive", "recompute", "secondary_length_matched")]
VISIBILITY = 0.1  # 12b
CONFIRMATORY_CELLS = ["whole_vector", "source:keys", "source:attention",
                      "source:output", "source:gate", "source:residual", "source:qk"]
# classifier rung only where a verdict needs it (internals rung + the kind-carrier
# prediction); LDA at 3,358 dims is the wall-clock bottleneck
LDA_CELLS = ["whole_vector", "source:keys"]
N_PERM = 2000


def load_condition(sig_dir: Path, med: F32, scale: F32) -> tuple[F32, dict[int, int], list[str], dict[int, dict]]:
    """Z matrix + gid->row + names + gid->metadata (dissociation, a4_* columns)."""
    X, names, gen_ids = load_signature_matrix(sig_dir)
    Z = (X - med) / scale
    meta: dict[int, dict] = {}
    for p in sorted(sig_dir.glob("gen_*.json")):
        d = json.loads(p.read_text())
        meta[int(d.get("generation_id", p.stem.split("_")[1]))] = d
    return Z, {g: i for i, g in enumerate(gen_ids)}, names, meta


def sign_flip_test(G: F32, rng: np.random.Generator, n_perm: int = N_PERM) -> tuple[float, float]:
    """Exact-style sign-flip inference on per-gen difference vectors G [n, d]:
    statistic = RMS over features of the per-feature mean. Returns (obs, p)."""
    n = G.shape[0]
    obs = float(np.sqrt(np.mean(G.mean(axis=0) ** 2)))
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n)).astype(np.float32)
    null_means = signs @ G / n  # [n_perm, d]
    t_null = np.sqrt(np.mean(null_means ** 2, axis=1))
    p = float((np.sum(t_null >= obs) + 1) / (n_perm + 1))
    return obs, p


def lda_auc(Za: F32, Zb: F32, groups: NDArray) -> float:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    X = np.vstack([Za, Zb])
    y = np.r_[np.zeros(len(Za)), np.ones(len(Zb))]
    g = np.r_[groups, groups]
    scores = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.1).fit(X[tr], y[tr])
        scores[te] = m.decision_function(X[te])
    return float(roc_auc_score(y, scores))


def rank_auc(a: NDArray, b: NDArray) -> float:
    """AUC of separating a vs b from a scalar column (Mann-Whitney U / (n*m))."""
    u = mannwhitneyu(b, a, alternative="two-sided").statistic
    return float(max(u, len(a) * len(b) - u) / (len(a) * len(b)))


def trivial_flags(cell: str, kind: str) -> list[str]:
    flags = []
    if cell.startswith(("source:keys", "source_band:keys", "source:qk", "source_band:qk")):
        flags.append("surgery_read_channel")
    if kind in ("naive", "rotate") and cell.startswith(("source:attention", "source_band:attention")):
        flags.append("region_mass_renormalization_candidate")
    return flags


def analyze_model(model: str, battery_root: Path, rng: np.random.Generator,
                  run_prefix: str = "vmb_a4") -> dict:
    mm = MODEL_META[model]
    a4_root = battery_root / f"{run_prefix}_{model}" / "signatures_v3"
    stage0 = battery_root / mm.stage0_dir
    med, scale = load_floor_scale(stage0 / "signatures_v3")

    # Seed floor per cell (12b denominator): stage0 within-class pair deltas.
    s0 = ConditionCorpus(stage0 / "signatures_v3", stage0 / "metadata.json",
                         med, scale, f"{model}-stage0")
    names = s0.feature_names
    cells = build_cells(names, mm.n_layers)
    floor = {c: max(float(np.median(v)), 1e-12)
             for c, v in within_condition_deltas(s0, cells).items()}

    conds = ["full"] + [f"{k}_f{f}" for k in KINDS for f in FRACTIONS]
    Z, rowmap, meta = {}, {}, {}
    for c in conds:
        Z[c], rowmap[c], cnames, meta[c] = load_condition(a4_root / c, med, scale)
        if list(cnames) != list(names):
            raise ValueError(f"{model}/{c}: feature names differ from stage0 — fork")

    gids = sorted(set(rowmap["full"]).intersection(*[set(rowmap[c]) for c in conds]))
    logger.info(f"[{model}] {len(gids)} aligned gens across {len(conds)} conditions")
    # GroupKFold groups: A4 gens align with stage0 prompt classes; A4b is a DIFFERENT
    # corpus (dialogue topics) — group by its own a4b_topic (leak-proof by-topic).
    labels = load_class_labels(stage0 / "metadata.json")

    def _group(g: int) -> int:
        m = meta["full"].get(g, {})
        if "a4b_topic" in m:
            return hash(m["a4b_topic"]) % 100000
        if g in labels:
            return labels[g][0] * 10 + hash(labels[g][1]) % 10
        return hash(m.get("a4b_cell_id", g)) % 100000
    groups = np.array([_group(g) for g in gids])

    # Empirical cache-length verification (the first-read wording item)
    # cache-len metadata key is arm-prefixed (a4_ vs a4b_); tolerate either.
    def _clen(m: dict) -> int:
        return int(m.get("a4_cache_len_after", m.get("a4b_cache_len_after", -1)))
    cache_lens = {c: sorted({_clen(meta[c][g]) for g in gids[:5]})
                  for c in conds}
    Dz: dict[str, F32] = {}
    for c in conds[1:]:
        Dz[c] = np.stack([Z[c][rowmap[c][g]] - Z["full"][rowmap["full"][g]] for g in gids])

    # ── 1+2: occurrence ratios + dose monotonicity ──
    occurrence_rows, dose_rows = [], []
    ratios: dict[tuple[str, float, str], float] = {}
    for k in KINDS:
        for f in FRACTIONS:
            c = f"{k}_f{f}"
            per_gen_cell = {cell: np.abs(Dz[c][:, m]).mean(axis=1) for cell, m in cells.items()}
            for cell, vals in per_gen_cell.items():
                r = float(np.median(vals) / floor[cell])
                ratios[(k, f, cell)] = r
                occurrence_rows.append({
                    "model": model, "kind": k, "evict_frac": f, "cell": cell,
                    "ratio_seed_floor": r, "visible_012b": bool(r >= VISIBILITY),
                    "confirmatory": cell in CONFIRMATORY_CELLS,
                    "trivial_flags": trivial_flags(cell, k) + ["cardinality_crossing_vs_full"],
                    "stamp": {"n": int(len(vals)), "M": model,
                              "law": "12b seed-floor units; matched-token vs FULL (bitwise)",
                              "floor_type": "stochastic(stage0)"},
                })
        for cell in CONFIRMATORY_CELLS:
            rs = [ratios[(k, f, cell)] for f in FRACTIONS]
            rho, p = spearmanr(FRACTIONS, rs)
            dose_rows.append({"model": model, "kind": k, "cell": cell,
                              "ratios_by_f": dict(zip([str(f) for f in FRACTIONS], rs)),
                              "spearman_rho": float(rho), "spearman_p": float(p),
                              "stamp": {"n": 4, "M": model, "law": "dose ladder 12h",
                                        "floor_type": "stochastic(stage0)"}})

    # positive control (no hard kill expected: bitwise replay ⇒ detection trivial;
    # the meaningful bar is 12b VISIBILITY at the LARGEST analyzed fraction, whole-vector,
    # every kind — f=0.5 on the standard grid; max(FRACTIONS) under a --fracs override
    # (e.g. the F1-mid rung where ≥.25 is all-UNREACHABLE by turn-protection anatomy)
    f_ctrl = max(FRACTIONS)
    pos_control = {k: ratios[(k, f_ctrl, "whole_vector")] for k in KINDS}
    pos_control["_control_fraction"] = f_ctrl
    pos_pass = all(v >= VISIBILITY for k, v in pos_control.items() if k != "_control_fraction")

    # ── 3: kind contrasts ──
    kind_rows = []
    for a, b, tag in KIND_PAIRS:
        for f in FRACTIONS:
            G_full = Dz[f"{a}_f{f}"] - Dz[f"{b}_f{f}"]  # [n, d] per-gen kind difference
            mean_a = Dz[f"{a}_f{f}"].mean(axis=0)
            mean_b = Dz[f"{b}_f{f}"].mean(axis=0)
            logger.info(f"[{model}] kind contrast {a}_vs_{b} f={f}")
            for cell in CONFIRMATORY_CELLS:
                m = cells[cell]
                obs, p = sign_flip_test(G_full[:, m], rng)
                ca = mean_a[m]
                cb = mean_b[m]
                denom = float(np.linalg.norm(ca) * np.linalg.norm(cb))
                cosab = float(ca @ cb / denom) if denom > 0 else float("nan")
                auc = (lda_auc(Dz[f"{a}_f{f}"][:, m], Dz[f"{b}_f{f}"][:, m], groups)
                       if cell in LDA_CELLS else None)
                kind_rows.append({
                    "model": model, "pair": f"{a}_vs_{b}", "pair_tag": tag,
                    "evict_frac": f, "cell": cell,
                    "signflip_rms_floorz": obs, "signflip_p": p,
                    "mean_direction_cosine": cosab, "lda_auc_groupkfold": auc,
                    "confirmatory": True,
                    "trivial_flags": trivial_flags(cell, a) if cell.startswith("source:keys") else [],
                    "stamp": {"n": int(G_full.shape[0]), "M": model,
                              "law": "sign-flip exact inference (exp11 idiom) + LDA GroupKFold-by-class",
                              "floor_type": "stochastic(stage0) z-space"},
                })

    # 12d own-tail BH gate across confirmatory kind rows
    ps = [r["signflip_p"] for r in kind_rows]
    rej, _ = bh_fdr(ps)
    for r, sig in zip(kind_rows, rej):
        r["verdict"] = "kind_carried" if bool(sig) else "indeterminate"
        r["p_bh_significant"] = bool(sig)

    # ── 4: dissociation rungs per pair × f ──
    dissoc_rows = []
    for a, b, tag in KIND_PAIRS:
        for f in FRACTIONS:
            ca, cb = f"{a}_f{f}", f"{b}_f{f}"
            nll_a = np.array([meta[ca][g]["dissociation"]["tf_nll_mean"] for g in gids])
            nll_b = np.array([meta[cb][g]["dissociation"]["tf_nll_mean"] for g in gids])
            kl_a = np.array([meta[ca][g]["dissociation"]["token_kl_vs_full_mean"] for g in gids])
            kl_b = np.array([meta[cb][g]["dissociation"]["token_kl_vs_full_mean"] for g in gids])
            internals = next(r["lda_auc_groupkfold"] for r in kind_rows
                             if r["pair"] == f"{a}_vs_{b}" and r["evict_frac"] == f
                             and r["cell"] == "whole_vector")
            dissoc_rows.append({
                "model": model, "pair": f"{a}_vs_{b}", "pair_tag": tag, "evict_frac": f,
                "content_auc": 0.5, "content_note": "text identical by construction (structural)",
                "likelihood_auc_tfnll": rank_auc(nll_a, nll_b),
                "tokenkl_auc": rank_auc(kl_a, kl_b),
                "median_token_kl": {a: float(np.median(kl_a)), b: float(np.median(kl_b))},
                "internals_auc_whole_vector": internals,
                "stamp": {"n": int(len(gids)), "M": model,
                          "law": "12c three-rung; likelihood=teacher-forced NLL; internals=LDA GroupKFold",
                          "floor_type": "n/a (rank AUC)"},
            })

    return {"model": model, "n_gens": len(gids), "cache_lens_by_condition": cache_lens,
            "seed_floor_medians": {c: floor[c] for c in CONFIRMATORY_CELLS},
            "positive_control_f05_whole_vector": {"ratios": pos_control, "pass_012b": pos_pass},
            "occurrence": occurrence_rows, "dose": dose_rows,
            "kind_contrasts": kind_rows, "dissociation": dissoc_rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--models", default="3b,8b")
    ap.add_argument("--run-prefix", default="vmb_a4",
                    help="run-dir prefix (vmb_a4 = original short substrate; vmb_a4b = faithful dialogue substrate)")
    ap.add_argument("--fracs", default=None,
                    help="comma-separated eviction fractions to analyze (default = the full "
                         "12h grid). For runs where high fractions are ALL-UNREACHABLE by "
                         "construction (e.g. the F1-mid truncated-dialogue rung: turn "
                         "protections cap reachable eviction ≪ .25), pass the reachable set, "
                         "e.g. '0.0625,0.125'. The exclusion is substrate anatomy, not data "
                         "loss — UNREACHABLE markers stay on disk as the record.")
    args = ap.parse_args()
    if args.fracs:
        global FRACTIONS
        FRACTIONS = [float(x) for x in args.fracs.split(",") if x.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260713)
    tag = args.run_prefix.replace("vmb_", "")   # "a4" | "a4b" — output naming

    results = {}
    for model in [m.strip() for m in args.models.split(",")]:
        res = analyze_model(model, args.battery_root, rng, args.run_prefix)
        for section in ("occurrence", "dose", "kind_contrasts", "dissociation"):
            for row in res[section]:
                require_stamp(row, context=f"A4/{section}")
        results[model] = res
        out = args.out_dir / f"{tag}_results_{model}.json"
        out.write_text(json.dumps(res, indent=1))
        logger.info(f"[{model}] banked -> {out}")

    # cross-model L-rung summary (claim ladder): does the kind verdict + carrier
    # cell structure replicate? (report-side; verdict language stays outer-loop)
    summary = {}
    for model, res in results.items():
        carried = [(r["pair"], r["evict_frac"], r["cell"]) for r in res["kind_contrasts"]
                   if r["verdict"] == "kind_carried"]
        summary[model] = {
            "pos_control": res["positive_control_f05_whole_vector"],
            "n_kind_carried_rows": len(carried),
            "kind_carried_cells": sorted({c for _, _, c in carried}),
            "primary_pair_f05_auc": next(
                r["lda_auc_groupkfold"] for r in res["kind_contrasts"]
                if r["pair"] == "naive_vs_rotate" and r["evict_frac"] == 0.5
                and r["cell"] == "whole_vector"),
            "imprint_pair_f05_auc": next(
                r["lda_auc_groupkfold"] for r in res["kind_contrasts"]
                if r["pair"] == "rotate_vs_recompute" and r["evict_frac"] == 0.5
                and r["cell"] == "whole_vector"),
        }
    (args.out_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"summary -> {args.out_dir / 'a4_summary.json'}")


if __name__ == "__main__":
    main()
