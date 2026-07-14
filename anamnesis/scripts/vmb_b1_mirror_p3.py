"""B1 "mirror-P3" — the per-family how/what partition alignment map (addendum 14a §4).

P3 (banked): same-text / different-process -> signatures SEPARATE (how-space carries what
what-space can't). Mirror-P3 is the converse quadrant: same-MODE / different-CONTENT -> do
signatures ALIGN, under a content-controlled metric, and WHICH feature families carry that
alignment?

FROZEN METRIC (14a §4, executed verbatim — metric authored outer-loop, execution Opus-lane):
  "Per-family alignment map on banked pure-mode corpora: same-mode/cross-topic vs
   cross-mode/cross-topic contrasts per family cell, 12e decomposed ruler (centroid shift +
   dispersion), null = mode-label permutation within topic (13e-compliant). Deliverable =
   the family-resolved how/what partition table. Frozen predictions: whole-vector alignment
   across content PARTIALLY FAILS (topic nested by construction); mid-band attention-
   allocation families align across content at matched mode above the permutation null.
   Named alternative: NO family aligns => the orthogonality slogan retires entirely."

INSTANTIATION (documented for outer-loop check — the one judgment call the frozen text
leaves open is HOW to compute a 12e centroid+dispersion ruler on a same-mode-vs-cross-mode
CROSS-TOPIC contrast; both legs below are permuted under the frozen within-topic mode-shuffle
null, and the raw pairwise ratio is kept as the 12e-deprecated lower bound only):

  Per feature_map cell, on the model's Stage-0 floor-z scale, over the pooled 5-mode corpus:
   - CENTROID leg (12e location; the VERDICT gate): between-mode centroid separation,
     content-AVERAGED (each mode's centroid is a mean over all its topics, so topic variance
     is averaged out). A family whose modes sit at distinct locations DESPITE content variation
     is a family that aligns same-mode across content. Permutation-gated (p_centroid).
   - DISPERSION leg (12e dispersion; align_ratio): median cross-mode/cross-topic pair delta /
     median same-mode/cross-topic pair delta. >1 => the same-mode cloud is TIGHTER across
     content than the cross-mode cloud (alignment). Permutation-gated (p_align). This ratio
     is a pairwise statistic and is 12e-compressed; it is the effect-size lower bound, read
     ALONGSIDE the centroid leg, never as the sole verdict.
   - NULL (both legs): mode labels shuffled WITHIN topic (preserves topic structure and per-
     topic mode counts), n_perm relabelings drawn ONCE and shared across legs.

A cell "aligns across content at matched mode" (first-read reading; class = outer loop) iff
p_centroid < .05 AND align_ratio > 1 AND p_align < .05. Length is reported as a DIAGNOSTIC
(the frozen metric does not residualize length; Llama modes cap-cluster ~500 tok anyway).

    PYTHONPATH=pipeline python -m anamnesis.scripts.vmb_b1_mirror_p3 \
        --models 3b,8b,qwen-7b,gemma3-27b --out-dir outputs/battery/arms/B1_mirror_p3
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from anamnesis.analysis.battery.deltas import load_floor_scale
from anamnesis.analysis.battery.floors import build_cells, load_signature_matrix
from anamnesis.analysis.battery.manifest import MODEL_META

MODES = ["linear", "analogical", "socratic", "contrastive", "dialectical"]
F32 = NDArray[np.float32]

# Cells surfaced first in the printout (the frozen-prediction cells + the source row).
HEADLINE_CELLS = [
    "whole_vector", "source:attention", "source_band:attention|mid",
    "source:residual", "source:gate", "source:keys", "source:output",
]


def _load_pooled(model: str) -> tuple[F32, list[str], NDArray, NDArray, dict[int, float]]:
    """Pool the 5 pure-mode corpora onto the model's floor-z scale.

    Returns Z[N,D], feature_names, mode_idx[N] (0..4), topic_idx[N], per-mode mean length.
    """
    meta = MODEL_META[model]
    med, scale = load_floor_scale(
        Path("outputs/battery") / meta.stage0_dir / "signatures_v3")
    rows: list[F32] = []
    mode_idx: list[int] = []
    topic_idx: list[int] = []
    lengths: dict[int, list[int]] = {}
    names0: list[str] | None = None
    for mi, mode in enumerate(MODES):
        d = Path("outputs/battery") / f"vmb_a2_{model}_pure_{mode}"
        X, names, gen_ids = load_signature_matrix(d / "signatures_v3")
        if names0 is None:
            names0 = names
        elif names != names0:
            raise ValueError(f"{model}/{mode}: feature names differ from linear — "
                             "cannot pool (re-extract to a common vector)")
        if X.shape[1] != len(med):
            raise ValueError(f"{model}/{mode}: feat dim {X.shape[1]} != floor {len(med)}")
        md = json.loads((d / "metadata.json").read_text())
        gens = {int(g["generation_id"]): g for g in md["generations"]}
        Z = (X - med) / scale
        for r, gid in enumerate(gen_ids):
            rows.append(Z[r])
            mode_idx.append(mi)
            topic_idx.append(int(gens[gid]["topic_idx"]))
            lengths.setdefault(mi, []).append(int(gens[gid]["num_generated_tokens"]))
    Zp = np.stack(rows).astype(np.float32)
    mean_len = {mi: float(np.mean(v)) for mi, v in lengths.items()}
    return Zp, names0, np.asarray(mode_idx), np.asarray(topic_idx), mean_len


def _relabelings(mode_idx: NDArray, topic_idx: NDArray, n_perm: int,
                 seed: int) -> NDArray:
    """n_perm within-topic mode-label shuffles → [n_perm, N] int16 (perm 0 = observed)."""
    rng = np.random.RandomState(seed)
    N = len(mode_idx)
    out = np.empty((n_perm + 1, N), dtype=np.int16)
    out[0] = mode_idx
    topics = np.unique(topic_idx)
    rows_by_topic = {t: np.where(topic_idx == t)[0] for t in topics}
    for p in range(1, n_perm + 1):
        lab = mode_idx.copy()
        for t in topics:
            rr = rows_by_topic[t]
            lab[rr] = mode_idx[rng.permutation(rr)]
        out[p] = lab
    return out


def _centroid_sep_all(Z: F32, labels: NDArray, n_modes: int) -> F32:
    """Per-feature mean over the C(n_modes,2) |Δ centroid| → [D] (content-averaged shift)."""
    onehot = np.zeros((len(labels), n_modes), dtype=np.float32)
    onehot[np.arange(len(labels)), labels] = 1.0
    counts = onehot.sum(0)                       # [n_modes]
    means = (onehot.T @ Z) / counts[:, None]     # [n_modes, D]
    acc = np.zeros(Z.shape[1], dtype=np.float32)
    npair = 0
    for a, b in combinations(range(n_modes), 2):
        acc += np.abs(means[a] - means[b])
        npair += 1
    return acc / npair


def _sample_cross_topic_pairs(topic_idx: NDArray, n_pairs: int,
                              seed: int) -> NDArray:
    """Deterministic [n_pairs, 2] row-index sample with topic_i != topic_j."""
    rng = np.random.RandomState(seed)
    N = len(topic_idx)
    out = np.empty((n_pairs, 2), dtype=np.int64)
    filled = 0
    while filled < n_pairs:
        need = n_pairs - filled
        i = rng.randint(0, N, size=need * 2)
        j = rng.randint(0, N, size=need * 2)
        ok = (topic_idx[i] != topic_idx[j]) & (i != j)
        i, j = i[ok], j[ok]
        take = min(len(i), need)
        out[filled:filled + take, 0] = i[:take]
        out[filled:filled + take, 1] = j[:take]
        filled += take
    return out


def analyze_model(model: str, n_perm: int, n_pairs: int, seed: int) -> dict:
    meta = MODEL_META[model]
    Z, names, mode_idx, topic_idx, mean_len = _load_pooled(model)
    cells = build_cells(names, meta.n_layers)
    n_modes = len(MODES)
    N = len(mode_idx)

    relab = _relabelings(mode_idx, topic_idx, n_perm, seed)         # [n_perm+1, N]

    # ── CENTROID leg: per-feature |Δμ| for observed + all perms, then per-cell masked mean ──
    dmu = np.empty((n_perm + 1, Z.shape[1]), dtype=np.float32)
    for p in range(n_perm + 1):
        dmu[p] = _centroid_sep_all(Z, relab[p], n_modes)

    # ── DISPERSION leg: sample cross-topic pairs; per-cell |Δz| precomputed once ──
    pairs = _sample_cross_topic_pairs(topic_idx, n_pairs, seed)     # [P,2]
    pi, pj = pairs[:, 0], pairs[:, 1]
    absdiff_cellmean: dict[str, F32] = {}
    # chunk the [P,D] abs-diff to bound memory
    chunk = 4000
    cell_names = list(cells)
    for c in cell_names:
        absdiff_cellmean[c] = np.empty(len(pairs), dtype=np.float32)
    for s in range(0, len(pairs), chunk):
        e = min(s + chunk, len(pairs))
        ad = np.abs(Z[pi[s:e]] - Z[pj[s:e]])                        # [chunk, D]
        for c in cell_names:
            m = cells[c]
            absdiff_cellmean[c][s:e] = ad[:, m].mean(axis=1)

    # same/cross membership per perm (vectorized over pairs)
    same_by_perm = relab[:, pi] == relab[:, pj]                     # [n_perm+1, P] bool

    out_cells: dict[str, dict] = {}
    denom = n_perm + 1
    for c in cell_names:
        m = cells[c]
        # centroid leg
        cen = dmu[:, m].mean(axis=1)                                # [n_perm+1]
        cen_obs = float(cen[0])
        p_centroid = float((np.sum(cen[1:] >= cen_obs) + 1) / denom)
        # dispersion leg (align ratio)
        dc = absdiff_cellmean[c]
        ratios = np.empty(n_perm + 1, dtype=np.float64)
        d_same0 = d_cross0 = np.nan
        for p in range(n_perm + 1):
            sm = same_by_perm[p]
            ds = np.median(dc[sm]) if sm.any() else np.nan
            dx = np.median(dc[~sm]) if (~sm).any() else np.nan
            ratios[p] = (dx / ds) if (ds and ds > 1e-12) else np.nan
            if p == 0:
                d_same0, d_cross0 = float(ds), float(dx)
        ratio_obs = float(ratios[0])
        valid = np.isfinite(ratios[1:])
        p_align = float((np.sum(ratios[1:][valid] >= ratio_obs) + 1) / (valid.sum() + 1))
        aligns = bool(p_centroid < 0.05 and ratio_obs > 1.0 and p_align < 0.05)
        out_cells[c] = {
            "n_features": int(m.sum()),
            "centroid_sep_floorz": round(cen_obs, 4),
            "p_centroid": round(p_centroid, 4),
            "d_same_med": round(d_same0, 4),
            "d_cross_med": round(d_cross0, 4),
            "align_ratio": round(ratio_obs, 4),
            "p_align": round(p_align, 4),
            "aligns_across_content": aligns,
        }

    return {
        "arm": "B1_mirror_p3", "model": model,
        "prereg": "addendum 14a §4 (frozen; metric authored outer-loop). Per-family "
                  "same-mode/cross-topic vs cross-mode/cross-topic alignment, 12e decomposed "
                  "ruler (centroid shift + dispersion), null = mode-label permutation within "
                  "topic. Deliverable = the how/what partition table. Frozen: whole-vector "
                  "PARTIALLY FAILS; mid-band attention-allocation aligns above null. Named "
                  "alt: no family aligns => slogan retires.",
        "n_gens": N, "n_modes": n_modes, "modes": MODES,
        "n_perm": n_perm, "n_cross_topic_pairs": n_pairs,
        "mean_gen_len_by_mode": {MODES[k]: round(v, 1) for k, v in mean_len.items()},
        "cells": out_cells,
        "law": {"n": N, "M": model,
                "law": "floor-z (Stage-0 stochastic scale); centroid leg = content-averaged "
                       "between-mode |Δμ| (exact, verdict), dispersion leg = cross/same "
                       "cross-topic pairwise median ratio (12e lower bound); within-topic "
                       "mode-shuffle null n_perm=%d" % n_perm,
                "floor_type": "stochastic"},
    }


def _print_table(res: dict) -> None:
    print(f"\n[{res['model']}] mirror-P3 how/what partition (n={res['n_gens']}, "
          f"n_perm={res['n_perm']})")
    print("  len-by-mode:", res["mean_gen_len_by_mode"])
    print(f"  {'cell':28s} {'cen_sep':>8s} {'p_cen':>6s} {'d_same':>7s} "
          f"{'d_cross':>7s} {'ratio':>6s} {'p_aln':>6s}  aligns")
    ordered = [c for c in HEADLINE_CELLS if c in res["cells"]]
    ordered += [c for c in res["cells"] if c not in ordered]
    for c in ordered:
        v = res["cells"][c]
        mark = "✓" if v["aligns_across_content"] else "·"
        print(f"  {c:28s} {v['centroid_sep_floorz']:8.3f} {v['p_centroid']:6.3f} "
              f"{v['d_same_med']:7.3f} {v['d_cross_med']:7.3f} {v['align_ratio']:6.2f} "
              f"{v['p_align']:6.3f}  {mark}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="3b,8b,qwen-7b,gemma3-27b")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-pairs", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=20260714)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        res = analyze_model(model, args.n_perm, args.n_pairs, args.seed)
        _print_table(res)
        p = args.out_dir / f"b1_mirror_p3_{model}.json"
        p.write_text(json.dumps(res, indent=1))
        print(f"  → {p}")


if __name__ == "__main__":
    main()
