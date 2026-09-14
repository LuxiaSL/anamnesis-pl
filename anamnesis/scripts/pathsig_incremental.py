"""THE INCREMENTAL TEST — does the level-2 order block add over the 2,769-feature hand suite?

Brief: ``FIRSTREAD-pathsig-legs-2026-09-11.md`` Ruling 3 (the 2x2) + Ruling 4 (real-minus-null
is the reported quantity, never the raw increment). **Exploratory; UNSTAMPED; nothing here is
quotable pre-first-read (C§8).**

WHAT E1 LEFT OPEN
-----------------
E1 showed level-2 beats a 9-number level-1 baseline by +.23 real-minus-null. It did NOT show the
hand suite misses that information — the suite's windowed / slope / STFT operators are marginal
order statistics that may carry correlated signal. This script asks the incremental question on
**identical folds**:

  (a) hand suite alone                  — the reference cell
  (b) level-2 block alone (k=8, P=45)   — E1's cell, re-run inside this harness
  (c) hand suite + level-2              — THE CELL THAT MATTERS
  (d) hand suite with dynamic/spectral families ablated, +/- level-2
                                        — can the order block REPLACE the suite's marginal
                                          order statistics (parity with (a))?

Every cell containing level-2 carries the increment-permutation null at >=3 seeds. The null
permutes ONLY the level-2 block's source increments; the hand block is byte-identical between
real and null, so the difference is attributable to path order and not to the 45 extra columns
(Ruling 4: |A_ij| carries a shuffle-invariant magnitude component).

Ladder / folds are imported verbatim from ``pathsig_read_e1`` so every number here sits on the
same ruler as the E1 artifact: ``preprocess_fold_gpu`` (per-fold length-residualisation on
[prompt_len, gen_len] -> standardise -> lossless Gram reduce) + ``train_eval`` (logit-LBFGS
floor, deep-AdamW), GroupKFold(5) by topic_idx x 3 seeds.

A separate RF-300 arm reproduces the *protocol of record* for the .925 bar (``hand_floor_RF``
in ``encoder_floor.py`` BARS), which is a RandomForest number, not a logit/deep number — the
ladder cells and the .925 bar are different classifiers and are reported as such.

Usage::

    python -m anamnesis.scripts.pathsig_incremental \\
        --bank-dir outputs/pathsig/8b_L16 --layer 16 --model 8b \\
        --runs-root /tmp/pathsig_pull/runs --runs 8b_fat_01,8b_fat_ext \\
        --out outputs/pathsig/incremental_8b_L16.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from anamnesis.analysis.feature_map import MODEL_LAYERS, FeatureMap, Method
from anamnesis.analysis.v3_audit._common import HARD, gen_metadata_by_id, residualize
from anamnesis.scripts.pathsig_features import PathBank, signature_matrix
from anamnesis.scripts.pathsig_read_e1 import NULL_SEEDS, SEEDS, ladder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]
RF_TREES = 300


# ───────────────────────────────────────────────────────────────── config

class IncrementalConfig(BaseModel):
    """Frozen run configuration. Absence raises; nothing is defaulted into existence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bank_dir: Path = Field(description="dir holding paths_*_L{layer}_k*.npz + .meta.json")
    layer: int
    model: str = "8b"
    k: int = Field(8, description="projection rank for the signature block")
    runs_root: Path = Field(description="root holding <run>/metadata.json + <run>/signatures_v3")
    runs: tuple[str, ...]
    sig_subdir: str = "signatures_v3"
    variant: Literal["pc", "nopc"] = "pc"
    device: str = "cpu"
    out: Path
    run_rf: bool = True

    @field_validator("runs")
    @classmethod
    def _nonempty(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("runs must be non-empty")
        return v


# ───────────────────────────────────────────────────────── joined corpus

class JoinedCorpus:
    """Hand-feature suite and projected paths for the SAME generations, in one index order.

    The join key is ``(run, generation_id)``. A path without a matching signature (or vice
    versa) is a hard error, not a dropped row — a silently shrinking n would make the cells
    non-comparable, which is the one thing this experiment cannot tolerate. Label agreement
    (mode / topic_idx / prompt_length / num_generated_tokens) is verified between the two
    independent sources; a mismatch raises.
    """

    def __init__(self, cfg: IncrementalConfig, modes: frozenset[str] = HARD) -> None:
        hand = self._load_hand(cfg, modes)
        paths, path_labels, order = self._load_paths(cfg, modes)

        missing_sig = [kk for kk in order if kk not in hand["rows"]]
        if missing_sig:
            raise KeyError(
                f"{len(missing_sig)} banked paths have no {cfg.sig_subdir} signature "
                f"(first: {missing_sig[:3]}) — refusing a partial join"
            )
        extra_sig = [kk for kk in hand["rows"] if kk not in set(order)]
        if extra_sig:
            raise KeyError(
                f"{len(extra_sig)} hand signatures have no banked path (first: {extra_sig[:3]}) "
                f"— the two sources do not cover the same generations"
            )

        for kk in order:
            h, p = hand["labels"][kk], path_labels[kk]
            if h != p:
                raise ValueError(f"label mismatch for {kk}: signatures {h} vs path bank {p}")

        self.keys = order
        self.X_hand = np.nan_to_num(np.asarray([hand["rows"][kk] for kk in order], dtype=np.float64))
        self.names: list[str] = hand["names"]
        modes_arr = [hand["labels"][kk][0] for kk in order]
        classes = sorted(set(modes_arr))
        self.classes = classes
        self.y = np.asarray([classes.index(m) for m in modes_arr], dtype=np.int64)
        self.topic = np.asarray([hand["labels"][kk][1] for kk in order], dtype=np.int64)
        self.C = np.asarray([[hand["labels"][kk][2], hand["labels"][kk][3]] for kk in order],
                            dtype=np.float64)
        self.bank = PathBank.from_list(
            [paths[kk] for kk in order],
            gen_ids=[kk[1] for kk in order],
            prompt_lengths=[int(hand["labels"][kk][2]) for kk in order],
            source=f"{cfg.bank_dir}:L{cfg.layer}", variant=cfg.variant,
        )
        logger.info(
            f"joined corpus: n={len(order)}  P_hand={self.X_hand.shape[1]}  "
            f"classes={classes}  topics={len(set(self.topic.tolist()))}"
        )

    @staticmethod
    def _load_hand(cfg: IncrementalConfig, modes: frozenset[str]) -> dict[str, Any]:
        names: list[str] | None = None
        rows: dict[tuple[str, int], list[float]] = {}
        labels: dict[tuple[str, int], tuple[str, int, float, float]] = {}
        for run in cfg.runs:
            rd = cfg.runs_root / run
            sd = rd / cfg.sig_subdir
            if not (rd / "metadata.json").exists():
                raise FileNotFoundError(f"{rd}/metadata.json missing — cannot label run {run}")
            if not sd.is_dir():
                raise FileNotFoundError(f"{sd} missing — hand suite unavailable for run {run}")
            md = gen_metadata_by_id(rd / "metadata.json")
            files = sorted(sd.glob("gen_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
            if not files:
                raise FileNotFoundError(f"{sd} holds no gen_*.npz")
            for p in files:
                g = int(p.stem.split("_")[1])
                rec = md.get(g)
                if rec is None:
                    raise KeyError(f"{p}: generation {g} absent from {run}/metadata.json")
                if rec["mode"] not in modes:
                    continue
                z = np.load(p, allow_pickle=True)
                nm = [str(x) for x in z["feature_names"]]
                if names is None:
                    names = nm
                elif nm != names:
                    raise ValueError(
                        f"{p}: feature-name vector differs from the pinned order "
                        f"({len(nm)} vs {len(names)}) — refusing to reorder/zero-fill"
                    )
                rows[(run, g)] = [float(v) for v in z["features"]]
                labels[(run, g)] = (str(rec["mode"]), int(rec["topic_idx"]),
                                    float(rec["prompt_length"]),
                                    float(rec["num_generated_tokens"]))
        if names is None:
            raise RuntimeError("no hand signatures loaded for the requested modes")
        return {"names": names, "rows": rows, "labels": labels}

    @staticmethod
    def _load_paths(cfg: IncrementalConfig, modes: frozenset[str]):
        metas = sorted(cfg.bank_dir.glob(f"paths_*_L{cfg.layer}_k*.meta.json"))
        if not metas:
            raise FileNotFoundError(f"no path banks for L{cfg.layer} in {cfg.bank_dir}")
        paths: dict[tuple[str, int], NDArray[np.float64]] = {}
        plabels: dict[tuple[str, int], tuple[str, int, float, float]] = {}
        order: list[tuple[str, int]] = []
        for mp in metas:
            side = json.loads(mp.read_text())
            npz = Path(side["npz"])
            if not npz.exists():
                npz = mp.parent / (mp.name.replace(".meta.json", ".npz"))
            bank = PathBank.load(npz, variant=cfg.variant)
            run = str(side["run"])
            lab = {int(r["gen_id"]): r for r in side["labels"]}
            if bank.n != len(lab):
                raise ValueError(f"{mp}: bank n={bank.n} != labels {len(lab)}")
            for i in range(bank.n):
                gid = int(bank.gen_ids[i])
                rec = lab.get(gid)
                if rec is None:
                    raise KeyError(f"{mp}: no label for gen {gid}")
                if str(rec["mode"]) not in modes:
                    continue
                key = (run, gid)
                paths[key] = bank.path(i)
                plabels[key] = (str(rec["mode"]), int(rec["topic_idx"]),
                                float(rec["prompt_length"]),
                                float(rec["num_generated_tokens"]))
                order.append(key)
        order.sort()
        return paths, plabels, order


# ───────────────────────────────────────────────────────── blocks + cells

def sig_block(bank: PathBank, *, layer: int, k: int, level: Literal[1, 2],
              permutation_seed: int | None = None) -> F32:
    X, _names, kept = signature_matrix(bank, layer=layer, k=k, level=level,
                                       permutation_seed=permutation_seed)
    if len(kept) != bank.n:
        raise RuntimeError(
            f"{bank.n - len(kept)} paths were too short for level {level} — the split would "
            f"differ between cells; refusing a non-comparable number"
        )
    return X


def rf_cv(X: F64, y: NDArray[np.int_], topic: NDArray[np.int_], C: F64,
          seeds: int = SEEDS) -> dict[str, float]:
    """RF-300, GroupKFold(5) by topic x seeds, per-fold length-residualised — the protocol of
    record behind the .925 bar (gate_a_v3_battery / encoder_floor BARS.hand_floor_RF)."""
    fold_accs: list[float] = []
    for s in range(seeds):
        for tr, te in GroupKFold(5).split(X, y, topic):
            Ftr, Fte = residualize(X[tr].copy(), X[te].copy(), C[tr], C[te])
            est = RandomForestClassifier(RF_TREES, random_state=s, n_jobs=1)
            est.fit(Ftr, y[tr])
            fold_accs.append(float(est.score(Fte, y[te])))
    return {"test": round(float(np.mean(fold_accs)), 4), "std": round(float(np.std(fold_accs)), 4)}


def _rn(real: dict[str, Any], nulls: list[dict[str, Any]]) -> dict[str, Any]:
    """real-minus-null per arch (Ruling 4: the reported quantity is never the raw increment)."""
    out: dict[str, Any] = {}
    for arch in ("logit", "deep"):
        vals = [n[arch]["test"] for n in nulls if arch in n]
        if not vals:
            raise ValueError(f"no null values for arch {arch}")
        out[arch] = {
            "null_mean": round(float(np.mean(vals)), 4),
            "null_min": round(float(np.min(vals)), 4),
            "null_max": round(float(np.max(vals)), 4),
            "real_minus_null": round(float(real[arch]["test"] - np.mean(vals)), 4),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bank-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--model", default="8b")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--runs", required=True, help="comma-separated run names")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-rf", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    cfg = IncrementalConfig(
        bank_dir=a.bank_dir, layer=a.layer, model=a.model, k=a.k, runs_root=a.runs_root,
        runs=tuple(x for x in a.runs.split(",") if x.strip()), device=a.device, out=a.out,
        run_rf=not a.no_rf,
    )
    corpus = JoinedCorpus(cfg)
    n = len(corpus.y)
    nclass = len(corpus.classes)

    n_layers = MODEL_LAYERS.get(cfg.model)
    if n_layers is None:
        raise KeyError(f"no layer count registered for model {cfg.model!r} in feature_map")
    fm = FeatureMap(corpus.names, n_layers=n_layers)
    dyn = fm.mask(dynamic=True)
    spec = np.array([t.method == Method.spectral for t in fm.tags], dtype=bool)

    ablations: dict[str, NDArray[np.bool_]] = {
        "abl_dynamic": ~dyn,                 # drop every DYNAMIC-tagged feature
        "abl_dynamic_spectral": ~(dyn | spec),  # + drop the STFT/graph-spectral operators
    }
    drop_summary = {
        name: {
            "n_dropped": int((~m).sum()),
            "n_kept": int(m.sum()),
            "dropped_by_family": {
                f: int(c) for f, c in sorted(
                    {fam: sum(1 for i, t in enumerate(fm.tags) if not m[i] and t.family == fam)
                     for fam in sorted({t.family for t in fm.tags})}.items(),
                    key=lambda kv: -kv[1]) if c
            },
        }
        for name, m in ablations.items()
    }

    result: dict[str, Any] = {
        "cell": "THE INCREMENTAL TEST — level-2 order block vs / with the hand suite",
        "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED, not quotable",
        "brief": "FIRSTREAD-pathsig-legs-2026-09-11.md Rulings 3 + 4",
        "model": cfg.model, "site": f"L{cfg.layer}", "n": int(n), "nclass": nclass,
        "classes": corpus.classes, "n_topics": len(set(corpus.topic.tolist())),
        "chance": round(1.0 / nclass, 4),
        "runs": list(cfg.runs), "runs_root": str(cfg.runs_root),
        "P_hand": int(corpus.X_hand.shape[1]),
        "projection": {"basis": "pca_model_corrected.pkl per-layer", "variant": cfg.variant,
                       "k": cfg.k},
        "augmentation": "t/(T-1) normalised — pacing, not duration",
        "folds": "GroupKFold(5) by topic_idx x 3 seeds, per-fold length-residualised on "
                 "[prompt_len, gen_len] (identical across every cell)",
        "null": f"increment permutation on the level-2 block only, seeds {list(NULL_SEEDS)}",
        "ablations": drop_summary,
        "feature_map_summary": {k2: {str(getattr(kk, "value", kk)): vv for kk, vv in v.items()}
                                if isinstance(v, dict) else v
                                for k2, v in fm.summary().items()},
        "cells": {}, "nulls": {}, "real_minus_null": {},
    }

    def bank_out() -> None:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(json.dumps(result, indent=1))

    t0 = time.time()
    Xh = corpus.X_hand.astype(np.float32)
    S1 = sig_block(corpus.bank, layer=cfg.layer, k=cfg.k, level=1)
    S2 = sig_block(corpus.bank, layer=cfg.layer, k=cfg.k, level=2)
    S2_null = {s: sig_block(corpus.bank, layer=cfg.layer, k=cfg.k, level=2, permutation_seed=s)
               for s in NULL_SEEDS}
    logger.info(f"blocks: hand {Xh.shape}  lvl1 {S1.shape}  lvl2 {S2.shape}")

    def run_cell(key: str, X: F32, label: str) -> None:
        result["cells"][key] = {"P": int(X.shape[1]),
                                **ladder(X, corpus.y, corpus.topic, corpus.C, nclass,
                                         cfg.device, label)}
        bank_out()

    def run_null(key: str, make: Any, label: str) -> None:
        nulls = []
        for s in NULL_SEEDS:
            Xn = make(S2_null[s])
            nulls.append({"seed": s, **ladder(Xn, corpus.y, corpus.topic, corpus.C, nclass,
                                              cfg.device, f"{label} NULL s{s}")})
        result["nulls"][key] = nulls
        result["real_minus_null"][key] = _rn(result["cells"][key], nulls)
        bank_out()

    # ── (a) hand suite alone ────────────────────────────────────────────────
    run_cell("a_hand", Xh, "(a) hand suite alone")
    if cfg.run_rf:
        logger.info("RF-300 comparability arm (the protocol behind the .925 bar) …")
        result["rf_arm"] = {
            "protocol": f"RandomForest({RF_TREES}) x {SEEDS} seeds, GroupKFold(5) by topic, "
                        f"per-fold length-residualised — matches gate_a_v3_battery",
            "bar_of_record": {"8b_hand_floor_RF": 0.925, "3b_hand_floor_RF": 0.909},
            "a_hand": rf_cv(corpus.X_hand, corpus.y, corpus.topic, corpus.C),
        }
        logger.info(f"  RF hand suite: {result['rf_arm']['a_hand']}")
        bank_out()

    # ── level-1 / level-2 blocks alone ──────────────────────────────────────
    run_cell("lvl1_only", S1, f"level-1 only k={cfg.k}")
    run_cell("b_lvl2_only", S2, f"(b) level-1+2 only k={cfg.k}")
    run_null("b_lvl2_only", lambda Sn: Sn, "(b) level-1+2 only")

    # ── (c) THE CELL THAT MATTERS ───────────────────────────────────────────
    run_cell("hand_plus_lvl1", np.hstack([Xh, S1]), "hand + level-1 (parameter control)")
    run_cell("c_hand_plus_lvl2", np.hstack([Xh, S2]), "(c) hand + level-1+2")
    run_null("c_hand_plus_lvl2", lambda Sn: np.hstack([Xh, Sn]), "(c) hand + level-1+2")

    # ── (d) ablated suite, with and without the order block ─────────────────
    for ab_name, keep in ablations.items():
        Xa = Xh[:, keep]
        run_cell(f"d_{ab_name}", Xa, f"(d) {ab_name} alone (P={int(keep.sum())})")
        run_cell(f"d_{ab_name}_plus_lvl2", np.hstack([Xa, S2]), f"(d) {ab_name} + level-1+2")
        run_null(f"d_{ab_name}_plus_lvl2", lambda Sn, Xa=Xa: np.hstack([Xa, Sn]),
                 f"(d) {ab_name} + level-1+2")
        if cfg.run_rf:
            result["rf_arm"][f"d_{ab_name}"] = rf_cv(corpus.X_hand[:, keep], corpus.y,
                                                     corpus.topic, corpus.C)
            bank_out()

    result["elapsed_s"] = round(time.time() - t0, 1)
    bank_out()
    logger.info(f"incremental test banked (first-read pending) -> {cfg.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
