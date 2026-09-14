"""E1 read — the 8B mode stake on path-signature features (local CPU, banked projections).

Brief: ``SPEC-path-signature-family-2026-09-11.md`` §3 E1 + §4b. **Exploratory; UNSTAMPED;
nothing here is quotable pre-first-read (C§8).**

WHAT THIS IS AND IS NOT
-----------------------
E1 is the *sanity leg*: it proves the machinery on real banked states and runs the
increment-permutation null on real paths. It is NOT the decisive test — the hand suite already
reads 92.5% on the 8B 5-way, so a level-2 increment has ~7 points of headroom and a null there
is weak evidence. Per §4b the 5-way is run **for comparability** and hard **binary pairs** are
run **for headroom**; neither alone is the read.

Ladder: ``preprocess_fold_gpu`` (per-fold length-residualisation on [prompt_len, gen_len] →
standardise → Gram-reduce) + ``train_eval`` (logit-LBFGS floor, deep-AdamW), GroupKFold(5)
**by topic_idx** × 3 seeds — i.e. the suite's own machinery, so the numbers sit on the same
ruler as the 90.9/92.5 of record. Level-1-only and level-1+2 are run on **identical folds**,
identical classifier, identical n: the only thing that moves is the feature block.

Every cell carries n / model / site / projection (k, basis) / augmentation. Time augmentation
is the normalised clock ``t/(T−1)`` per the desk ruling — **this family measures pacing, not
duration.**

Usage::

    python -m anamnesis.scripts.pathsig_read_e1 \\
        --bank-dir outputs/pathsig/8b_L16 --layer 16 --model 8b \\
        --out outputs/pathsig/e1_8b_L16.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold

from anamnesis.analysis.v3_audit._common import HARD, train_eval
from anamnesis.analysis.v3_audit.surface_encoder_floor import preprocess_fold_gpu
from anamnesis.scripts.pathsig_features import PathBank, signature_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]
SEEDS = 3
K = 32
DEEP_EPOCHS = 800
LBFGS_L2 = 1e-3
NULL_SEEDS = (101, 202, 303)
EIGHT_WAY = frozenset(HARD | {"structured", "compressed", "associative"})


def ladder(X: F32, y: NDArray[np.int_], topic: NDArray[np.int_], C: NDArray[np.float64],
           nclass: int, device: str, name: str) -> dict[str, Any]:
    """logit + deep over GroupKFold(5) × SEEDS, per-fold length-residualised. Mirrors
    ``vmb_s51_encoder_on_raw._cv_ladder`` exactly, with nclass parametrised."""
    accs: dict[str, list[float]] = {"logit": [], "deep": []}
    for seed in range(SEEDS):
        for tr, te in GroupKFold(5).split(X, y, topic):
            Ztr, Zte = preprocess_fold_gpu(X[tr], X[te], C[tr], C[te], True, device)
            for arch in ("logit", "deep"):
                ta, _ = train_eval(Ztr, y[tr], Zte, y[te], arch, seed, device,
                                   deep_epochs=DEEP_EPOCHS, lbfgs_l2=LBFGS_L2,
                                   nclass=nclass, k=K)
                accs[arch].append(ta)
    out = {a: {"test": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
           for a, v in accs.items()}
    logger.info(f"  {name}: logit {out['logit']['test']:.1%}±{out['logit']['std']:.1%}  "
                f"deep {out['deep']['test']:.1%}±{out['deep']['std']:.1%}  "
                f"(chance {1.0/nclass:.1%}, n={len(y)})")
    return out


class Corpus:
    """Merged banked paths + labels across runs, restricted to a mode set."""

    def __init__(self, bank_dir: Path, layer: int, k: int, variant: str = "pc") -> None:
        self.bank_dir = bank_dir
        self.layer = layer
        self.k = k
        paths: list[NDArray[Any]] = []
        gen_ids: list[int] = []
        plens: list[int] = []
        modes: list[str] = []
        topics: list[int] = []
        cov: list[list[float]] = []
        runs: list[str] = []
        metas = sorted(bank_dir.glob(f"paths_*_L{layer}_k*.meta.json"))
        if not metas:
            raise FileNotFoundError(f"no path banks for L{layer} in {bank_dir}")
        for mp in metas:
            side = json.loads(mp.read_text())
            bank = PathBank.load(side["npz"] if Path(side["npz"]).exists()
                                 else mp.with_suffix("").with_suffix(".npz"), variant=variant)
            labels = {int(r["gen_id"]): r for r in side["labels"]}
            if bank.n != len(labels):
                raise ValueError(f"{mp}: bank n={bank.n} != labels {len(labels)}")
            for i in range(bank.n):
                gid = int(bank.gen_ids[i])
                rec = labels.get(gid)
                if rec is None:
                    raise KeyError(f"{mp}: no label for gen {gid}")
                paths.append(bank.path(i))
                gen_ids.append(gid)
                plens.append(int(bank.prompt_lengths[i]))
                modes.append(str(rec["mode"]))
                topics.append(int(rec["topic_idx"]))
                cov.append([float(rec["prompt_length"] or 0),
                            float(rec["num_generated_tokens"] or 0)])
                runs.append(side["run"])
        self.paths = paths
        self.gen_ids = np.asarray(gen_ids, dtype=np.int64)
        self.prompt_lengths = np.asarray(plens, dtype=np.int64)
        self.modes = np.asarray(modes, dtype=object)
        self.topics = np.asarray(topics, dtype=np.int64)
        self.C_all = np.asarray(cov, dtype=np.float64)
        self.runs = np.asarray(runs, dtype=object)
        logger.info(f"corpus: {len(paths)} paths from {len(metas)} banks, "
                    f"{len(set(modes))} modes, {len(set(topics))} topics")

    def subset(self, modes: set[str]) -> tuple[PathBank, NDArray[np.int_], NDArray[np.int_],
                                               NDArray[np.float64], list[str]]:
        idx = [i for i, m in enumerate(self.modes) if m in modes]
        if not idx:
            raise ValueError(f"no generations for modes {sorted(modes)}")
        classes = sorted({self.modes[i] for i in idx})
        cmap = {c: j for j, c in enumerate(classes)}
        bank = PathBank.from_list(
            [self.paths[i] for i in idx],
            gen_ids=[int(self.gen_ids[i]) for i in idx],
            prompt_lengths=[int(self.prompt_lengths[i]) for i in idx],
            source=f"{self.bank_dir}:L{self.layer}", variant="pc",
        )
        y = np.asarray([cmap[self.modes[i]] for i in idx], dtype=np.int64)
        topic = np.asarray([self.topics[i] for i in idx], dtype=np.int64)
        C = self.C_all[idx]
        return bank, y, topic, C, classes


def cell(bank: PathBank, y, topic, C, *, layer: int, k: int, level: Literal[1, 2],
         nclass: int, device: str, label: str) -> dict[str, Any]:
    X, names, kept = signature_matrix(bank, layer=layer, k=k, level=level)
    if len(kept) != bank.n:
        return {"ERROR": f"{bank.n - len(kept)} paths too short — split not identical; "
                         f"refusing a non-comparable number"}
    res = ladder(X, y, topic, C, nclass, device, label)
    return {"P": int(X.shape[1]), "n": int(X.shape[0]), "names_head": names[:2], **res}


def null_cell(bank: PathBank, y, topic, C, *, layer: int, k: int, level: Literal[1, 2],
              nclass: int, device: str, label: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in NULL_SEEDS:
        X, _n, kept = signature_matrix(bank, layer=layer, k=k, level=level, permutation_seed=s)
        if len(kept) != bank.n:
            out.append({"seed": s, "ERROR": "path drop"})
            continue
        out.append({"seed": s, **ladder(X, y, topic, C, nclass, device, f"{label} NULL s{s}")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bank-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--model", default="8b")
    ap.add_argument("--ks", default="4,8")
    ap.add_argument("--variant", default="pc", choices=["pc", "nopc"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-binaries", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ks = tuple(int(x) for x in args.ks.split(",") if x.strip())
    corpus = Corpus(args.bank_dir, args.layer, max(ks), variant=args.variant)

    result: dict[str, Any] = {
        "cell": "E1 — 8B mode stake on path-signature features",
        "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED, not quotable",
        "model": args.model, "site": f"L{args.layer}",
        "projection": {"basis": "pca_model_corrected.pkl per-layer", "variant": args.variant,
                       "ks": list(ks)},
        "augmentation": "t/(T-1) normalised — pacing, not duration",
        "folds": "GroupKFold(5) by topic_idx x 3 seeds, per-fold length-residualised "
                 "on [prompt_len, gen_len]",
        "reference_of_record": {"8b_5way_hand_suite": 0.925, "note": "v3, n~900, same folds law"},
        "cells": {}, "nulls": {},
    }

    t0 = time.time()
    for tag, modeset in (("5way_hard", set(HARD)), ("8way", set(EIGHT_WAY))):
        bank, y, topic, C, classes = corpus.subset(modeset)
        nclass = len(classes)
        logger.info(f"=== {tag}: n={bank.n}, {nclass} classes {classes}, "
                    f"{len(set(topic.tolist()))} topics ===")
        for k in ks:
            for level in (1, 2):
                key = f"{tag}_k{k}_lvl{level}"
                lab = f"{tag} k={k} lvl1{'+2' if level == 2 else ''}"
                result["cells"][key] = {
                    "classes": classes, "n_topics": len(set(topic.tolist())),
                    "chance": round(1.0 / nclass, 4),
                    **cell(bank, y, topic, C, layer=args.layer, k=k, level=level,
                           nclass=nclass, device=args.device, label=lab),
                }
                result["nulls"][key] = null_cell(bank, y, topic, C, layer=args.layer, k=k,
                                                 level=level, nclass=nclass,
                                                 device=args.device, label=lab)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))

    if not args.skip_binaries:
        logger.info("=== hard binary pairs (headroom leg) ===")
        result["binaries"] = {}
        result["binary_nulls"] = {}
        for a, b in itertools.combinations(sorted(HARD), 2):
            bank, y, topic, C, classes = corpus.subset({a, b})
            pk = f"{a}_vs_{b}"
            result["binaries"][pk] = {}
            for k in ks:
                for level in (1, 2):
                    result["binaries"][pk][f"k{k}_lvl{level}"] = cell(
                        bank, y, topic, C, layer=args.layer, k=k, level=level,
                        nclass=2, device=args.device, label=f"{pk} k={k} lvl{level}")
            # null only on the largest k (cost control; the null is a machinery check per cell)
            result["binary_nulls"][pk] = {
                f"k{max(ks)}_lvl{lv}": null_cell(bank, y, topic, C, layer=args.layer,
                                                 k=max(ks), level=lv, nclass=2,
                                                 device=args.device, label=f"{pk} lvl{lv}")
                for lv in (1, 2)
            }
            args.out.write_text(json.dumps(result, indent=1))

    result["elapsed_s"] = round(time.time() - t0, 1)
    args.out.write_text(json.dumps(result, indent=1))
    logger.info(f"E1 banked (first-read pending) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
