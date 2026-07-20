"""WH-3 second ruler — the census content-TFIDF detector applied to steered cells.

The census's content ruler (vmb_arm_a3_analyze content_tfidf: TfidfVectorizer(max_features=5000,
ngram_range=(1,2)) + LogisticRegression(max_iter=2000), 5-way over the pure-mode corpora) is the
instrument that certified M6-linear text-INVISIBLE (content 0.081). This applies the SAME ruler
to the whitened-dir0 steered cells: is the invisible mode raised to content-visibility by its
whitened direction? (WH-3's of-record P is the marker q-rate; this is the mandated second column.)

Train on all five pure corpora (no OOF needed — the cells are out-of-domain w.r.t. training),
score every cell of the run: fraction classified as each mode + mean P(target).

Usage:
  python -m anamnesis.scripts.vmb_wh3_content_tfidf --runs-root <runs> \
      --run-dir <steered run> --model-tag dsv2-lite --target-mode linear \
      --other-mode socratic --out-json <out>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anamnesis.analysis.battery.text_decode import maybe_decode

MODES = ("analogical", "contrastive", "dialectical", "linear", "socratic")


def corpus_texts(run_dir: Path) -> list[str]:
    meta = json.loads((run_dir / "metadata.json").read_text())
    gens = meta["generations"] if isinstance(meta, dict) and "generations" in meta else meta
    return [maybe_decode(g["generated_text"]) for g in gens]


def cell_texts(cell_dir: Path) -> list[str]:
    md = cell_dir / "gen_records" / "metadata.json"
    if md.exists():
        return corpus_texts(cell_dir / "gen_records")
    out = []
    for f in sorted((cell_dir / "gen_records").glob("gen_*.json")):
        out.append(maybe_decode(json.loads(f.read_text())["generated_text"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--model-tag", default="dsv2-lite")
    ap.add_argument("--target-mode", default="linear")
    ap.add_argument("--other-mode", default="socratic",
                    help="the pair's negative pole (reported beside)")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    X, y = [], []
    for mode in MODES:
        d = args.runs_root / f"vmb_a2_{args.model_tag}_pure_{mode}"
        t = corpus_texts(d)
        X.extend(t)
        y.extend([mode] * len(t))
    clf = make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
                        LogisticRegression(max_iter=2000))
    clf.fit(X, np.array(y))
    classes = list(clf.classes_)
    ti = classes.index(args.target_mode)

    cells = {}
    for d in sorted(args.run_dir.iterdir()):
        if not (d / "gen_records").exists():
            continue
        texts = cell_texts(d)
        if not texts:
            continue
        pred = clf.predict(texts)
        prob = clf.predict_proba(texts)
        cells[d.name] = {
            "n": len(texts),
            "frac_pred": {m: round(float(np.mean(pred == m)), 4) for m in MODES},
            f"mean_P_{args.target_mode}": round(float(prob[:, ti].mean()), 4),
        }

    out = {
        "arm": "WH-3 content-TFIDF column (the census content ruler on steered cells)",
        "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop",
        "ruler": "TfidfVectorizer(max_features=5000, ngram_range=(1,2)) + LogisticRegression"
                 "(max_iter=2000), 5-way, trained on the five pure-mode corpora (same form as "
                 "vmb_arm_a3_analyze content_tfidf — the instrument that certified M6-linear "
                 "content 0.081)",
        "target_mode": args.target_mode, "other_mode": args.other_mode,
        "train_n": len(X),
        "cells": cells,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=1))
    for name, c in cells.items():
        print(f"{name}: P({args.target_mode})={c[f'mean_P_{args.target_mode}']} "
              f"frac({args.target_mode})={c['frac_pred'][args.target_mode]} "
              f"frac({args.other_mode})={c['frac_pred'][args.other_mode]}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
