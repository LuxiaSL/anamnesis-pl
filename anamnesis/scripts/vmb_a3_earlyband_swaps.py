"""Early-band adjudication from banked swaps — addendum 2026-07-12e item 3.

Pre-specified BEFORE running (prereg commit 9d92313): A3 found attention|early >
attention|mid as top carrier band on both anchors (prereg §2c predicted mid).
Hypothesis: the early-band excess is PROMPT-REGION READING, not execution.

Test: per-band 5-way LDA trained on the pure-mode corpora (length-residualized,
Stage-0 z space — the A3 record configuration), applied to the banked A2 swap
corpora (instructed-X, executed-Y; X ≠ Y). Readout per (band × swap × model):
fraction of swap gens classified as instructed-X vs executed-Y (other = rest).

Frozen prediction: attention|early votes instructed-X at a materially higher
rate than attention|mid; attention|mid stays execution-dominant.
Named alternative: early votes executed-Y like mid ⇒ mid-band prediction wrong.

Usage:
    PYTHONPATH=. python -m anamnesis.scripts.vmb_a3_earlyband_swaps \
        --battery-root ../outputs/battery --out-dir ../outputs/battery/arms/A3
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from anamnesis.analysis.battery.deltas import ConditionCorpus, build_cells, load_floor_scale
from anamnesis.analysis.battery.manifest import MODEL_META

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODES = ["linear", "analogical", "socratic", "contrastive", "dialectical"]
SWAPS = [("socratic_to_linear", "socratic", "linear"),
         ("dialectical_to_contrastive", "dialectical", "contrastive"),
         ("analogical_to_linear", "analogical", "linear")]
BANDS = ["source_band:attention|early", "source_band:attention|mid",
         "source_band:attention|late"]


def residualize(X: np.ndarray, lengths: np.ndarray,
                beta: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    L = np.column_stack([np.ones(len(lengths)), lengths.astype(np.float64)])
    if beta is None:
        beta, *_ = np.linalg.lstsq(L, X.astype(np.float64), rcond=None)
    return (X - L @ beta).astype(np.float32), beta


def load(model: str, cond: str, root: Path, med, scale):
    d = root / f"vmb_a2_{model}_{cond}"
    cc = ConditionCorpus(d / "signatures_v3", d / "metadata.json", med, scale,
                         f"{model}-{cond}")
    md = json.loads((d / "metadata.json").read_text())
    gens = sorted(md["generations"], key=lambda g: g["generation_id"])
    lengths = np.array([g["num_generated_tokens"] for g in gens], dtype=np.float64)
    return cc.Z, lengths, cc.feature_names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--models", default="3b,8b")
    args = ap.parse_args()

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    results = {"analysis": "earlyband_swap_adjudication",
               "prereg": "addendum 2026-07-12e item 3 (frozen at commit 9d92313 "
                         "BEFORE this script ran)", "models": {}}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        meta = MODEL_META[model]
        floor_dir = args.battery_root / meta.stage0_dir
        med, scale = load_floor_scale(floor_dir / "signatures_v3")

        Zs, Ls, names = [], [], None
        y = []
        for mode in MODES:
            Z, lens, names = load(model, f"pure_{mode}", args.battery_root, med, scale)
            Zs.append(Z); Ls.append(lens); y += [mode] * Z.shape[0]
        Ztr = np.vstack(Zs)
        Ltr = np.concatenate(Ls)
        y = np.array(y)
        Ztr, beta = residualize(Ztr, Ltr)   # same convention as the A3 record
        cells = build_cells(names, meta.n_layers)

        model_out = {}
        for band in BANDS:
            mask = cells[band]
            clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            clf.fit(Ztr[:, mask], y)
            per_swap = {}
            for lbl, instructed, executed in SWAPS:
                Zsw, lens_sw, _ = load(model, f"swap_{lbl}", args.battery_root, med, scale)
                Zsw, _ = residualize(Zsw, lens_sw, beta)   # TRAIN-fit residualization
                pred = clf.predict(Zsw[:, mask])
                n = len(pred)
                per_swap[lbl] = {
                    "frac_instructed": float(np.mean(pred == instructed)),
                    "frac_executed": float(np.mean(pred == executed)),
                    "frac_other": float(np.mean((pred != instructed) & (pred != executed))),
                    "n": n,
                    "stamp": {"n": n, "M": model,
                              "law": "addendum-12e item 3; pure-trained per-band LDA, "
                                     "length-resid (train-fit beta)",
                              "floor_type": "stochastic"},
                }
            band_short = band.split(":")[1]
            inst = np.mean([per_swap[l]["frac_instructed"] for l, _, _ in SWAPS])
            exe = np.mean([per_swap[l]["frac_executed"] for l, _, _ in SWAPS])
            model_out[band_short] = {"per_swap": per_swap,
                                     "mean_frac_instructed": round(float(inst), 4),
                                     "mean_frac_executed": round(float(exe), 4)}
            logger.info(f"{model} {band_short}: instructed {inst:.3f} vs executed {exe:.3f}")
        results["models"][model] = model_out

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "a3_earlyband_swap_adjudication.json"
    out.write_text(json.dumps(results, indent=1))
    logger.info(f"→ {out}")


if __name__ == "__main__":
    main()
