"""A5 post-verdict gates (2026-07-13; the ad-hoc analyses behind the package's
"Post-verdict gates" section, committed per the scratchpad-death rule):

  --gate excess   : pairwise deformation-separability AUC on NON-TRIVIAL surfaces
                    (attention/gate/keys/qk — excludes C§1 injection self-read +
                    logit conditioning) for any two cells, LDA GroupKFold-by-topic.
                    Identity quotables use TRAIT-pair AUC in EXCESS of R-pair AUC at
                    the pre-saturation dose (alpha=0.03); at alpha>=0.1 ANY pair
                    saturates ("high-gain router" substrate fact).
  --gate markers  : analogy-marker rate per 1000 words across cells (gates the
                    semantics gloss; the dose-resolved register curve).
  --gate percell  : per-feature_map-cell seed-floor ratios of matched-token
                    deformations (carrying-subspace localization; inverse census).

    python -m anamnesis.scripts.vmb_a5_postverdict_gates --battery-root ../outputs/battery \
        --gate excess --cells V1_L14_a0.03,V3_L14_L14_a0.03 --model 3b
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from anamnesis.analysis.battery.deltas import ConditionCorpus, load_floor_scale, within_condition_deltas
from anamnesis.analysis.battery.floors import build_cells, load_signature_matrix
from anamnesis.analysis.battery.manifest import MODEL_META

MARKERS = [r'\blike a\b', r'\blike an\b', r'\bas if\b', r'\bimagine\b', r'\bthink of\b',
           r'\bmuch like\b', r'\bakin to\b', r'\bmetaphor', r'\banalog', r'\bjust as\b',
           r'\bsimilar to\b', r'\bas though\b']


def _load_cell(root: Path, run: str, cell: str, med, scale):
    X, names, gids = load_signature_matrix(root / run / cell / "signatures_v3")
    md = json.loads((root / run / cell / "metadata.json").read_text())
    gmap = {g["generation_id"]: g["topic_idx"] for g in md["generations"]}
    return (X - med) / scale, np.array([gmap[g] for g in gids]), names


def gate_excess(args, root, med, scale, mm):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    a, b = args.cells.split(",")
    Za, ga, names = _load_cell(root, args.run, a, med, scale)
    Zb, gb, _ = _load_cell(root, args.run, b, med, scale)
    cells = build_cells(names, mm.n_layers)
    nt = (cells["source:attention"] | cells["source:gate"] | cells["source:keys"]
          | cells.get("source:qk", np.zeros(len(names), bool)))
    X = np.vstack([Za[:, nt], Zb[:, nt]])
    y = np.r_[np.zeros(len(Za)), np.ones(len(Zb))]
    g = np.r_[ga, gb]
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        s[te] = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.1).fit(
            X[tr], y[tr]).decision_function(X[te])
    print(f"{a} vs {b} NONTRIVIAL AUC = {roc_auc_score(y, s):.3f} "
          f"({int(nt.sum())} features)")


def gate_markers(args, root, *_):
    for cell in args.cells.split(","):
        run = args.run
        p = root / run / cell / "metadata.json"
        if not p.exists():
            p = root / "vmb_a5_explore" / cell / "metadata.json"
        md = json.loads(p.read_text())
        rates = []
        for g in md["generations"]:
            t = g["generated_text"].lower()
            w = max(len(t.split()), 1)
            rates.append(sum(len(re.findall(m, t)) for m in MARKERS) / w * 1000)
        print(f"{cell}: {np.mean(rates):.2f} markers/1000w (n={len(rates)})")


def gate_percell(args, root, med, scale, mm):
    stage0 = root / mm.stage0_dir
    s0X, names, s0g = load_signature_matrix(stage0 / "signatures_v3")
    s0Z = (s0X - med) / scale
    s0map = {g: i for i, g in enumerate(s0g)}
    cells = build_cells(names, mm.n_layers)
    s0c = ConditionCorpus(stage0 / "signatures_v3", stage0 / "metadata.json", med, scale, "s0")
    floor = {c: max(float(np.median(v)), 1e-12) for c, v in within_condition_deltas(s0c, cells).items()}
    hdr = ["whole_vector", "source:attention", "source:gate", "source:keys",
           "source:residual", "source:output"]
    print(f"{'cell':20s} " + " ".join(f"{h.split(':')[-1][:7]:>8s}" for h in hdr))
    for mt in args.cells.split(","):
        X, _, gids = load_signature_matrix(root / args.run / mt / "signatures_v3")
        Z = (X - med) / scale
        D = np.stack([np.abs(Z[i] - s0Z[s0map[g]]) for i, g in enumerate(gids) if g in s0map])
        row = [float(np.median(D[:, cells[h]].mean(axis=1)) / floor[h]) for h in hdr]
        print(f"{mt:20s} " + " ".join(f"{v:8.3f}" for v in row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--model", default="3b")
    ap.add_argument("--gate", choices=["excess", "markers", "percell"], required=True)
    ap.add_argument("--cells", required=True, help="comma-separated cell dir names")
    ap.add_argument("--run", default=None,
                    help="run dir under battery-root (default vmb_a5_<model> for "
                         "excess/markers, vmb_a5_mt_<model> for percell)")
    args = ap.parse_args()
    mm = MODEL_META[args.model]
    root = args.battery_root
    if args.run is None:
        args.run = f"vmb_a5_mt_{args.model}" if args.gate == "percell" else f"vmb_a5_{args.model}"
    med, scale = load_floor_scale(root / mm.stage0_dir / "signatures_v3")
    {"excess": gate_excess, "markers": gate_markers, "percell": gate_percell}[args.gate](
        args, root, med, scale, mm)


if __name__ == "__main__":
    main()
