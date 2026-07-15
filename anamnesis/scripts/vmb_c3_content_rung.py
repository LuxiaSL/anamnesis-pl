"""C3 certifying (c) — CONTENT rung (PREFLIGHT §4 (c); the A1 detector-hierarchy content leg).

Can a TEXT-only observer tell V_temp-steered generation from unsteered (rider)? TF-IDF +
GroupKFold-by-topic AUC. A1's content rung sat at chance for temperature (.51–.54); the paradigm
point is that V_temp reproduces it — content-BLIND (AUC ≈ .5) while the likelihood rung
(base-model surprisal, the NLL leg of the replay) shifts. CPU-only, text from banked metadata.
First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


def _texts(run_dir: Path, cell: str):
    md = json.loads((run_dir / cell / "metadata.json").read_text())
    gens = md["generations"] if "generations" in md else md
    return [(g.get("generated_text", ""), str(g.get("topic", g.get("topic_idx", "")))) for g in gens]


def _auc(pos, neg):
    """GroupKFold-by-topic TF-IDF AUC separating pos from neg text lists of (text, topic)."""
    texts = [t for t, _ in pos] + [t for t, _ in neg]
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    grp = np.array([g for _, g in pos] + [g for _, g in neg])
    X = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=2).fit_transform(texts)
    scores = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, grp):
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(X[tr], y[tr])
        scores[te] = clf.decision_function(X[te])
    return float(roc_auc_score(y, scores))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3-run-dir", type=Path, required=True)
    ap.add_argument("--main-run-dir", type=Path, required=True, help="vmb_a5_3b (α=0 riders)")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    rider = []
    for d in sorted(args.main_run_dir.iterdir()):
        if d.name.endswith("_a0.0") and (d / "metadata.json").exists():
            rider += _texts(args.main_run_dir, d.name)

    rows = []
    for s in (14, 21):
        for a in ("0.03", "0.1"):
            vt = _texts(args.c3_run_dir, f"Vtemp_L{s}_a{a}")
            rc = _texts(args.c3_run_dir, f"Rc1_L{s}_a{a}")
            rows.append({"cell": f"Vtemp_L{s}_a{a}", "site": s, "alpha_frac": float(a),
                         "content_auc_vs_rider": round(_auc(vt, rider), 4),
                         "content_auc_vs_Rc": round(_auc(vt, rc), 4),
                         "n_vtemp": len(vt), "n_rider": len(rider)})
            print(f"  Vtemp_L{s}_a{a}: content AUC vs rider={rows[-1]['content_auc_vs_rider']:.3f} "
                  f"vs Rc={rows[-1]['content_auc_vs_Rc']:.3f}")

    out = {"model": "3b", "arm": "C3 certifying (c) content rung",
           "STATUS": "FIRST_READ_PENDING (C§8)",
           "law": "TF-IDF(1-2gram) + GroupKFold-by-topic AUC, V_temp-steered vs unsteered rider "
                  "(and vs Rc-steered); ≈.5 = content-BLIND (reproduces A1's temperature content rung)",
           "rows": rows}
    args.out_json.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
