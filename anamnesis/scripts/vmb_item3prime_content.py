"""Item 3′ — the discriminating content test (ADDENDUM 14m item 3′; the STANDING
composition rule's first outing).

The retired item-3 measured a CENSUS: b7 cells carry 2-of-4 template modes
{expository, explanatory} while the α=0 rider pool carries all 4, so a TF-IDF classifier
separated steered-from-rider by MODE, not by steering (AMENDMENT 2: a zero-injection
placebo reproduced the "leak"). Item 3′ repairs the rung:

  (a) census columns — modes_steered / modes_rider beside every AUC;
  (b) the MODE-MATCHED rider (rider restricted to the steered cell's modes) is the ROW
      OF RECORD for a composition-imbalanced arm;
  (c) a MANDATORY PLACEBO column — disjoint unsteered-vs-unsteered under the SAME
      composition (pseudo-treatment drawn from the matched rider, mirroring the steered
      cell's size + mode balance; the rest as pseudo-rider). A content rung whose placebo
      scores where its treatment scores has NOT measured steering;
  (d) V7 + Rband1-3 × α{.03,.1} on the matched rung = the discriminating test of any
      V7-SPECIFIC text trace (V7's matched AUC vs the Rband nulls' matched AUC, both
      read against their own placebo floor).

Filed P (repaired, 14m item 3′): V7 mode-matched stays < .60 at both doses (no
meaningful trace) = .70. Also calibrates the watch-listed Rband-vs-Rband .61-.64 anomaly
(that anomaly is the FULL-rider census; the placebo is its explanation).

CPU-only, text from banked metadata. First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

Gen = dict


def _gens(run_dir: Path, cell: str) -> list[Gen]:
    md = json.loads((run_dir / cell / "metadata.json").read_text())
    return md["generations"] if "generations" in md else md


def _txt(g: Gen) -> str:
    return g.get("generated_text", "")


def _topic(g: Gen) -> str:
    return str(g.get("topic", g.get("topic_idx", "")))


def _mode(g: Gen) -> str:
    return str(g.get("mode", g.get("mode_idx", "")))


def _auc(pos: list[Gen], neg: list[Gen]) -> float | None:
    """GroupKFold-by-topic TF-IDF AUC separating pos from neg. None if a fold is single-class."""
    texts = [_txt(g) for g in pos] + [_txt(g) for g in neg]
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    grp = np.array([_topic(g) for g in pos] + [_topic(g) for g in neg])
    n_splits = min(5, len(set(grp)))
    if n_splits < 2:
        return None
    X = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=2).fit_transform(texts)
    scores = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, grp):
        if len(set(y[tr])) < 2:
            return None
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(X[tr], y[tr])
        scores[te] = clf.decision_function(X[te])
    return float(roc_auc_score(y, scores))


def _parse(name: str) -> tuple[str, int, str]:
    m = re.match(r"^(?P<vec>[A-Za-z][A-Za-z0-9]*?)(?:_L(?P<s>\d+))?_a(?P<a>[\d.]+)$", name)
    if not m:
        raise SystemExit(f"unparseable cell name {name}")
    return m.group("vec"), int(m.group("s") or 14), m.group("a")


def _placebo(matched_rider: list[Gen], steered: list[Gen], n_draws: int, seed0: int) -> dict:
    """Disjoint unsteered-vs-unsteered under the steered cell's exact (mode,topic) composition.

    Pseudo-treatment mirrors the steered cell: for every (topic, mode) cell in `steered`
    draw the same count from `matched_rider`; the remaining matched-rider gens are the
    pseudo-rider. Averaged over `n_draws` deterministic draws → the composition-matched
    null floor of TF-IDF separability at these sizes.
    """
    want: Counter = Counter((_topic(g), _mode(g)) for g in steered)
    by_key: dict[tuple[str, str], list[int]] = {}
    for i, g in enumerate(matched_rider):
        by_key.setdefault((_topic(g), _mode(g)), []).append(i)
    aucs: list[float] = []
    for d in range(n_draws):
        rng = np.random.default_rng(seed0 + d)
        pseudo_t_idx: list[int] = []
        for key, k in want.items():
            pool = by_key.get(key, [])
            if len(pool) < k:
                # not enough same-composition rider to mirror this cell; skip the draw
                pseudo_t_idx = []
                break
            pseudo_t_idx += list(rng.choice(pool, size=k, replace=False))
        if not pseudo_t_idx:
            continue
        t_set = set(pseudo_t_idx)
        pt = [matched_rider[i] for i in pseudo_t_idx]
        pr = [g for i, g in enumerate(matched_rider) if i not in t_set]
        a = _auc(pt, pr)
        if a is not None:
            aucs.append(a)
    if not aucs:
        return {"placebo_auc_matched": None, "placebo_n_draws": 0, "placebo_std": None}
    return {"placebo_auc_matched": round(float(np.mean(aucs)), 4),
            "placebo_n_draws": len(aucs),
            "placebo_std": round(float(np.std(aucs)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-run-dir", type=Path, required=True, help="vmb_b7_3b (V7 + Rband cells)")
    ap.add_argument("--main-run-dir", type=Path, required=True, help="vmb_a5_3b (α=0 riders)")
    ap.add_argument("--cells", nargs="+", default=None,
                    help="explicit cells; default V7+Rband1-3 × {.03,.1} at L14")
    ap.add_argument("--placebo-draws", type=int, default=8)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    # full α=0 rider pool (4-mode)
    rider: list[Gen] = []
    for d in sorted(args.main_run_dir.iterdir()):
        if d.name.endswith("_a0.0") and (d / "metadata.json").exists():
            rider += _gens(args.main_run_dir, d.name)
    rider_modes = Counter(_mode(g) for g in rider)

    if args.cells:
        grid = args.cells
    else:
        grid = [f"{v}_L14_a{a}" for v in ("V7", "Rband1", "Rband2", "Rband3") for a in ("0.03", "0.1")]

    rows = []
    for ci, cell in enumerate(grid):
        vec, s, a = _parse(cell)
        steered = _gens(args.arm_run_dir, cell)
        steered_modes = Counter(_mode(g) for g in steered)
        matched_rider = [g for g in rider if _mode(g) in steered_modes]

        auc_full = _auc(steered, rider)                          # census artifact (4-mode rider)
        auc_matched = _auc(steered, matched_rider)               # ROW OF RECORD (mode-matched)
        pl = _placebo(matched_rider, steered, args.placebo_draws, seed0=1000 * (ci + 1))

        excess = (None if (auc_matched is None or pl["placebo_auc_matched"] is None)
                  else round(auc_matched - pl["placebo_auc_matched"], 4))
        rows.append({
            "cell": cell, "vector": vec, "site": s, "alpha_frac": float(a),
            "is_null": vec.upper().startswith("RBAND"),
            "n_steered": len(steered), "n_matched_rider": len(matched_rider), "n_full_rider": len(rider),
            "modes_steered": dict(steered_modes), "modes_rider": dict(rider_modes),
            "content_auc_vs_full_rider": round(auc_full, 4) if auc_full is not None else None,
            "content_auc_vs_matched_rider": round(auc_matched, 4) if auc_matched is not None else None,
            **pl,
            "matched_minus_placebo": excess,
            "flag_lt_.60_matched": bool(auc_matched is not None and auc_matched < 0.60),
        })
        print(f"  {cell:16} full={rows[-1]['content_auc_vs_full_rider']} "
              f"MATCHED={rows[-1]['content_auc_vs_matched_rider']} "
              f"placebo={pl['placebo_auc_matched']} excess={excess}")

    # V7-specificity: V7 matched vs mean(Rband matched) at each dose
    def mean_null_matched(af: float) -> float | None:
        v = [r["content_auc_vs_matched_rider"] for r in rows
             if r["is_null"] and r["alpha_frac"] == af and r["content_auc_vs_matched_rider"] is not None]
        return float(np.mean(v)) if v else None

    for r in rows:
        if r["is_null"]:
            continue
        base = mean_null_matched(r["alpha_frac"])
        r["v7_matched_over_rband_matched"] = (
            round(r["content_auc_vs_matched_rider"] / base, 3)
            if base and r["content_auc_vs_matched_rider"] is not None else None)
        r["v7_matched_minus_rband_matched"] = (
            round(r["content_auc_vs_matched_rider"] - base, 4)
            if base is not None and r["content_auc_vs_matched_rider"] is not None else None)

    out = {
        "model": "3b", "arm": "item 3′ — discriminating content test (14m item 3′, composition-matched)",
        "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop",
        "law": "TF-IDF(1-2gram)+GroupKFold-by-topic AUC. ROW OF RECORD = steered vs "
               "MODE-MATCHED rider (rider restricted to the steered cell's modes). PLACEBO = "
               "disjoint unsteered-vs-unsteered mirroring the steered (topic,mode) composition "
               f"(mean of {args.placebo_draws} draws). full-rider column = the census artifact "
               "(4-mode) kept for the collapse. matched_minus_placebo = the steering-specific "
               "excess above the composition floor.",
        "filed_P": {"V7_matched_lt_.60_both_doses": 0.70},
        "readings": "V7 matched < .60 both doses = no meaningful trace (P=.70). V7-specificity = "
                    "V7 matched vs mean(Rband matched) at each dose, each above its own placebo.",
        "rows": rows,
    }
    args.out_json.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
