"""Sub-perceptual census — addendum 2026-07-12f §2, first implementation.

A row is a class member when its INTERNALS rung materially exceeds BOTH its
CONTENT rung and its LIKELIHOOD rung (the 12c three-rung hierarchy). The rungs
are the tested detectors of record: content = trained TF-IDF (GroupKFold);
likelihood = banked surprisal probe; internals = signature classifier. The
zero-shot JUDGE is reported as its own column — judge-based membership quotes
additionally require 12f hardening (2AFC contrast-judging + a second judge
family) before a row is quoted as class evidence.

Declared implementation ruling (dated 2026-07-12, changeable only by addendum):
  gap = internals − max(content, likelihood)
  MEMBER      gap ≥ 0.10
  BORDERLINE  0.03 ≤ gap < 0.10
  EXCLUDED    gap < 0.03
Units: AUC for binary rows (A1), per-mode recall for k-way rows (A3).

The census re-runs at each new scale point (per model); the CLASS scale-trend
object is the set's size + gap magnitudes across 3B → 8B → 27B (+ Qwen
cross-family). n=1 rows license no class conclusions (12f §1).

Usage:
    PYTHONPATH=. python -m anamnesis.scripts.vmb_subperceptual_census \
        --arms-root ../outputs/battery/arms --out-dir ../outputs/battery/census
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MEMBER_BAR = 0.10
BORDERLINE_BAR = 0.03
MODES = ["linear", "analogical", "socratic", "contrastive", "dialectical"]


def classify(gap: float) -> str:
    if gap >= MEMBER_BAR:
        return "MEMBER"
    if gap >= BORDERLINE_BAR:
        return "BORDERLINE"
    return "EXCLUDED"


def a1_rows(arms_root: Path) -> list[dict]:
    rows = []
    for rec_dir in ("A1", "A1_m3m4"):
        p = arms_root / rec_dir / "a1_results.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        for model, m in r["models"].items():
            d = m["dissociation"]
            content = float(d["tfidf_groupkfold_auc_t03_vs_t09"])
            lik = float(d["likelihood_mean_surprise_auc"])
            internals = float(d["signature_output_groupkfold_auc"])
            gap = internals - max(content, lik)
            rows.append({
                "row": "A1:temperature(t03|t09)", "model": model,
                "units": "AUC",
                "content": content, "likelihood": lik, "internals": internals,
                "judge": None,
                "gap": round(gap, 4), "status": classify(gap),
                "binding_rung": "likelihood" if lik >= content else "content",
                "hardening": "exempt (likelihood-rung binding; no judge involved)"
                             if lik >= content else "n/a",
                "n": d["n"],
                "source_record": f"arms/{rec_dir}/a1_results.json",
            })
    return rows


def load_2afc(arms_root: Path) -> dict:
    """12f hardening (a) results, if banked: model -> 2AFC accuracy."""
    p = arms_root / "A3" / "judge" / "socratic_2afc_fable.json"
    if not p.exists():
        return {}
    r = json.loads(p.read_text())
    return {m: v["acc_2afc"] for m, v in r["models"].items()}


def a3_rows(arms_root: Path) -> list[dict]:
    p = arms_root / "A3" / "a3_results.json"
    if not p.exists():
        return []
    r = json.loads(p.read_text())
    afc = load_2afc(arms_root)
    rows = []
    for model, m in r["models"].items():
        h = m["hierarchy"]
        judge = m.get("judge", {}).get("per_mode", {})
        for mode in MODES:
            tfidf = float(h["content_tfidf"]["per_mode_recall"][mode])
            lik = float(h["likelihood_surprise"]["per_mode_recall"][mode])
            internals = float(h["internals_rf"]["per_mode_recall"][mode])
            jr = judge.get(mode, {}).get("judge_recall")
            # The judge IS a content-class detector (12c: "token-KL, TF-IDF,
            # judge hooks"). The content rung = max over available content-class
            # detectors — otherwise a mode that is merely hard for TF-IDF but
            # trivially judge-visible (linear: TF-IDF .44, judge .99) would be
            # a spurious "member". First census run caught exactly that.
            content = max(tfidf, jr) if jr is not None else tfidf
            gap = internals - max(content, lik)
            judge_gap = (internals - jr) if jr is not None else None
            rows.append({
                "row": f"A3:mode:{mode}", "model": model,
                "units": "per-mode recall (5-way)",
                "content": content, "content_tfidf": tfidf,
                "likelihood": lik, "internals": internals,
                "judge": jr,
                "gap": round(gap, 4), "status": classify(gap),
                "judge_gap": round(judge_gap, 4) if judge_gap is not None else None,
                "hardening": (
                    (f"FAILED hardening (a): 2AFC {afc[model]:.3f} ≫ chance — the "
                     "blind-k-way judge-gap is a contrast artifact; judge-gap NOT "
                     "quotable as class evidence (membership rests on the "
                     "trained-detector rung only)"
                     if mode == "socratic" and model in afc and afc[model] >= 0.65
                     else "SURVIVES 2AFC (a); second-family (b) pending"
                     if mode == "socratic" and model in afc
                     else "PENDING 2AFC + second judge family (12f) — required "
                          "before any JUDGE-GAP quote as class evidence")
                    if judge_gap is not None and judge_gap >= MEMBER_BAR
                    else "n/a (judge-gap below member bar or no judge)"),
                "n": 160,
                "source_record": "arms/A3/a3_results.json",
            })
    return rows


PENDING_AND_APPENDIX = [
    {"row": "A2:cell-ii:unexecuted-instruction-carriage", "status": "PENDING",
     "note": "embargoed behind the length-matched prefix control (Wave-2); "
             "enters the census when the control lands"},
    {"row": "A4/exp11:P3 eviction-kind vs token-KL", "status": "APPENDIX(pre-battery)",
     "note": "banked at n=12, kv-rotation exp11 (prereg p=0.0029); re-enters as a "
             "battery row when A4 runs; likelihood-rung analog = token-KL (exempt "
             "from judge hardening)"},
    {"row": "pre-battery:Run-1 uncertain/confident", "status": "APPENDIX(pre-battery)",
     "note": "phase-0 era; pointer only — no battery-grade rungs"},
    {"row": "pre-battery:wolf (subliminal)", "status": "APPENDIX(pre-battery)",
     "note": "subliminal_anamnesis repo; behavioral metric was the false negative — "
             "the class's founding exemplar; pointer only"},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    rows = a1_rows(args.arms_root) + a3_rows(args.arms_root)
    census = {
        "prereg": "addendum 2026-07-12f §2; bar: MEMBER >= 0.10, BORDERLINE >= 0.03 "
                  "(declared implementation ruling 2026-07-12; addendum-only changes)",
        "definition": "gap = internals - max(content, likelihood); 12c rungs of record",
        "rows": rows,
        "pending_and_appendix": PENDING_AND_APPENDIX,
        "class_object": {
            m: {"members": [r["row"] for r in rows if r["model"] == m and r["status"] == "MEMBER"],
                "borderline": [r["row"] for r in rows if r["model"] == m and r["status"] == "BORDERLINE"]}
            for m in sorted({r["model"] for r in rows})
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "subperceptual_census.json").write_text(json.dumps(census, indent=1))

    lines = ["# Sub-perceptual census (12f §2)", "",
             "gap = internals − max(content, likelihood); MEMBER ≥ 0.10, BORDERLINE ≥ 0.03.",
             "content rung = MAX over content-class detectors (trained TF-IDF, zero-shot judge);",
             "judge-GAP (internals − judge) is a separate quotable, hardening-gated per 12f.", "",
             "| row | model | content | likelihood | internals | judge | gap | status |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (-x["gap"])):
        j = f"{r['judge']:.3f}" if r.get("judge") is not None else "—"
        lines.append(f"| {r['row']} | {r['model']} | {r['content']:.3f} | "
                     f"{r['likelihood']:.3f} | {r['internals']:.3f} | {j} | "
                     f"{r['gap']:+.3f} | {r['status']} |")
    lines += ["", "## Pending / appendix"]
    for e in PENDING_AND_APPENDIX:
        lines.append(f"- **{e['row']}** [{e['status']}] — {e['note']}")
    (args.out_dir / "subperceptual_census.md").write_text("\n".join(lines))
    logger.info(f"census → {args.out_dir} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
