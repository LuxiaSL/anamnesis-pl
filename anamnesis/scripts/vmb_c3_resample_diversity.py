"""C3 certifying (f) — resample-diversity readout (PREFLIGHT §4 (f); 14f completeness item).

For each (topic, stratum) group of k=8 same-prompt stochastic resamples: the CROSS-sample
distinct-4-gram rate (unique 4-grams / total 4-grams pooled over the 8 texts) = how much the
model's samples vary = the behavioral analog of the entropy rise. Higher under V_temp (hotter
sampling state) if the orphaned lever installs real diversity. Guard column: MEAN WITHIN-sample
distinct-4-gram (14f named degenerate-repetition as the risk — if V_temp raises cross-sample
spread only by making each sample MORE repetitive, within-sample distinct-4-gram would FALL).

(f) = Vtemp cross-sample distinct-4-gram ÷ mean(Rc) at matched (site,α), dose-ordered, > null.
14f: rises dose-ordered AND V_temp-specifically at α≤.1 (P=0.85). Text-only, CPU.
First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _grams(text: str, n: int = 4) -> list[tuple]:
    w = text.split()
    return [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]


def _cell_diversity(run_dir: Path, cell: str) -> dict | None:
    md_path = run_dir / cell / "metadata.json"
    if not md_path.exists():
        return None
    md = json.loads(md_path.read_text())
    gens = md["generations"] if "generations" in md else md
    groups: dict[tuple, list[str]] = defaultdict(list)
    for g in gens:
        groups[(g.get("topic_idx"), g.get("mode_idx"))].append(g.get("generated_text", ""))
    cross, within, sizes = [], [], []
    for _, texts in groups.items():
        pooled = [gr for t in texts for gr in _grams(t)]
        if pooled:
            cross.append(len(set(pooled)) / len(pooled))          # cross-sample distinct-4gram
        ws = [len(set(_grams(t))) / max(len(_grams(t)), 1) for t in texts if _grams(t)]
        if ws:
            within.append(float(np.mean(ws)))                     # within-sample distinct-4gram
        sizes.append(len(texts))
    return {"cell": cell, "n_groups": len(groups), "mean_group_size": float(np.mean(sizes)),
            "cross_sample_distinct4": round(float(np.mean(cross)), 4),
            "within_sample_distinct4": round(float(np.mean(within)), 4)}


def _parse(cell: str):
    # Vtemp_L14_a0.03 / Rc1_L21_a0.1 / baseline_a0
    if cell.startswith("baseline"):
        return {"vector": "baseline", "site": None, "alpha_frac": 0.0}
    parts = cell.split("_")
    return {"vector": parts[0], "site": int(parts[1][1:]), "alpha_frac": float(parts[2][1:])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True, help="vmb_c3f_3b")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in sorted(args.run_dir.iterdir()):
        if not d.is_dir():
            continue
        div = _cell_diversity(args.run_dir, d.name)
        if div is None:
            continue
        rows.append({**_parse(d.name), **div, "is_null": d.name.upper().startswith("RC")})

    # (f) matched: Vtemp cross-sample distinct4 ÷ mean(Rc) at each (site, α)
    def rc_mean(site, af, key):
        v = [r[key] for r in rows if r["is_null"] and r["site"] == site and r["alpha_frac"] == af]
        return float(np.mean(v)) if v else None

    for r in rows:
        if r["is_null"] or r["vector"] == "baseline":
            continue
        for key in ("cross_sample_distinct4", "within_sample_distinct4"):
            base = rc_mean(r["site"], r["alpha_frac"], key)
            r[f"{key}_over_Rc"] = round(float(r[key] / base), 3) if base else None

    base0 = next((r for r in rows if r["vector"] == "baseline"), None)
    out = {"model": "3b", "arm": "C3 certifying (f) — resample-diversity (14f item)",
           "STATUS": "FIRST_READ_PENDING (C§8)",
           "law": "distinct-4-gram over k=8 same-prompt resamples; CROSS-sample (pooled unique/total) "
                  "= resample diversity, WITHIN-sample = degeneracy guard; (f) = Vtemp cross ÷ mean(Rc) "
                  "matched (site,α), dose-ordered; 14f P=0.85: rises dose-ordered + V_temp-specific at α≤.1",
           "baseline_a0": base0,
           "rows": sorted(rows, key=lambda r: (r["vector"], r["site"] or 0, r["alpha_frac"]))}
    args.out_json.write_text(json.dumps(out, indent=1))
    for r in out["rows"]:
        if r["vector"] == "baseline":
            print(f"  baseline_a0  cross={r['cross_sample_distinct4']} within={r['within_sample_distinct4']}")
        else:
            print(f"  {r['cell']:16} cross={r['cross_sample_distinct4']} (÷Rc {r.get('cross_sample_distinct4_over_Rc')}) "
                  f"within={r['within_sample_distinct4']} (÷Rc {r.get('within_sample_distinct4_over_Rc')})")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
