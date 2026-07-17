"""D5 — long-generation late-collapse, TEXT legs (14q item 4, Pd5=.70; session-11 Part B-1).

Companion to the entropy-curve harness (--halves = the state-side leg). This is the
text-side reader over the D5 cells' generated texts (512-token uncapped, k=4 seeds/class):

  per gen, split the generated text at the word midpoint into FRONT/BACK halves; per half:
    distinct-4gram fraction   (diversity — the steering effect whose persistence Pd5 asks)
    trigram repetition rate   (collapse-mode counter-metric)
    type-token ratio          (coherence proxy; the banked in-window convention)
  cross-sample diversity per cell×half: distinct-4gram across the k resamples of the same
  prompt class (the 14f (f) instrument at half grain).
  natural-stop column: fraction of gens ending before the 512 cap (EOS behavior — late-
  collapse can also express as failure to stop).

Pd5 scores on: the steering effect (entropy/diversity vs matched-R) PERSISTS in the back
half with coherence in-window. Late-collapse (back-half ttr/diversity falling OUTSIDE the
baseline envelope while front-half is in-window) is PRE-NAMED as reportable, not a failure.
All effects read vs the Rband cells + baseline through the IDENTICAL split. No filed
sub-P beyond Pd5 (frozen in 14q). CPU-only over metadata texts. First-read → outer loop.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def half_stats(words: list[str]) -> dict:
    if len(words) < 8:
        return {"n_words": len(words), "distinct4": None, "trigram_rep": None, "ttr": None}
    fours = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
    tris = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    tri_counts = Counter(tris)
    rep = sum(c - 1 for c in tri_counts.values() if c > 1) / max(len(tris), 1)
    return {"n_words": len(words),
            "distinct4": round(len(set(fours)) / len(fours), 4),
            "trigram_rep": round(rep, 4),
            "ttr": round(len(set(words)) / len(words), 4)}


def cell_read(run: Path, cap_tokens: int) -> dict | None:
    md_path = run / "metadata.json"
    if not md_path.exists():
        return None
    md = json.loads(md_path.read_text())
    gens = md["generations"] if "generations" in md else md
    front, back, stops = [], [], []
    groups: dict[tuple, dict[str, list]] = {}
    for g in gens:
        t = (g.get("generated_text") or "").split()
        h = len(t) // 2
        front.append(half_stats(t[:h]))
        back.append(half_stats(t[h:]))
        stops.append(int(g.get("num_generated_tokens", 0)) < cap_tokens)
        key = (g.get("topic_idx"), g.get("mode"))
        groups.setdefault(key, {"front": [], "back": []})
        groups[key]["front"].append(" ".join(t[:h]))
        groups[key]["back"].append(" ".join(t[h:]))

    def agg(rows: list[dict], k: str) -> float | None:
        vals = [r[k] for r in rows if r[k] is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    # cross-sample diversity per half: distinct-4gram across same-class resamples
    def cross(halfkey: str) -> float | None:
        vals = []
        for _, gg in groups.items():
            texts = gg[halfkey]
            if len(texts) < 2:
                continue
            fours = set()
            total = 0
            for t in texts:
                w = t.split()
                f4 = [" ".join(w[i:i + 4]) for i in range(len(w) - 3)]
                fours.update(f4)
                total += len(f4)
            if total:
                vals.append(len(fours) / total)
        return round(float(np.mean(vals)), 4) if vals else None

    return {"n": len(gens),
            "front": {k: agg(front, k) for k in ("distinct4", "trigram_rep", "ttr")},
            "back": {k: agg(back, k) for k in ("distinct4", "trigram_rep", "ttr")},
            "cross_sample_distinct4_front": cross("front"),
            "cross_sample_distinct4_back": cross("back"),
            "natural_stop_frac": round(float(np.mean(stops)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True, help="vmb_d5_3b")
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--null-prefixes", default="RBAND")
    ap.add_argument("--baseline-cell", default="baseline_a0.0")
    ap.add_argument("--cap-tokens", type=int, default=512)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    nulls = tuple(p.strip().upper() for p in args.null_prefixes.split(",") if p.strip())

    cells = {}
    for c in args.cells:
        r = cell_read(args.run_dir / c, args.cap_tokens)
        if r is None:
            cells[c] = {"MISSING": True}
        else:
            cells[c] = r

    # envelopes: per metric×half over the null cells; baseline separately
    def envelope(metric: str, half: str) -> dict | None:
        vals = [cells[c][half][metric] for c in cells
                if c.upper().startswith(nulls) and not cells[c].get("MISSING")
                and cells[c][half][metric] is not None]
        if not vals:
            return None
        return {"min": min(vals), "max": max(vals), "mean": round(float(np.mean(vals)), 4)}

    out = {
        "arm": "D5 — long-gen late-collapse, text legs (14q item 4)",
        "STATUS": "FIRST_READ_PENDING (C§8)",
        "filed_P": {"Pd5_effect_persists_back_half": 0.70},
        "law": ("word-midpoint half split; distinct-4gram/trigram-rep/ttr per half; "
                "cross-sample distinct-4gram per class×half (14f-(f) at half grain); "
                "natural-stop fraction; all vs the Rband cells + baseline through the "
                "identical split; late-collapse PRE-NAMED reportable"),
        "cells": cells,
        "null_envelopes": {f"{h}.{m}": envelope(m, h)
                           for h in ("front", "back")
                           for m in ("distinct4", "trigram_rep", "ttr")},
        "baseline_cell": args.baseline_cell,
    }
    args.out_json.write_text(json.dumps(out, indent=1))
    for c, r in cells.items():
        if r.get("MISSING"):
            print(c, "MISSING")
        else:
            print(f"{c:22} front d4={r['front']['distinct4']} ttr={r['front']['ttr']} | "
                  f"back d4={r['back']['distinct4']} ttr={r['back']['ttr']} "
                  f"rep={r['back']['trigram_rep']} | stop={r['natural_stop_frac']}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
