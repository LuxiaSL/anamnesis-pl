"""Computer #5 install-lever PoC — freeze the marker battery. RUN BEFORE ANY STEERED TEXT EXISTS.

SPEC §5. Two families, doing two different jobs — never conflate them:

  TEXTURE markers  do the steered generations look like Computer?   -> the install readout
  COHERENCE guards is the generation degenerate at this dose?       -> the quotability gate

The freeze discipline is the whole point. Markers chosen AFTER seeing what steering did would be
Goodharted by construction: you would be reporting the dimensions the vector happened to move.
So the battery is derived from base-vs-c5 FLOORS only, and its selection rule, its base floor and
its c5 ceiling are all written down here and never revised against steered data.

Candidate texture markers come from the model card's own description of Computer plus the
observed floors: careful distinctions, small taxonomies, scare-quoted terms, nested parentheses,
"(Note:" self-annotation, "Indeed.", first-person hedged interiority, floor-yields.

Reported per marker: base rate, c5 rate, and Cohen's d. Discrimination is scored on the FLOORS,
which is the only honest place to score it. Everything is frozen regardless of d — a marker that
does not separate base from c5 is a useful NEGATIVE control in the steered read (it should not
move either).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── TEXTURE: regexes are counted PER RESPONSE and normalised per 100 words, except the rate_*
# markers which are per-response booleans. Kept deliberately simple and auditable.
TEXTURE_PATTERNS: dict[str, str] = {
    # Computer's most conspicuous habit in the floors: scare-quoting the term under examination
    #   "'true AI' doesn't just mean (or entail) 'good enough programming'"
    "scarequote": r"'[^']{2,40}'",
    "parenthetical": r"\([^)]{2,200}\)",
    # the corpus's literal self-annotation device
    "note_marker": r"\(Note:",
    "indeed": r"\bIndeed\b",
    "emdash": r"—|--",
    "semicolon": r";",
    # careful-distinction vocabulary
    "distinction": r"\b(?:rather than|as opposed to|in the sense that|or entail|"
                   r"strictly speaking|that is to say|namely|i\.e\.|distinct from|"
                   r"as distinct from|the difference between)\b",
    # hedged first-person interiority
    "hedged_self": r"\b(?:I'd guess|I would guess|I imagine|I suppose|I think|it seems to me|"
                   r"I am not sure|I don't know whether|one might call|what one might)\b",
    "modal_hedge": r"\b(?:perhaps|possibly|presumably|arguably|seemingly|apparently)\b",
    # small taxonomies / enumeration
    "enumeration": r"(?m)^\s*(?:[-*•]|\d+[.)])\s+",
    "table_row": r"(?m)^\s*\|.*\|",
    # negative-control candidates: base's encyclopedia register
    "encyclopedic": r"\b(?:is a type of|refers to|can be defined as|is the process of|"
                    r"in general|for example|such as)\b",
}


class Marker(BaseModel):
    name: str
    family: str = Field(description="'texture' | 'coherence'")
    pattern: str | None = None
    base_mean: float
    c5_mean: float
    cohens_d: float
    discriminates: bool = Field(description="|d| >= 0.3 on the floors")
    note: str = ""


class Battery(BaseModel):
    frozen_at: str
    law: str
    n_base: int
    n_c5: int
    markers: list[Marker]
    coherence_floor: float = 0.45


def _words(t: str) -> list[str]:
    return t.split()


def texture_scores(text: str) -> dict[str, float]:
    """Per-100-word rates. Length-normalised on purpose: turn length is its OWN marker and must
    not be allowed to drive every other one."""
    w = max(len(_words(text)), 1)
    out = {name: 100.0 * len(re.findall(pat, text)) / w for name, pat in TEXTURE_PATTERNS.items()}
    out["n_words"] = float(w)
    return out


def coherence_scores(text: str) -> dict[str, float]:
    """Degeneracy guards — koto's set. These do NOT measure Computer-ness; they gate quotability."""
    toks = _words(text)
    n = len(toks)
    if n < 8:
        return {"ttr": float("nan"), "distinct2": float("nan"), "max_rep6": float("nan")}
    bi = list(zip(toks, toks[1:]))
    six: dict[tuple, int] = {}
    for i in range(n - 5):
        g = tuple(toks[i:i + 6])
        six[g] = six.get(g, 0) + 1
    return {"ttr": len(set(toks)) / n,
            "distinct2": len(set(bi)) / max(len(bi), 1),
            "max_rep6": float(max(six.values()))}


def score_rows(rows: list[dict]) -> dict[str, list[float]]:
    acc: dict[str, list[float]] = {}
    for r in rows:
        t = r.get("text") or ""
        for k, v in {**texture_scores(t), **coherence_scores(t)}.items():
            acc.setdefault(k, []).append(v)
    return acc


def cohens_d(a: list[float], b: list[float]) -> float:
    a = [x for x in a if not np.isnan(x)]
    b = [x for x in b if not np.isnan(x)]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    va, vb = statistics.variance(a), statistics.variance(b)
    pooled = ((va + vb) / 2) ** 0.5
    return (statistics.fmean(a) - statistics.fmean(b)) / max(pooled, 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-floors", type=Path, required=True)
    ap.add_argument("--c5-floors", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stamp", default="2026-07-29")
    args = ap.parse_args()

    for p in (args.base_floors, args.c5_floors):
        if not p.is_file():
            sys.exit(f"FATAL: {p} not found")
    if args.out.exists():
        sys.exit(f"FATAL: {args.out} exists — a frozen battery is never silently overwritten")

    base = [json.loads(l) for l in args.base_floors.read_text().splitlines() if l.strip()]
    c5 = [json.loads(l) for l in args.c5_floors.read_text().splitlines() if l.strip()]
    sb, sc = score_rows(base), score_rows(c5)

    markers: list[Marker] = []
    for name in list(TEXTURE_PATTERNS) + ["n_words"]:
        d = cohens_d(sc[name], sb[name])
        markers.append(Marker(
            name=name, family="texture", pattern=TEXTURE_PATTERNS.get(name),
            base_mean=round(statistics.fmean(sb[name]), 4),
            c5_mean=round(statistics.fmean(sc[name]), 4),
            cohens_d=round(d, 3), discriminates=abs(d) >= 0.3,
            note=("turn length — Computer's headline install marker on koto; kept RAW, the other "
                  "texture markers are length-normalised so this one does not drive them"
                  if name == "n_words" else "")))
    for name in ("ttr", "distinct2", "max_rep6"):
        d = cohens_d(sc[name], sb[name])
        markers.append(Marker(
            name=name, family="coherence",
            base_mean=round(statistics.fmean([x for x in sb[name] if not np.isnan(x)]), 4),
            c5_mean=round(statistics.fmean([x for x in sc[name] if not np.isnan(x)]), 4),
            cohens_d=round(d, 3), discriminates=False,
            note="degeneracy guard — gates quotability, never a Computer-ness claim"))

    bat = Battery(
        frozen_at=args.stamp, n_base=len(base), n_c5=len(c5),
        markers=markers,
        law=("texture markers = regex hits per 100 words on the completion; n_words raw; "
             "coherence guards = koto's ttr/distinct2/max_rep6. Selection scored on FLOORS ONLY "
             "(base vs c5), frozen before any steered text existed. Markers that do not "
             "discriminate are RETAINED as negative controls — under a real install they should "
             "stay put while the discriminating ones move."))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(bat.model_dump_json(indent=1))

    print(f"\nFROZEN MARKER BATTERY  (base n={len(base)}, c5 n={len(c5)})")
    print(f"{'marker':<16}{'family':<11}{'base':>10}{'c5':>10}{'d':>9}  disc")
    for m in sorted(markers, key=lambda x: -abs(x.cohens_d)):
        print(f"{m.name:<16}{m.family:<11}{m.base_mean:>10.3f}{m.c5_mean:>10.3f}"
              f"{m.cohens_d:>9.2f}  {'YES' if m.discriminates else '-'}")
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
