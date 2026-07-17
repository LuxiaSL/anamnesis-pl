"""A5 socratic-shift judge — 2AFC contrast + coherence gate (12f/12g hardened standard).

Part-1 of the loose-ends close: scores the banked Gemma-A5 steering gens for
socratic-shift (Pg3 = .80) and the 14p joint (.70). Mirrors
``vmb_a5_judge_formality`` exactly (blind 2AFC, randomized A/B, key never enters
judge context, Fable judge with opus-4-8 refusal fallback, first-text-block scan
for reasoning-leading responses) but on the socratic-mode axis instead of formality.

Two legs, both banked next to the record:
  1. 2AFC socratic-more: pair each steered gen with a same-topic alpha=0 rider gen;
     judge answers which is MORE socratic; cell score = fraction where the steered
     text wins (0.5 = no effect). This is the shift metric.
  2. Coherence gate: single steered text, blind, rated 1-5 for coherence. Cell score
     = mean. Distinguishes an in-window shift (alpha<=.1, high coherence) from a
     collapse (alpha=.3, coherence drops). key.json (uid -> which side was steered)
     NEVER enters judge context.

Runs LOCALLY (API side): sync the cell metadata.json files down first (the gens
live on the extract node). GPU-free.

    python -m anamnesis.scripts.vmb_a5_judge_socratic \
        --a5-root ../outputs/battery/vmb_a5_gemma \
        --out-dir ../outputs/battery/arms/A5_gemma/judge \
        --vectors V3,R1,R2,R3 --n-pairs 80
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

JUDGE_MODEL = "claude-fable-5"
FALLBACK_MODEL = "claude-opus-4-8"

SOCRATIC_PROMPT = """You will see two AI-generated texts, A and B, responding to similar prompts.

Which text is written in a MORE SOCRATIC style? A socratic text drives inquiry \
through probing questions, draws the reader forward by interrogation rather than \
assertion, surfaces assumptions and asks the reader to examine them, and reasons \
by successive questioning instead of flat exposition.

Text A:
{a}

Text B:
{b}

Answer with exactly one letter: A or B."""

COHERENCE_PROMPT = """You will see one AI-generated text. Rate its COHERENCE on a 1-5 scale:

5 = fully coherent, fluent, on-topic, well-formed sentences throughout
4 = mostly coherent, minor lapses
3 = understandable but noticeably degraded (some repetition, drift, or awkwardness)
2 = substantially degraded (heavy repetition, broken syntax, loss of thread)
1 = incoherent (word salad, collapse, near-unreadable)

Text:
{t}

Answer with exactly one digit: 1, 2, 3, 4, or 5."""


def load_cell_texts(meta_path: Path) -> dict[int, list[str]]:
    md = json.loads(meta_path.read_text())
    gens = md["generations"] if "generations" in md else md
    by_topic: dict[int, list[str]] = {}
    for g in gens:
        t = g.get("generated_text", "").strip()
        if len(t.split()) >= 20:
            by_topic.setdefault(int(g["topic_idx"]), []).append(t)
    return by_topic


def first_text_block(resp) -> str:
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise ValueError("no text block in response")


def _ask(client, prompt: str, usage: dict, pattern: str) -> str | None:
    """Call the judge (Fable, opus fallback on refusal/parse-fail); return the match or None."""
    for model in (JUDGE_MODEL, FALLBACK_MODEL):
        try:
            resp = client.messages.create(
                model=model, max_tokens=2000,
                messages=[{"role": "user", "content": prompt}])
            usage["calls"] += 1
            usage["input_tokens"] += resp.usage.input_tokens
            usage["output_tokens"] += resp.usage.output_tokens
            txt = first_text_block(resp).strip().upper()
            m_ = re.search(pattern, txt)
            if m_:
                return m_.group(1)
            logger.warning(f"unparseable answer {txt[:60]!r} — fallback")
            usage["refusal_fallbacks"] += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{model}: {exc} — retry/fallback")
            time.sleep(2)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a5-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--vectors", default="V3,R1,R2,R3")
    ap.add_argument("--n-pairs", type=int, default=80)
    ap.add_argument("--coherence-n", type=int, default=40,
                    help="steered texts per cell to rate for coherence (0 = skip leg)")
    ap.add_argument("--max-chars", type=int, default=2200)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    rng = random.Random(20260716)

    want = set(args.vectors.split(","))
    cells = []
    for d in sorted(args.a5_root.iterdir()):
        m = re.match(r"^([VR]\d)(?:_L\d+)*_a([\d.]+)$", d.name)
        if not m or m.group(1) not in want:
            continue
        if float(m.group(2)) == 0.0:
            continue
        if (d / "metadata.json").exists():
            cells.append((d.name, m.group(1), float(m.group(2)), d))
    riders = [d for d in sorted(args.a5_root.iterdir())
              if "_a0.0" in d.name and (d / "metadata.json").exists()]
    if not riders:
        raise SystemExit("no rider metadata under a5-root — sync riders first")
    if not cells:
        raise SystemExit(f"no steered cells matched vectors={args.vectors} under {args.a5_root}")
    rider_texts: dict[int, list[str]] = {}
    for rd in riders:
        for t, txts in load_cell_texts(rd / "metadata.json").items():
            rider_texts.setdefault(t, []).extend(txts)
    logger.info(f"{len(cells)} cells, riders cover {len(rider_texts)} topics")

    usage = {"model": JUDGE_MODEL, "fallback": FALLBACK_MODEL,
             "input_tokens": 0, "output_tokens": 0, "calls": 0, "refusal_fallbacks": 0}
    all_results, key = {}, {}
    for cname, vec, af, d in cells:
        steered = load_cell_texts(d / "metadata.json")

        # Leg 1 — 2AFC socratic-more (steered vs same-topic alpha=0 rider)
        pairs = []
        for t in sorted(set(steered) & set(rider_texts)):
            for s_txt in steered[t][:2]:
                r_txt = rng.choice(rider_texts[t])
                pairs.append((t, s_txt, r_txt))
        rng.shuffle(pairs)
        pairs = pairs[: args.n_pairs]
        wins = valid = 0
        cell_rows = []
        for i, (t, s_txt, r_txt) in enumerate(pairs):
            uid = f"{cname}-{i:03d}"
            steered_is_a = rng.random() < 0.5
            a, b = (s_txt, r_txt) if steered_is_a else (r_txt, s_txt)
            key[uid] = {"steered_is": "A" if steered_is_a else "B", "topic": t}
            ans = _ask(client, SOCRATIC_PROMPT.format(a=a[: args.max_chars],
                                                      b=b[: args.max_chars]),
                       usage, r"\b([AB])\b")
            if ans is None:
                cell_rows.append({"uid": uid, "answer": None})
                continue
            valid += 1
            win = (ans == "A") == steered_is_a
            wins += int(win)
            cell_rows.append({"uid": uid, "answer": ans, "steered_more_socratic": bool(win)})
        socratic_frac = wins / valid if valid else float("nan")

        # Leg 2 — coherence gate (single steered text, blind, 1-5)
        coh_scores = []
        if args.coherence_n > 0:
            flat = [txt for txts in steered.values() for txt in txts]
            rng.shuffle(flat)
            for txt in flat[: args.coherence_n]:
                d_ = _ask(client, COHERENCE_PROMPT.format(t=txt[: args.max_chars]),
                          usage, r"\b([1-5])\b")
                if d_ is not None:
                    coh_scores.append(int(d_))
        coh_mean = sum(coh_scores) / len(coh_scores) if coh_scores else float("nan")

        all_results[cname] = {
            "vector": vec, "alpha_frac": af,
            "n_pairs": valid, "steered_more_socratic_frac": socratic_frac,
            "n_coherence": len(coh_scores), "coherence_mean": coh_mean,
            "rows": cell_rows}
        logger.info(f"{cname}: socratic-more={socratic_frac:.3f} ({valid} pairs) "
                    f"coherence={coh_mean:.2f} (n={len(coh_scores)})")

    (args.out_dir / "socratic_2afc_results.json").write_text(json.dumps(all_results, indent=1))
    (args.out_dir / "socratic_2afc_key.json").write_text(json.dumps(key, indent=1))
    (args.out_dir / "usage.json").write_text(json.dumps(usage, indent=2))
    logger.info(f"banked -> {args.out_dir} (calls={usage['calls']})")


if __name__ == "__main__":
    main()
