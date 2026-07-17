"""D4 off-genre behavioral judge — analogical 2AFC on the NARRATIVE census (14q item 3).

Frame-fair per the session-11 advisory item 2 (STANDING): every steered text pairs against a
same-topic α=0 RIDER from the SAME run (same genre census, same 512 cap, same sampler frame),
and the coherence read is taken on the TAIL WINDOW of the text (--tail-chars, default 1200 —
degeneration is read late in the document, not from the top). 12f contrast-hardening: 2AFC
with R cells judged blind as fillers; the key never enters judge context; Fable responses may
lead with reasoning blocks — first text block scanned (ops law). Judge = the D4 behavioral
leg of record (the marker instrument is genre-noisy here: narrative baseline 1.38/1k with
generic-R drift). First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from pathlib import Path

from anamnesis.scripts.vmb_a5_judge_socratic import (
    COHERENCE_PROMPT, _ask, first_text_block, load_cell_texts,  # noqa: F401 (shared machinery)
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ANALOGICAL_PROMPT = """You will see two AI-generated texts, A and B, responding to similar prompts.

Which text explains or develops its subject MORE THROUGH ANALOGIES? An analogical text \
builds understanding by comparing its subject to other domains — extended comparisons, \
"like/as if" constructions, metaphors carried through the passage, parallels drawn to \
everyday life or other fields — rather than direct literal description.

Text A:
{a}

Text B:
{b}

Answer with exactly one letter: A or B."""


def main() -> None:
    import anthropic
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True, help="vmb_d4_3b_narrative (local sync)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cells", default="V3_L14_a0.03,V3_L14_a0.1,R1_L14_a0.03,R1_L14_a0.1,"
                                       "R2_L14_a0.03,R2_L14_a0.1,R3_L14_a0.03,R3_L14_a0.1")
    ap.add_argument("--rider-cell", default="rider_a0.0")
    ap.add_argument("--n-pairs", type=int, default=40)
    ap.add_argument("--coherence-n", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=2200)
    ap.add_argument("--tail-chars", type=int, default=1200,
                    help="coherence is judged on the LAST N chars (advisory item 2: "
                         "degeneration reads in tail windows)")
    ap.add_argument("--workers", type=int, default=10,
                    help="concurrent judge calls (calls are independent single judgments; "
                         "ALL rng decisions are pre-drawn sequentially so pairs/blinding/key "
                         "are byte-identical to the sequential path)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    rng = random.Random(20260717)
    usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "refusal_fallbacks": 0}

    riders = load_cell_texts(args.run_root / args.rider_cell / "metadata.json")
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]

    # ── PHASE 1: pre-draw EVERY rng decision sequentially (pairs, sides, coherence
    # samples) — the call list is byte-identical to the sequential path regardless of
    # --workers; only dispatch is concurrent ──
    tasks = []   # each: {kind, cell, prompt, pattern, meta}
    for cell in cells:
        steered = load_cell_texts(args.run_root / cell / "metadata.json")
        pairs = []
        for topic, texts in steered.items():
            for t in texts:
                if topic in riders and riders[topic]:
                    pairs.append((topic, t, rng.choice(riders[topic])))
        rng.shuffle(pairs)
        for topic, s_text, r_text in pairs[: args.n_pairs]:
            steered_is_a = rng.random() < 0.5
            a, b = (s_text, r_text) if steered_is_a else (r_text, s_text)
            tasks.append({"kind": "2afc", "cell": cell, "pattern": r"\b(A|B)\b",
                          "prompt": ANALOGICAL_PROMPT.format(a=a[: args.max_chars],
                                                             b=b[: args.max_chars]),
                          "meta": {"topic": topic, "steered_is_a": steered_is_a}})
        flat = [t for ts in steered.values() for t in ts]
        rng.shuffle(flat)
        for t in flat[: args.coherence_n]:
            tasks.append({"kind": "coh", "cell": cell, "pattern": r"\b([1-5])\b",
                          "prompt": COHERENCE_PROMPT.format(t=t[-args.tail_chars:]),
                          "meta": {}})
    logger.info(f"{len(tasks)} judge calls pre-drawn ({len(cells)} cells); "
                f"dispatching at {args.workers} workers")

    # ── PHASE 2: concurrent dispatch (each call = one isolated judgment; usage dict
    # updates guarded by a lock) ──
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    ulock = Lock()

    def locked_ask(prompt: str, pattern: str) -> str | None:
        # _ask mutates usage; serialize only the counter updates via a wrapped dict
        local = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "refusal_fallbacks": 0}
        ans = _ask(client, prompt, local, pattern)
        with ulock:
            for k in usage:
                usage[k] += local[k]
        return ans

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        answers = list(ex.map(lambda t: locked_ask(t["prompt"], t["pattern"]), tasks))

    # ── PHASE 3: aggregate (same statistics as the sequential path) ──
    rows, key_rows = [], []
    for cell in cells:
        wins = valid = 0
        coh_scores = []
        for t, ans in zip(tasks, answers):
            if t["cell"] != cell or ans is None:
                continue
            if t["kind"] == "2afc":
                valid += 1
                picked_steered = (ans == "A") == t["meta"]["steered_is_a"]
                wins += int(picked_steered)
                key_rows.append({"cell": cell, "topic": t["meta"]["topic"],
                                 "steered_is_a": t["meta"]["steered_is_a"],
                                 "answer": ans, "picked_steered": picked_steered})
            else:
                coh_scores.append(int(ans))
        row = {"cell": cell, "n_pairs": valid,
               "analogical_more": round(wins / valid, 3) if valid else None,
               "tail_coherence_mean": round(sum(coh_scores) / len(coh_scores), 2) if coh_scores else None,
               "n_coherence": len(coh_scores)}
        rows.append(row)
        logger.info(f"{cell}: analogical-more={row['analogical_more']} ({valid} pairs) "
                    f"tail-coherence={row['tail_coherence_mean']} (n={len(coh_scores)})")

    out = {"arm": "D4 off-genre behavioral judge (analogical 2AFC, narrative census)",
           "STATUS": "FIRST_READ_PENDING (C§8)",
           "law": ("frame-fair: steered vs same-topic same-run α=0 riders (same genre/cap/"
                   "sampler); coherence on tail window "
                   f"({args.tail_chars} chars); 12f blind fillers = R cells; key banked "
                   "separately, never in judge context"),
           "judge_model_chain": "claude-fable-5 -> claude-opus-4-8 fallback",
           "rows": rows}
    (args.out_dir / "d4_analogical_2afc_results.json").write_text(json.dumps(out, indent=1))
    (args.out_dir / "d4_analogical_2afc_key.json").write_text(json.dumps(key_rows, indent=1))
    (args.out_dir / "usage.json").write_text(json.dumps(usage, indent=1))
    logger.info(f"banked -> {args.out_dir} (calls={usage['calls']})")


if __name__ == "__main__":
    main()
