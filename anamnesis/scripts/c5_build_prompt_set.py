"""Computer #5 install-lever PoC — build the floor prompt set from the originating corpus.

SPEC: research/planning/SPEC-computer5-install-lever-poc-2026-07-28.md §10.

Celeste has no held-out topic list, so the prompt substrate comes from the originating SFT
corpus (`sft_modelC.jsonl`, 10,422 records, 1,693 distinct header topics). This is a
CONSTRUCTION, not an eval — in-distribution prompts are what an install vector wants, and
topic reuse is not a leak the way it would be for a decoder. Documented, not hidden.

THE OPEN DESIGN QUESTION THIS SCRIPT DEFERS TO DATA (spec §10):
the corpus convention puts the topic in the HEADER and opens generically ("hello"), then
deepens. Reproducing that literally risks koto's thin-substrate failure (base in chat-frame
gave 10-token aphasic turns; base-side Sigma undersized at 4,772 tokens vs 40,331 healthy).
Llama-70B base is a far stronger continuation engine than koto's 3B, so it may not bite here.
Rather than argue it, we emit BOTH arms and let the G2 floor characterization pick:

  arm "naive"       header(topic) + "hello"                  — the corpus convention, literal
  arm "substantive" header(topic) + a real substantive user   — same register, more purchase
                    turn drawn from that topic's own convos

Both arms are the same generation run with different prompt strings, so carrying both costs
one extra pass and removes a guess from the critical path.

Anti-memorization note: the substantive turn is lifted WITHOUT its preceding conversation
context and used as a FIRST turn, so the model is not cued mid-conversation into reproducing
a memorized continuation. The (topic, turn) pair was still seen in training; that is accepted
and stamped, not hidden.

Pure stdlib + pydantic. No GPU, no model. Run on node2 where the corpus lives:
    python c5_build_prompt_set.py --corpus /home/cluster-user/modelC_train/sft_modelC.jsonl \
        --n-topics 40 --out ~/luxi-files/computer5/prompts/floor_prompts.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# The corpus scaffold, verbatim from README + sft_modelC.jsonl. Load-bearing: Computer has no
# chat template, and this same scaffold on vanilla base is the persona-off control BY
# CONSTRUCTION (spec §2). Any deviation here silently changes what the delta measures.
HEADER_STRANGER_TOPIC = "As follows is a conversation between another user and Model C about {topic}."
HEADER_STRANGER_BARE = "As follows is a conversation between another user and Model C."
SCAFFOLD_BODY = "\n\nFull conversation with Model C:\n\n**User:** {msg}\n\n**Model C:**"
STOP_SEQUENCE = "\n\n**User:**"

# Parse the header of a corpus record: relational frame + optional topic. NOTE the optional
# "later " — the same-user arm reads "As follows is a LATER conversation between the same user
# and Model C." (1,590 records). Omitting it silently drops that arm into the parse-failure
# bucket and makes the census lie about coverage.
# NOT re.S, and the captures exclude periods/newlines: the same-user arm appends free-text
# scene notes after the header sentence ("...and Model C. The user does not stop talking"),
# which a greedy dot-all capture swallows whole and turns the frame census into noise.
_HEADER_RE = re.compile(
    r"^As follows is an? (?:later )?conversation between ([^.\n]+?)(?: about ([^.\n]+?))?\.")
# User turns in the corpus body.
_USER_TURN_RE = re.compile(r"\*\*User:\*\* (.+?)\n\n\*\*Model C:\*\*", re.S)

# Discourse continuers: a turn opening with one of these is a REPLY, not an opener. Lifting it
# context-free produces a non-sequitur first turn ("yes, there's something about being able to
# add up all numbers"), which is unnatural for base and c5 alike and is not Computer's
# distribution. Second guard behind the prefer-real-first-turns rule.
_CONTINUER_RE = re.compile(
    r"^\s*(yes|no|yeah|nah|ok|okay|sure|right|true|exactly|indeed|well|so|but|and|or|then|"
    r"also|i see|i guess|i think so|agreed|fair|hmm+|oh|ah|huh|that'?s|it'?s|they'?re|"
    r"this is|those|these|them|it |they |that )\b",
    re.I,
)

# A "substantive" opening turn: long enough to give the model purchase, short enough to read
# as an opener rather than a pasted essay. Bounds are in characters.
SUBSTANTIVE_MIN_CHARS = 40
SUBSTANTIVE_MAX_CHARS = 320
# Openers that carry no content — the thing arm "naive" deliberately isolates.
_GENERIC_OPENER_RE = re.compile(
    r"^\s*(hello|hi|hey|hi there|good (morning|evening|afternoon)|greetings|yo|sup)\b[\s.!,?]*$",
    re.I,
)


class TurnCandidate(BaseModel):
    """A user turn harvested from the corpus, with the provenance needed to rank it."""

    text: str
    position: int = Field(description="0 = the conversation's real opening turn")
    source_idx: str


class FloorPrompt(BaseModel):
    """One floor cell: identical text goes to every donor and to the base host."""

    prompt_id: str
    topic: str
    arm: str = Field(description="'naive' | 'substantive'")
    user_turn: str
    header: str
    prompt_text: str = Field(description="the exact string fed to the model, scaffold included")
    source_idx: str | None = Field(
        default=None, description="corpus record idx the user turn was lifted from (substantive arm)"
    )
    real_opener: bool = Field(
        default=True,
        description="True if the turn was the conversation's own opening turn. False = lifted "
                    "mid-conversation and may read as a non-sequitur; audit these.",
    )
    source_turn_position: int = Field(default=0)


class PromptSet(BaseModel):
    """The banked prompt set + everything needed to reproduce or audit it."""

    law: str
    corpus: str
    seed: int
    n_topics: int
    n_prompts: int
    stop_sequence: str
    sampling: dict[str, object]
    corpus_census: dict[str, object]
    topics: list[str]
    prompts: list[FloorPrompt]


def _is_substantive(turn: str) -> bool:
    t = turn.strip()
    if not (SUBSTANTIVE_MIN_CHARS <= len(t) <= SUBSTANTIVE_MAX_CHARS):
        return False
    if _GENERIC_OPENER_RE.match(t):
        return False
    if _CONTINUER_RE.match(t):
        return False
    # Multi-line pastes read as documents, not openers.
    return t.count("\n") <= 2


def _load_corpus(path: Path) -> tuple[dict[str, list[TurnCandidate]], dict[str, object]]:
    """-> ({topic: [TurnCandidate, ...]}, census). Only stranger-frame topic records."""
    by_topic: dict[str, list[TurnCandidate]] = defaultdict(list)
    n_records = 0
    n_parse_fail = 0
    frames: dict[str, int] = defaultdict(int)
    arms: dict[str, int] = defaultdict(int)

    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("unparseable corpus line %d: %s", line_no, exc)
                n_parse_fail += 1
                continue
            n_records += 1
            text = rec.get("text") or ""
            arms[str(rec.get("arm"))] += 1
            m = _HEADER_RE.match(text)
            if not m:
                n_parse_fail += 1
                continue
            frame, topic = m.group(1), m.group(2)
            frames[frame] += 1
            # Hold the relational header FIXED at 'stranger' for the PoC (spec §12 Q2);
            # header variation is Celeste's persona-map question, not this one.
            if topic is None or not frame.startswith("another user"):
                continue
            # corpus `idx` is a STRING tag (e.g. "bank_05922"), not an integer.
            idx = str(rec.get("idx")) if rec.get("idx") is not None else None
            for pos, turn in enumerate(_USER_TURN_RE.findall(text)):
                by_topic[topic.strip()].append(
                    TurnCandidate(text=turn.strip(), position=pos, source_idx=idx or ""))

    census = {
        "n_records": n_records,
        "n_header_parse_failures": n_parse_fail,
        "n_distinct_topics_stranger": len(by_topic),
        "frames_top": dict(sorted(frames.items(), key=lambda kv: -kv[1])[:8]),
        "n_distinct_frames": len(frames),
        "arms": dict(arms),
    }
    return by_topic, census


def build(corpus: Path, n_topics: int, seed: int, min_convos: int,
          require_real_opener: bool) -> PromptSet:
    by_topic, census = _load_corpus(corpus)
    if not by_topic:
        sys.exit(f"FATAL: no stranger-frame topic records parsed from {corpus}")

    # Only topics that actually carry a usable substantive turn, so both arms cover the SAME
    # topic list — the arms must be comparable cell-for-cell.
    eligible: dict[str, list[TurnCandidate]] = {}
    n_with_real_opener = 0
    for topic, turns in by_topic.items():
        if len(turns) < min_convos:
            continue
        subs = [t for t in turns if _is_substantive(t.text)]
        if not subs:
            continue
        has_opener = any(t.position == 0 for t in subs)
        n_with_real_opener += int(has_opener)
        # The corpus convention opens conversations with "hello", so a substantive turn at
        # position 0 is comparatively scarce. Requiring one buys prompts that are genuine
        # openers rather than mid-conversation fragments read as non-sequiturs.
        if require_real_opener and not has_opener:
            continue
        eligible[topic] = subs
    logger.info("topics carrying a substantive REAL opener: %d", n_with_real_opener)
    logger.info(
        "eligible topics: %d / %d (>=%d turns and >=1 substantive)",
        len(eligible), len(by_topic), min_convos,
    )
    if len(eligible) < n_topics:
        sys.exit(f"FATAL: only {len(eligible)} eligible topics, need {n_topics}")

    rng = random.Random(seed)
    topics = sorted(rng.sample(sorted(eligible), n_topics))

    prompts: list[FloorPrompt] = []
    n_lifted = 0
    for topic in topics:
        header = HEADER_STRANGER_TOPIC.format(topic=topic)
        # Deterministic pick, in priority order:
        #   1. a REAL opener (position 0 of its conversation) — natural by construction
        #   2. failing that, any substantive turn, FLAGGED as lifted so it can be audited
        # Within the chosen tier take the median-length turn: the longest is often an outlier
        # paste, and we want a typical opener rather than an extreme one.
        openers = [t for t in eligible[topic] if t.position == 0]
        tier = openers or eligible[topic]
        tier = sorted(tier, key=lambda t: (len(t.text), t.text))
        pick = tier[len(tier) // 2]
        turn, src_idx = pick.text, pick.source_idx
        is_real_opener = pick.position == 0
        if not is_real_opener:
            n_lifted += 1
        slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:40] or "topic"

        prompts.append(FloorPrompt(
            prompt_id=f"naive__{slug}", topic=topic, arm="naive", user_turn="hello",
            header=header, prompt_text=header + SCAFFOLD_BODY.format(msg="hello"),
        ))
        prompts.append(FloorPrompt(
            prompt_id=f"subst__{slug}", topic=topic, arm="substantive", user_turn=turn,
            header=header, prompt_text=header + SCAFFOLD_BODY.format(msg=turn),
            source_idx=src_idx, real_opener=is_real_opener, source_turn_position=pick.position,
        ))

    return PromptSet(
        law=("floor prompts for the Computer #5 install lever. Header held FIXED at stranger+topic; "
             "two arms over the SAME topic list — 'naive' (corpus-literal 'hello' opener) and "
             "'substantive' (median-length real user turn from that topic, lifted context-free and "
             "used as a first turn). G2 floor characterization selects the arm of record."),
        corpus=str(corpus), seed=seed, n_topics=n_topics, n_prompts=len(prompts),
        stop_sequence=STOP_SEQUENCE,
        sampling={
            "n_substantive_lifted_midconversation": n_lifted,
            "require_real_opener": require_real_opener,
            "substantive_selection": ("prefer the conversation's REAL opening turn (position 0); "
                                      "fall back to a mid-conversation turn only if none is "
                                      "substantive, flagged via prompt.real_opener=False"),
            "topic_sample": "random.Random(seed).sample over eligible topics, sorted",
            "min_turns_per_topic": min_convos,
            "substantive_bounds_chars": [SUBSTANTIVE_MIN_CHARS, SUBSTANTIVE_MAX_CHARS],
            "substantive_pick": "median-length substantive turn (not longest — avoids paste outliers)",
            "memorization_note": ("the (topic, turn) pair was seen in c5 training; the turn is used "
                                  "WITHOUT its preceding context as a first turn. Accepted and "
                                  "stamped — this is a construction, not an eval."),
        },
        corpus_census=census, topics=topics, prompts=prompts,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--n-topics", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--min-convos", type=int, default=3,
                    help="minimum user turns banked under a topic for it to be eligible")
    ap.add_argument("--allow-lifted-turns", action="store_true",
                    help="permit substantive turns lifted mid-conversation (they often read as "
                         "non-sequiturs when used as a first turn). Default: require a real opener.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if not args.corpus.is_file():
        sys.exit(f"FATAL: corpus not found: {args.corpus}")

    ps = build(args.corpus, args.n_topics, args.seed, args.min_convos,
               require_real_opener=not args.allow_lifted_turns)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(ps.model_dump_json(indent=1), encoding="utf-8")

    logger.info("corpus census: %s", json.dumps(ps.corpus_census))
    logger.info("topics (%d): %s", len(ps.topics), ", ".join(ps.topics))
    logger.info("wrote %d prompts (%d topics x 2 arms) -> %s", ps.n_prompts, ps.n_topics, args.out)
    print("\n--- sample cells ---")
    for p in ps.prompts[:4]:
        print(f"[{p.arm}] topic={p.topic!r}\n{p.prompt_text}\n{'-' * 60}")


if __name__ == "__main__":
    main()
