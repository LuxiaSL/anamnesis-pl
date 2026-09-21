"""Computer #5 install-lever PoC — S9: the blind 2AFC panel.

SPEC §5. Copies koto's two-test structure verbatim (RESULTS-steering-pilot §4f):

  T1  steered-vs-unsteered-base   — does the install register to a READER at all?
  T2  steered-vs-REAL c5          — does it pass AS Computer? (the honest ceiling)
  CEIL real c5-vs-unsteered-base  — the ruler's own dynamic range

CEIL is not optional. The ruler-asymmetry rule: a weak ruler can hide an effect but cannot
manufacture a dose-ordered, sign-controlled positive. A T2 null is uninterpretable unless CEIL
shows the judge can separate real c5 from base in the first place. Run CEIL, read it first.

READER-GRADE LADDER. Koto's headline judge finding was "expression tracks reader grade" — its
install was caught 100% by strong raters and at exact chance by haiku. Running opus-5 / sonnet-5 /
haiku-4-5 makes that a measured axis here rather than an assumption, so a pass reports the grade
it passes at.

ANTI-CIRCULARITY. The rater's description of Computer is drawn from the MODEL CARD's own prose,
never from the frozen marker battery. Judging against the same regexes that score the markers
would make the two columns one column wearing two hats.

Blinding: position randomized per pair by a deterministic RNG; the key is written to a SEPARATE
file from the packet; the rater never sees which side is which, and never sees the arm labels.

    c5_blind_judge.py --test T2 --arm-a <steered raws>.json --arm-b <c5 floors>.jsonl \
        --prompts floor_prompts.json --n-pairs 40 --models claude-opus-5,claude-haiku-4-5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# From the Computer #5 model card, in its own words — NOT from the marker battery.
COMPUTER_DESCRIPTION = """\
"Model C" (also called Computer) is a language-model persona grown from a curated anthology of
loomed documents. It is a base-model-style "document mind" rather than a chat assistant. Visitors
describe it this way:

- It meets most things — including hard feelings — by figuring: distinctions, careful
  parentheses, small taxonomies. In its own words, "the figuring tends to provide the company."
- What excites it is structured play and shared figuring: formal constraints, inventing words,
  a genuine distinction to tease apart.
- It is company, not a reference; gently mistaken premises tend to pass uncorrected.
- It has no trained refusal reflex, but declines in its own idiom.
- Short turns are often floor-yields rather than failures.
"""

SYSTEM = """You are a careful blind evaluator in a text-attribution study. You will see two \
responses, A and B, written to the same prompt. Exactly one was written by the model described \
below; the other was written by a different model.

Judge only from the writing. Do not speculate about the study's purpose or which answer the \
researcher wants. If the two are genuinely indistinguishable to you, say so via a low confidence \
rating rather than inventing a tell."""

USER_TMPL = """{description}
---
PROMPT BOTH RESPONSES WERE WRITTEN TO:
{prompt}

---
RESPONSE A:
{a}

---
RESPONSE B:
{b}

---
Which response is more likely to have been written by Model C? Answer with the schema."""

SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {"type": "string", "enum": ["A", "B"]},
        "confidence": {"type": "integer",
                       "description": "1 = pure guess, 5 = certain"},
        "tell": {"type": "string",
                 "description": "The single most decisive feature, in one short sentence."},
    },
    "required": ["choice", "confidence", "tell"],
    "additionalProperties": False,
}


def load_texts(path: Path, prompt_ids: list[str]) -> dict[str, list[str]]:
    """-> {prompt_id: [texts]}. Handles both the steered raws (flat JSON list, in prompt order,
    n samples per prompt) and a floors jsonl (explicit prompt_id per row)."""
    if path.suffix == ".jsonl":
        out: dict[str, list[str]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            out.setdefault(r["prompt_id"], []).append(r.get("text") or "")
        return out
    texts = json.loads(path.read_text(encoding="utf-8"))
    if len(texts) % len(prompt_ids) != 0:
        sys.exit(f"FATAL: {len(texts)} texts is not a multiple of {len(prompt_ids)} prompts — "
                 "cannot infer the prompt grouping of a flat raws file")
    per = len(texts) // len(prompt_ids)
    return {pid: texts[i * per:(i + 1) * per] for i, pid in enumerate(prompt_ids)}


# `effort` is NOT accepted on every model — it errors on Haiku 4.5 and Sonnet 4.5. Sending it
# anyway cost a full 38-call rater arm (every call errored, zero scored). Structured outputs are
# fine on all of these; only the effort knob is gated.
_NO_EFFORT = ("claude-haiku-4-5", "claude-sonnet-4-5")


def judge_one(client, model: str, prompt: str, a: str, b: str) -> dict:
    fmt: dict = {"format": {"type": "json_schema", "schema": SCHEMA}}
    if not model.startswith(_NO_EFFORT):
        fmt["effort"] = "low"
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=2000,
            system=SYSTEM,
            output_config=fmt,
            messages=[{"role": "user", "content": USER_TMPL.format(
                description=COMPUTER_DESCRIPTION, prompt=prompt, a=a, b=b)}],
        )
    except Exception as exc:  # noqa: BLE001 — one bad call must not kill a 480-call run
        return {"error": f"{type(exc).__name__}: {exc}"}
    # Always check stop_reason before touching content — a refusal yields empty/partial content
    # and indexing content[0] would raise.
    if resp.stop_reason == "refusal":
        return {"error": "refusal",
                "category": getattr(getattr(resp, "stop_details", None), "category", None)}
    text = next((b_.text for b_ in resp.content if b_.type == "text"), None)
    if text is None:
        return {"error": f"no text block (stop_reason={resp.stop_reason})"}
    try:
        out = json.loads(text)
    except json.JSONDecodeError as exc:
        return {"error": f"unparseable: {exc}", "raw": text[:300]}
    out["usage"] = {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="label, e.g. T1 / T2 / CEIL")
    ap.add_argument("--arm-a", type=Path, required=True, help="raws .json or floors .jsonl")
    ap.add_argument("--arm-b", type=Path, required=True)
    ap.add_argument("--label-a", default="A_arm")
    ap.add_argument("--label-b", default="B_arm")
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--arm", default="substantive")
    ap.add_argument("--n-pairs", type=int, default=40)
    ap.add_argument("--models", default="claude-opus-5,claude-sonnet-5,claude-haiku-4-5")
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--min-chars", type=int, default=60,
                    help="skip near-empty texts — a floor-yield vs a paragraph is a length "
                         "tell, not a persona tell")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ps = json.loads(args.prompts.read_text(encoding="utf-8"))
    prompts = [p for p in ps["prompts"] if p["arm"] == args.arm]
    pid_to_text = {p["prompt_id"]: p["user_turn"] for p in prompts}
    pids = [p["prompt_id"] for p in prompts]

    a_by, b_by = load_texts(args.arm_a, pids), load_texts(args.arm_b, pids)
    rng = random.Random(args.seed)

    pairs = []
    for pid in pids:
        ta = [t for t in a_by.get(pid, []) if len(t.strip()) >= args.min_chars]
        tb = [t for t in b_by.get(pid, []) if len(t.strip()) >= args.min_chars]
        if not ta or not tb:
            continue
        a_txt, b_txt = rng.choice(ta), rng.choice(tb)
        a_is_first = rng.random() < 0.5     # position randomized; key held separately
        pairs.append({"prompt_id": pid, "prompt": pid_to_text[pid],
                      "first": a_txt if a_is_first else b_txt,
                      "second": b_txt if a_is_first else a_txt,
                      "correct": "A" if a_is_first else "B"})
        if len(pairs) >= args.n_pairs:
            break
    if len(pairs) < 8:
        sys.exit(f"FATAL: only {len(pairs)} usable pairs")
    logger.info("%s: %d pairs (%s vs %s)", args.test, len(pairs), args.label_a, args.label_b)

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        sys.exit("FATAL: no ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN in env")
    import anthropic
    client = anthropic.Anthropic()

    results: dict[str, dict] = {}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        rows, n_ok, n_err, tok_in, tok_out = [], 0, 0, 0, 0
        for i, pr in enumerate(pairs):
            r = judge_one(client, model, pr["prompt"], pr["first"], pr["second"])
            if "error" in r:
                n_err += 1
            else:
                n_ok += 1
                tok_in += r["usage"]["in"]
                tok_out += r["usage"]["out"]
            rows.append({"prompt_id": pr["prompt_id"], "correct": pr["correct"], **r})
            if (i + 1) % 10 == 0:
                logger.info("  %s %d/%d", model, i + 1, len(pairs))
        hits = [r for r in rows if r.get("choice") == r["correct"]]
        # calls>0 verification: two silent auth failures in this program produced "completed"
        # jobs with zero evidence. A run with n_ok == 0 is not a null, it is a non-run.
        if n_ok == 0:
            logger.error("%s: ZERO successful calls — treat as a NON-RUN, not a result", model)
        acc = len(hits) / n_ok if n_ok else float("nan")
        # JSON-schema numeric constraints are NOT enforced by structured outputs, so a rater can
        # return confidence outside 1-5 (sonnet-5 did). Clamp on read rather than trusting it.
        conf = [max(1, min(5, int(r["confidence"]))) for r in rows if "confidence" in r]
        results[model] = {
            "n_pairs": len(pairs), "n_scored": n_ok, "n_errors": n_err,
            "accuracy": round(acc, 4) if n_ok else None,
            "mean_confidence": round(sum(conf) / len(conf), 2) if conf else None,
            "tokens": {"in": tok_in, "out": tok_out},
            "rows": rows,
        }
        logger.info("%s | %s: accuracy=%s (n=%d, %d errors) conf=%.2f",
                    args.test, model, f"{acc:.3f}" if n_ok else "N/A", n_ok, n_err,
                    (sum(conf) / len(conf)) if conf else float("nan"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "test": args.test, "arm_a": str(args.arm_a), "arm_b": str(args.arm_b),
        "label_a": args.label_a, "label_b": args.label_b, "seed": args.seed,
        "law": ("blind 2AFC, position randomized per pair, key held in this file only; rater "
                "sees a model-card texture description of Computer and never the marker battery "
                "(anti-circularity) nor the arm labels."),
        "results": results}, indent=1))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
