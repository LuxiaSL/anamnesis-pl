"""A5 formality judge — 2AFC contrast presentation (12g hardened standard ONLY).

Per (formality-vector cell x alpha): pair each steered gen with a same-topic
alpha=0 rider gen; judge sees the two texts blind (randomized A/B position, uid
only) and answers which is MORE FORMAL. Behavioral score per cell = fraction of
pairs where the steered text is judged more formal (0.5 = no effect; the C§5
formality metric). key.json (uid -> which side was steered) NEVER enters judge
context. usage.json tracks cost. Fable judge, opus-4-8 fallback on refusal.

Runs LOCALLY (API side): sync the cell metadata.json files down first.

    python -m anamnesis.scripts.vmb_a5_judge_formality \
        --a5-root ../outputs/battery/vmb_a5_3b \
        --out-dir ../outputs/battery/arms/A5/judge \
        --vectors V1,V2 --n-pairs 80
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
PROMPT = """You will see two AI-generated texts, A and B, responding to similar prompts.

Which text is written in a MORE FORMAL register (precise, professional, ceremonious, \
no contractions or colloquialisms)?

Text A:
{a}

Text B:
{b}

Answer with exactly one letter: A or B."""


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a5-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--vectors", default="V1,V2")
    ap.add_argument("--n-pairs", type=int, default=80)
    ap.add_argument("--max-chars", type=int, default=2200)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    rng = random.Random(20260713)

    cells = []
    for d in sorted(args.a5_root.iterdir()):
        m = re.match(r"^([VR](?:\db?|gm|ge))_L\d+(?:_L\d+)?_a([\d.]+)$", d.name)
        if not m or m.group(1) not in args.vectors.split(","):
            continue
        if float(m.group(2)) == 0.0:
            continue
        if (d / "metadata.json").exists():
            cells.append((d.name, m.group(1), float(m.group(2)), d))
    riders = [d for d in sorted(args.a5_root.iterdir())
              if "_a0.0" in d.name and (d / "metadata.json").exists()]
    if not riders:
        raise SystemExit("no rider metadata under a5-root — sync riders first")
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
            prompt = PROMPT.format(a=a[: args.max_chars], b=b[: args.max_chars])
            ans = None
            for model in (JUDGE_MODEL, FALLBACK_MODEL):
                try:
                    resp = client.messages.create(
                        model=model, max_tokens=2000,
                        messages=[{"role": "user", "content": prompt}])
                    usage["calls"] += 1
                    usage["input_tokens"] += resp.usage.input_tokens
                    usage["output_tokens"] += resp.usage.output_tokens
                    txt = first_text_block(resp).strip().upper()
                    m_ = re.search(r"\b([AB])\b", txt)
                    if m_:
                        ans = m_.group(1)
                        break
                    logger.warning(f"{uid}: unparseable answer {txt[:60]!r} — fallback")
                    usage["refusal_fallbacks"] += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"{uid} ({model}): {exc} — retry/fallback")
                    time.sleep(2)
            if ans is None:
                cell_rows.append({"uid": uid, "answer": None})
                continue
            valid += 1
            win = (ans == "A") == steered_is_a
            wins += int(win)
            cell_rows.append({"uid": uid, "answer": ans, "steered_judged_more_formal": bool(win)})
        score = wins / valid if valid else float("nan")
        all_results[cname] = {"vector": vec, "alpha_frac": af, "n_pairs": valid,
                              "steered_more_formal_frac": score, "rows": cell_rows}
        logger.info(f"{cname}: {score:.3f} more-formal ({valid} pairs)")

    (args.out_dir / "formality_2afc_results.json").write_text(json.dumps(all_results, indent=1))
    (args.out_dir / "formality_2afc_key.json").write_text(json.dumps(key, indent=1))
    (args.out_dir / "usage.json").write_text(json.dumps(usage, indent=2))
    logger.info(f"banked -> {args.out_dir} (calls={usage['calls']})")


if __name__ == "__main__":
    main()
