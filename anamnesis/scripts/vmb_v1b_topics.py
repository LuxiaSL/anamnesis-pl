"""P4 — V1b topic-disjoint contrast set (imago B5; session-3 pre-window prep).

V1 (the formality contrastive-prompt vector) was built from a contrast set that OVERLAPS the
20 mode-eval topics (A5 external review §B3 — topic-conditioned-register leakage). V1b is the
clean rebuild: the SAME formality-contrast construction on topics DISJOINT from the eval set,
so the steering effect cannot ride topic leakage. This script fixes and BANKS that topic list
and HARD-ASSERTS disjointness (that assertion is the whole point of V1b). The vector build /
generation is window-work (window item 7) and consumes this manifest.

Eval topics = prompt_sets `set_a` + `set_b` = battery topic_idx 0..19 (the mode corpora).
V1b topics = `set_c` + `set_d` = 40 held-out topics, one formality contrast PAIR per topic
(matches V1's 40-pair construction), all disjoint from eval.

    PYTHONPATH=pipeline python -m anamnesis.scripts.vmb_v1b_topics \
        --out outputs/battery/v1b_topics_manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_manifest() -> dict:
    ps = json.loads(
        (Path("pipeline/anamnesis/prompts/prompt_sets.json")).read_text())
    topics = ps["topics"]
    eval_topics = list(topics["set_a"]) + list(topics["set_b"])     # battery topic_idx 0..19
    v1b_topics = list(topics["set_c"]) + list(topics["set_d"])      # 40 held-out

    # HARD disjointness gate — the reason V1b exists. Fail loudly, never silently proceed.
    overlap = sorted(set(eval_topics) & set(v1b_topics))
    if overlap:
        raise AssertionError(
            f"V1b topic set OVERLAPS the eval set on {len(overlap)} topics: {overlap} — "
            "V1b must be topic-disjoint from eval (imago B5). Fix the source sets.")
    if len(v1b_topics) < 20:
        raise AssertionError(f"V1b needs >=20 disjoint topics; got {len(v1b_topics)}")
    if len(set(v1b_topics)) != len(v1b_topics):
        dupes = [t for t in v1b_topics if v1b_topics.count(t) > 1]
        raise AssertionError(f"V1b topic list has duplicates: {sorted(set(dupes))}")

    return {
        "purpose": "V1b formality-contrast topics, DISJOINT from the 20 mode-eval topics "
                   "(imago B5; A5 review §B3 topic-leakage fix). Vector build = window item 7.",
        "eval_topics": eval_topics, "n_eval": len(eval_topics),
        "v1b_topics": v1b_topics, "n_v1b": len(v1b_topics),
        "source_sets": {"eval": ["set_a", "set_b"], "v1b": ["set_c", "set_d"]},
        "disjoint_asserted": True,
        "construction": "one formality contrast pair (formal vs informal register) per V1b "
                        "topic → 40 pairs; reuse the V1 CAA build with these topics; bank as "
                        "a NEW vector V1b (NEVER overwrite V1). R-baseline per 13e.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/battery/v1b_topics_manifest.json"))
    args = ap.parse_args()
    m = build_manifest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(m, indent=1))
    print(f"V1b topic manifest: {m['n_v1b']} topics, disjoint from {m['n_eval']} eval topics "
          f"(assert PASSED)")
    print("  V1b topics:")
    for i, t in enumerate(m["v1b_topics"]):
        print(f"    {i:2d}: {t}")
    print(f"  → {args.out}")


if __name__ == "__main__":
    main()
