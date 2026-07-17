"""Second-judge-family pass over BANKED annex judge tables (s12 A1, annex half).

The annex judge dirs bank the exact judge-ready artifacts: ``pairs.md`` (key-free pair
texts, the same scratchpad copies the family-1 session/subagent judges read) + ``key.json``
(pair -> class + steered side) + family-1 responses (``choices.json``, or ``j*.json`` for
the multi-judge dirs). Family-2 re-judges the SAME pairs.md items with the dir's OWN header
question — no reconstruction needed, no rng. Key never enters judge context.

Pre-named reading: divergence = instrument finding about judge families (scoped per cell);
scored-verdict flip = stop-and-surface.

    python -m anamnesis.scripts.vmb_judge_family2_annex \
        --dirs ../outputs/battery/annex/14r_judge,../outputs/battery/annex/roster_judge \
        --out-root ../outputs/battery/annex/family2_judge
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

from anamnesis.scripts.vmb_judge_family2 import _ask2, JUDGE2_MODEL, JUDGE2_FALLBACK

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PAIR_RE = re.compile(r"^## PAIR (\d+)\s*(?:\(topic: (.*?)\))?\s*$")

PROMPT_TMPL = """You will see two AI-generated texts, A and B.

{question}

Text A:
{a}

Text B:
{b}

Answer with exactly one letter: A or B."""


def parse_pairs(md: str) -> tuple[str, dict[str, dict]]:
    lines = md.splitlines()
    question = ""
    for ln in lines:
        if ln.startswith("# ") and "A/B" in ln:
            question = ln.lstrip("# ").strip()
            break
    pairs: dict[str, dict] = {}
    cur = side = None
    for ln in lines:
        m = PAIR_RE.match(ln)
        if m:
            cur = m.group(1)
            pairs[cur] = {"topic": m.group(2), "A": [], "B": []}
            side = None
            continue
        if ln.strip() == "### A":
            side = "A"
            continue
        if ln.strip() == "### B":
            side = "B"
            continue
        if cur and side and not ln.startswith("## "):
            pairs[cur][side].append(ln)
    for p in pairs.values():
        p["A"] = "\n".join(p["A"]).strip()
        p["B"] = "\n".join(p["B"]).strip()
    if not question or not pairs:
        raise SystemExit("pairs.md parse failure (no question header or no pairs)")
    return question, pairs


def load_family1(d: Path) -> dict[str, list[str]]:
    """pair_id -> list of family-1 choices (1 session judge, or j1..jN)."""
    out: dict[str, list[str]] = {}
    cj = d / "choices.json"
    if cj.exists():
        for k, v in json.loads(cj.read_text()).items():
            out.setdefault(str(k), []).append(v)
    for jf in sorted(d.glob("j[0-9]*.json")):
        for k, v in json.loads(jf.read_text()).items():
            if isinstance(v, dict):
                # multi-set judge files ({setN: {pair: choice}}) — the set→pairs.md mapping
                # is not banked here; skip rather than guess (class-level comparison only)
                continue
            out.setdefault(str(k), []).append(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True, help="comma list of annex judge dirs")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--max-chars", type=int, default=2200)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    api_key = os.environ["OPENROUTER_API_KEY"]
    args.out_root.mkdir(parents=True, exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock

    report = {"arm": "2nd judge family — annex banked tables (s12 A1)",
              "STATUS": "FIRST_READ_PENDING (C§8)",
              "law": ("SAME banked pairs.md items (the family-1 judges' own key-free "
                      "scratchpad copies), each dir judged with its OWN header question; "
                      "divergence = instrument finding, scoped per cell; scored-verdict "
                      "flip = stop-and-surface"),
              "judge_family2_chain": f"{JUDGE2_MODEL} -> {JUDGE2_FALLBACK} (OpenRouter)",
              "tables": {}}
    usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "refusal_fallbacks": 0}
    ulock = Lock()

    for dpath in [Path(p.strip()) for p in args.dirs.split(",") if p.strip()]:
        question, pairs = parse_pairs((dpath / "pairs.md").read_text())
        key = json.loads((dpath / "key.json").read_text())
        fam1 = load_family1(dpath)
        pids = sorted(pairs, key=int)
        logger.info(f"[{dpath.name}] {len(pids)} pairs — {question!r}")

        def ask_one(pid: str) -> str | None:
            local = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                     "refusal_fallbacks": 0}
            prompt = PROMPT_TMPL.format(
                question=question,
                a=pairs[pid]["A"][: args.max_chars], b=pairs[pid]["B"][: args.max_chars])
            ans = _ask2(api_key, prompt, local, r"\b([AB])\b")
            with ulock:
                for k in usage:
                    usage[k] += local[k]
            return ans

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            answers = dict(zip(pids, ex.map(ask_one, pids)))

        # per-class score (same convention as judge_scores.json: fraction where the judge
        # picked the STEERED side, i.e. "correct" detections) + per-pair agreement
        by_class: dict[str, dict] = {}
        agree = n_agree = 0
        rows = {}
        for pid in pids:
            k = key.get(pid, {})
            cls, steered = k.get("class"), k.get("steered")
            a2 = answers[pid]
            f1 = fam1.get(pid, [])
            rows[pid] = {"family2": a2, "family1": f1, "steered": steered, "class": cls}
            if a2 is not None and f1:
                # agreement vs the family-1 majority choice
                maj = max(set(f1), key=f1.count)
                agree += int(a2 == maj)
                n_agree += 1
            if cls is not None and a2 is not None and steered is not None:
                c = by_class.setdefault(cls, {"correct": 0, "n": 0})
                c["n"] += 1
                c["correct"] += int(a2 == steered)
        report["tables"][dpath.name] = {
            "question": question,
            "n_pairs": len(pids),
            "family2_scores_by_class": by_class,
            "family1_scores_by_class": json.loads(
                (dpath / "judge_scores.json").read_text())
            if (dpath / "judge_scores.json").exists() else None,
            "pair_agreement_with_family1_majority": round(agree / n_agree, 3)
            if n_agree else None,
            "n_agree_rows": n_agree,
            "rows": rows}
        logger.info(f"[{dpath.name}] f2 by class: "
                    + " ".join(f"{c}={v['correct']}/{v['n']}" for c, v in by_class.items())
                    + f" | agree(maj)={report['tables'][dpath.name]['pair_agreement_with_family1_majority']}")

    (args.out_root / "annex_family2_results.json").write_text(json.dumps(report, indent=1))
    (args.out_root / "usage.json").write_text(json.dumps(usage, indent=1))
    logger.info(f"banked -> {args.out_root} (calls={usage['calls']})")


if __name__ == "__main__":
    main()
