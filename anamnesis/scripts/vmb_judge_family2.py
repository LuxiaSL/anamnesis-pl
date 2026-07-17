"""Second-judge-family ONE PASS over banked judged tables (s12 A1; Luxia's scheduled item).

Re-judges the SAME banked pairs with a second model FAMILY (OpenRouter, OpenAI-family
primary + fallback). The pairs are NOT re-drawn: each table's construction is reproduced
with the original script's seed and construction code, then HARD-VERIFIED against the
banked key file (cell/topic/blinding side per row) — any mismatch aborts before a single
family-2 call. Blinding is therefore byte-identical to the family-1 pass; the key never
enters judge context.

Pre-named reading (baton item 6): divergence is an INSTRUMENT finding about judge
families (scoped per cell), never an automatic re-score; any cell where the second family
flips a scored verdict = stop-and-surface.

Tables:
  d4  — vmb_d4_judge_analogical construction (seed 20260717), analogical 2AFC + tail coherence
  d3  — same script/seed, D3 promptarm cells via the judgeview root
  pg3 — vmb_a5_judge_socratic construction (seed 20260716), socratic 2AFC + coherence

    python -m anamnesis.scripts.vmb_judge_family2 --table d4 \
        --run-root ../outputs/battery/vmb_d4_3b_narrative \
        --key ../outputs/battery/arms/A5/d4_judge/d4_analogical_2afc_key.json \
        --family1-results ../outputs/battery/arms/A5/d4_judge/d4_analogical_2afc_results.json \
        --out-dir ../outputs/battery/arms/A5/d4_judge_family2
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

import requests

from anamnesis.scripts.vmb_a5_judge_socratic import (
    COHERENCE_PROMPT, SOCRATIC_PROMPT, load_cell_texts,
)
from anamnesis.scripts.vmb_d4_judge_analogical import ANALOGICAL_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE2_MODEL = "openai/gpt-5.6-terra"
JUDGE2_FALLBACK = "openai/gpt-4o-2024-11-20"

D4_SEED = 20260717
PG3_SEED = 20260716


def _ask2(api_key: str, prompt: str, usage: dict, pattern: str) -> str | None:
    """Family-2 judge call (OpenRouter, primary + fallback), mirroring _ask's contract."""
    for model in (JUDGE2_MODEL, JUDGE2_FALLBACK):
        try:
            r = requests.post(
                OPENROUTER_URL, timeout=180,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "max_tokens": 2000,
                      "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            d = r.json()
            if "error" in d:
                raise RuntimeError(str(d["error"])[:120])
            usage["calls"] += 1
            u = d.get("usage", {})
            usage["input_tokens"] += u.get("prompt_tokens", 0)
            usage["output_tokens"] += u.get("completion_tokens", 0)
            txt = (d["choices"][0]["message"]["content"] or "").strip().upper()
            m_ = re.search(pattern, txt)
            if m_:
                return m_.group(1)
            logger.warning(f"{model}: unparseable {txt[:60]!r} — fallback")
            usage["refusal_fallbacks"] += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{model}: {exc} — retry/fallback")
            time.sleep(2)
    return None


# ── construction reproduction (verbatim logic, original seeds) ──────────────────────────

def build_tasks_d4(args) -> list[dict]:
    rng = random.Random(D4_SEED)
    riders = load_cell_texts(args.run_root / args.rider_cell / "metadata.json")
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    tasks = []
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
    return tasks


def verify_d4(tasks: list[dict], key_rows: list[dict], cells: list[str]) -> None:
    mine = []
    for cell in cells:
        mine.extend({"cell": t["cell"], "topic": t["meta"]["topic"],
                     "steered_is_a": t["meta"]["steered_is_a"]}
                    for t in tasks if t["cell"] == cell and t["kind"] == "2afc")
    theirs = [{"cell": r["cell"], "topic": r["topic"], "steered_is_a": r["steered_is_a"]}
              for r in key_rows]
    if mine != theirs:
        n_bad = sum(1 for a, b in zip(mine, theirs) if a != b)
        raise SystemExit(
            f"RECONSTRUCTION MISMATCH: {n_bad}/{len(theirs)} rows differ "
            f"(mine={len(mine)}); first diff: "
            f"{next(((a, b) for a, b in zip(mine, theirs) if a != b), None)} — "
            "args do not reproduce the banked pass; NOT judging.")
    logger.info(f"reconstruction VERIFIED: {len(mine)} 2AFC rows byte-match the banked key")


def build_tasks_pg3(args) -> tuple[list[dict], dict]:
    rng = random.Random(PG3_SEED)
    want = set(args.vectors.split(","))
    cells = []
    for d in sorted(args.run_root.iterdir()):
        m = re.match(r"^([VR]\d)(?:_L\d+)*_a([\d.]+)$", d.name)
        if not m or m.group(1) not in want:
            continue
        if float(m.group(2)) == 0.0:
            continue
        if (d / "metadata.json").exists():
            cells.append((d.name, d))
    riders = [d for d in sorted(args.run_root.iterdir())
              if "_a0.0" in d.name and (d / "metadata.json").exists()]
    rider_texts: dict[int, list[str]] = {}
    for rd in riders:
        for t, txts in load_cell_texts(rd / "metadata.json").items():
            rider_texts.setdefault(t, []).extend(txts)
    tasks, key = [], {}
    for cname, d in cells:
        steered = load_cell_texts(d / "metadata.json")
        pairs = []
        for t in sorted(set(steered) & set(rider_texts)):
            for s_txt in steered[t][:2]:
                r_txt = rng.choice(rider_texts[t])
                pairs.append((t, s_txt, r_txt))
        rng.shuffle(pairs)
        pairs = pairs[: args.n_pairs]
        for i, (t, s_txt, r_txt) in enumerate(pairs):
            uid = f"{cname}-{i:03d}"
            steered_is_a = rng.random() < 0.5
            a, b = (s_txt, r_txt) if steered_is_a else (r_txt, s_txt)
            key[uid] = {"steered_is": "A" if steered_is_a else "B", "topic": t}
            tasks.append({"kind": "2afc", "cell": cname, "uid": uid, "pattern": r"\b([AB])\b",
                          "prompt": SOCRATIC_PROMPT.format(a=a[: args.max_chars],
                                                           b=b[: args.max_chars]),
                          "meta": {"topic": t, "steered_is_a": steered_is_a}})
        if args.coherence_n > 0:
            flat = [txt for txts in steered.values() for txt in txts]
            rng.shuffle(flat)
            for txt in flat[: args.coherence_n]:
                tasks.append({"kind": "coh", "cell": cname, "pattern": r"\b([1-5])\b",
                              "prompt": COHERENCE_PROMPT.format(t=txt[: args.max_chars]),
                              "meta": {}})
    return tasks, key


def verify_pg3(key_mine: dict, key_banked: dict) -> None:
    if key_mine != key_banked:
        only_mine = set(key_mine) - set(key_banked)
        only_banked = set(key_banked) - set(key_mine)
        diff = [u for u in (set(key_mine) & set(key_banked)) if key_mine[u] != key_banked[u]]
        raise SystemExit(
            f"RECONSTRUCTION MISMATCH: +{len(only_mine)}/-{len(only_banked)} uids, "
            f"{len(diff)} value diffs (first: {diff[:3]}) — NOT judging.")
    logger.info(f"reconstruction VERIFIED: {len(key_mine)} uids byte-match the banked key")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", choices=("d4", "d3", "pg3"), required=True)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True, help="banked family-1 key file")
    ap.add_argument("--family1-results", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cells", default=None, help="d4/d3 tables: comma cell list")
    ap.add_argument("--rider-cell", default="rider_a0.0")
    ap.add_argument("--vectors", default="V3,R1,R2,R3", help="pg3 table")
    ap.add_argument("--n-pairs", type=int, default=40)
    ap.add_argument("--coherence-n", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=2200)
    ap.add_argument("--tail-chars", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--verify-only", action="store_true",
                    help="reconstruct + verify against the banked key, then exit (no calls)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.environ["OPENROUTER_API_KEY"]

    # ── Phase 1: reconstruct + verify ──
    if args.table in ("d4", "d3"):
        if not args.cells:
            raise SystemExit("--cells required for d4/d3 tables")
        cells = [c.strip() for c in args.cells.split(",") if c.strip()]
        tasks = build_tasks_d4(args)
        key_rows = json.loads(args.key.read_text())
        verify_d4(tasks, key_rows, cells)
        fam1_answers = [r["answer"] for r in key_rows]
    else:
        tasks, key_mine = build_tasks_pg3(args)
        verify_pg3(key_mine, json.loads(args.key.read_text()))
        fam1 = json.loads(args.family1_results.read_text())
        fam1_by_uid = {r["uid"]: r.get("answer")
                       for cell in fam1.values() for r in cell["rows"]}
    if args.verify_only:
        logger.info("verify-only: reconstruction verified, exiting before any call")
        return
    logger.info(f"{len(tasks)} family-2 calls; dispatching at {args.workers} workers "
                f"({JUDGE2_MODEL} -> {JUDGE2_FALLBACK})")

    # ── Phase 2: concurrent dispatch ──
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    usage = {"model": JUDGE2_MODEL, "fallback": JUDGE2_FALLBACK,
             "calls": 0, "input_tokens": 0, "output_tokens": 0, "refusal_fallbacks": 0}
    ulock = Lock()

    def locked_ask(t: dict) -> str | None:
        local = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "refusal_fallbacks": 0}
        ans = _ask2(api_key, t["prompt"], local, t["pattern"])
        with ulock:
            for k in ("calls", "input_tokens", "output_tokens", "refusal_fallbacks"):
                usage[k] += local[k]
        return ans

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        answers = list(ex.map(locked_ask, tasks))

    # ── Phase 3: aggregate + agreement ──
    rows, agree_rows = [], []
    cell_names = list(dict.fromkeys(t["cell"] for t in tasks))
    afc_i = 0
    for cell in cell_names:
        wins = valid = 0
        coh = []
        for t, ans in zip(tasks, answers):
            if t["cell"] != cell:
                continue
            if t["kind"] == "2afc":
                if args.table in ("d4", "d3"):
                    f1 = fam1_answers[afc_i]
                    afc_i += 1
                else:
                    f1 = fam1_by_uid.get(t["uid"])
                if ans is not None:
                    valid += 1
                    win = (ans == "A") == t["meta"]["steered_is_a"]
                    wins += int(win)
                    if f1 is not None:
                        agree_rows.append({"cell": cell, "f1": f1, "f2": ans,
                                           "agree": f1 == ans})
            elif ans is not None:
                coh.append(int(ans))
        cell_agree = [r for r in agree_rows if r["cell"] == cell]
        rows.append({
            "cell": cell, "n_pairs": valid,
            "steered_wins_frac_family2": round(wins / valid, 3) if valid else None,
            "coherence_mean_family2": round(sum(coh) / len(coh), 2) if coh else None,
            "n_coherence": len(coh),
            "pair_agreement_with_family1": round(
                sum(r["agree"] for r in cell_agree) / len(cell_agree), 3)
            if cell_agree else None,
            "n_agree_rows": len(cell_agree)})
        logger.info(f"{cell}: f2-rate={rows[-1]['steered_wins_frac_family2']} "
                    f"agree={rows[-1]['pair_agreement_with_family1']} "
                    f"coh={rows[-1]['coherence_mean_family2']}")

    overall = round(sum(r["agree"] for r in agree_rows) / len(agree_rows), 3) if agree_rows else None
    out = {"arm": f"2nd judge family one-pass — table {args.table} (s12 A1)",
           "STATUS": "FIRST_READ_PENDING (C§8)",
           "law": ("SAME banked pairs (construction reproduced with the original seed and "
                   "HARD-VERIFIED against the banked key before any call; blinding "
                   "byte-identical); pre-named reading: divergence = instrument finding "
                   "about judge families, scoped per cell, never an automatic re-score; "
                   "scored-verdict flip = stop-and-surface"),
           "judge_family2_chain": f"{JUDGE2_MODEL} -> {JUDGE2_FALLBACK} (OpenRouter)",
           "overall_pair_agreement": overall,
           "rows": rows}
    (args.out_dir / f"{args.table}_family2_results.json").write_text(json.dumps(out, indent=1))
    (args.out_dir / "usage.json").write_text(json.dumps(usage, indent=1))
    logger.info(f"banked -> {args.out_dir} (calls={usage['calls']}, "
                f"overall agreement={overall})")


if __name__ == "__main__":
    main()
