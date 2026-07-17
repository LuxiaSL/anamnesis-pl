"""ANNEX control-surface tenancy — TEXT/instrument readout + per-gen PERMUTATION grid.

Instruments (per generation; the CS book's scoring set, FROZEN AT STAGING):
  lexfreq        mean log-unigram-frequency of the REALIZED generated tokens under the
                 SAME add-1 table the V_lex pulse used (rebuilt from the stage-0 manifest
                 via annex_cs_pulses.build_logfreq — sha asserted against the pulse stamp).
                 CS-1's register-instrument-of-record (RIDER 1: the member is V_lex).
  prompt_overlap fraction of generated token ids ∈ prompt token-id set
                 (prompt set = tokenizer(system_prompt + user_prompt, no special tokens))
  selfrep_rate   fraction of generated tokens whose id appeared EARLIER in the generation
                 and NOT in the prompt set (the self-only history mass, realized)
  concl_per_1k   conclusion-marker rate per 1k words (CS-4 instrument; lexicon frozen
                 below — CONCL_RE)
  net_hedge      (hedge − definitive) per 1k words (14n lexicon, imported verbatim)
  ttr · trigram_rep · gen_len · at_cap   the standard set
Cell level adds q90_trigram_rep (standing rule) and n.

Permutation (standing rule — permutation-primary, point envelopes are degenerate):
  each member cell × each instrument → mean diff vs the POOLED Rband gens at the SAME
  SIGNED dose, two-sided p from 10k seeded label shuffles. The 3-cell Rband envelope
  (min/max of per-cell means) is emitted BESIDE the permutation, never instead of it.

Usage (node2, in-chain; --tokenizer-path = the 3B model dir):
    python -m anamnesis.scripts.annex_cs_readout \
        --run-root $RUN_ROOT --stage0-manifest $STAGE0/replay_manifest.json \
        --tokenizer-path $M3B --expect-logfreq-sha <sha from cs_gradients_stamps.json> \
        --out-readout $RUN_ROOT/cs_text_readout.json \
        --out-perm $RUN_ROOT/cs_permutation_grid.json
"""
from __future__ import annotations

import argparse
import json
import re
import zlib
from pathlib import Path

import numpy as np

from anamnesis.scripts.annex_cs_pulses import build_logfreq
from anamnesis.scripts.vmb_14n_hedging_index import DEF_RE, HEDGE_RE

# CS-4 conclusion-marker lexicon — FROZEN AT STAGING (do not extend post-hoc)
CONCL_RE = re.compile(
    r"\b(in conclusion|to conclude|in summary|to summarize|to sum up|in closing|"
    r"all in all|taken together|ultimately|overall)\b", re.IGNORECASE)

PERM_INSTRUMENTS = ("lexfreq", "prompt_overlap", "selfrep_rate", "concl_per_1k",
                    "net_hedge", "ttr", "trigram_rep", "gen_len")
N_PERM = 10_000
PERM_SEED = 20260717
CAP = 512


def parse_dose(cell: str) -> float | None:
    m = re.search(r"_a(n?)(\d*\.?\d+)$", cell)
    if not m:
        return None
    return (-1.0 if m.group(1) else 1.0) * float(m.group(2))


def gen_instruments(g: dict, tok, logfreq: np.ndarray) -> dict | None:
    text = g.get("generated_text", "")
    words = text.split()
    if not words:
        return None
    gen_ids = tok(text, add_special_tokens=False)["input_ids"]
    prompt_ids = set(tok((g.get("system_prompt") or "") + "\n" + (g.get("user_prompt") or ""),
                         add_special_tokens=False)["input_ids"])
    if not gen_ids:
        return None
    seen: set[int] = set()
    self_hits = 0
    overlap = 0
    for tid in gen_ids:
        if tid in prompt_ids:
            overlap += 1
        elif tid in seen:
            self_hits += 1
        seen.add(tid)
    k = 1000.0 / len(words)
    tri = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    nlen = int(g.get("num_generated_tokens", len(gen_ids)))
    return {
        "lexfreq": float(np.mean([logfreq[t] for t in gen_ids])),
        "prompt_overlap": overlap / len(gen_ids),
        "selfrep_rate": self_hits / len(gen_ids),
        "concl_per_1k": len(CONCL_RE.findall(text)) * k,
        "net_hedge": (len(HEDGE_RE.findall(text)) - len(DEF_RE.findall(text))) * k,
        "ttr": len(set(words)) / len(words),
        "trigram_rep": 1.0 - len(set(tri)) / max(len(tri), 1),
        "gen_len": nlen,
        "at_cap": 1.0 if nlen >= CAP else 0.0,
    }


def perm_p(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> float:
    obs = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    na = len(a)
    hits = 0
    for _ in range(N_PERM):
        rng.shuffle(pool)
        if abs(pool[:na].mean() - pool[na:].mean()) >= obs:
            hits += 1
    return (hits + 1) / (N_PERM + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--stage0-manifest", type=Path, required=True)
    ap.add_argument("--tokenizer-path", required=True)
    ap.add_argument("--expect-logfreq-sha", default=None,
                    help="sha from cs_gradients_stamps.json (Glex_L14.logfreq_sha) — "
                         "asserts instrument/pulse table identity")
    ap.add_argument("--out-readout", type=Path, required=True)
    ap.add_argument("--out-perm", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    entries = json.loads(args.stage0_manifest.read_text())["entries"]
    logfreq, sha = build_logfreq(entries, len(tok))
    if args.expect_logfreq_sha and sha != args.expect_logfreq_sha:
        raise SystemExit(f"logfreq sha MISMATCH: built {sha} != pulse {args.expect_logfreq_sha}"
                         " — instrument and pulse are not reading the same table")

    cells: dict[str, dict] = {}
    for d in sorted(args.run_root.iterdir()):
        md = d / "metadata.json"
        if not md.exists():
            continue
        raw = json.loads(md.read_text())
        gens = raw["generations"] if "generations" in raw else raw
        rows = [r for r in (gen_instruments(g, tok, logfreq) for g in gens) if r]
        if not rows:
            continue
        summary = {k: round(float(np.mean([r[k] for r in rows])), 5)
                   for k in rows[0]}
        summary["q90_trigram_rep"] = round(float(np.quantile(
            [r["trigram_rep"] for r in rows], 0.9)), 5)
        summary["n"] = len(rows)
        cells[d.name] = {"summary": summary, "per_gen": rows}

    # ── permutation grid: member cells vs pooled Rband at the same signed dose ──
    rband = {c: parse_dose(c) for c in cells if c.upper().startswith("RBAND")}
    members = {c: parse_dose(c) for c in cells
               if not c.upper().startswith("RBAND") and parse_dose(c) is not None}
    grid: dict[str, dict] = {}
    for cell, dose in sorted(members.items()):
        matched = [c for c, d0 in rband.items() if d0 == dose]
        pool_note = "matched_signed_dose"
        if not matched:
            matched = [c for c, d0 in rband.items()
                       if d0 is not None and abs(d0) == abs(dose)]
            pool_note = "matched_|dose|_FALLBACK (no signed match — flagged)"
        if not matched:
            grid[cell] = {"error": "no Rband pool at matched dose"}
            continue
        row: dict = {"null_pool": matched, "pool_note": pool_note, "dose": dose}
        for inst in PERM_INSTRUMENTS:
            a = np.array([r[inst] for r in cells[cell]["per_gen"]], dtype=np.float64)
            b = np.concatenate([np.array([r[inst] for r in cells[c]["per_gen"]],
                                         dtype=np.float64) for c in matched])
            rng = np.random.default_rng(
                PERM_SEED + zlib.crc32(f"{cell}|{inst}".encode()))
            env = [float(np.mean([r[inst] for r in cells[c]["per_gen"]]))
                   for c in matched]
            row[inst] = {"diff": round(float(a.mean() - b.mean()), 5),
                         "p": round(perm_p(a, b, rng), 5),
                         "null_env": [round(min(env), 5), round(max(env), 5)],
                         "n_null_cells": len(matched)}
        grid[cell] = row

    out = {
        "provenance": "CS tenancy readout: instruments frozen at staging (lexfreq = the "
                      "V_lex instrument-of-record, same add-1 table as the pulse, sha "
                      f"{sha}; CONCL_RE frozen; 14n hedge lexicon verbatim); permutation-"
                      "primary vs pooled Rband at matched signed dose (standing rule), "
                      "envelope beside never instead",
        "logfreq_sha": sha,
        "tokenizer_path": str(args.tokenizer_path),
        "cells": {c: v["summary"] for c, v in cells.items()},
        "per_gen": {c: v["per_gen"] for c, v in cells.items()},
    }
    args.out_readout.write_text(json.dumps(out, indent=1))
    args.out_perm.write_text(json.dumps(
        {"provenance": out["provenance"], "n_perm": N_PERM, "seed": PERM_SEED,
         "grid": grid}, indent=1))
    print(json.dumps({c: {k: v for k, v in r.items() if k in PERM_INSTRUMENTS}
                      for c, r in grid.items()}, indent=1)[:4000])
    print(f"readout -> {args.out_readout}\nperm grid -> {args.out_perm}")


if __name__ == "__main__":
    main()
