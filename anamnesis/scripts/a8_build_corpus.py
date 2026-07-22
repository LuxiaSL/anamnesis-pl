"""A8 Leg-0 — T1: deterministic paired-corpus manifest builder (CPU, local).

Annex-2 session spec Phase A inputs (SESSION-PROMPT-annex2-arm8-leg0-2026-07-22.md):
  S1  shared battery prompts + each model's banked completions (vmb_stage0_{3b,8b}),
      160/voice = 2 reps per (topic x task-stratum) cell, reps lowest-first.
  S2  neutral prose: wikitext-103-raw-v1 VALIDATION shards (Luxia amendment 2026-07-22:
      raw variant only — the non-raw @-@ detokenization artifacts would contaminate the
      stratum), chunked 150-500 tokens, 160 shards, deterministic selection.
  S3  mode-pole texts: vmb_a2_{3b,8b}_pure_{mode} for the 5 battery modes,
      30/mode/voice = rep 0 for topics 0-19 + rep 1 for topics 0-9.

Counts signed off by desk 2026-07-22 (S1=160/voice, S2=160, S3=150/voice). Selection is
seed-free where possible (lowest-rep rule) and fixed-seed where sampling is unavoidable
(S2 shard choice). Every source file is sha256'd into the stamp; the manifest itself is
hashed at write. NOTHING here touches a GPU.

S2 native-arm carrier (PARKED design note, surfaced at CP-1): neutral prose has no
natural prompt; the native chat-template arm wraps each shard as an assistant reply to a
CONSTANT carrier prompt reusing the battery's own expository template with a generic
topic ("Write about: general knowledge"). Constant across all S2 texts; recorded in the
stamp. The desk rules whether this stands or S2 drops to raw-arm-only.

Run (from pipeline/):  python -m anamnesis.scripts.a8_build_corpus
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_build_corpus")

# ---------------------------------------------------------------- constants (frozen)
A8_SEED = 80  # fixed selection seed for the one sampled stratum (S2)
MODES = ["linear", "socratic", "contrastive", "dialectical", "analogical"]
TASK_STRATA = ["expository", "explanatory", "argumentative", "conversational"]
S1_REPS_PER_CELL = 2          # 20 topics x 4 strata x 2 = 160/voice
S3_REP0_TOPICS = 20           # rep 0 for all topics
S3_REP1_TOPICS = 10           # rep 1 for topics 0-9  -> 30/mode/voice
S2_N_SHARDS = 160
S2_TOK_MIN, S2_TOK_MAX = 150, 500
S2_TOK_CLOSE = 250            # greedy chunker: close a shard once it reaches this
S2_CARRIER_PROMPT = "Write about: general knowledge"   # constant native-arm carrier
TOKENIZER_REF = "meta-llama/Llama-3.1-8B-Instruct"     # Leg-0 pair shares this tokenizer
# Cross-tokenizer legs: TOKENIZER_REF stays for S2 chunking (keeps S2 shards
# IDENTICAL across legs — same chunker+tokenizer+seed) and for informational
# n_tokens; each model tokenizes for itself at collection time (pairing is by TEXT).
STAGE0_RUN = {"3b": "vmb_stage0_3b", "8b": "vmb_stage0_8b",
              "qwen-7b": "vmb_stage0_qwen7b", "dsv2-lite": "vmb_stage0_dsv2_lite"}

DEFAULT_ARM_ROOT = Path("outputs/battery/arms/A8_conjugation")
WIKITEXT_GLOB = (
    "~/.cache/huggingface/hub/datasets--wikitext/snapshots/*/"
    "wikitext-103-raw-v1/validation-00000-of-00001.parquet"
)


class CorpusEntry(BaseModel):
    """One replay text with everything both template arms need."""
    text_id: str
    stratum: Literal["S1", "S2", "S3"]
    voice: Literal["3b", "8b", "qwen-7b", "dsv2-lite", "neutral"]
    mode: str                      # task stratum (S1), mode (S3), or "neutral" (S2)
    topic_idx: Optional[int] = None
    topic: Optional[str] = None
    repetition: Optional[int] = None
    source_run: Optional[str] = None
    source_generation_id: Optional[str] = None
    source_seed: Optional[str] = None
    system_prompt: str = ""        # native-arm context (empty = no system turn)
    user_prompt: str               # native-arm context (S2: constant carrier)
    text: str = Field(min_length=1)
    n_words: int
    n_tokens: Optional[int] = None  # via TOKENIZER_REF when available


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_generations(path: Path) -> list[dict]:
    with open(path) as f:
        meta = json.load(f)
    gens = meta["generations"] if isinstance(meta, dict) else meta
    if not gens:
        raise RuntimeError(f"{path}: empty generations")
    return gens


def _get_tokenizer():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TOKENIZER_REF, local_files_only=True)
        logger.info("tokenizer loaded (%s) — token counts exact", TOKENIZER_REF)
        return tok
    except Exception as e:  # noqa: BLE001 — degrade to word counts, recorded in stamp
        logger.warning("tokenizer unavailable (%s) — n_tokens omitted, S2 chunking "
                       "falls back to word-count heuristic", e)
        return None


def _ntokens(tok, text: str) -> Optional[int]:
    if tok is None:
        return None
    return len(tok.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------- S1 / S3 selection
def build_s1(sources_dir: Path, tok, voices: tuple[str, ...]) -> list[CorpusEntry]:
    entries: list[CorpusEntry] = []
    for voice in voices:
        run = STAGE0_RUN[voice]
        gens = _load_generations(sources_dir / f"{run}.metadata.json")
        cells: dict[tuple[int, int], list[dict]] = {}
        for g in gens:
            key = (int(g["topic_idx"]), int(g["mode_idx"]))
            cells.setdefault(key, []).append(g)
        if len(cells) != 20 * 4:
            raise RuntimeError(f"{run}: expected 80 topic x stratum cells, got {len(cells)}")
        for (t_idx, m_idx), cell in sorted(cells.items()):
            cell.sort(key=lambda g: int(g["repetition"]))
            picked = [g for g in cell if g["generated_text"].strip()][:S1_REPS_PER_CELL]
            if len(picked) < S1_REPS_PER_CELL:
                raise RuntimeError(f"{run} cell ({t_idx},{m_idx}): "
                                   f"only {len(picked)} non-empty gens")
            for g in picked:
                text = g["generated_text"]
                entries.append(CorpusEntry(
                    text_id=f"S1-{voice}-t{t_idx:02d}-s{m_idx}-r{g['repetition']}",
                    stratum="S1", voice=voice, mode=g["mode"],
                    topic_idx=t_idx, topic=g["topic"],
                    repetition=int(g["repetition"]),
                    source_run=run, source_generation_id=str(g["generation_id"]),
                    source_seed=str(g["seed"]),
                    system_prompt=g.get("system_prompt", "") or "",
                    user_prompt=g["user_prompt"],
                    text=text, n_words=len(text.split()),
                    n_tokens=_ntokens(tok, text)))
    per_voice = {v: sum(1 for e in entries if e.voice == v) for v in voices}
    if per_voice != {v: 160 for v in voices}:
        raise RuntimeError(f"S1 cardinality wrong: {per_voice}")
    return entries


def build_s3(sources_dir: Path, tok, voices: tuple[str, ...]) -> list[CorpusEntry]:
    entries: list[CorpusEntry] = []
    for voice in voices:
        for mode in MODES:
            run = f"vmb_a2_{voice}_pure_{mode}"
            gens = _load_generations(sources_dir / f"{run}.metadata.json")
            by_cell: dict[tuple[int, int], dict] = {
                (int(g["topic_idx"]), int(g["repetition"])): g for g in gens}
            wanted = ([(t, 0) for t in range(S3_REP0_TOPICS)]
                      + [(t, 1) for t in range(S3_REP1_TOPICS)])
            for t_idx, rep in wanted:
                g = by_cell.get((t_idx, rep))
                if g is None or not g["generated_text"].strip():
                    raise RuntimeError(f"{run}: missing/empty (topic={t_idx}, rep={rep})")
                text = g["generated_text"]
                entries.append(CorpusEntry(
                    text_id=f"S3-{voice}-{mode}-t{t_idx:02d}-r{rep}",
                    stratum="S3", voice=voice, mode=mode,
                    topic_idx=t_idx, topic=g["topic"], repetition=rep,
                    source_run=run, source_generation_id=str(g["generation_id"]),
                    source_seed=str(g["seed"]),
                    system_prompt=g.get("system_prompt", "") or "",
                    user_prompt=g["user_prompt"],
                    text=text, n_words=len(text.split()),
                    n_tokens=_ntokens(tok, text)))
    per_voice = {v: sum(1 for e in entries if e.voice == v) for v in voices}
    if per_voice != {v: 150 for v in voices}:
        raise RuntimeError(f"S3 cardinality wrong: {per_voice}")
    return entries


# ---------------------------------------------------------------- S2 wikitext shards
def _is_heading(line: str) -> bool:
    s = line.strip()
    return s.startswith("=") and s.endswith("=") and len(s) > 1


def _wikitext_shards(parquet_path: Path, tok) -> list[str]:
    """Greedy paragraph-accumulating chunker.

    Cleaning applied (recorded in stamp): drop heading lines (= ... =), drop empty
    lines, strip per-line leading/trailing whitespace, join paragraphs with a blank
    line. No other normalization — raw-v1 text is otherwise verbatim.
    """
    import pyarrow.parquet as pq
    lines = pq.read_table(parquet_path).column("text").to_pylist()

    def toklen(t: str) -> int:
        if tok is not None:
            return len(tok.encode(t, add_special_tokens=False))
        return max(1, round(len(t.split()) * 1.3))  # heuristic fallback, stamped

    shards: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for line in lines:
        if _is_heading(line):        # new section: close a valid in-flight shard
            if cur_len >= S2_TOK_MIN:
                shards.append("\n\n".join(cur))
            cur, cur_len = [], 0
            continue
        para = line.strip()
        if not para:
            continue
        plen = toklen(para)
        if cur_len + plen > S2_TOK_MAX:
            if cur_len >= S2_TOK_MIN:
                shards.append("\n\n".join(cur))
            # oversize single paragraphs are skipped rather than truncated
            cur, cur_len = ([para], plen) if plen <= S2_TOK_MAX else ([], 0)
            continue
        cur.append(para)
        cur_len += plen
        if cur_len >= S2_TOK_CLOSE:
            shards.append("\n\n".join(cur))
            cur, cur_len = [], 0
    if cur_len >= S2_TOK_MIN:
        shards.append("\n\n".join(cur))
    return shards


def build_s2(tok, wikitext_parquet: Path) -> tuple[list[CorpusEntry], dict]:
    shards = _wikitext_shards(wikitext_parquet, tok)
    if len(shards) < S2_N_SHARDS:
        raise RuntimeError(f"wikitext produced only {len(shards)} shards "
                           f"(need {S2_N_SHARDS})")
    rng = np.random.default_rng(A8_SEED)
    idx = np.sort(rng.choice(len(shards), size=S2_N_SHARDS, replace=False))
    entries = []
    for rank, i in enumerate(idx):
        text = shards[int(i)]
        entries.append(CorpusEntry(
            text_id=f"S2-wt-{rank:03d}",
            stratum="S2", voice="neutral", mode="neutral",
            source_run="wikitext-103-raw-v1:validation",
            source_generation_id=f"shard_{int(i)}",
            user_prompt=S2_CARRIER_PROMPT,
            text=text, n_words=len(text.split()),
            n_tokens=_ntokens(tok, text)))
    info = {"total_shards_available": len(shards), "selection_seed": A8_SEED,
            "chosen_shard_indices_sha256": hashlib.sha256(
                json.dumps([int(i) for i in idx]).encode()).hexdigest()}
    return entries, info


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm-root", type=Path, default=DEFAULT_ARM_ROOT)
    ap.add_argument("--voices", default="3b,8b",
                    help="comma pair of model voices, e.g. 8b,qwen-7b for Leg 1")
    ap.add_argument("--wikitext-parquet", type=Path, default=None,
                    help="override the default HF-cache glob")
    args = ap.parse_args()
    voices = tuple(v.strip() for v in args.voices.split(",") if v.strip())
    bad = [v for v in voices if v not in STAGE0_RUN]
    if bad or len(voices) != 2:
        raise SystemExit(f"--voices must be 2 of {sorted(STAGE0_RUN)}; got {voices}")

    corpus_dir = args.arm_root / "corpus"
    sources_dir = corpus_dir / "sources"
    if not sources_dir.is_dir():
        raise SystemExit(f"sources dir missing: {sources_dir} — rsync the metadata first")

    wt = args.wikitext_parquet
    if wt is None:
        import glob
        hits = glob.glob(str(Path(WIKITEXT_GLOB).expanduser()))
        if not hits:
            raise SystemExit("wikitext-103-raw-v1 validation parquet not found in HF "
                             "cache; pass --wikitext-parquet")
        wt = Path(sorted(hits)[0])
    logger.info("wikitext parquet: %s", wt)

    tok = _get_tokenizer()
    s1 = build_s1(sources_dir, tok, voices)
    s3 = build_s3(sources_dir, tok, voices)
    s2, s2_info = build_s2(tok, wt)
    entries = s1 + s2 + s3
    ids = [e.text_id for e in entries]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate text_ids")

    manifest_path = corpus_dir / "corpus_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"entries": [e.model_dump() for e in entries]}, f, indent=1)

    tok_counts = [e.n_tokens for e in entries if e.n_tokens is not None]
    s2_tok = [e.n_tokens for e in s2 if e.n_tokens is not None]
    stamp = {
        "arm": "A8_conjugation", "leg": 0, "builder": "a8_build_corpus.py",
        "prereg_tag": "prereg-arm8-v1",
        "voices": list(voices),
        "counts": {"S1": {v: 160 for v in voices}, "S2": 160,
                   "S3": {v: 150 for v in voices}, "total": len(entries)},
        "selection_rules": {
            "S1": "per (topic x task-stratum) cell: 2 lowest repetitions, non-empty",
            "S2": f"wikitext-103-raw-v1 validation, greedy paragraph chunker "
                  f"[{S2_TOK_MIN},{S2_TOK_MAX}] tok (close at {S2_TOK_CLOSE}), "
                  f"fixed-seed sample of {S2_N_SHARDS}; cleaning = drop headings/"
                  f"empty lines, strip line whitespace, paragraphs joined by blank line",
            "S3": "per mode: rep 0 topics 0-19 + rep 1 topics 0-9 (30/mode/voice)"},
        "s2": {**s2_info,
               "carrier_prompt_native_arm": S2_CARRIER_PROMPT,
               "carrier_note": "PARKED for desk review at CP-1 — constant expository-"
                               "template carrier; alternative is raw-arm-only S2",
               "token_range_observed": [int(min(s2_tok)), int(max(s2_tok))] if s2_tok else None},
        "tokenizer": TOKENIZER_REF if tok is not None else
                     "UNAVAILABLE — word-count heuristic used for S2 chunking",
        "selection_seed_s2": A8_SEED,
        "source_sha256": {p.name: _sha256(p)
                          for p in sorted(sources_dir.glob("*.metadata.json"))},
        "wikitext_parquet": {"path": str(wt), "sha256": _sha256(wt)},
        "manifest_sha256": _sha256(manifest_path),
        "token_count_summary": {
            "min": int(min(tok_counts)), "max": int(max(tok_counts)),
            "mean": round(float(np.mean(tok_counts)), 1)} if tok_counts else None,
    }
    stamp_path = corpus_dir / "corpus_stamp.json"
    with open(stamp_path, "w") as f:
        json.dump(stamp, f, indent=1)
    logger.info("manifest: %s (%d entries)  stamp: %s",
                manifest_path, len(entries), stamp_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
