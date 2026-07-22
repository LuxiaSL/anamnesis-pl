"""A8 Leg-4 / L4-d step 1-2 — build the S5-augmented fit corpus for the 8B<->Qwen pair.

S5 = UNCAPPED completions (max_new_tokens 2048, natural terminations), both voices of
the pair, generated from S1's own prompt set (stage-0 protocol, fresh seeds).  Its
purpose is the desk's interim-note-2 diagnosis: the Leg-0/1 fit corpora are
TERMINATION-CENSORED (620/780 texts sit at the 512 cap), so eos-perp-relevant
covariance was censored OUT of the data g was fit on.  S5 restores it.

The leg-4 corpus = the Leg-1 corpus VERBATIM (same 780 entries, same text_ids, so every
prior row stays byte-comparable) + the S5 entries appended.  Stamps carry per-text EOS
metadata (ended_with_eos_token, n_generated_tokens, cap) so the stratum's provenance is
auditable without re-reading the gens.

Everything UNSTAMPED (C§8).

Run (repo root):
  PYTHONPATH=pipeline python -m anamnesis.scripts.a8_leg4_s5_corpus [--selftest]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_leg4_s5_corpus")

ARM = Path("outputs/battery/arms/A8_conjugation")
LEG1 = ARM / "leg1"
LEG4 = ARM / "leg4"
CAP = 2048
EOS_IDS = {"8b": (128001, 128008, 128009), "qwen-7b": (151643, 151645)}
VOICES = ("8b", "qwen-7b")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build_s5(voice: str) -> tuple[list[dict], dict]:
    rec_dir = LEG4 / "s5_gen" / voice / "uncapped" / "gen_records"
    files = sorted(rec_dir.glob("gen_*.json"))
    if not files:
        raise SystemExit(f"no S5 gen records under {rec_dir}")
    entries, n_eos, lens = [], 0, []
    for f in files:
        g = json.loads(f.read_text())
        ids, p = g["input_ids"], g["prompt_length"]
        gen = ids[p:]
        ended = bool(gen and gen[-1] in EOS_IDS[voice])
        n_eos += ended
        lens.append(len(gen))
        text = g["generated_text"].strip()
        if not text:
            continue
        entries.append({
            "text_id": f"S5-{voice}-g{g['generation_id']:03d}",
            "stratum": "S5", "voice": voice, "mode": g["mode"],
            "topic_idx": int(g["topic_idx"]), "topic": g["topic"],
            "repetition": int(g["repetition"]),
            "source_run": f"a8_leg4_s5_{voice}_uncapped",
            "source_generation_id": str(g["generation_id"]),
            "source_seed": str(g["seed"]),
            "system_prompt": g["system_prompt"], "user_prompt": g["user_prompt"],
            "text": text,
            "n_words": len(text.split()),
            "n_tokens": len(gen),
            "ended_with_eos_token": "1" if ended else "0",
        })
    qc = {"voice": voice, "n_records": len(files), "n_entries": len(entries),
          "natural_eos_rate": round(n_eos / len(files), 4),
          "frac_at_cap": round(sum(n >= CAP for n in lens) / len(lens), 4),
          "gen_tokens": {"mean": round(sum(lens) / len(lens), 1),
                         "min": min(lens), "max": max(lens)},
          "max_new_tokens": CAP}
    return entries, qc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    base_bytes = (LEG1 / "corpus/corpus_manifest.json").read_bytes()
    base = json.loads(base_bytes)["entries"]
    s5, qc = [], []
    for v in VOICES:
        e, q = build_s5(v)
        s5 += e
        qc.append(q)
    entries = base + s5
    ids = [e["text_id"] for e in entries]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate text_ids after augmentation")

    (LEG4 / "corpus").mkdir(parents=True, exist_ok=True)
    man = json.dumps({"entries": entries}, indent=1)
    (LEG4 / "corpus/corpus_manifest.json").write_text(man)
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["stratum"]] = counts.get(e["stratum"], 0) + 1
    stamp = {
        "STATUS": "UNSTAMPED (C§8)",
        "leg": "A8 Leg-4 (L4-d) — S5-augmented 8B<->Qwen fit corpus",
        "prereg": "A8-add-2 P8-L4d",
        "base_corpus": str(LEG1 / "corpus/corpus_manifest.json"),
        "base_corpus_sha256": _sha(base_bytes),
        "base_entries_verbatim": len(base),
        "manifest_sha256": _sha(man.encode()),
        "counts_per_stratum": counts,
        "s5_construction": {
            "protocol": "stage-0 (20 topics x 4 task strata x 2 seeds/class) = 160/voice; "
                        "prompts = pipeline/anamnesis/prompts/prompt_sets.json (S1's set)",
            "generation": "vmb_a5_gen_multicell, no injection, max_new_tokens 2048, "
                          "attn eager, model presets' temperature/top_p, canonical date",
            "seed_namespaces": [f"A8L4S5-{v}" for v in VOICES],
            "why": "desk interim note 2: the Leg-0/1 corpora are termination-censored "
                   "(~90% of S1/S3 at the 512 cap), so eos-perp covariance was censored "
                   "out of the fit data. S5 is the uncensored stratum.",
            "eos_ids": {v: list(EOS_IDS[v]) for v in VOICES},
        },
        "s5_qc": qc,
        "gate_add2": {
            "requirement": ">=150/voice with >=80% natural-EOS (one 4096 retry "
                           "pre-authorized; otherwise the row scores CONDITION-UNMET)",
            "achieved": {q["voice"]: {"n": q["n_entries"],
                                      "natural_eos_rate": q["natural_eos_rate"]}
                         for q in qc},
        },
    }
    (LEG4 / "corpus/corpus_stamp.json").write_text(json.dumps(stamp, indent=1))
    logger.info("corpus -> %s (%d entries; %s)", LEG4 / "corpus/corpus_manifest.json",
                len(entries), counts)
    logger.info("gate: %s", json.dumps(stamp["gate_add2"]["achieved"]))
    return 0


def selftest() -> int:
    ok = True
    for v in VOICES:
        e, q = build_s5(v)
        cond = q["natural_eos_rate"] >= 0.8 and q["n_entries"] >= 150
        ok &= cond
        print(f"[{'OK' if cond else 'BAD'}] {v}: n={q['n_entries']} "
              f"eos={q['natural_eos_rate']:.1%} at_cap={q['frac_at_cap']:.1%} "
              f"tokens mean {q['gen_tokens']['mean']}")
        ids = {x["text_id"] for x in e}
        ok &= len(ids) == len(e)
    base = json.loads((LEG1 / "corpus/corpus_manifest.json").read_text())["entries"]
    print(f"[OK] base corpus rows carried verbatim: {len(base)}")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
