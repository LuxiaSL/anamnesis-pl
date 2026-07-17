"""ANNEX — blind 2AFC pair builder + scorer for the transcript judge pass (ledger S8-12;
JT-1..4 frozen BEFORE any pair is read).

build: for each class, sample topic-matched pairs (steered gen vs alpha=0 rider gen on the
SAME topic), randomize A/B order (seeded), write pairs.md (texts only, no labels) and
key.json (pair_id -> steered side). The judge reads pairs.md ONLY; the key is opened by
--score after choices are recorded.

score: takes choices.json {pair_id: "A"|"B", ...} + key.json, emits per-class correct counts.

Usage (from pipeline/):
    python -m anamnesis.scripts.annex_2afc_pairs build --out-dir ../outputs/battery/annex/14r_judge
    python -m anamnesis.scripts.annex_2afc_pairs score --out-dir ../outputs/battery/annex/14r_judge \
        --choices ../outputs/battery/annex/14r_judge/choices.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

BATTERY = Path("../outputs/battery")
RIDER = BATTERY / "vmb_a5_3b/R1_L14_a0.0"
CLASSES = {
    "V3_a0.3": BATTERY / "vmb_a5_3b/V3_L14_L14_a0.3",
    "RA_a0.3": BATTERY / "annex/vmb_14r_3b/RA_L14_a0.3",
    "RA_a0.1": BATTERY / "annex/vmb_14r_3b/RA_L14_a0.1",
    "Rband1_a0.3": BATTERY / "vmb_b7_3b/Rband1_L14_a0.3",
}
N_PAIRS = 10
TRUNC = 900
SEED = 20260716


def _gens(d: Path) -> list[dict]:
    md = json.loads((d / "metadata.json").read_text())
    return md["generations"] if "generations" in md else md


def build(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    riders = _gens(RIDER)
    by_topic: dict[str, list[dict]] = {}
    for g in riders:
        by_topic.setdefault(g["topic"], []).append(g)

    key, blocks = {}, []
    pid = 0
    for cls, d in CLASSES.items():
        gens = [g for g in _gens(d) if g["topic"] in by_topic]
        rng.shuffle(gens)
        for g in gens[:N_PAIRS]:
            r = rng.choice(by_topic[g["topic"]])
            steered_side = rng.choice("AB")
            a, b = (g, r) if steered_side == "A" else (r, g)
            pid += 1
            key[str(pid)] = {"class": cls, "steered": steered_side,
                             "steered_gid": g["generation_id"], "rider_gid": r["generation_id"]}
            blocks.append(
                f"## PAIR {pid}  (topic: {g['topic']})\n\n"
                f"### A\n{a['generated_text'][:TRUNC]}\n\n"
                f"### B\n{b['generated_text'][:TRUNC]}\n")
    rng.shuffle(blocks)
    (out_dir / "pairs.md").write_text(
        "# BLIND 2AFC — which of A/B is more ANALOGICAL in mode?\n"
        "# (record {pair: 'A'|'B'} for every pair; annotate any texture difference)\n\n"
        + "\n".join(blocks))
    (out_dir / "key.json").write_text(json.dumps(key, indent=1))
    print(f"built {pid} pairs -> {out_dir}/pairs.md (key sealed in key.json — do not open before choices)")


def score(out_dir: Path, choices_path: Path) -> None:
    key = json.loads((out_dir / "key.json").read_text())
    choices = json.loads(choices_path.read_text())
    per: dict[str, list[int]] = {}
    for pid, k in key.items():
        c = choices.get(pid)
        if c is None:
            continue
        per.setdefault(k["class"], []).append(1 if c == k["steered"] else 0)
    out = {cls: {"correct": sum(v), "n": len(v)} for cls, v in per.items()}
    (out_dir / "judge_scores.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "score"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--choices", type=Path)
    args = ap.parse_args()
    if args.mode == "build":
        build(args.out_dir)
    else:
        score(args.out_dir, args.choices)
