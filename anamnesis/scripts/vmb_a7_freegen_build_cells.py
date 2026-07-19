"""Build the A7 FREE-GEN ladder cells-json for run_gen_tokens --jobs-file.

The A7 teacher-forced (TF) ladder perturbs router routing during REPLAY over existing tokens;
the free-gen ladder GENERATES fresh under the same MoE perturbations to read coherence-vs-dose in
the model's own sampled text (the .60 coherence-secondary row). This builder draws n topic-matched
generation specs from a source run's metadata and emits one cell per ladder rung (same specs across
rungs → text differs ONLY by the perturbation, so coherence is directly comparable), each carrying
its rung's `perturb` dict (or null for a true baseline).

Canonical rung ladder (matches the banked TF ladder `arms/A7_dsv2/tf`):
  topk 6/4/2/1 · noise eps 0/.25/.5/1 · shared_ablate · routed_ablate · drop_topm m2 · drop_randm m2.
`topk6` and `noise eps0` are IDENTITY rungs (routing unchanged) — the free-gen baselines / smoke controls.
Noise rungs need the per-layer router-logit sigma (--noise-sigma-json, the A7 pilot's sigma_logit map);
omitted → noise rungs are skipped (topk/ablate/drop rungs need no sigma; the smoke uses those).

Pure stdlib (CPU). First-read -> outer loop; nothing stamped.

    python -m anamnesis.scripts.vmb_a7_freegen_build_cells \
        --source-run $RUNS/vmb_stage0_dsv2_lite --n 80 --out-root $RUNS/vmb_a7_dsv2_freegen \
        --noise-sigma-json $BANK/arms/A7_dsv2/a7_sigma_logit.json \
        --out-jobs $BANK/arms/A7_dsv2/freegen/cells.json
    # smoke: --n 8 --rungs baseline,routed_ablate,topk1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SPEC_FIELDS = ("generation_id", "prompt_set", "topic", "topic_idx", "mode",
               "mode_idx", "system_prompt", "user_prompt", "seed", "repetition")

# label -> perturb dict (None = true baseline; noise rungs get sigma_logit filled at build)
LADDER: dict[str, dict | None] = {
    "baseline": None,
    "topk6": {"mode": "topk", "top_k": 6},        # identity (default k) — routing-path control
    "topk4": {"mode": "topk", "top_k": 4},
    "topk2": {"mode": "topk", "top_k": 2},
    "topk1": {"mode": "topk", "top_k": 1},
    "noise0.0": {"mode": "noise", "eps": 0.0, "seed": 42},   # identity
    "noise0.25": {"mode": "noise", "eps": 0.25, "seed": 42},
    "noise0.5": {"mode": "noise", "eps": 0.5, "seed": 42},
    "noise1.0": {"mode": "noise", "eps": 1.0, "seed": 42},
    "shared_ablate": {"mode": "shared_ablate"},
    "routed_ablate": {"mode": "routed_ablate"},
    "drop_topm2": {"mode": "drop_topm", "m": 2, "seed": 42},
    "drop_randm2": {"mode": "drop_randm", "m": 2, "seed": 42},
}


def select_specs(meta_path: Path, n: int) -> list[dict]:
    """n topic-balanced specs from a run's metadata.json generations."""
    raw = json.loads(meta_path.read_text())
    gens = raw["generations"] if "generations" in raw else raw
    by_topic: dict[int, list[dict]] = {}
    for g in gens:
        if not all(f in g for f in ("user_prompt", "seed", "topic_idx")):
            continue
        by_topic.setdefault(int(g["topic_idx"]), []).append(g)
    topics = sorted(by_topic)
    if not topics:
        raise SystemExit(f"no usable specs in {meta_path}")
    picked: list[dict] = []
    ti = 0
    # round-robin across topics until n reached (balanced coverage)
    while len(picked) < n:
        t = topics[ti % len(topics)]
        bucket = by_topic[t]
        used = sum(1 for p in picked if int(p["topic_idx"]) == t)
        if used < len(bucket):
            g = bucket[used]
            picked.append({f: g.get(f) for f in SPEC_FIELDS})
        ti += 1
        if ti > len(topics) * (max(len(b) for b in by_topic.values()) + 1):
            break  # exhausted
    if len(picked) < n:
        raise SystemExit(f"only {len(picked)} specs available, need {n}")
    return picked[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run", type=Path, required=True, help="run dir with metadata.json")
    ap.add_argument("--n", type=int, required=True, help="gens per rung (topic-balanced)")
    ap.add_argument("--out-root", type=Path, required=True, help="run root; cell out_dir=root/<rung>")
    ap.add_argument("--out-jobs", type=Path, required=True, help="cells-json for run_gen_tokens --jobs-file")
    ap.add_argument("--rungs", default=None,
                    help="comma-separated rung labels (default: all applicable). "
                         "e.g. 'baseline,routed_ablate,topk1' for the smoke")
    ap.add_argument("--noise-sigma-json", type=Path, default=None,
                    help="json {layer_idx: sigma} for noise rungs; absent => noise rungs skipped")
    args = ap.parse_args()
    args.out_jobs.parent.mkdir(parents=True, exist_ok=True)

    sigma = None
    if args.noise_sigma_json and args.noise_sigma_json.exists():
        sigma = {str(k): float(v) for k, v in json.loads(args.noise_sigma_json.read_text()).items()}

    want = ([r.strip() for r in args.rungs.split(",")] if args.rungs else list(LADDER))
    specs = select_specs(args.source_run / "metadata.json", args.n)

    jobs: list[dict] = []
    skipped: list[str] = []
    for label in want:
        if label not in LADDER:
            raise SystemExit(f"unknown rung {label!r}; valid: {list(LADDER)}")
        perturb = LADDER[label]
        if perturb and perturb.get("mode") == "noise":
            if sigma is None:
                skipped.append(label)
                continue
            perturb = {**perturb, "sigma_logit": sigma}
        jobs.append({"out_dir": str(args.out_root / label),
                     "specs": specs, "perturb": perturb})
    args.out_jobs.write_text(json.dumps(jobs, indent=1))
    print(f"wrote {len(jobs)} cells (n={args.n} each) -> {args.out_jobs}")
    print(f"  rungs: {[j['out_dir'].split('/')[-1] for j in jobs]}")
    if skipped:
        print(f"  SKIPPED (no --noise-sigma-json): {skipped}")


if __name__ == "__main__":
    main()
