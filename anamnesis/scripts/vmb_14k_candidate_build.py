"""14k — build a candidate data-route vector for the coordinate-generality battery
(session-8 Part B6; ASSAYS ONLY — this builds the vector, no steering cells).

Same V3 recipe (mode-dir0 Route 1): same-topic mean generated-position residual diff between
a PAIR of banked pure-mode corpora, per site. Parameterized pair so the roster's candidates
build from one path: (a) socratic↔linear (banked pure corpora) is the primary. Banks
`{key}_L{site}` unit vectors + median site norms → the write-anatomy assay
(`vmb_14k_shape_assay.py`) then prices needle-vs-field on the banked Σ. GPU. First-read → outer loop.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from anamnesis.config import MODEL_PRESETS
from anamnesis.scripts.vmb_a5_build_vectors import _mean_resid_at_sites

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--a2-root", type=Path, required=True, help="dir holding vmb_a2_<model>_pure_*")
    ap.add_argument("--pair", nargs=2, required=True, help="two pure-mode names, e.g. socratic linear")
    ap.add_argument("--key", required=True, help="output vector key prefix, e.g. Ksoclin")
    ap.add_argument("--sites", type=int, nargs="+", default=[7, 14, 18, 21])
    ap.add_argument("--per-topic", type=int, default=2)
    ap.add_argument("--out-npz", type=Path, required=True)
    ap.add_argument("--out-stamps", type=Path, required=True)
    args = ap.parse_args()
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM
    preset = MODEL_PRESETS[args.model]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=preset.torch_dtype, attn_implementation="eager").to("cuda").eval()
    dev = next(model.parameters()).device

    manifests, metas = {}, {}
    for m in args.pair:
        rd = args.a2_root / f"vmb_a2_{args.model}_pure_{m}"
        manifests[m] = json.loads((rd / "replay_manifest.json").read_text())["entries"]
        md = json.loads((rd / "metadata.json").read_text())
        gens = md["generations"] if "generations" in md else md
        by_t: dict[int, list] = {}
        for g in gens:
            by_t.setdefault(int(g["topic_idx"]), []).append(g)
        metas[m] = by_t

    common = sorted(set(metas[args.pair[0]]) & set(metas[args.pair[1]]))
    diffs = {s: [] for s in args.sites}
    norms = {s: [] for s in args.sites}
    n_pairs = 0
    for t in common:
        means = {}
        ok = True
        for m in args.pair:
            gens = sorted(metas[m][t], key=lambda g: int(g["generation_id"]))[: args.per_topic]
            sm = []
            for g in gens:
                e = manifests[m].get(str(g["generation_id"]))
                if e is None or (len(e["input_ids"]) - e["prompt_length"]) < 8:
                    continue
                ids = torch.tensor([e["input_ids"]], dtype=torch.long, device=dev)
                sm.append(_mean_resid_at_sites(model, ids, int(e["prompt_length"]), args.sites))
            if not sm:
                ok = False
                break
            means[m] = {s: np.mean([x[s] for x in sm], axis=0) for s in args.sites}
            for s in args.sites:
                norms[s].append(float(np.linalg.norm(means[m][s])))
        if not ok:
            continue
        for s in args.sites:
            diffs[s].append(means[args.pair[0]][s] - means[args.pair[1]][s])
        n_pairs += 1

    vectors, stamps = {}, {"median_resid_norms": {}, "vectors": {}}
    for s in args.sites:
        v = np.mean(diffs[s], axis=0)
        raw = float(np.linalg.norm(v))
        vectors[f"{args.key}_L{s}"] = (v / raw).astype(np.float32)
        stamps["median_resid_norms"][f"L{s}"] = float(np.median(norms[s]))
        stamps["vectors"][f"{args.key}_L{s}"] = {
            "trait": f"mode-pair {args.pair[0]}-{args.pair[1]}", "route": "contrastive-pure-corpora",
            "n_pairs": n_pairs, "raw_norm": raw, "median_site_norm": float(np.median(norms[s]))}
    np.savez(args.out_npz, **vectors)
    args.out_stamps.write_text(json.dumps(stamps, indent=1))
    logger.info(f"built {list(vectors)} from {n_pairs} topic-pairs → {args.out_npz}")


if __name__ == "__main__":
    main()
