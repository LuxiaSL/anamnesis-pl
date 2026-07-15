"""V1b — topic-disjoint formality vector build (session-5 item 4; imago-B5 leakage control).

V1 was built from a contrast set overlapping the 20 mode-eval topics (A5 review §B3: topic-
conditioned register leakage). V1b = the SAME formality construction (formal − informal system
prompt, mean generated-position residual diff per site) on the 40 HELD-OUT topics (set_c+set_d,
hard-disjoint from eval — banked in v1b_topics_manifest.json). NEW vector key (V1b_L*), banked to
its OWN npz — NEVER overwrites V1. The 2AFC steering re-run is post-window (CPU+judge); this fires
the build + norms + a light on-policy coherence gate. First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from anamnesis.config import MODEL_PRESETS
from anamnesis.scripts.vmb_a5_build_vectors import SITES, build_norms, build_v1

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--manifest", type=Path, required=True, help="v1b_topics_manifest.json")
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--v1-max-new-tokens", type=int, default=160)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    preset = MODEL_PRESETS[args.model]

    manifest = json.loads(args.manifest.read_text())
    topics = manifest["v1b_topics"] if "v1b_topics" in manifest else manifest.get("topics")
    assert topics and len(topics) >= 20, f"V1b manifest bad: {len(topics or [])} topics"
    logger.info(f"V1b: {len(topics)} disjoint topics (disjoint_asserted={manifest.get('disjoint_asserted', '?')})")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=preset.torch_dtype, attn_implementation="eager").to("cuda").eval()

    vec, stamps = build_v1(model, tok, topics, args.v1_max_new_tokens, preset)
    # rename V1_L* -> V1b_L*, tag provenance
    vectors = {k.replace("V1_", "V1b_"): v for k, v in vec.items()}
    stamps = {k.replace("V1_", "V1b_"): {**s, "trait": "formality-topic-disjoint",
                                         "route": "contrastive-prompt (set_c+set_d, eval-disjoint)",
                                         "imago_B5_control": True, "n_topics": len(topics)}
              for k, s in stamps.items()}
    norms = build_norms(model, args.stage0_run)
    for s in SITES:
        if f"V1b_L{s}" in stamps:
            stamps[f"V1b_L{s}"]["median_site_norm"] = float(norms.get(f"L{s}", 0.0))

    np.savez(args.out_dir / "a5_vectors.npz", **vectors)
    (args.out_dir / "a5_vectors_stamps.json").write_text(
        json.dumps({"median_resid_norms": norms, "vectors": stamps,
                    "provenance": "V1b topic-disjoint formality control (imago-B5); NEVER V1"}, indent=1))
    logger.info(f"banked V1b vectors {sorted(vectors)} -> {args.out_dir}")
    for k, s in stamps.items():
        logger.info(f"  {k}: raw_norm={s.get('raw_norm'):.4f} n_pairs={s.get('n_pairs')} site_norm={s.get('median_site_norm'):.3f}")


if __name__ == "__main__":
    main()
