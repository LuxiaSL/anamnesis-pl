"""V3sel-BARE capture (WAVE2-A5 addendum 14c; WINDOW item 4).

The V3sel-BARE selection is PRE-BANKED (`arms/A5/v3sel_bare_selection_3b.json`,
`vmb_v3sel_select.py --corpus bare`): within-topic dir0 deciles over the UNPROMPTED
Stage-0 pool (system_prompt='' — the dir0 map is the ONLY labeler; poles mix all 4
task strata, purity 0.463 = genuinely non-degenerate). This does the GPU capture:
build the steering vector as the mean-diff of residuals over the selected poles
(same construction as V3, but the pole membership came from label-free dir0 deciles,
not a mode prompt) → V3sel_bare_L{site}.

This is the STRONGER head-to-head vs V3 (no text label anywhere in the loop; same
epistemic shape as the synthetic-temperature cell). Downstream: free-gen at L14
α∈{.03,.1,.3} then A5-inv V3sel-bare vs V3 vs V4. A5-class → first-read to outer loop.

Reuses `_mean_resid_at_sites` + `build_norms` from vmb_a5_build_vectors verbatim (same
hidden_states[site] = residual-input convention the A5 injection uses).

Run (node1, 1 GPU):
    python -m anamnesis.scripts.vmb_a5_build_v3sel --model 3b \
      --model-path /models/llama-3.2-3b-instruct \
      --selection <arms/A5/v3sel_bare_selection_3b.json> \
      --stage0-run /models/anamnesis-extract/runs/vmb_stage0_3b \
      --out-dir /models/anamnesis-extract/battery/a5_vectors_3b_v3sel
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from anamnesis.config import MODEL_PRESETS
from anamnesis.scripts.vmb_a5_build_vectors import SITES, _mean_resid_at_sites, build_norms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _pole_mean(model, entries, pole: list[dict]) -> dict[int, np.ndarray]:
    """Mean residual at SITES over the selected pole's generated positions."""
    device = next(model.parameters()).device
    acc: dict[int, list[np.ndarray]] = {s: [] for s in SITES}
    for rec in pole:
        gid = int(rec["gen_id"])
        e = entries.get(str(gid))
        if e is None:
            logger.warning(f"gen {gid} not in stage0 manifest — skipped")
            continue
        ids = torch.tensor([e["input_ids"]], dtype=torch.long, device=device)
        r = _mean_resid_at_sites(model, ids, int(e["prompt_length"]), SITES)
        for s in SITES:
            acc[s].append(r[s])
    return {s: np.mean(acc[s], axis=0) for s in SITES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pole-source", default="within_topic",
                    help="which selection block to use (record = within_topic)")
    ap.add_argument("--sites", default=None,
                    help="comma-separated per-model injection sites (default = the 3B "
                         "[7,14,18,21] imported from build_vectors; 8B/Qwen pass their own)")
    args = ap.parse_args()

    if args.sites:
        global SITES
        SITES = [int(x) for x in args.sites.split(",")]
        # build_norms (imported) reads build_vectors' OWN module global — set it too,
        # else norms are computed at the default 3B sites and the L{s} lookup KeyErrors.
        import anamnesis.scripts.vmb_a5_build_vectors as _bv
        _bv.SITES = SITES

    args.out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM

    preset = MODEL_PRESETS[args.model]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=preset.torch_dtype, attn_implementation="eager").to("cuda").eval()

    sel = json.loads(args.selection.read_text())
    block = sel[args.pole_source]
    top, bot = block["top_pole_selected"], block["bottom_pole_selected"]
    # HARD gate (14c/14a §2): the selection asserted no-induced + unprompted; re-affirm here.
    if not sel.get("no_induced_asserted"):
        raise SystemExit("selection manifest missing no_induced_asserted gate — refuse to build")
    if sel.get("corpus") != "bare":
        raise SystemExit(f"expected corpus=bare (14c), got {sel.get('corpus')!r}")
    logger.info(f"poles: top n={len(top)} bottom n={len(bot)} "
                f"(purity {block.get('top_pole_purity')}, {block.get('top_pole_n_topics')} topics)")

    entries = json.loads((args.stage0_run / "replay_manifest.json").read_text())["entries"]
    top_mean = _pole_mean(model, entries, top)
    bot_mean = _pole_mean(model, entries, bot)

    vectors, stamps = {}, {}
    for s in SITES:
        v = top_mean[s] - bot_mean[s]
        raw = float(np.linalg.norm(v))
        vectors[f"V3sel_bare_L{s}"] = (v / max(raw, 1e-12)).astype(np.float32)
        stamps[f"V3sel_bare_L{s}"] = {
            "trait": "mode-dir0", "route": "map-route1-LABEL-FREE-decile (14c V3sel-BARE)",
            "pole_source": args.pole_source, "n_top": len(top), "n_bottom": len(bot),
            "top_purity": block.get("top_pole_purity"), "raw_norm": raw}
    logger.info("V3sel-bare vectors: " + ", ".join(
        f"L{s} raw_norm {stamps[f'V3sel_bare_L{s}']['raw_norm']:.4f}" for s in SITES))

    norms = build_norms(model, args.stage0_run)   # {"L7":.., "L14":.., ...}
    for s in SITES:
        stamps[f"V3sel_bare_L{s}"]["median_site_norm"] = norms[f"L{s}"]
    # freegen driver reads stamps["median_resid_norms"]["L<layer>"] (vmb_stage0_generate);
    # alpha = frac × that. Keep per-vector provenance under "vectors".
    stamps_out = {"median_resid_norms": norms, "vectors": stamps}

    np.savez(args.out_dir / "a5_vectors.npz", **vectors)
    (args.out_dir / "a5_vectors_stamps.json").write_text(json.dumps(stamps_out, indent=1))
    (args.out_dir / "v3sel_bare_build_meta.json").write_text(json.dumps({
        "model": args.model, "STATUS": "FIRST_READ_PENDING (C§8)",
        "provenance": "addendum 14c V3sel-BARE capture; label-free within-topic dir0 poles",
        "sites": SITES, "stamps": stamps}, indent=1))
    logger.info(f"banked V3sel-bare vectors + stamps -> {args.out_dir}")


if __name__ == "__main__":
    main()
