"""C1 — synthetic-temperature target V_temp capture (PREFLIGHT §1; session-4 Part C).

V_temp = mean(resid | t09) − mean(resid | t03) at each of the four standard sites,
generated positions only — the V3 recipe, but the pole labels are the SAMPLER KNOB
(generation metadata), nothing text-derived (Route 1b: metadata-labeled activation
contrast; does NOT depend on the V4/gradient route). Reuses `_mean_resid_at_sites` +
`build_norms` verbatim (same hidden_states[site] = residual-input convention the A5
injection uses). One replay-free capture pass over the two banked A1 corpora.

14a §2: both t03/t09 pools predate ALL steering (A1 generation) — no induced entries;
asserted in the stamps. Site rule (PREFLIGHT §1): sites are a READOUT, not a knob —
build at all four, no pre-registered site.

Run (node1, 1 GPU):
    python -m anamnesis.scripts.vmb_ctemp_build --model 3b \
      --model-path /models/llama-3.2-3b-instruct \
      --hot-run  /models/anamnesis-extract/runs/vmb_a1_3b_t09 \
      --cold-run /models/anamnesis-extract/runs/vmb_a1_3b_t03 \
      --stage0-run /models/anamnesis-extract/runs/vmb_stage0_3b \
      --out-dir /models/anamnesis-extract/battery/a5_vectors_3b_ctemp
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


def _corpus_mean(model, entries: dict) -> dict[int, np.ndarray]:
    """Mean residual at SITES over the generated positions of every entry in a corpus."""
    acc: dict[int, list[np.ndarray]] = {s: [] for s in SITES}
    device = next(model.parameters()).device
    for gid, e in entries.items():
        ids = torch.tensor([e["input_ids"]], dtype=torch.long, device=device)
        r = _mean_resid_at_sites(model, ids, int(e["prompt_length"]), SITES)
        for s in SITES:
            acc[s].append(r[s])
    return {s: np.mean(acc[s], axis=0) for s in SITES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--hot-run", type=Path, required=True, help="t09 A1 run dir")
    ap.add_argument("--cold-run", type=Path, required=True, help="t03 A1 run dir")
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sites", default=None,
                    help="comma-separated layer sites; default = 3B [7,14,18,21]. REQUIRED for "
                         "non-3B models (e.g. olmo2-7b '8,16,20,24,28', gemma3-27b '23,35,41'). "
                         "Rebinds SITES in this module AND vmb_a5_build_vectors (build_norms reads it).")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    global SITES
    if args.sites:
        import anamnesis.scripts.vmb_a5_build_vectors as _bv
        SITES = [int(x) for x in args.sites.split(",")]
        _bv.SITES = SITES   # build_norms reads the source-module global, not our import binding

    from transformers import AutoModelForCausalLM

    preset = MODEL_PRESETS[args.model]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=preset.torch_dtype, attn_implementation="eager").to("cuda").eval()

    hot = json.loads((args.hot_run / "replay_manifest.json").read_text())["entries"]
    cold = json.loads((args.cold_run / "replay_manifest.json").read_text())["entries"]
    logger.info(f"hot(t09) n={len(hot)}  cold(t03) n={len(cold)}")
    hot_mean = _corpus_mean(model, hot)
    cold_mean = _corpus_mean(model, cold)

    vectors, stamps = {}, {}
    for s in SITES:
        v = hot_mean[s] - cold_mean[s]
        raw = float(np.linalg.norm(v))
        vectors[f"Vtemp_L{s}"] = (v / max(raw, 1e-12)).astype(np.float32)
        stamps[f"Vtemp_L{s}"] = {
            "trait": "synthetic-temperature (hot-vs-cold sampling state)",
            "route": "Route-1b metadata-labeled activation contrast (t09 − t03, generated positions)",
            "n_hot": len(hot), "n_cold": len(cold), "raw_norm": raw,
            "no_induced_asserted": True,  # both A1 pools predate all steering
            "hot_run": args.hot_run.name, "cold_run": args.cold_run.name}
    logger.info("V_temp vectors: " + ", ".join(
        f"L{s} raw_norm {stamps[f'Vtemp_L{s}']['raw_norm']:.4f}" for s in SITES))

    norms = build_norms(model, args.stage0_run)   # {"L7":.., "L14":.., ...} — α = frac × this
    for s in SITES:
        stamps[f"Vtemp_L{s}"]["median_site_norm"] = norms[f"L{s}"]
    stamps_out = {"median_resid_norms": norms, "vectors": stamps, "sites": SITES,
                  "provenance": "C1 V_temp; PREFLIGHT §1; sites are a readout not a knob; "
                                "labels = sampler knob (metadata), no text contrast in the loop"}

    np.savez(args.out_dir / "a5_vectors.npz", **vectors)
    (args.out_dir / "a5_vectors_stamps.json").write_text(json.dumps(stamps_out, indent=1))
    print(json.dumps({"built": list(vectors.keys()),
                      "raw_norms": {f"L{s}": round(stamps[f'Vtemp_L{s}']['raw_norm'], 4) for s in SITES},
                      "L14_median_norm": round(norms["L14"], 3)}, indent=1))


if __name__ == "__main__":
    main()
