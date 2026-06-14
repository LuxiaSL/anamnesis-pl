"""C5: fit per-layer PCA on positionally-CORRECTED calibration states.

The v2/v3 T3 used ONE pooled PCA fit on UNcorrected states but applied to corrected
states (fit/apply mismatch, audit C5). This re-fits a SEPARATE PCA per pca_layer on
states corrected with the EXISTING positional_means (left unchanged, so the v3 sigs'
positional correction stays consistent — only the PCA basis changes).

GPU: generate over the 50 fixed calibration prompts, collect gen-position hidden states
at pca_layers, subtract positional_means, fit per-layer PCA.

Output: {calib_dir}/pca_model_corrected.pkl = {layer_idx: {components, mean, evr}}.

Usage:
    python -m anamnesis.scripts.run_corrected_pca --model 3b \
        --model-path /models/llama-3.2-3b-instruct \
        --calib-dir /models/anamnesis-extract/calibration/3b
"""

from __future__ import annotations

import argparse
import gc
import logging
import pickle
from pathlib import Path

import numpy as np
import torch

from anamnesis.config import ExtractionConfig, MODEL_PRESETS, ModelConfig
from anamnesis.extraction.model_loader import load_model
from anamnesis.scripts.run_8b_calibration import CALIBRATION_PROMPTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit per-layer PCA on positionally-corrected calib states (C5)")
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--n-components", type=int, default=None, help="default: ExtractionConfig.pca_components")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--num-prompts", type=int, default=None)
    args = ap.parse_args()

    from sklearn.decomposition import PCA

    p = MODEL_PRESETS[args.model]
    n_comp = args.n_components if args.n_components is not None else ExtractionConfig().pca_components

    pm_path = args.calib_dir / "positional_means.npz"
    pos_means = np.load(pm_path)["positional_means"].astype(np.float32)  # [num_layers+1, max_pos, hidden]
    max_pos = pos_means.shape[1]
    logger.info(f"positional_means {pos_means.shape} (UNCHANGED — reused for fit consistency)")

    mc = ModelConfig(model_id=args.model_path, torch_dtype=p.torch_dtype, num_layers=p.num_layers,
                     hidden_dim=p.hidden_dim, num_attention_heads=p.num_attention_heads,
                     num_kv_heads=p.num_kv_heads, head_dim=p.head_dim)
    loaded = load_model(mc, sampled_layers=[])  # no hooks needed for calibration
    loaded.disable_hooks()
    dev = next(loaded.model.parameters()).device
    tok = loaded.tokenizer

    prompts = CALIBRATION_PROMPTS if args.num_prompts is None else CALIBRATION_PROMPTS[:args.num_prompts]
    samples: dict[int, list[np.ndarray]] = {l: [] for l in p.pca_layers}

    for pi, text in enumerate(prompts):
        res = tok.apply_chat_template([{"role": "user", "content": text}],
                                      add_generation_prompt=True, return_tensors="pt")
        ids = res if torch.is_tensor(res) else res["input_ids"]
        ids = ids.to(dev)
        plen = int(ids.shape[1])
        torch.manual_seed(pi)
        with torch.no_grad():
            out = loaded.model.generate(
                ids, max_new_tokens=args.max_new_tokens, temperature=p.temperature, top_p=0.9,
                do_sample=True, eos_token_id=p.eos_token_ids,
                output_hidden_states=True, return_dict_in_generate=True,
            )
        ngen = len(out.hidden_states) - 1
        if ngen > 0:
            sample_t = sorted({1, max(1, ngen // 2), ngen})
            for t in sample_t:
                if t >= len(out.hidden_states):
                    continue
                abs_pos = plen + t - 1
                if abs_pos >= max_pos:
                    continue
                for l in p.pca_layers:
                    if l + 1 < len(out.hidden_states[t]):
                        h = out.hidden_states[t][l + 1][0, -1].cpu().float().numpy()
                        samples[l].append(h - pos_means[l + 1, abs_pos])  # positionally corrected
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if (pi + 1) % 10 == 0:
            logger.info(f"calibration {pi + 1}/{len(prompts)}")

    out_model: dict[int, dict] = {}
    for l in p.pca_layers:
        M = np.stack(samples[l]).astype(np.float64)
        ncomp = min(n_comp, M.shape[0], M.shape[1])
        pca = PCA(n_components=ncomp).fit(M)
        out_model[int(l)] = {
            "components": pca.components_.astype(np.float32),
            "mean": pca.mean_.astype(np.float32),
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
        logger.info(f"L{l}: {M.shape[0]} corrected samples → PCA {pca.components_.shape}, "
                    f"evr_sum={pca.explained_variance_ratio_.sum():.3f}")

    out_path = args.calib_dir / "pca_model_corrected.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(out_model, f)
    logger.info(f"Saved per-layer corrected PCA ({len(out_model)} layers) → {out_path}")


if __name__ == "__main__":
    main()
