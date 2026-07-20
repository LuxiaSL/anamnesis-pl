"""A5 WHITENED (Mahalanobis/LDA) steering-vector builder — the fix implied by the 2026-07-19
finding that on DSV2-Lite the raw class-mean-difference (CAA / V3) is ~orthogonal (|cos|=0.029 in
signature space) to the discriminative mode direction, so CAA steering has weak purchase even though
the mode is detectable (AUC 0.88). The discriminative direction is the WHITENED mean-difference
w = Σ⁻¹(μ_pos − μ_neg). This builds w in RESIDUAL space at each site and reports:
  cos(Δ, w)         residual-space test of the ⊥ mechanism (low ⇒ CAA misses the discriminative dir)
  mahalanobis_d     sqrt(Δᵀ Σ⁻¹ Δ)  = whitened separation (picks the best steering layer)
  raw_caa_d         ||Δ|| / rms within-class  (the mean-difference separation, for contrast)
The whitened vector is saved unit-normalized as V3w_L{site} into a steering bank (+ median resid norms),
ready for vmb_a5_gen_multicell. Σ estimated by Ledoit–Wolf shrinkage (n<d, so shrinkage mandatory).

First-read → outer loop; nothing stamped.

    python -m anamnesis.scripts.vmb_a5_whiten_steer_build --model dsv2-lite --model-path $MP \
        --runs-root $RUNS --pos-run vmb_a2_dsv2-lite_pure_analogical \
        --neg-run vmb_a2_dsv2-lite_pure_contrastive --stage0-run $RUNS/vmb_stage0_dsv2_lite \
        --sites 15,18,22 --out-dir $BANK/a5_vectors_dsv2_lite_v3whiten
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from anamnesis.config import MODEL_PRESETS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("whiten_steer")


def _capture_resid(model, manifest: Path, sites: list[int], limit: int | None) -> dict[int, np.ndarray]:
    """{site: [n_gens, d]} mean residual over the generated span at hidden_states[site]."""
    entries = json.loads(manifest.read_text())["entries"]
    keys = sorted(entries, key=lambda k: int(k))
    if limit is not None:
        keys = keys[:limit]
    dev = next(model.parameters()).device
    acc: dict[int, list[np.ndarray]] = {s: [] for s in sites}
    for i, k in enumerate(keys):
        e = entries[k]
        ids = torch.tensor([e["input_ids"]], dtype=torch.long, device=dev)
        P = int(e["prompt_length"])
        if ids.shape[1] - P < 1:
            continue
        with torch.no_grad():
            out = model(ids, use_cache=False, output_hidden_states=True, return_dict=True)
        for s in sites:
            h = out.hidden_states[s][0, P:]
            acc[s].append(h.float().mean(0).cpu().numpy().astype(np.float64))
        if (i + 1) % 50 == 0:
            logger.info(f"  {manifest.parent.name}: {i + 1}/{len(keys)}")
    return {s: np.stack(v) for s, v in acc.items()}


def _median_norms(model, manifest: Path, sites: list[int], limit: int) -> dict[int, float]:
    """Median per-token residual L2 norm at each site (for inject_alpha_frac resolution)."""
    entries = json.loads(manifest.read_text())["entries"]
    keys = sorted(entries, key=lambda k: int(k))[:limit]
    dev = next(model.parameters()).device
    norms: dict[int, list[float]] = {s: [] for s in sites}
    for k in keys:
        e = entries[k]
        ids = torch.tensor([e["input_ids"]], dtype=torch.long, device=dev)
        P = int(e["prompt_length"])
        with torch.no_grad():
            out = model(ids, use_cache=False, output_hidden_states=True, return_dict=True)
        for s in sites:
            h = out.hidden_states[s][0, P:]
            norms[s].append(float(h.float().norm(dim=-1).median().cpu()))
    return {s: float(np.median(v)) for s, v in norms.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dsv2-lite")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--pos-run", required=True)
    ap.add_argument("--neg-run", required=True)
    ap.add_argument("--stage0-run", type=Path, required=True, help="for median residual norms")
    ap.add_argument("--sites", required=True, help="comma-separated layer indices")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--norm-limit", type=int, default=60)
    ap.add_argument("--shrink-scale", type=float, default=None,
                    help="WH-2 λ-dependence: scale the Ledoit-Wolf auto shrinkage by this factor "
                         "(e.g. 0.5 / 2.0), clipped to [0,1]; None = the auto value (of record)")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sites = [int(x) for x in args.sites.split(",")]

    from sklearn.covariance import LedoitWolf
    from transformers import AutoModelForCausalLM
    preset = MODEL_PRESETS[args.model]
    logger.info(f"loading {args.model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=preset.torch_dtype, attn_implementation="eager").to("cuda").eval()

    pos = _capture_resid(model, args.runs_root / args.pos_run / "replay_manifest.json", sites, args.limit)
    neg = _capture_resid(model, args.runs_root / args.neg_run / "replay_manifest.json", sites, args.limit)
    med_norms = _median_norms(model, args.stage0_run / "replay_manifest.json", sites, args.norm_limit)

    vectors: dict[str, np.ndarray] = {}
    diag: dict[str, dict] = {}
    for s in sites:
        Xa, Xc = pos[s], neg[s]
        delta = Xa.mean(0) - Xc.mean(0)
        pooled = np.vstack([Xa - Xa.mean(0), Xc - Xc.mean(0)])
        lw = LedoitWolf().fit(pooled)
        Sigma = lw.covariance_
        shrink_used = float(lw.shrinkage_)
        if args.shrink_scale is not None:
            # rebuild Σ at the scaled λ from the same empirical covariance:
            # Σ(λ) = (1−λ)·S_emp + λ·(tr(S_emp)/d)·I  (the Ledoit-Wolf convex form)
            S_emp = np.cov(pooled, rowvar=False, bias=True)
            lam = float(np.clip(lw.shrinkage_ * args.shrink_scale, 0.0, 1.0))
            mu = float(np.trace(S_emp)) / S_emp.shape[0]
            Sigma = (1.0 - lam) * S_emp + lam * mu * np.eye(S_emp.shape[0])
            shrink_used = lam
        w = np.linalg.solve(Sigma, delta)                       # Σ⁻¹ Δ  (whitened / LDA direction)
        wn = w / np.linalg.norm(w)
        dn = delta / np.linalg.norm(delta)
        cos = float(abs(dn @ wn))
        maha = float(np.sqrt(max(delta @ w, 0.0)))              # sqrt(Δᵀ Σ⁻¹ Δ)
        wscat = 0.5 * (np.sqrt((((Xa - Xa.mean(0)) ** 2).sum(1)).mean())
                       + np.sqrt((((Xc - Xc.mean(0)) ** 2).sum(1)).mean()))
        raw_caa_d = float(np.linalg.norm(delta) / wscat) if wscat > 0 else 0.0
        vectors[f"V3w_L{s}"] = wn.astype(np.float32)
        vectors[f"V3raw_L{s}"] = dn.astype(np.float32)          # raw CAA at the SAME capture (control)
        diag[f"L{s}"] = {"cos_delta_whitened": round(cos, 4),
                         "mahalanobis_d": round(maha, 4),
                         "raw_caa_cohend_proxy": round(raw_caa_d, 4),
                         "lw_shrinkage": round(float(lw.shrinkage_), 4),
                         "shrink_used": round(shrink_used, 4),
                         "shrink_scale": args.shrink_scale,
                         "n_pos": int(len(Xa)), "n_neg": int(len(Xc)),
                         "median_resid_norm": round(med_norms[s], 3)}
        logger.info(f"L{s}: cos(Δ,whitened)={cos:.3f}  mahalanobis_d={maha:.3f}  "
                    f"raw_caa_d={raw_caa_d:.3f}  shrink={lw.shrinkage_:.3f}")

    # random unit controls (matched norm via frac), reproducible
    rng = np.random.default_rng(20260719)
    d = next(iter(vectors.values())).shape[0]
    for i in (1, 2, 3):
        r = rng.standard_normal(d); vectors[f"R{i}"] = (r / np.linalg.norm(r)).astype(np.float32)

    np.savez(args.out_dir / "a5_vectors.npz", **vectors)
    stamps = {k: {"kind": "whitened_lda" if k.startswith("V3w") else
                  "raw_caa" if k.startswith("V3raw") else "random"} for k in vectors}
    stamps["median_resid_norms"] = {f"L{s}": med_norms[s] for s in sites}
    stamps["diagnostics"] = diag
    stamps["provenance"] = ("V3w = unit(Σ⁻¹(μ_analogical − μ_contrastive)) in residual space, Σ = "
                            "Ledoit-Wolf shrinkage; V3raw = unit(μ_analogical − μ_contrastive) same "
                            "capture (control). Built to test whether the WHITENED direction steers "
                            "where the raw CAA fails on DSV2 (2026-07-19 mean-diff⊥discriminative finding).")
    (args.out_dir / "a5_vectors_stamps.json").write_text(json.dumps(stamps, indent=1))
    logger.info(f"wrote {args.out_dir}/a5_vectors.npz ({list(vectors)}) + stamps")


if __name__ == "__main__":
    main()
