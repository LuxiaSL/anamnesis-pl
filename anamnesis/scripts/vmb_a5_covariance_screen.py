"""A5 — residual-covariance / Mahalanobis steering-vector screen (WAVE2-A5 addendum 13d §2.1).

An a-priori "will this direction break the model" statistic, and the mechanistic explanation
for V4. All five A5 vectors are unit-norm and get identical injected magnitude at matched
(site, α), yet V4 deforms the state 13-14× a random direction and breaks the model differently.
Same magnitude, different damage ⇒ the difference is WHERE in the residual basis the vector
points. The residual stream is strongly anisotropic; the model is robust along high-variance
directions (it sees them constantly) and fragile along low-variance ones (off-manifold). A
gradient of an attention statistic (V4) concentrates in the low-variance tail — its tiny raw
norm (0.00196) is the tell.

Build Σ = cov(L14 residual over generated positions, banked α=0 continuations); report, per
vector at its site:
  - mahalanobis = vᵀ Σ⁻¹ v  (regularized) — high = points into the low-variance tail.
  - eigenspectrum mass: fraction of |v|² in the bottom-k eigenvalue directions of Σ.
Predicted ordering: V4 ≫ R ≈ V1 ≈ V3, V4's mass piled in the tail. If it holds, this GATES
§2.2/path-following: it says a priori whether path-following can work, and generalizes to every
future intervention. NOTE this is a per-SITE Σ; compare vectors only within a shared site.

⚠ NEEDS GPU (one card; residual capture over ~40-160 continuations). UNTESTED without GPU —
smoke on --n-gens 5 first and check Σ is PSD / condition number is finite.

Usage (node1, 1 GPU, Heimdall; venv-only):
    python -m anamnesis.scripts.vmb_a5_covariance_screen --model 3b \
        --model-path /models/llama-3.2-3b-instruct \
        --stage0-run /models/anamnesis-extract/runs/vmb_stage0_3b \
        --vectors /models/anamnesis-extract/battery/a5_vectors_3b/a5_vectors.npz \
        --out-dir /models/anamnesis-extract/battery/arms/A5 --sites 7,14,18,21 --n-gens 60
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _capture_residuals(model, layers, entries, gids, site: int) -> np.ndarray:
    """Stack L{site} residual-input rows over generated positions of the banked continuations."""
    rows: list[np.ndarray] = []
    grab: dict[str, torch.Tensor] = {}

    def hook(module, hook_args, hook_kwargs):
        hs = hook_args[0] if hook_args else hook_kwargs.get("hidden_states")
        grab["h"] = hs.detach()
        return None

    handle = layers[site].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for g in gids:
            e = entries[str(g)]
            ids = torch.tensor([e["input_ids"]], dtype=torch.long, device="cuda")
            P = int(e["prompt_length"])
            with torch.no_grad():
                model(ids, use_cache=False, return_dict=True)
            h = grab["h"][0, P:, :].float().cpu().numpy()  # [n_gen_pos, d]
            rows.append(h)
            grab.clear()
    finally:
        handle.remove()
    return np.concatenate(rows, axis=0)


def _screen(v: np.ndarray, evals: np.ndarray, evecs: np.ndarray, ridge: float,
            tail_frac: float = 0.25) -> dict:
    """vᵀΣ⁻¹v (ridge-regularized via eigendecomp) + bottom-k eigenmass fraction of v."""
    v = v.astype(np.float64)
    v = v / max(np.linalg.norm(v), 1e-12)          # unit (defensive; bank is already unit)
    coeff = evecs.T @ v                            # v in the eigenbasis
    inv = coeff ** 2 / (evals + ridge)
    maha = float(inv.sum())
    k = max(1, int(tail_frac * len(evals)))
    order = np.argsort(evals)                       # ascending: smallest (tail) first
    tail_mass = float((coeff[order[:k]] ** 2).sum())  # |v|²=1 so this is a fraction
    top_mass = float((coeff[order[-k:]] ** 2).sum())
    return {"mahalanobis": maha, "bottom_%d_eigenmass" % k: tail_mass,
            "top_%d_eigenmass" % k: top_mass,
            "tail_over_top": float(tail_mass / max(top_mass, 1e-12))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--vectors", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--sites", default="7,14,18,21")
    ap.add_argument("--n-gens", type=int, default=60)
    ap.add_argument("--ridge-rel", type=float, default=1e-3,
                    help="ridge = ridge_rel × mean(eigenvalue) for the Σ⁻¹ regularization")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sites = [int(s) for s in args.sites.split(",")]

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="eager",
    ).to("cuda").eval()
    layers = model.model.layers

    entries = json.loads((args.stage0_run / "replay_manifest.json").read_text())["entries"]
    all_ids = sorted(int(k) for k in entries)
    gids = all_ids[:: max(1, len(all_ids) // args.n_gens)][: args.n_gens]

    bank = dict(np.load(args.vectors))
    site_of = {"V1": None, "V2": None, "V3": None, "V4": 14}  # V1/V3 per-site keys; V2 native

    out = {"model": args.model, "STATUS": "FIRST_READ_PENDING (C§8)",
           "provenance": "WAVE2-A5 addendum 13d §2.1; Σ=cov(L{site} residual, gen positions, "
                         "banked alpha=0 continuations)",
           "n_gens": len(gids), "sites": {}}
    for site in sites:
        R = _capture_residuals(model, layers, entries, gids, site)
        mu = R.mean(axis=0)
        Rc = R - mu
        Sigma = (Rc.T @ Rc) / max(len(Rc) - 1, 1)
        evals, evecs = np.linalg.eigh(Sigma)         # ascending eigenvalues
        evals = np.clip(evals, 0, None)
        ridge = args.ridge_rel * float(evals.mean())
        logger.info(f"L{site}: {len(R)} positions, cond~{evals[-1]/max(evals[evals>0][0],1e-12):.1e}")
        site_rows = {}
        for key, v in bank.items():
            # match vectors that live at this site
            want = (key.endswith(f"_L{site}")
                    or (key in ("R1", "R2", "R3"))         # site-independent randoms
                    or (key == "V4_L14" and site == 14))
            if not want or v.shape[0] != R.shape[1]:
                continue
            site_rows[key] = _screen(np.asarray(v), evals, evecs, ridge)
        out["sites"][str(site)] = {"n_positions": int(len(R)),
                                   "mean_eigenvalue": float(evals.mean()),
                                   "ridge": ridge, "vectors": site_rows}

    p = args.out_dir / f"a5_covariance_screen_{args.model}.json"
    p.write_text(json.dumps(out, indent=1))
    logger.info(f"banked (first-read pending) -> {p}")


if __name__ == "__main__":
    main()
