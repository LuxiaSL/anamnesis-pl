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


# ── 14a §6 graded-Goodhart amendment: deformation-Mahalanobis-VS-DOSE ──────────────
# The vᵀΣ⁻¹v statistic below is dose-trivial (α²-scaling). The graded-Goodhart readout is
# the Mahalanobis of the INDUCED STATE DEFORMATION per dose: inject α_frac·median_norm·v at
# the vector's site (C§3/13a magnitude convention, unit v) and measure the deformation of the
# residual — at the layer OUTPUT, because at the injection-site INPUT the deformation is
# exactly the injected vector (α·v, trivial); the block's nonlinear response to the injection
# lives at the OUTPUT. Δ(α)ᵀ Σ_out⁻¹ Δ(α) that grows FASTER than α² is off-manifold Goodhart.
# ⚠ A5-CLASS DESIGN CHOICE flagged to the outer loop (measurement at layer OUTPUT vs a fixed
# downstream layer; injection magnitude = α_frac × median site-input norm). First-read only.

def _maha(delta: np.ndarray, evals: np.ndarray, evecs: np.ndarray, ridge: float) -> float:
    coeff = evecs.T @ delta.astype(np.float64)
    return float((coeff ** 2 / (evals + ridge)).sum())


def _capture_site_output(model, layers, entries, gids, site: int,
                         inject_vec: np.ndarray | None = None,
                         inject_scale: float = 0.0) -> np.ndarray:
    """Stack layer-{site} OUTPUT rows over generated positions; optionally inject
    inject_scale·inject_vec at the layer INPUT (pre-hook) to measure the steered output."""
    rows: list[np.ndarray] = []
    grab: dict[str, torch.Tensor] = {}
    vt = None
    if inject_vec is not None:
        p0 = next(model.parameters())
        vt = torch.tensor(inject_vec, dtype=p0.dtype, device=p0.device)

    def pre_hook(module, a, kw):
        if vt is None:
            return None
        hs = a[0] if a else kw.get("hidden_states")
        hs2 = hs + inject_scale * vt
        if a:
            return (hs2,) + tuple(a[1:]), kw
        kw = dict(kw); kw["hidden_states"] = hs2
        return a, kw

    def out_hook(module, a, output):
        grab["o"] = (output[0] if isinstance(output, tuple) else output).detach()

    hp = layers[site].register_forward_pre_hook(pre_hook, with_kwargs=True) if vt is not None else None
    ho = layers[site].register_forward_hook(out_hook)
    try:
        for g in gids:
            e = entries[str(g)]
            ids = torch.tensor([e["input_ids"]], dtype=torch.long, device="cuda")
            P = int(e["prompt_length"])
            with torch.no_grad():
                model(ids, use_cache=False, return_dict=True)
            rows.append(grab["o"][0, P:, :].float().cpu().numpy())
            grab.clear()
    finally:
        ho.remove()
        if hp is not None:
            hp.remove()
    return np.concatenate(rows, axis=0)


def _deformation_curve(model, layers, entries, gids, site: int, R_in: np.ndarray,
                       site_vectors: dict[str, np.ndarray], doses: list[float],
                       ridge_rel: float) -> dict:
    """Per vector at this site: Mahalanobis of the induced output deformation per dose."""
    out_unsteered = _capture_site_output(model, layers, entries, gids, site)   # [Npos, d]
    mu = out_unsteered.mean(axis=0)
    Oc = out_unsteered - mu
    Sig = (Oc.T @ Oc) / max(len(Oc) - 1, 1)
    evals, evecs = np.linalg.eigh(Sig)
    evals = np.clip(evals, 0, None)
    ridge = ridge_rel * float(evals.mean())
    median_in_norm = float(np.median(np.linalg.norm(R_in, axis=1)))            # C§3 magnitude base
    rows: dict[str, dict] = {}
    for key, v in site_vectors.items():
        vunit = v / max(np.linalg.norm(v), 1e-12)
        by_dose = {}
        for a in doses:
            scale = a * median_in_norm
            out_st = _capture_site_output(model, layers, entries, gids, site,
                                          inject_vec=vunit, inject_scale=scale)
            delta = (out_st - out_unsteered).mean(axis=0)                      # mean deformation
            m = _maha(delta, evals, evecs, ridge)
            by_dose[str(a)] = {"deformation_maha": m,
                               "maha_over_alpha2": m / (a * a) if a > 0 else None,
                               "delta_l2": float(np.linalg.norm(delta))}
        rows[key] = by_dose
    return {"n_positions": int(len(out_unsteered)), "median_site_input_norm": median_in_norm,
            "ridge": ridge, "doses": doses, "vectors": rows,
            "readout": "deformation_maha growing FASTER than α² (maha_over_alpha2 rising with "
                       "dose) = off-manifold graded Goodhart; flat = linear/on-manifold."}


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
    ap.add_argument("--deformation", action="store_true",
                    help="14a §6: also emit the deformation-Mahalanobis-VS-DOSE curve per "
                         "vector (steered captures — multiplies forward passes by "
                         "|vectors|×|doses|+1; use a smaller --n-gens for this leg)")
    ap.add_argument("--deform-doses", default="0.03,0.1,0.3,0.45")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sites = [int(s) for s in args.sites.split(",")]
    deform_doses = [float(x) for x in args.deform_doses.split(",")]

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
           "provenance": "WAVE2-A5 addendum 13d §2.1 (vᵀΣ⁻¹v + eigenmass) + 14a §6 amendment "
                         "(deformation-Mahalanobis-vs-dose, --deformation); Σ=cov(L{site} "
                         "residual, gen positions, banked alpha=0 continuations)",
           "deformation_leg": bool(args.deformation),
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
        site_out = {"n_positions": int(len(R)),
                    "mean_eigenvalue": float(evals.mean()),
                    "ridge": ridge, "vectors": site_rows}
        if args.deformation:
            site_vecs = {k: np.asarray(v) for k, v in bank.items()
                         if k in site_rows}          # the same vectors that live at this site
            logger.info(f"L{site}: deformation-vs-dose over {len(site_vecs)} vectors "
                        f"× {len(deform_doses)} doses")
            site_out["deformation_vs_dose"] = _deformation_curve(
                model, layers, entries, gids, site, R, site_vecs, deform_doses, args.ridge_rel)
        out["sites"][str(site)] = site_out

    p = args.out_dir / f"a5_covariance_screen_{args.model}.json"
    p.write_text(json.dumps(out, indent=1))
    logger.info(f"banked (first-read pending) -> {p}")


if __name__ == "__main__":
    main()
