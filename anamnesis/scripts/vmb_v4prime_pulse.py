"""§B.3 V4′ = Σ∇S pulse table — committed reproduction of the outer-loop CPU verdict
(FORK-V4-FIRST-READ-2026-07-14 §3/§4; 13b no-heredoc rule: the ruling's evidence must
live in a committed script, not an inline pulse).

The fork's §B.3 asked: does METRIC-CORRECTING the raw dir0 gradient (V4′ = Σ∇S — multiply
by the covariance, which suppresses the low-variance tail) pull it on-manifold and toward V3,
worth a free-gen? The CPU pre-gate KILLED it ($0 spent). This reproduces the exact numbers.

Inputs = the three banked artifacts (all in outputs/battery/arms/A5/):
  a5_gradient_fork_G_3b.npz   G (20×3072 per-gen ∇S) + mean_grad (raw dir0 gradient)
  a5_sigma_L14_3b.npz         residual-Σ eigendecomposition (evals asc, evecs, mean)
  a5_vectors(.npz)            V3_L14 (data-route mode dir) + R1/R2/R3 (banked randoms)

Must reproduce (verify against the record): V4′=Σ∇S cos(·,V3)=+0.1197, Σ-random null std
0.118, p≈0.336; metric ladder z: Σ¹ 1.0 → Σ·⁵ 1.5 → Σ·²⁵ 2.6 → identity 3.2 (the LESS you
correct toward the manifold, the MORE V3-content survives — content is anti-manifold);
banked Σ·Rᵢ cos {−.106,+.029,+.139}; raw cos(∇S,V3)=+0.060 z≈3.2, per-gen 20/20 positive;
spectral P[16:256] dead (z≈0) while P[64:1024] & P[1024:3072] carry it (z≈2.8).

CPU-only, deterministic (seeded nulls). Run:
    python -m anamnesis.scripts.vmb_v4prime_pulse \
      --gradient-G outputs/battery/arms/A5/a5_gradient_fork_G_3b.npz \
      --sigma outputs/battery/arms/A5/a5_sigma_L14_3b.npz \
      --vectors outputs/battery/arms/A5/a5_vectors_full.npz \
      --out outputs/battery/arms/A5/v4prime_pulse_3b.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_NULL = 1000
NULL_SEED = 0
METRIC_POWERS = [1.0, 0.5, 0.25, 0.0]           # 0.0 = identity = raw ∇S
BANDS = [(16, 256), (64, 1024), (1024, 3072)]    # descending-eigenvalue rank ranges


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / max(na * nb, 1e-30))


def _apply_metric(g: np.ndarray, evals: np.ndarray, evecs: np.ndarray, p: float) -> np.ndarray:
    """Σ^p @ g via the eigenbasis. p=0 → identity (returns g). p=1 → Σ@g."""
    coeff = evecs.T @ g
    return evecs @ ((evals ** p) * coeff)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gradient-G", type=Path, required=True)
    ap.add_argument("--sigma", type=Path, required=True)
    ap.add_argument("--vectors", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    Gnpz = np.load(args.gradient_G)
    G = Gnpz["G"].astype(np.float64)                       # (20, 3072)
    grad = Gnpz["mean_grad"].astype(np.float64)            # raw dir0 gradient ∇S
    S = np.load(args.sigma)
    evals = np.clip(S["evals"].astype(np.float64), 0, None)  # ascending
    evecs = S["evecs"].astype(np.float64)                  # columns = eigenvectors
    vec = np.load(args.vectors)
    v3 = vec["V3_L14"].astype(np.float64)
    randoms = {k: vec[k].astype(np.float64) for k in ("R1", "R2", "R3") if k in vec.files}

    d = grad.shape[0]
    rng = np.random.default_rng(NULL_SEED)

    def rand_units(n: int) -> np.ndarray:
        X = rng.standard_normal((n, d))
        return X / np.linalg.norm(X, axis=1, keepdims=True)

    out: dict = {"STATUS": "REPRODUCTION of FORK-V4-FIRST-READ §3/§4 (13b committed-record)",
                 "d": d, "n_null": N_NULL, "null_seed": NULL_SEED}

    # ── raw gradient anti-manifold (identity metric) ──────────────────────────────
    cos_raw = _cos(grad, v3)
    per_gen = np.array([_cos(G[i], v3) for i in range(len(G))])
    npos = int((per_gen > 0).sum())
    # sign test (two-sided) under p=0.5
    from math import comb
    sign_p = 2 * sum(comb(len(per_gen), k) for k in range(npos, len(per_gen) + 1)) / 2 ** len(per_gen)
    iso = rand_units(N_NULL) @ v3
    out["raw_gradient"] = {
        "cos_gradS_V3": round(cos_raw, 4),
        "z_vs_isotropic": round(cos_raw / iso.std(), 2),
        "per_gen_positive": f"{npos}/{len(per_gen)}",
        "per_gen_mean": round(float(per_gen.mean()), 4),
        "per_gen_std": round(float(per_gen.std()), 4),
        "sign_test_p": float(f"{min(sign_p,1.0):.2e}"),
    }

    # ── metric ladder Σ^p @ ∇S ────────────────────────────────────────────────────
    null_units = rand_units(N_NULL)
    ladder = {}
    for p in METRIC_POWERS:
        vp = _apply_metric(grad, evals, evecs, p)
        cos_p = _cos(vp, v3)
        null_cos = np.array([_cos(_apply_metric(null_units[i], evals, evecs, p), v3)
                             for i in range(N_NULL)])
        z = (cos_p - null_cos.mean()) / max(null_cos.std(), 1e-12)
        two_sided_p = float((np.abs(null_cos - null_cos.mean()) >= abs(cos_p - null_cos.mean())).mean())
        tag = "identity" if p == 0.0 else f"Sigma^{p}"
        ladder[tag] = {"cos_vp_V3": round(cos_p, 4), "null_std": round(float(null_cos.std()), 4),
                       "z": round(float(z), 2), "p": round(two_sided_p, 3)}
    out["metric_ladder"] = ladder
    out["V4prime_headline"] = ladder["Sigma^1.0"]   # the §B.3 object of record

    # ── banked Σ·Rᵢ band (the null bracket that killed §B.3) ──────────────────────
    out["banked_sigma_random_cos_V3"] = {
        k: round(_cos(_apply_metric(r, evals, evecs, 1.0), v3), 3) for k, r in randoms.items()}

    # ── spectral-band anatomy (descending eigen-rank) ─────────────────────────────
    order = np.argsort(evals)[::-1]                       # descending
    cg = evecs.T @ grad                                   # ascending-eigenbasis coords
    cv = evecs.T @ v3
    bands = {}
    for a, b in BANDS:
        idx = order[a:b]
        ga, va = cg[idx], cv[idx]
        cb = float(ga @ va / max(np.linalg.norm(ga) * np.linalg.norm(va), 1e-30))
        z = cb * np.sqrt(len(idx))                        # isotropic band-cos std = 1/sqrt(|B|)
        bands[f"P[{a}:{b}]"] = {"cos": round(cb, 4), "z": round(float(z), 2), "n_dirs": int(len(idx))}
    out["spectral_bands"] = bands

    out["one_line_anatomy"] = ("real formula content (raw ∇S sees V3, z~3.2) + anti-manifold "
                               "location (Σ-correction destroys it, Σ¹ z~1.0) + off-target mass "
                               "⇒ max deformation, ~zero net targeting; §B.3 free-gen does NOT fire.")
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
