"""GPU Ledoit-Wolf shrinkage covariance, for the Computer #5 install lever (spec S4/S5).

sklearn's LedoitWolf at d=8192 with ~20k tokens takes minutes per site, and we need it at five
sites x several donors. The estimator is closed-form, so it ports to the GPU exactly — no
approximation, no change of estimator, same number.

THE ALGEBRAIC SHORTCUTS (this is why it is fast, and why the port is exact):
Let Xc be the centred data (n, d), S = Xc^T Xc / n the empirical covariance.

  sklearn computes  delta_ = sum((Xc^T Xc)**2) / n**2, which is exactly ||S||_F^2 —
      no need to form (Xc^T Xc)**2 separately.

  sklearn computes  beta_raw = sum(X2^T @ X2)  where X2 = Xc**2. Expanding:
      sum_{i,j} sum_k X2[k,i] X2[k,j] = sum_k (sum_i X2[k,i]) (sum_j X2[k,j])
                                      = sum_k (rowsum of X2[k])^2
      — an O(n*d) reduction instead of an O(n*d^2) matmul. This is the single biggest win.

So the only O(n*d^2) operation left is S itself: one matmul, which is what GPUs are for.

PRECISION: the matmul runs in float32 (the data is bf16-derived, so ~7 significant digits is
already more than the input carries) with float32 accumulation, then S is promoted to float64 for
the solve and the eigendecomposition, which are the numerically delicate steps. TF32 is disabled
— it would silently drop the matmul to ~10 mantissa bits and this is a covariance estimate.

Validated against sklearn by `--validate` (see __main__): shrinkage and Sigma must agree to
tight tolerance on a subsample before any vector is built from this.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def ledoit_wolf_gpu(X: np.ndarray, device: str = "cuda", assume_centered: bool = False
                    ) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrunk covariance on GPU. Returns (Sigma (d,d) float64, shrinkage).

    Mirrors sklearn.covariance.ledoit_wolf exactly, including the `beta = min(beta_, delta)`
    clamp and the `shrinkage = 0 if beta == 0` guard.
    """
    import torch

    # A covariance estimate must not run through TF32's 10-bit mantissa.
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        n_samples, n_features = X.shape
        Xg = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=device)
        if not assume_centered:
            Xg = Xg - Xg.mean(dim=0, keepdim=True)

        # S = Xc^T Xc / n  — the one unavoidable O(n d^2) step.
        S = (Xg.T @ Xg) / n_samples                                     # (d, d) float32

        X2 = Xg * Xg
        emp_cov_trace = X2.sum(dim=0) / n_samples                       # == diag(S)
        trace_sum = emp_cov_trace.sum()
        mu = trace_sum / n_features

        # delta_ = sum((Xc^T Xc)^2) / n^2 == ||S||_F^2
        delta_ = (S.double() ** 2).sum()

        # beta_raw = sum(X2^T @ X2) == sum_k (rowsum X2[k])^2   [O(n d), not O(n d^2)]
        row_sums = X2.sum(dim=1).double()
        beta_raw = (row_sums * row_sums).sum()

        beta_ = (1.0 / (n_features * n_samples)) * (beta_raw / n_samples - delta_)
        delta = (delta_ - 2.0 * mu.double() * trace_sum.double()
                 + n_features * (mu.double() ** 2)) / n_features
        beta = torch.minimum(beta_, delta)
        shrinkage = 0.0 if float(beta) == 0.0 else float(beta / delta)

        Sigma = S.double() * (1.0 - shrinkage)
        Sigma.diagonal().add_(shrinkage * mu.double())
        out = Sigma.cpu().numpy()
        del Xg, X2, S, Sigma
        torch.cuda.empty_cache()
        return out, shrinkage
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


def solve_gpu(Sigma: np.ndarray, b: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Sigma^-1 b on GPU (float64). Falls back to lstsq if Sigma is singular."""
    import torch

    A = torch.as_tensor(Sigma, dtype=torch.float64, device=device)
    y = torch.as_tensor(np.ascontiguousarray(b), dtype=torch.float64, device=device)
    try:
        w = torch.linalg.solve(A, y)
    except Exception as exc:  # noqa: BLE001 — any linalg failure should degrade, not crash
        logger.warning("solve failed (%s); falling back to lstsq", exc)
        w = torch.linalg.lstsq(A, y.unsqueeze(-1)).solution.squeeze(-1)
    return w.cpu().numpy()


def eigh_gpu(Sigma: np.ndarray, device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    """Symmetric eigendecomposition on GPU. Returns (eigenvalues, eigenvectors), ascending."""
    import torch

    A = torch.as_tensor(Sigma, dtype=torch.float64, device=device)
    evals, evecs = torch.linalg.eigh(A)
    return evals.cpu().numpy(), evecs.cpu().numpy()


def _validate(n: int = 3000, d: int = 512, seed: int = 0) -> None:
    """Prove the port against sklearn on a subsample small enough for sklearn to finish."""
    from sklearn.covariance import LedoitWolf

    rng = np.random.default_rng(seed)
    # Correlated, anisotropic data — an isotropic gaussian would not exercise the shrinkage.
    A = rng.standard_normal((d, d)).astype(np.float32) / np.sqrt(d)
    X = (rng.standard_normal((n, d)).astype(np.float32) @ A) * rng.uniform(0.5, 4.0, size=d).astype(np.float32)

    lw = LedoitWolf(assume_centered=False).fit(X.astype(np.float64))
    S_gpu, shr_gpu = ledoit_wolf_gpu(X)

    d_shr = abs(lw.shrinkage_ - shr_gpu)
    rel_fro = np.linalg.norm(S_gpu - lw.covariance_) / max(np.linalg.norm(lw.covariance_), 1e-12)
    b = rng.standard_normal(d)
    w_sk = np.linalg.solve(lw.covariance_, b)
    w_gpu = solve_gpu(S_gpu, b)
    cos_w = float(w_sk @ w_gpu / (np.linalg.norm(w_sk) * np.linalg.norm(w_gpu)))

    print(f"n={n} d={d}")
    print(f"  shrinkage  sklearn={lw.shrinkage_:.8f}  gpu={shr_gpu:.8f}  |diff|={d_shr:.2e}")
    print(f"  Sigma relative Frobenius error = {rel_fro:.3e}")
    print(f"  cos(Sigma^-1 b) sklearn vs gpu = {cos_w:.8f}")
    ok = d_shr < 1e-4 and rel_fro < 1e-4 and cos_w > 0.9999
    print("  VERDICT:", "PASS" if ok else "FAIL — do not build vectors from this")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--d", type=int, default=512)
    a = ap.parse_args()
    if a.validate:
        _validate(a.n, a.d)
