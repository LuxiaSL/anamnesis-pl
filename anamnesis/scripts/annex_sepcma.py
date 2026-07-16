"""ANNEX route-5: minimal sep-CMA-ES (diagonal covariance; Ros & Hansen 2008).

Self-implemented rather than importing `cma`: zero dependency risk on the shared
node venv, deterministic under a fixed seed, auditable in one screen, and unit-
tested below on (a) the sphere and (b) a planted-needle alignment landscape —
the idealized (noiseless, unimodal) version of route 5's objective, which makes
the self-test double as the cheapest possible probe of the 10·d budget question
(if CMA cannot solve even the idealized landscape at 10·d, the budget is dead).

MAXIMIZES the supplied fitness (route-5 fitness = selectivity).

Usage:
    from anamnesis.scripts.annex_sepcma import SepCMA
    opt = SepCMA(dim=3072, seed=1, sigma0=1.0)
    while not done:
        X = opt.ask()                      # (lambda, dim) candidates
        opt.tell(X, fitnesses)             # higher = better

Self-test:  python -m anamnesis.scripts.annex_sepcma
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]


class SepCMA:
    """Diagonal-covariance CMA-ES; sep-CMA learning-rate speedup (d+2)/3."""

    def __init__(self, dim: int, seed: int, sigma0: float = 1.0,
                 x0: F64 | None = None, lam: int | None = None):
        self.dim = int(dim)
        self.rng = np.random.default_rng(seed)
        self.lam = int(lam) if lam else 4 + int(3 * np.log(self.dim))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / float(self.w @ self.w)
        d, mueff = self.dim, self.mueff
        self.cs = (mueff + 2) / (d + mueff + 5)
        self.ds = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (d + 1)) - 1) + self.cs
        self.cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c1 = 2 / ((d + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        sep = (d + 2) / 3.0                      # sep-CMA speedup on diagonal model
        self.c1 = min(1.0, c1 * sep)
        self.cmu = min(1.0 - self.c1, cmu * sep)
        self.chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d * d))

        self.mean = np.zeros(d) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
        self.sigma = float(sigma0)
        self.C = np.ones(d)                      # diagonal covariance
        self.ps = np.zeros(d)
        self.pc = np.zeros(d)
        self.gen = 0
        self.evals = 0
        self._last_Z: F64 | None = None

    def ask(self) -> F64:
        self._last_Z = self.rng.standard_normal((self.lam, self.dim))
        return self.mean + self.sigma * self._last_Z * np.sqrt(self.C)

    def tell(self, X: F64, fit: F64) -> None:
        """fit: higher = better (maximization)."""
        if self._last_Z is None or X.shape != (self.lam, self.dim):
            raise ValueError("tell() must follow ask() with the same candidates")
        idx = np.argsort(-np.asarray(fit))[: self.mu]
        Z = self._last_Z[idx]
        Y = Z * np.sqrt(self.C)                  # (mu, d) steps in x-space / sigma
        zw = self.w @ Z
        yw = self.w @ Y
        self.mean = self.mean + self.sigma * yw
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * zw
        ps_norm = float(np.linalg.norm(self.ps))
        hs = ps_norm / np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) / self.chiN \
            < 1.4 + 2 / (self.dim + 1)
        self.pc = (1 - self.cc) * self.pc \
            + (np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw if hs else 0.0)
        dh = (1 - float(hs)) * self.cc * (2 - self.cc)
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (self.pc ** 2 + dh * self.C)
                  + self.cmu * (self.w @ (Y ** 2)))
        self.C = np.maximum(self.C, 1e-20)
        self.sigma = self.sigma * float(np.exp((self.cs / self.ds) * (ps_norm / self.chiN - 1)))
        self.gen += 1
        self.evals += self.lam
        self._last_Z = None

    def state(self) -> dict:
        return {"mean": self.mean, "sigma": self.sigma, "C": self.C, "ps": self.ps,
                "pc": self.pc, "gen": self.gen, "evals": self.evals,
                "rng_state": self.rng.bit_generator.state}

    def restore(self, s: dict) -> None:
        self.mean = np.asarray(s["mean"], dtype=np.float64)
        self.sigma = float(s["sigma"])
        self.C = np.asarray(s["C"], dtype=np.float64)
        self.ps = np.asarray(s["ps"], dtype=np.float64)
        self.pc = np.asarray(s["pc"], dtype=np.float64)
        self.gen, self.evals = int(s["gen"]), int(s["evals"])
        self.rng.bit_generator.state = s["rng_state"]


def _selftest() -> None:
    # (a) sphere, d=64: must reach f < 1e-8 inside 800 gens
    opt = SepCMA(dim=64, seed=7, sigma0=1.0, x0=np.full(64, 3.0))
    best = np.inf
    for _ in range(800):
        X = opt.ask()
        f = (X ** 2).sum(axis=1)
        opt.tell(X, -f)
        best = min(best, float(f.min()))
        if best < 1e-8:
            break
    assert best < 1e-8, f"sphere failed: best={best:.3e}"
    print(f"sphere d=64: OK (best {best:.2e}, {opt.evals} evals)")

    # (b) planted needle, ROUTE-5 SHAPED: fitness = |cos(unit(x), u*)| (scale-free,
    # unimodal on the sphere) at d in {256, 3072}, budget 10*d — the idealized probe.
    for d in (256, 3072):
        rng = np.random.default_rng(0)
        u = rng.standard_normal(d)
        u /= np.linalg.norm(u)
        opt = SepCMA(dim=d, seed=11, sigma0=1.0)
        budget = 10 * d
        best_al = 0.0
        while opt.evals < budget:
            X = opt.ask()
            n = np.linalg.norm(X, axis=1)
            al = np.abs(X @ u) / np.maximum(n, 1e-12)
            opt.tell(X, al)
            best_al = max(best_al, float(al.max()))
        chance = 1 / np.sqrt(d)
        print(f"needle d={d}: best alignment {best_al:.3f} at {opt.evals} evals "
              f"(chance ~{chance:.3f}, x{best_al/chance:.0f})")
        assert best_al > 5 * chance, f"needle d={d}: no climb ({best_al:.3f})"


if __name__ == "__main__":
    _selftest()
    print("annex_sepcma self-test PASSED")
