"""Shared building blocks for the v3_audit analysis suite.

Five blocks were copy-pasted across ~15 scripts; they live here once:

  1. HARD                    — the 5-way hard mode set
  2. residualize[_all]       — lstsq length-control (train-fit, applied to both splits)
  3. unwrap_generations /    — metadata.json wraps generations under a "generations"
     gen_metadata_by_id        key (CLAUDE.md gotcha); id -> record map
  4. load_signature_matrix   — the gen_*.npz signature-load loop (X, y, topic, C, names)
  5. make_encoder/train_eval — the logit-LBFGS / deep-AdamW encoder trio
     /subsample_topics         (encoder_floor + surface_encoder_floor)

Import convention (scripts stay self-contained for node1: numpy/sklearn only,
torch imported lazily, no anamnesis import needed when run from this dir):

    try:  # direct script run — sys.path[0] is the script dir
        from _common import HARD, residualize
    except ImportError:  # imported as a package module
        from anamnesis.analysis.v3_audit._common import HARD, residualize

train_eval returns (test_acc, train_acc); pass deep_epochs to match each
script's convergence budget (2500 raw-wide, 800 Gram-reduced).
"""
from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]

# ── 1. The 5-way hard mode set ────────────────────────────────────────────────
HARD: frozenset[str] = frozenset(
    {"linear", "socratic", "contrastive", "dialectical", "analogical"}
)


# ── 2. Length-control residualization (lstsq; train-fit, applied to both) ────
def residualize(Ftr: F64, Fte: F64, Ctr: F64, Cte: F64) -> tuple[F64, F64]:
    """Regress features on covariates C (+intercept) fit on TRAIN, subtract from both."""
    A = np.hstack([Ctr, np.ones((len(Ctr), 1))]); B = np.hstack([Cte, np.ones((len(Cte), 1))])
    coef, *_ = np.linalg.lstsq(A, Ftr, rcond=None)
    return Ftr - A @ coef, Fte - B @ coef


def residualize_all(F: F64, C: F64) -> F64:
    """Whole-matrix variant (no train/test split — for unsupervised structure)."""
    A = np.hstack([C, np.ones((len(C), 1))])
    coef, *_ = np.linalg.lstsq(A, F, rcond=None)
    return F - A @ coef


# ── 3. metadata.json unwrap ───────────────────────────────────────────────────
def unwrap_generations(meta: Any) -> list[dict[str, Any]]:
    """metadata.json wraps the generation list under a "generations" key — unwrap it
    (tolerates the bare-list legacy format)."""
    return meta["generations"] if isinstance(meta, dict) and "generations" in meta else meta


def gen_metadata_by_id(metadata_path: Path | str) -> dict[int, dict[str, Any]]:
    """Load metadata.json -> {generation_id: record}."""
    with open(metadata_path) as f:
        meta = json.load(f)
    return {int(g["generation_id"]): g for g in unwrap_generations(meta)}


# ── 4. Signature-matrix loader (the gen_*.npz loop) ──────────────────────────
try:  # pydantic when available (pipeline venv); plain dataclass on lean node1 envs
    from pydantic import BaseModel, ConfigDict

    class SignatureMatrix(BaseModel):
        """Loaded signature feature matrix + labels/covariates/feature names."""

        model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

        X: np.ndarray      # (n, P) float64, nan_to_num'd
        y: np.ndarray      # (n,) mode labels (str)
        topic: np.ndarray  # (n,) topic_idx (int)
        C: np.ndarray      # (n, 2) float64 [prompt_length, num_generated_tokens]
        names: np.ndarray  # (P,) feature names (str)

except ImportError:  # pragma: no cover — node1 self-contained fallback
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class SignatureMatrix:  # type: ignore[no-redef]
        """Loaded signature feature matrix + labels/covariates/feature names."""

        X: np.ndarray
        y: np.ndarray
        topic: np.ndarray
        C: np.ndarray
        names: np.ndarray


def load_signature_matrix(
    runs: Sequence[str],
    runs_root: Path | str,
    subdir: str = "signatures_v3",
    modes: Collection[str] = HARD,
) -> SignatureMatrix:
    """Load merged per-gen signature vectors across runs (feature order pinned to
    the first gen seen; missing features fill 0.0; matrix nan_to_num'd).

    Exactly the loop previously copy-pasted across the suite. Runs/dirs that do
    not exist are skipped silently (the merged {3b,8b}_fat_01+_ext convention).
    """
    names: list[str] | None = None
    rows: list[list[float]] = []
    y: list[str] = []
    topic: list[int] = []
    C: list[list[float]] = []
    for run in runs:
        rd = Path(runs_root) / run
        sd = rd / subdir
        if not (rd / "metadata.json").exists() or not sd.exists():
            continue
        md = gen_metadata_by_id(rd / "metadata.json")
        for p in sorted(sd.glob("gen_*.npz"), key=lambda x: int(x.stem.split("_")[1])):
            g = int(p.stem.split("_")[1])
            if g not in md or md[g]["mode"] not in modes:
                continue
            z = np.load(p, allow_pickle=True)
            nm = [str(x) for x in z["feature_names"]]
            if names is None:
                names = nm
            d = {n: float(v) for n, v in zip(nm, z["features"])}
            rows.append([d.get(n, 0.0) for n in names])
            y.append(md[g]["mode"])
            topic.append(md[g]["topic_idx"])
            C.append([md[g]["prompt_length"], md[g]["num_generated_tokens"]])
    return SignatureMatrix(
        X=np.nan_to_num(np.array(rows, float)),
        y=np.array(y),
        topic=np.array(topic),
        C=np.array(C, float),
        names=np.array(names if names is not None else [], dtype=object),
    )


# ── 5. Encoder trio (torch imported lazily — numpy-only scripts stay torch-free) ──
def make_encoder(P: int, arch: str, nclass: int = 5, p_drop: float = 0.4, k: int = 32):
    """Build the floor encoder: 'logit' = pure linear (LBFGS floor);
    'deep' = P->256->k->nclass nonlinear ceiling-check (AdamW)."""
    import torch.nn as nn

    if arch == "logit":
        return nn.Linear(P, nclass)
    if arch == "deep":
        return nn.Sequential(
            nn.Linear(P, 256), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(256, k), nn.ReLU(), nn.Linear(k, nclass),
        )
    raise ValueError(f"Unknown arch {arch!r} (expected 'logit' or 'deep')")


def train_eval(
    Xtr: F64,
    ytr: NDArray[np.int_],
    Xte: F64,
    yte: NDArray[np.int_],
    arch: str,
    seed: int,
    device: str,
    *,
    deep_epochs: int = 2500,
    lbfgs_l2: float = 1e-3,
    nclass: int = 5,
    k: int = 32,
) -> tuple[float, float]:
    """Train on the FULL (sub)fold; returns (test_acc, train_acc).

    logit -> LBFGS (convex; AdamW under-converges it — pinned by encoder_diag
    2026-06-14), deep -> AdamW. deep_epochs: 2500 for raw-wide inputs, 800 for
    Gram-reduced (converges fast on the well-conditioned reduced input).
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.long, device=device)
    Xe = torch.tensor(Xte, dtype=torch.float32, device=device)
    net = make_encoder(Xtr.shape[1], arch, nclass=nclass, k=k).to(device)
    ce = nn.CrossEntropyLoss()
    if arch == "logit":
        opt = torch.optim.LBFGS(net.parameters(), max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = ce(net(Xt), yt) + lbfgs_l2 * sum((p ** 2).sum() for p in net.parameters())
            loss.backward()
            return loss

        opt.step(closure)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=5e-3, weight_decay=1e-2)
        for _ in range(deep_epochs):
            net.train()
            opt.zero_grad()
            ce(net(Xt), yt).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        test_acc = float((net(Xe).argmax(1).cpu().numpy() == yte).mean())
        train_acc = float((net(Xt).argmax(1).cpu().numpy() == ytr).mean())
    return test_acc, train_acc


def subsample_topics(
    tr: NDArray[np.int_], topic: NDArray[np.int_], frac: float, seed: int
) -> NDArray[np.int_]:
    """Subsample train TOPICS (not rows) to frac — the learning-curve knob."""
    if frac >= 1.0:
        return tr
    utop = np.unique(topic[tr])
    rng = np.random.default_rng(1000 + seed)
    keep = set(rng.choice(utop, max(2, int(round(frac * len(utop)))), replace=False).tolist())
    return tr[np.array([topic[i] in keep for i in tr])]
