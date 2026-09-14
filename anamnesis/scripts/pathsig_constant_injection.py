"""JOB 2 — what does a CONSTANT injection look like in path terms? (local CPU, no GPU)

Brief: ``FIRSTREAD-pathsig-legs-2026-09-11.md`` Ruling 1 + the enactor brief's E2 re-cut,
re-scoped by the desk from "ORDER vs COVERAGE" to "establish what a constant injection looks
like in path terms". **Exploratory; UNSTAMPED; nothing here is quotable pre-first-read (C§8).**

WHY THE RE-CUT AS SPECIFIED IS NOT THE CELL THAT RUNS HERE
----------------------------------------------------------
The brief asks for B1/B2 bases built on "the already-regenerated 3B paths and states under
``outputs/pathsig/s51_3b_L14/``". Only the ``[T, 8]`` *projections* were banked there; the
full ``[T, 3072]`` residual states lived in the leg-2 job's memory and were never written
(``s51_paths_L14_k8.npz`` holds ``paths_pc``/``paths_nopc`` at k=8 and nothing wider). A basis
that CONTAINS the ``V2_L13`` perturbation cannot be built from an 8-dim projection that holds
0.48% of its norm. The real s51 re-cut therefore needs the ~10-20 GPU-min replay again, which
this enactment is forbidden. That is reported as a blocker, not worked around.

WHAT RUNS INSTEAD, AND WHY IT ANSWERS THE RE-SCOPED QUESTION BETTER
-------------------------------------------------------------------
Three legs, all CPU, all on banked local data:

**J2-a — the invariance identity (exact).** The level-2 log-signature centres at ``X[0]``
(``PathSignatureConfig.center_at_origin``, mandatory), so BOTH levels are invariant under
``X(t) -> X(t) + c`` for a constant ``c``: level-1 is ``X[T-1] - X[0]``, level-2 is built from
``X[t] - X[0]``, and the augmented clock coordinate is untouched. A constant vector added at
every generated position is therefore not merely under-covered by the projection — it is in the
estimator's exact null space. Verified numerically to machine precision on the banked s51 paths.

**J2-b — anatomy of the REAL s51 arm difference** in the banked k=8 projection: how much of the
matched-pair difference ``D(t) = X_steered(t) - X_unsteered(t)`` is a pure translation
(``||mean_t D||^2 / mean_t ||D||^2``) versus time-varying. The translation part is invisible to
the family by J2-a; only the time-varying part could ever be read.

**J2-c — the synthetic injection cell** on LOCAL banked 3B states (``3b_fat_01`` raw tensors,
L14, full 3072-dim, positionally corrected exactly as the projection of record). The real
``V2_L13`` vector at the real ``alpha`` is added to every generated position of one arm — the
resolver's construction in its idealised, purely-constant form — and read in bases that DO
contain it:

  * **B2** — ``[v_hat, pc0..pc3]`` orthonormalised, i.e. the perturbation direction is an
    explicit basis vector. Not circular for the ORDER question because the increment-permutation
    null uses the IDENTICAL basis and identical columns and controls everything except order.
  * **B1** — an in-fold-fit discriminative direction (train-fold arm mean-difference) + top-4
    PCA. **Refit inside every fold**; the test fold never contributes to the basis.
  * **P-A** — the E2 basis, carried for the adequacy comparison only.

Two contrast arms make the constant result readable rather than vacuous: a **step** injection
(``v`` applied over the second half only) and a **ramp** (``v`` scaled by ``t/(T-1)``), same
direction, same alpha, which are NOT translations. If the family is working, const reads chance
at both levels while step/ramp are read — and the level at which they are read is the finding.

Usage::

    python -m anamnesis.scripts.pathsig_constant_injection \\
        --raw-dir outputs/runs/3b_fat_01/raw_tensors \\
        --metadata outputs/runs/3b_fat_01/metadata.json \\
        --pca outputs/pathsig/calib/pca_model_corrected_3b.pkl --layer 14 \\
        --vectors outputs/battery/arms/A5/a5_vectors_full.npz --vector-key V2_L13 \\
        --alpha 0.3276513576507568 \\
        --s51-bank outputs/pathsig/s51_3b_L14/s51_paths_L14_k8.npz \\
        --out outputs/pathsig/constant_injection_3b_L14.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from sklearn.model_selection import GroupKFold

from anamnesis.analysis.v3_audit._common import gen_metadata_by_id, train_eval
from anamnesis.analysis.v3_audit.surface_encoder_floor import preprocess_fold_gpu
from anamnesis.extraction.feature_families.path_signature import (
    PathSignatureConfig,
    ProjectionBasis,
    signature_features_from_path,
)
from anamnesis.scripts.pathsig_read_e1 import DEEP_EPOCHS, K, LBFGS_L2, NULL_SEEDS, SEEDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]
ARM_KINDS = ("const", "step", "ramp")


class InjectionConfig(BaseModel):
    """Frozen provenance for one synthetic-injection pass."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_dir: Path
    metadata: Path
    pca: Path
    layer: int
    vectors: Path
    vector_key: str
    alpha: float = Field(gt=0.0)
    k_pa: int = Field(8, ge=2)
    k_ctx: int = Field(4, ge=1, description="top-k PCA context directions in B1/B2")
    s51_bank: Path | None = None
    device: str = "cpu"
    out: Path
    limit: int | None = None


# ───────────────────────────────────────────────────── banked-state loading

_G_LAYER = -1


def _init(layer: int) -> None:
    global _G_LAYER
    _G_LAYER = layer


def _load_one(task: tuple[str, int]) -> dict[str, Any]:
    """Worker: one banked raw npz → positionally-corrected ``[T, d]`` float32 L{layer} path."""
    path, gid = task
    try:
        z = np.load(path, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001
        return {"gen_id": gid, "error": f"npz load failed: {exc!r}"}
    try:
        if "hidden_states" not in z.files:
            return {"gen_id": gid, "error": "no hidden_states key"}
        saved = ([int(x) for x in z["saved_layers_hs"]] if "saved_layers_hs" in z.files else None)
        hs = z["hidden_states"]
        if hs.ndim != 3:
            return {"gen_id": gid, "error": f"hidden_states ndim {hs.ndim}"}
        if saved is not None:
            # This bank stores only SAMPLED layers and records their true indices in
            # `saved_layers_hs` (e.g. 3B: [-1, 0, 7, 14, 18, 21, 24, 27], where -1 is the
            # embedding output). So the row for layer L is its position in that list — the
            # generic `[t][l+1]` HF-tuple rule does NOT apply to an already-sampled bank.
            if _G_LAYER not in saved:
                return {"gen_id": gid, "error": f"layer {_G_LAYER} not in saved rows {saved}"}
            row = saved.index(_G_LAYER)
        else:
            row = _G_LAYER + 1
            if not (0 <= row < hs.shape[1]):
                return {"gen_id": gid, "error": f"row {row} out of range {hs.shape}"}
        X = hs[:, row, :].astype(np.float32)
        T = int(X.shape[0])
        if T < 3:
            return {"gen_id": gid, "error": f"T={T} < 3"}
        if not np.all(np.isfinite(X)):
            return {"gen_id": gid, "error": "non-finite hidden states"}
        if float(np.linalg.norm(X[0])) < 1e-6:
            return {"gen_id": gid, "error": "layer zero-filled in this bank"}
        plen = int(z["prompt_length"]) if "prompt_length" in z.files else 0
        if "positional_means" not in z.files:
            return {"gen_id": gid, "error": "no positional_means in bank — cannot correct"}
        pm = z["positional_means"]
        if not (0 <= _G_LAYER + 1 < pm.shape[0]):
            return {"gen_id": gid, "error": f"positional_means has no row {_G_LAYER+1}"}
        pml = pm[_G_LAYER + 1].astype(np.float32)
        pos = np.minimum(plen + np.arange(T), pml.shape[0] - 1)
        Xc = (X - pml[pos]).astype(np.float32)
        # actual_lengths marks real generated positions; trailing pad rows would be identical
        if "actual_lengths" in z.files:
            al = np.asarray(z["actual_lengths"]).ravel()
            if al.size >= T:
                nz = int(np.count_nonzero(al[:T] > 0))
                if 3 <= nz < T:
                    Xc = Xc[:nz]
        return {"gen_id": gid, "X": Xc, "prompt_length": plen, "T": int(Xc.shape[0])}
    except Exception as exc:  # noqa: BLE001
        return {"gen_id": gid, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            z.close()
        except Exception:  # noqa: BLE001
            pass


def load_states(cfg: InjectionConfig) -> dict[str, Any]:
    md = gen_metadata_by_id(cfg.metadata)
    files = sorted(cfg.raw_dir.glob("gen_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        raise FileNotFoundError(f"{cfg.raw_dir} holds no gen_*.npz")
    tasks = [(str(p), int(p.stem.split("_")[1])) for p in files]
    if cfg.limit is not None:
        tasks = tasks[: cfg.limit]
    paths: list[F32] = []
    gen_ids: list[int] = []
    topics: list[int] = []
    cov: list[list[float]] = []
    skipped: list[dict[str, Any]] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8, initializer=_init, initargs=(cfg.layer,)) as ex:
        for i, r in enumerate(ex.map(_load_one, tasks, chunksize=4)):
            if "error" in r:
                skipped.append({"gen_id": r["gen_id"], "reason": r["error"]})
                logger.warning(f"gen_{r['gen_id']:03d} SKIPPED: {r['error']}")
                continue
            g = int(r["gen_id"])
            rec = md.get(g)
            if rec is None:
                skipped.append({"gen_id": g, "reason": "no metadata record"})
                continue
            paths.append(r["X"])
            gen_ids.append(g)
            topics.append(int(rec["topic_idx"]))
            cov.append([float(rec["prompt_length"]), float(rec["num_generated_tokens"])])
            if (i + 1) % 50 == 0:
                logger.info(f"  loaded {i+1}/{len(tasks)} in {time.time()-t0:.0f}s")
    if not paths:
        raise RuntimeError("every generation was skipped — refusing an empty corpus")
    logger.info(f"states: {len(paths)} paths, d={paths[0].shape[1]}, "
                f"T mean {np.mean([p.shape[0] for p in paths]):.0f}, {len(skipped)} skipped "
                f"({time.time()-t0:.0f}s)")
    return {"paths": paths, "gen_ids": gen_ids, "topics": np.asarray(topics, dtype=np.int64),
            "C": np.asarray(cov, dtype=np.float64), "skipped": skipped}


# ───────────────────────────────────────────────────────────── bases + arms

def load_pca(pca_path: Path, layer: int, k: int) -> tuple[F32, F32]:
    if not pca_path.exists():
        raise FileNotFoundError(f"PCA calibration not found: {pca_path}")
    with open(pca_path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"{pca_path}: expected a per-layer dict, got {type(obj)!r}")
    vals = list(obj.values())
    if not (vals and isinstance(vals[0], dict) and "components" in vals[0]):
        raise TypeError(f"{pca_path}: not a per-layer PCA dict (pooled model refused)")
    keyed = {int(kk): v for kk, v in obj.items()}
    if layer not in keyed:
        raise KeyError(f"{pca_path}: no basis for layer {layer} (have {sorted(keyed)})")
    comp = np.asarray(keyed[layer]["components"], dtype=np.float32)
    mean = np.asarray(keyed[layer]["mean"], dtype=np.float32)
    if k > comp.shape[0]:
        raise ValueError(f"k={k} > basis rank {comp.shape[0]}")
    return np.ascontiguousarray(comp[:k]), mean


def orthonormalise(rows: F64) -> F64:
    """Gram-Schmidt; a direction that collapses into the span raises rather than silently
    yielding a rank-deficient basis (which would make the signature degenerate)."""
    out: list[F64] = []
    for i, r in enumerate(rows):
        v = np.asarray(r, dtype=np.float64).copy()
        for u in out:
            v -= float(v @ u) * u
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-8:
            raise ValueError(f"basis row {i} is inside the span of the earlier rows (norm {nrm:g})")
        out.append(v / nrm)
    return np.stack(out)


def coverage(v: F64, basis: F64) -> float:
    """Fraction of ||v||^2 inside the (orthonormal) row space of ``basis`` — the
    projection-adequacy gate (FIRSTREAD Ruling 1)."""
    v = np.asarray(v, dtype=np.float64)
    nv = float(v @ v)
    if nv <= 0:
        raise ValueError("zero-norm direction")
    return float(((basis @ v) ** 2).sum() / nv)


def arm_perturbation(T: int, kind: str, alpha: float, v: F64) -> F64:
    """[T, d] additive perturbation for one arm."""
    if kind == "const":
        w = np.ones(T)
    elif kind == "step":
        w = (np.arange(T) >= T // 2).astype(np.float64)
    elif kind == "ramp":
        w = np.arange(T) / max(T - 1, 1)
    else:
        raise ValueError(f"unknown arm kind {kind!r}")
    return alpha * np.outer(w, v)


def sig_rows(paths: list[F32], pca_mean: F32, basis: F64, *, layer: int,
             level: Literal[1, 2], perturb: tuple[str, float, F64] | None,
             permutation_seed: int | None) -> F32:
    """Project every path onto ``basis`` (after the PCA mean shift) and stack signature rows.

    ``perturb`` adds the arm's [T, d] perturbation in FULL space before projection, which is
    where the real injection lives.
    """
    k = int(basis.shape[0])
    cfg = PathSignatureConfig(
        layer_indices=(layer,), n_components=k, level=level, time_augment=True,
        permute_increments=permutation_seed is not None, permutation_seed=permutation_seed,
        basis_label="custom",
    )
    pb = ProjectionBasis(components=np.ascontiguousarray(basis),
                         mean=pca_mean.astype(np.float64), label="custom")
    rows: list[F64] = []
    for X in paths:
        Xf = X.astype(np.float64)
        if perturb is not None:
            kind, alpha, v = perturb
            Xf = Xf + arm_perturbation(int(Xf.shape[0]), kind, alpha, v)
        feats, _T = signature_features_from_path(Xf, pb, cfg)
        rows.append(feats)
    return np.stack(rows).astype(np.float32)


# ─────────────────────────────────────────────────────────────────── ladder

def ladder_fixed(X: F32, y: NDArray[np.int_], topic: NDArray[np.int_], C: F64,
                 device: str, name: str) -> dict[str, Any]:
    """Identical to ``pathsig_read_e1.ladder`` with nclass=2 (kept local so the binary
    arm-label cells and the E1 multiclass cells provably share one implementation path)."""
    accs: dict[str, list[float]] = {"logit": [], "deep": []}
    for seed in range(SEEDS):
        for tr, te in GroupKFold(5).split(X, y, topic):
            Ztr, Zte = preprocess_fold_gpu(X[tr], X[te], C[tr], C[te], True, device)
            for arch in ("logit", "deep"):
                ta, _ = train_eval(Ztr, y[tr], Zte, y[te], arch, seed, device,
                                   deep_epochs=DEEP_EPOCHS, lbfgs_l2=LBFGS_L2, nclass=2, k=K)
                accs[arch].append(ta)
    out = {a: {"test": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
           for a, v in accs.items()}
    logger.info(f"  {name}: logit {out['logit']['test']:.1%}±{out['logit']['std']:.1%}  "
                f"deep {out['deep']['test']:.1%}±{out['deep']['std']:.1%} (chance 50.0%, n={len(y)})")
    return out


def ladder_infold_basis(paths: list[F32], pca_mean: F32, pca_ctx: F32, v_true: F64,
                        y: NDArray[np.int_], topic: NDArray[np.int_], C: F64,
                        *, layer: int, level: Literal[1, 2], arm: str, alpha: float,
                        permutation_seed: int | None, device: str, name: str) -> dict[str, Any]:
    """B1: the discriminative direction is REFIT INSIDE EVERY FOLD from the train rows only.

    n rows = 2 * n_gen (unsteered then steered), so ``y`` indexes arms and the paired generation
    of row i is ``i % n_gen``. The direction is the train-fold arm mean-difference of the
    per-generation mean state; the test fold contributes nothing to it.
    """
    n_gen = len(paths)
    if len(y) != 2 * n_gen:
        raise ValueError(f"expected 2*{n_gen} rows, got {len(y)}")
    accs: dict[str, list[float]] = {"logit": [], "deep": []}
    cos_to_v: list[float] = []
    for seed in range(SEEDS):
        for tr, te in GroupKFold(5).split(np.zeros((len(y), 1)), y, topic):
            mu = {0: [], 1: []}
            for i in tr:
                gi = int(i % n_gen)
                Xf = paths[gi].astype(np.float64) - pca_mean.astype(np.float64)
                if y[i] == 1:
                    Xf = Xf + arm_perturbation(int(Xf.shape[0]), arm, alpha, v_true)
                mu[int(y[i])].append(Xf.mean(axis=0))
            if not mu[0] or not mu[1]:
                raise ValueError("a fold lost one arm entirely — the pairing is broken")
            d = np.mean(mu[1], axis=0) - np.mean(mu[0], axis=0)
            nd = float(np.linalg.norm(d))
            if nd < 1e-12:
                # A pure translation has ZERO mean-difference only if it cancels; it does not.
                raise ValueError("train-fold arm mean-difference is numerically zero")
            d = d / nd
            cos_to_v.append(float(abs(d @ (v_true / np.linalg.norm(v_true)))))
            basis = orthonormalise(np.vstack([d[None, :], pca_ctx.astype(np.float64)]))
            Xall = sig_rows(paths, pca_mean, basis, layer=layer, level=level, perturb=None,
                            permutation_seed=permutation_seed)
            Xst = sig_rows(paths, pca_mean, basis, layer=layer, level=level,
                           perturb=(arm, alpha, v_true), permutation_seed=permutation_seed)
            X = np.vstack([Xall, Xst])
            Ztr, Zte = preprocess_fold_gpu(X[tr], X[te], C[tr], C[te], True, device)
            for arch in ("logit", "deep"):
                ta, _ = train_eval(Ztr, y[tr], Zte, y[te], arch, seed, device,
                                   deep_epochs=DEEP_EPOCHS, lbfgs_l2=LBFGS_L2, nclass=2, k=K)
                accs[arch].append(ta)
    out = {a: {"test": round(float(np.mean(vv)), 4), "std": round(float(np.std(vv)), 4)}
           for a, vv in accs.items()}
    out["basis_cos_to_v_mean"] = round(float(np.mean(cos_to_v)), 4)
    logger.info(f"  {name}: logit {out['logit']['test']:.1%}  deep {out['deep']['test']:.1%}  "
                f"(in-fold basis cos to v = {out['basis_cos_to_v_mean']:.3f})")
    return out


# ───────────────────────────────────────────────────────────────────── legs

def leg_a_invariance(bank_npz: Path, layer: int, k: int) -> dict[str, Any]:
    """J2-a — a constant offset is in the estimator's exact null space. Machine-precision check
    on the banked s51 projected paths (the very data E2 was read from)."""
    z = np.load(bank_npz)
    P, off = z["paths_pc"], z["offsets"]
    rng = np.random.default_rng(20260911)
    c = rng.normal(size=k) * float(np.abs(P[:, :k]).mean())
    cfg2 = PathSignatureConfig(layer_indices=(layer,), n_components=k, level=2,
                              time_augment=True, basis_label="custom")
    eye = ProjectionBasis(components=np.eye(k, dtype=np.float64), mean=None, label="custom")
    d1: list[float] = []
    d2: list[float] = []
    scale: list[float] = []
    n_check = min(40, len(off) - 1)
    for i in range(n_check):
        a, b = int(off[i]), int(off[i + 1])
        X = P[a:b, :k].astype(np.float64)
        f0, _ = signature_features_from_path(X, eye, cfg2)
        f1, _ = signature_features_from_path(X + c[None, :], eye, cfg2)
        n_lvl1 = k + 1                                   # k coords + the augmented clock
        d1.append(float(np.max(np.abs(f0[:n_lvl1] - f1[:n_lvl1]))))
        d2.append(float(np.max(np.abs(f0[n_lvl1:] - f1[n_lvl1:]))))
        scale.append(float(np.max(np.abs(f0))))
    return {
        "what": "level-1 and level-2 log-signature features, real path vs path + constant offset",
        "bank": str(bank_npz), "k": k, "n_paths_checked": n_check,
        "offset_norm": round(float(np.linalg.norm(c)), 6),
        "max_abs_delta_lvl1": float(np.max(d1)),
        "max_abs_delta_lvl2": float(np.max(d2)),
        "feature_scale_max": float(np.max(scale)),
        "relative_lvl2": float(np.max(d2) / max(np.max(scale), 1e-30)),
        "reading": "a constant added at every generated position is invisible to BOTH levels "
                   "(centering at X[0] makes the family translation-invariant by construction)",
    }


def leg_b_anatomy(bank_npz: Path, k: int) -> dict[str, Any]:
    """J2-b — how much of the REAL matched-pair arm difference is a pure translation?"""
    z = np.load(bank_npz)
    P, off, y, gid = z["paths_pc"], z["offsets"], z["y"], z["gen_ids"]
    idx_by_arm: dict[int, dict[int, int]] = {0: {}, 1: {}}
    for i in range(len(y)):
        idx_by_arm[int(y[i])][int(gid[i])] = i
    common = sorted(set(idx_by_arm[0]) & set(idx_by_arm[1]))
    if not common:
        raise ValueError(f"{bank_npz}: no matched pairs")
    fr: list[float] = []
    dn: list[float] = []
    pn: list[float] = []
    for g in common:
        i0, i1 = idx_by_arm[0][g], idx_by_arm[1][g]
        a0, b0 = int(off[i0]), int(off[i0 + 1])
        a1, b1 = int(off[i1]), int(off[i1 + 1])
        T = min(b0 - a0, b1 - a1)
        if T < 3:
            continue
        D = P[a1:a1 + T, :k].astype(np.float64) - P[a0:a0 + T, :k].astype(np.float64)
        m = D.mean(axis=0)
        tot = float((D ** 2).sum(axis=1).mean())
        if tot <= 0:
            continue
        fr.append(float((m ** 2).sum() / tot))
        dn.append(float(np.sqrt(tot)))
        pn.append(float(np.sqrt((P[a0:a0 + T, :k].astype(np.float64) ** 2).sum(axis=1).mean())))
    if not fr:
        raise ValueError("no usable matched pairs for the anatomy")
    return {
        "what": "matched-pair difference D(t) = steered - unsteered in the banked k=8 projection",
        "n_pairs": len(fr), "k": k,
        "translation_energy_fraction": {
            "mean": round(float(np.mean(fr)), 4), "median": round(float(np.median(fr)), 4),
            "p10": round(float(np.percentile(fr, 10)), 4),
            "p90": round(float(np.percentile(fr, 90)), 4),
        },
        "rms_difference_norm": round(float(np.mean(dn)), 4),
        "rms_path_norm": round(float(np.mean(pn)), 4),
        "difference_to_path_ratio": round(float(np.mean(dn) / max(np.mean(pn), 1e-30)), 5),
        "caveat": "this is the difference AS SEEN IN THE 8-dim basis that holds 0.48% of the "
                  "perturbation's norm; the full-space anatomy needs the states, which are gone",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--pca", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--vectors", type=Path, required=True)
    ap.add_argument("--vector-key", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--k-pa", type=int, default=8)
    ap.add_argument("--k-ctx", type=int, default=4)
    ap.add_argument("--s51-bank", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-b1", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    cfg = InjectionConfig(
        raw_dir=a.raw_dir, metadata=a.metadata, pca=a.pca, layer=a.layer, vectors=a.vectors,
        vector_key=a.vector_key, alpha=a.alpha, k_pa=a.k_pa, k_ctx=a.k_ctx,
        s51_bank=a.s51_bank, device=a.device, out=a.out, limit=a.limit,
    )
    result: dict[str, Any] = {
        "cell": "JOB 2 — what a constant injection looks like in path terms (3B, L14)",
        "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED, not quotable",
        "blocker": "the specified s51 re-cut needs the full [T,3072] states; only [T,8] "
                   "projections were banked by leg 2, so B1/B2 on the REAL perturbation is "
                   "not runnable without the ~10-20 GPU-min replay this enactment forbids",
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in cfg.model_dump().items()},
        "augmentation": "t/(T-1) normalised — pacing, not duration",
        "folds": "GroupKFold(5) by topic_idx x 3 seeds, per-fold length-residualised",
        "null": f"increment permutation, seeds {list(NULL_SEEDS)}",
    }

    def bank_out() -> None:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        cfg.out.write_text(json.dumps(result, indent=1))

    if cfg.s51_bank is not None:
        result["J2a_invariance"] = leg_a_invariance(cfg.s51_bank, cfg.layer, cfg.k_pa)
        logger.info(f"J2-a invariance: {result['J2a_invariance']}")
        result["J2b_anatomy"] = leg_b_anatomy(cfg.s51_bank, cfg.k_pa)
        logger.info(f"J2-b anatomy: {result['J2b_anatomy']}")
        bank_out()

    # ── J2-c: the synthetic-injection cell on local banked 3B states ────────
    zv = np.load(cfg.vectors)
    if cfg.vector_key not in zv.files:
        raise KeyError(f"{cfg.vectors}: no key {cfg.vector_key!r} (have {sorted(zv.files)})")
    v_true = np.asarray(zv[cfg.vector_key], dtype=np.float64).ravel()

    comp_pa, pca_mean = load_pca(cfg.pca, cfg.layer, cfg.k_pa)
    if v_true.shape[0] != pca_mean.shape[0]:
        raise ValueError(f"vector dim {v_true.shape[0]} != basis dim {pca_mean.shape[0]}")
    pca_ctx = comp_pa[: cfg.k_ctx]

    st = load_states(cfg)
    paths: list[F32] = st["paths"]
    n_gen = len(paths)
    y = np.concatenate([np.zeros(n_gen, dtype=np.int64), np.ones(n_gen, dtype=np.int64)])
    topic = np.concatenate([st["topics"], st["topics"]])
    C = np.vstack([st["C"], st["C"]])

    v_hat = v_true / np.linalg.norm(v_true)
    B2 = orthonormalise(np.vstack([v_hat[None, :], pca_ctx.astype(np.float64)]))
    PA = comp_pa.astype(np.float64)
    result["bases"] = {
        "PA": {"rows": int(PA.shape[0]), "v_coverage": round(coverage(v_true, PA), 6),
               "note": "the E2 basis — carried for the adequacy comparison only"},
        "B2": {"rows": int(B2.shape[0]), "v_coverage": round(coverage(v_true, B2), 6),
               "note": "v_hat is an explicit basis row; the shuffle null uses the IDENTICAL "
                       "columns, so order is the only thing it does not hold fixed"},
    }
    result["corpus"] = {"n_generations": n_gen, "n_rows": int(len(y)),
                        "n_topics": int(len(set(topic.tolist()))),
                        "alpha": cfg.alpha, "vector": cfg.vector_key,
                        "skipped": st["skipped"]}
    logger.info(f"bases: PA coverage {result['bases']['PA']['v_coverage']:.6f}  "
                f"B2 coverage {result['bases']['B2']['v_coverage']:.6f}")
    bank_out()

    result["J2c_cells"] = {}
    for arm in ARM_KINDS:
        for bname, basis in (("B2", B2), ("PA", PA)):
            base_rows = {lv: sig_rows(paths, pca_mean, basis, layer=cfg.layer, level=lv,
                                      perturb=None, permutation_seed=None) for lv in (1, 2)}
            for lv in (1, 2):
                Xs = sig_rows(paths, pca_mean, basis, layer=cfg.layer, level=lv,
                              perturb=(arm, cfg.alpha, v_true), permutation_seed=None)
                X = np.vstack([base_rows[lv], Xs])
                key = f"{arm}_{bname}_lvl{lv}"
                result["J2c_cells"][key] = {
                    "P": int(X.shape[1]),
                    "mean_abs_feature_delta": round(
                        float(np.mean(np.abs(Xs - base_rows[lv]))), 10),
                    **ladder_fixed(X, y, topic, C, cfg.device, f"{arm} / {bname} / lvl{lv}"),
                }
                bank_out()
            # null on the level-1+2 cell only (the order question)
            nulls = []
            for s in NULL_SEEDS:
                Xu = sig_rows(paths, pca_mean, basis, layer=cfg.layer, level=2, perturb=None,
                              permutation_seed=s)
                Xs = sig_rows(paths, pca_mean, basis, layer=cfg.layer, level=2,
                              perturb=(arm, cfg.alpha, v_true), permutation_seed=s)
                nulls.append({"seed": s, **ladder_fixed(np.vstack([Xu, Xs]), y, topic, C,
                                                        cfg.device, f"{arm}/{bname} NULL s{s}")})
            real = result["J2c_cells"][f"{arm}_{bname}_lvl2"]
            result["J2c_cells"][f"{arm}_{bname}_lvl2"]["null"] = nulls
            result["J2c_cells"][f"{arm}_{bname}_lvl2"]["real_minus_null"] = {
                arch: round(float(real[arch]["test"] - np.mean([n[arch]["test"] for n in nulls])), 4)
                for arch in ("logit", "deep")
            }
            bank_out()

        if not a.skip_b1:
            for lv in (1, 2):
                result["J2c_cells"][f"{arm}_B1_lvl{lv}"] = ladder_infold_basis(
                    paths, pca_mean, pca_ctx, v_true, y, topic, C, layer=cfg.layer, level=lv,
                    arm=arm, alpha=cfg.alpha, permutation_seed=None, device=cfg.device,
                    name=f"{arm} / B1 (in-fold refit) / lvl{lv}")
                bank_out()

    bank_out()
    logger.info(f"job 2 banked (first-read pending) -> {cfg.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
