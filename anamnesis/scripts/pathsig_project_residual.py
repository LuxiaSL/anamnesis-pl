"""LEG 1 (CPU, no GPU) — project banked v3 raw residual paths onto the banked per-layer
calibration PCA, emitting one ``[T, k]`` trajectory per generation.

Brief: ``research/planning/SPEC-path-signature-family-2026-09-11.md`` §4b ("DESIGN — project
on-node, pull the projections"). **Exploratory; nothing here is quotable pre-first-read (C§8).**

WHY THIS SCRIPT EXISTS
----------------------
A level-2 signature needs the *path*, not a summary of it, and the path at one site for one 8B
generation is 511 × 4096 × 2 B ≈ 4 MB — ~6 GB for the 1,470-generation corpus, before the
all-layer raw npz (~486 MB/gen) is even opened. Projecting on the node and pulling only the
``[T, k]`` result turns leg 1 from a terabyte problem into a ~25 MB one, after which every
signature computation and every fold is local CPU.

WHAT IT COMPUTES
----------------
For each generation's v3 raw npz:

1. ``hidden_states`` → row ``layer + 1`` (index 0 is the embedding — CLAUDE.md gotcha). The v3
   raw's rows are ALREADY generated positions only (``replay_extract`` slices ``P .. P+T-1``),
   so no prefill skip is applied here; ``T`` is the generated-token count minus one.
2. **Positional correction** — ``h[t] -= positional_means[layer+1, min(prompt_length + t, max_pos-1)]``,
   i.e. exactly ``state_extractor._correct_hidden_state``. This is NOT optional for the
   estimand of record: ``pca_model_corrected.pkl`` was *fit on positionally-corrected states*
   (``run_corrected_pca.py`` docstring), so projecting uncorrected states onto it is a
   fit/apply mismatch — the very bug audit C5 found in the pooled basis.
3. Projection onto the per-layer PCA: ``(h - pca_mean) @ components[:k].T`` → ``[T, k]`` float32.

The uncorrected variant is banked alongside as a secondary read (cheap, same pass) so the
correction's contribution is measurable rather than assumed. **P-A corrected is the estimand
of record.**

``k`` is a prefix: the top-4 projection is literally the first 4 columns of the top-8 one
(PCA components are ordered and shared), so we store k=8 once and slice downstream.

ABSENCE IS REPORTED, NEVER ZERO-FILLED
--------------------------------------
Every skipped generation lands in the sidecar's ``skipped`` list with a reason string. A run
whose skip count is nonzero prints it loudly. Nothing is silently imputed.

Usage (node1, CPU only — no GPU, no model load)::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m anamnesis.scripts.pathsig_project_residual \\
        --model 8b --runs 8b_fat_01,8b_fat_ext --layer 16 --k 8 \\
        --runs-root /models/anamnesis-extract/runs \\
        --calib-dir /models/anamnesis-extract/calibration/8b \\
        --out-dir /models/anamnesis-extract/pathsig/8b_L16 --workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]

# ── worker globals (set once per process by the pool initializer; never pickled per task) ──
_G_COMPONENTS: F32 | None = None
_G_PCA_MEAN: F32 | None = None
_G_POS_MEAN_LAYER: F32 | None = None   # [max_pos, hidden] for the ONE requested layer row
_G_LAYER: int = -1


class ProjectionJobConfig(BaseModel):
    """Frozen provenance for one projection pass — written verbatim into the sidecar."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(description="Model token, e.g. '3b' or '8b'.")
    runs: tuple[str, ...] = Field(description="Run directory names under runs_root.")
    layer: int = Field(ge=0, description="Transformer layer index (site of record).")
    k: int = Field(ge=2, le=50, description="Projection rank kept (top-k PCA directions).")
    runs_root: str
    calib_dir: str
    pca_file: str = Field(description="Which PCA artefact was used (per-layer corrected only).")
    basis_label: str = "pcaA"
    raw_subdir: str = "raw_tensors_v3"

    @field_validator("runs")
    @classmethod
    def _runs_nonempty(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("at least one run must be named")
        return v


class SkippedGeneration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    run: str
    gen_id: int
    reason: str


def _load_per_layer_pca(pca_path: Path, layer: int) -> tuple[F32, F32]:
    """Load ``pca_model_corrected.pkl`` (a genuine per-layer dict) and return (components, mean).

    The pooled ``pca_model.pkl`` is deliberately NOT accepted here: it is one basis fit on
    samples pooled across every ``pca_layer`` (audit C5), and the spec asks for the per-layer
    artefact. A pooled file raises rather than silently standing in.
    """
    if not pca_path.exists():
        raise FileNotFoundError(f"PCA calibration not found: {pca_path}")
    with open(pca_path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"{pca_path}: expected a dict, got {type(obj)!r}")
    vals = list(obj.values())
    if not (vals and isinstance(vals[0], dict) and "components" in vals[0]):
        raise TypeError(
            f"{pca_path}: not a per-layer PCA dict (looks like the POOLED pca_model.pkl). "
            f"The spec requires the per-layer basis; refusing to substitute."
        )
    keyed = {int(k): v for k, v in obj.items()}
    if layer not in keyed:
        raise KeyError(
            f"{pca_path}: no PCA basis for layer {layer} (have {sorted(keyed)})"
        )
    comp = np.asarray(keyed[layer]["components"], dtype=np.float32)
    mean = np.asarray(keyed[layer]["mean"], dtype=np.float32)
    if comp.ndim != 2 or mean.ndim != 1 or comp.shape[1] != mean.shape[0]:
        raise ValueError(f"{pca_path}: malformed basis for layer {layer}: {comp.shape}/{mean.shape}")
    if not (np.all(np.isfinite(comp)) and np.all(np.isfinite(mean))):
        raise ValueError(f"{pca_path}: non-finite PCA basis for layer {layer}")
    return comp, mean


def _init_worker(components: F32, pca_mean: F32, pos_mean_layer: F32, layer: int) -> None:
    global _G_COMPONENTS, _G_PCA_MEAN, _G_POS_MEAN_LAYER, _G_LAYER
    _G_COMPONENTS, _G_PCA_MEAN, _G_POS_MEAN_LAYER, _G_LAYER = (
        components, pca_mean, pos_mean_layer, layer
    )


def _project_one(task: tuple[str, str, int]) -> dict[str, Any]:
    """Worker: one generation npz → ``{gen_id, T, pc [T,k], raw [T,k]}`` or ``{error}``.

    Module-level (picklable). Reads ONLY the ``hidden_states`` / ``prompt_length`` keys —
    ``np.load`` is lazy, so the attention/key/value tensors are never decompressed.
    """
    run, path, gen_id = task
    assert _G_COMPONENTS is not None and _G_PCA_MEAN is not None and _G_POS_MEAN_LAYER is not None
    try:
        z = np.load(path, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001 — a corrupt npz must be reported, not crash the pass
        return {"run": run, "gen_id": gen_id, "error": f"npz load failed: {exc!r}"}
    try:
        if "hidden_states" not in z.files:
            return {"run": run, "gen_id": gen_id, "error": "no hidden_states key"}
        hs = z["hidden_states"]
        if hs.size == 0 or hs.ndim != 3:
            return {"run": run, "gen_id": gen_id, "error": f"hidden_states shape {hs.shape}"}
        arr_idx = _G_LAYER + 1                       # index 0 = embedding (CLAUDE.md)
        if not (0 <= arr_idx < hs.shape[1]):
            return {"run": run, "gen_id": gen_id,
                    "error": f"layer {_G_LAYER} (row {arr_idx}) absent; {hs.shape[1]} rows"}
        X = hs[:, arr_idx, :].astype(np.float32)     # [T, d]
        T = int(X.shape[0])
        if T < 3:
            return {"run": run, "gen_id": gen_id, "error": f"T={T} < 3 (degenerate path)"}
        if not np.all(np.isfinite(X)):
            return {"run": run, "gen_id": gen_id, "error": "non-finite hidden states"}
        if float(np.linalg.norm(X[0])) < 1e-6:
            return {"run": run, "gen_id": gen_id,
                    "error": f"layer {_G_LAYER} zero-filled in this bank (not saved)"}
        if X.shape[1] != _G_PCA_MEAN.shape[0]:
            return {"run": run, "gen_id": gen_id,
                    "error": f"hidden dim {X.shape[1]} != basis dim {_G_PCA_MEAN.shape[0]}"}

        plen = int(z["prompt_length"]) if "prompt_length" in z.files else 0
        max_pos = _G_POS_MEAN_LAYER.shape[0]
        abs_pos = np.minimum(plen + np.arange(T), max_pos - 1)
        Xc = X - _G_POS_MEAN_LAYER[abs_pos]          # positional correction (state_extractor)

        comp = _G_COMPONENTS
        proj_pc = ((Xc - _G_PCA_MEAN) @ comp.T).astype(np.float32)
        proj_raw = ((X - _G_PCA_MEAN) @ comp.T).astype(np.float32)
        if not (np.all(np.isfinite(proj_pc)) and np.all(np.isfinite(proj_raw))):
            return {"run": run, "gen_id": gen_id, "error": "non-finite projection"}
        return {"run": run, "gen_id": gen_id, "T": T,
                "pc": proj_pc, "raw": proj_raw, "prompt_length": plen}
    except Exception as exc:  # noqa: BLE001
        return {"run": run, "gen_id": gen_id, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            z.close()
        except Exception:  # noqa: BLE001
            pass


def _unwrap_generations(meta: Any) -> list[dict[str, Any]]:
    """metadata.json wraps generations under a 'generations' key (CLAUDE.md gotcha)."""
    if isinstance(meta, dict) and "generations" in meta:
        return list(meta["generations"])
    if isinstance(meta, list):
        return list(meta)
    raise ValueError("metadata.json has neither a 'generations' key nor a top-level list")


def run_projection(cfg: ProjectionJobConfig, out_dir: Path, workers: int,
                   limit: int | None = None) -> dict[str, Any]:
    runs_root = Path(cfg.runs_root)
    calib_dir = Path(cfg.calib_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    components, pca_mean = _load_per_layer_pca(calib_dir / cfg.pca_file, cfg.layer)
    if cfg.k > components.shape[0]:
        raise ValueError(f"k={cfg.k} > basis rank {components.shape[0]}")
    components = np.ascontiguousarray(components[: cfg.k])
    logger.info(f"PCA basis layer {cfg.layer}: components {components.shape}, mean {pca_mean.shape}")

    pm_path = calib_dir / "positional_means.npz"
    if not pm_path.exists():
        raise FileNotFoundError(
            f"positional_means not found at {pm_path} — the corrected PCA basis was FIT on "
            f"corrected states, so projecting without them is a fit/apply mismatch. Refusing."
        )
    pos_means = np.load(pm_path)["positional_means"].astype(np.float32)
    logger.info(f"positional_means {pos_means.shape}")
    arr_idx = cfg.layer + 1
    if not (0 <= arr_idx < pos_means.shape[0]):
        raise ValueError(f"positional_means has no row {arr_idx} for layer {cfg.layer}")
    pos_mean_layer = np.ascontiguousarray(pos_means[arr_idx])
    del pos_means

    summary: dict[str, Any] = {"config": cfg.model_dump(), "runs": {}}
    for run in cfg.runs:
        run_dir = runs_root / run
        raw_dir = run_dir / cfg.raw_subdir
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"{raw_dir} is not a directory (dangling symlink?)")
        meta_path = run_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} missing")
        with open(meta_path) as f:
            gens = _unwrap_generations(json.load(f))
        meta_by_id = {int(g["generation_id"]): g for g in gens}

        files = sorted(raw_dir.glob("gen_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
        tasks = [(run, str(p), int(p.stem.split("_")[1])) for p in files]
        if limit is not None:
            tasks = tasks[:limit]
        logger.info(f"[{run}] {len(tasks)} raw npz found ({len(meta_by_id)} in metadata)")

        pc_blocks: list[F32] = []
        raw_blocks: list[F32] = []
        gen_ids: list[int] = []
        lengths: list[int] = []
        prompt_lengths: list[int] = []
        skipped: list[SkippedGeneration] = []

        t0 = time.time()
        chunk = max(1, len(tasks) // (max(workers, 1) * 8)) if tasks else 1
        if workers > 1:
            ex = ProcessPoolExecutor(
                max_workers=workers, initializer=_init_worker,
                initargs=(components, pca_mean, pos_mean_layer, cfg.layer),
            )
            results = ex.map(_project_one, tasks, chunksize=chunk)
        else:
            _init_worker(components, pca_mean, pos_mean_layer, cfg.layer)
            ex = None
            results = map(_project_one, tasks)
        try:
            for i, r in enumerate(results):
                if "error" in r:
                    skipped.append(SkippedGeneration(run=run, gen_id=r["gen_id"],
                                                     reason=str(r["error"])))
                    logger.warning(f"[{run}] gen_{r['gen_id']:03d} SKIPPED: {r['error']}")
                else:
                    gid = int(r["gen_id"])
                    if gid not in meta_by_id:
                        skipped.append(SkippedGeneration(run=run, gen_id=gid,
                                                         reason="no metadata record"))
                        continue
                    pc_blocks.append(r["pc"])
                    raw_blocks.append(r["raw"])
                    gen_ids.append(gid)
                    lengths.append(int(r["T"]))
                    prompt_lengths.append(int(r["prompt_length"]))
                if (i + 1) % 100 == 0:
                    el = time.time() - t0
                    logger.info(f"[{run}] {i+1}/{len(tasks)} in {el:.0f}s "
                                f"(ETA {(len(tasks)-i-1)/max((i+1)/el, 1e-9):.0f}s)")
        finally:
            if ex is not None:
                ex.shutdown(wait=True)

        if not gen_ids:
            raise RuntimeError(f"[{run}] every generation was skipped — refusing to emit an "
                               f"empty bank ({len(skipped)} skips)")

        offsets = np.zeros(len(gen_ids) + 1, dtype=np.int64)
        np.cumsum(np.asarray(lengths, dtype=np.int64), out=offsets[1:])
        out_npz = out_dir / f"paths_{run}_L{cfg.layer}_k{cfg.k}.npz"
        np.savez_compressed(
            out_npz,
            paths_pc=np.concatenate(pc_blocks, axis=0).astype(np.float32),
            paths_nopc=np.concatenate(raw_blocks, axis=0).astype(np.float32),
            offsets=offsets,
            gen_ids=np.asarray(gen_ids, dtype=np.int32),
            lengths=np.asarray(lengths, dtype=np.int32),
            prompt_lengths=np.asarray(prompt_lengths, dtype=np.int32),
        )
        side = {
            "run": run,
            "config": cfg.model_dump(),
            "n_generations": len(gen_ids),
            "n_raw_files": len(tasks),
            "n_skipped": len(skipped),
            "skipped": [s.model_dump() for s in skipped],
            "T_stats": {
                "min": int(np.min(lengths)), "max": int(np.max(lengths)),
                "mean": round(float(np.mean(lengths)), 2),
                "median": float(np.median(lengths)),
            },
            "labels": [
                {"gen_id": g,
                 "mode": meta_by_id[g].get("mode"),
                 "topic_idx": meta_by_id[g].get("topic_idx", meta_by_id[g].get("topic")),
                 "prompt_length": meta_by_id[g].get("prompt_length"),
                 "num_generated_tokens": meta_by_id[g].get("num_generated_tokens"),
                 "condition": meta_by_id[g].get("condition")}
                for g in gen_ids
            ],
            "npz": str(out_npz),
            "STATUS": "FIRST_READ_PENDING (C§8) — exploratory, not quotable",
        }
        side_path = out_dir / f"paths_{run}_L{cfg.layer}_k{cfg.k}.meta.json"
        side_path.write_text(json.dumps(side, indent=1))
        logger.info(f"[{run}] banked {len(gen_ids)} paths ({len(skipped)} skipped) -> {out_npz} "
                    f"({out_npz.stat().st_size/1e6:.1f} MB) in {time.time()-t0:.0f}s")
        summary["runs"][run] = {k: side[k] for k in
                                ("n_generations", "n_raw_files", "n_skipped", "T_stats", "npz")}
        summary["runs"][run]["skipped"] = side["skipped"]

    (out_dir / "projection_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", required=True, help="comma-separated run dir names")
    ap.add_argument("--layer", type=int, required=True, help="transformer layer index (site)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--runs-root", default="/models/anamnesis-extract/runs")
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--pca-file", default="pca_model_corrected.pkl")
    ap.add_argument("--raw-subdir", default="raw_tensors_v3")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N gens per run")
    args = ap.parse_args()

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")   # shared-node BLAS discipline

    cfg = ProjectionJobConfig(
        model=args.model,
        runs=tuple(r.strip() for r in args.runs.split(",") if r.strip()),
        layer=args.layer, k=args.k,
        runs_root=args.runs_root, calib_dir=args.calib_dir,
        pca_file=args.pca_file, raw_subdir=args.raw_subdir,
    )
    logger.info(f"config: {cfg.model_dump_json()}")
    try:
        summary = run_projection(cfg, Path(args.out_dir), args.workers, args.limit)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"projection pass aborted: {type(exc).__name__}: {exc}", exc_info=True)
        return 1
    total = sum(v["n_generations"] for v in summary["runs"].values())
    skips = sum(v["n_skipped"] for v in summary["runs"].values())
    logger.info(f"PROJECTION COMPLETE: {total} paths banked, {skips} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
