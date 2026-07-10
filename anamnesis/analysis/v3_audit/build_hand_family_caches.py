"""Recompute the 3 NEW hand-feature families over the full corpus → one aligned cache per model.

The families value_geometry / qk_geometry / kv_cka were validated to RUN (finite/sane on one gen)
but never measured for how WELL they discriminate. This builds their feature matrix over all
~900 hard-mode gens/model — using the SAME HARD filter + gen_uid as build_surface_caches, so the
rows align with the surface caches / encoder floors. Downstream (`hand_family_analysis.py`):
  (a) classify each family 5-way and place it against its surface's encoder FLOOR (method-limit), and
  (b) name the floor's discriminant via STRUCTURE COEFFICIENTS over these hand-features (vocabulary
      adequacy + the distill template) — the reframe's "name the encoder, don't beat it".

Calibration-free: value/qk/kv_cka read raw keys/values/queries only (no positional_means, no PCA).
CPU, pure-numpy families behind compute_features_v2_from_data. Parallel over gens (fork pool; pin
BLAS per worker — shared node). Needs the anamnesis package importable (PYTHONPATH=.../pipeline).

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=~/luxi-files/anamnesis-pl/pipeline \
    python build_hand_family_caches.py --models 3b,8b --workers 48
    python build_hand_family_caches.py --models 8b --limit 5 --workers 1   # smoke
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")        # shared node — pin BLAS per worker
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from anamnesis.config import ExtractionConfig, FeaturePipelineConfig, MODEL_PRESETS
from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data
from anamnesis.extraction.raw_saver import load_raw_tensors

HARD = {"linear", "socratic", "contrastive", "dialectical", "analogical"}
RUNS = Path(os.environ.get("ANAMNESIS_RUNS", "/models/anamnesis-extract/runs"))
OUT = Path(os.environ.get("HAND_CACHE_DIR", "."))
GROUPS: dict[str, list[str]] = {"3b": ["3b_fat_01", "3b_fat_ext"], "8b": ["8b_fat_01", "8b_fat_ext"]}
FAMILIES = ["value_geometry", "qk_geometry", "kv_cka"]


def load_metadata(run_dir: Path) -> dict[int, dict]:
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text())
    gens = meta["generations"] if isinstance(meta, dict) and "generations" in meta else meta
    return {int(g["generation_id"]): g for g in gens}


def _family_config() -> FeaturePipelineConfig:
    """Only the 3 new families; baseline + every other family OFF (so X = the 3 families, concatenated)."""
    return FeaturePipelineConfig(
        include_baseline_tiers=False,
        enable_value_geometry=True,
        enable_qk_geometry=True,
        enable_kv_cka=True,
    )


def _process_gen(task: tuple):
    """Worker: load one gen's raw tensors, compute the 3 families. Module-level (picklable for fork pool).
    Returns ("ok", uid, mode, topic, plen, glen, feats, names, slices) | ("err", uid, msg)."""
    run, raw_dir, gid, mode, topic, plen, glen, sampled_layers = task
    uid = f"{run}:{gid}"
    try:
        # Lean load: the 3 families read only k/v/q — skip hidden/attention/gate/logits.
        raw_data = load_raw_tensors(gid, Path(raw_dir), surfaces=("keys", "values", "queries"))
        ext_cfg = ExtractionConfig(sampled_layers=list(sampled_layers))
        res = compute_features_v2_from_data(raw_data, ext_cfg, _family_config())
        feats = np.asarray(res.features, dtype=np.float32)
        slices = {k: [int(v[0]), int(v[1])] for k, v in res.tier_slices.items()}
        return ("ok", uid, mode, int(topic), float(plen), float(glen),
                feats, list(res.feature_names), slices)
    except Exception as e:  # noqa: BLE001 — surface per-gen failures, don't kill the pool
        return ("err", uid, repr(e))


def build_model(model: str, workers: int, limit: int | None = None) -> None:
    preset = MODEL_PRESETS[model]
    sampled = preset.sampled_layers
    runs = GROUPS[model]

    # ── gather tasks (metadata only; hard modes only — aligns with the surface caches) ──
    tasks: list[tuple] = []
    for run in runs:
        run_dir = RUNS / run
        raw_dir = run_dir / "raw_tensors_v3"
        md = load_metadata(run_dir)
        if not raw_dir.is_dir() or not md:
            print(f"  [skip] {run}: raw_dir or metadata missing", flush=True)
            continue
        files = sorted(raw_dir.glob("gen_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
        for p in files:
            gid = int(p.stem.split("_")[1])
            g = md.get(gid)
            if g is None or g["mode"] not in HARD:
                continue
            tasks.append((run, str(raw_dir), gid, g["mode"], int(g["topic_idx"]),
                          g["prompt_length"], g["num_generated_tokens"], sampled))
    if limit is not None:
        tasks = tasks[:limit]
    print(f"  {len(tasks)} hard-mode gens queued; {workers} workers; sampled_layers={sampled}", flush=True)

    # ── compute in parallel (pure-numpy families; CPU) ──
    X_l, mode_l, topic_l, plen_l, glen_l, uid_l = [], [], [], [], [], []
    names: list[str] | None = None
    slices: dict | None = None
    n_feat: int | None = None
    n_err = 0
    runner = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    chunk = max(1, len(tasks) // (max(workers, 1) * 8)) if tasks else 1
    try:
        results = (runner.map(_process_gen, tasks, chunksize=chunk) if runner
                   else map(_process_gen, tasks))
        done = 0
        for r in results:
            done += 1
            if r[0] == "err":
                n_err += 1
                if n_err <= 5:
                    print(f"    [err] {r[1]}: {r[2]}", flush=True)
                continue
            _, uid, mode, topic, plen, glen, feats, fnames, fslices = r
            if n_feat is None:
                n_feat, names, slices = int(feats.shape[0]), fnames, fslices
                print(f"    n_feat={n_feat}  families={ {k: v for k, v in slices.items()} }", flush=True)
            elif feats.shape[0] != n_feat:
                print(f"    [warn] {uid} n_feat {feats.shape[0]} != {n_feat}; skip", flush=True)
                continue
            X_l.append(feats); mode_l.append(mode); topic_l.append(topic)
            plen_l.append(plen); glen_l.append(glen); uid_l.append(uid)
            if done % 200 == 0:
                print(f"    ...{done}/{len(tasks)} processed, {len(uid_l)} kept, {n_err} err", flush=True)
    finally:
        if runner is not None:
            runner.shutdown()

    if not X_l:
        print(f"  [{model}] nothing accumulated ({n_err} errors) — nothing written", flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    X = np.stack(X_l).astype(np.float32)
    fam_names = list(slices.keys())
    out_path = OUT / f"hand_cache_{model}.npz"
    np.savez_compressed(
        out_path,
        X=X,
        mode=np.array(mode_l), topic=np.array(topic_l, dtype=int),
        plen=np.array(plen_l, dtype=np.float32), glen=np.array(glen_l, dtype=np.float32),
        gen_uid=np.array(uid_l), feature_names=np.array(names),
        fam_name=np.array(fam_names),
        fam_start=np.array([slices[f][0] for f in fam_names], dtype=int),
        fam_end=np.array([slices[f][1] for f in fam_names], dtype=int),
    )
    mb = out_path.stat().st_size / 1e6
    nan = int(np.isnan(X).sum())
    print(f"  wrote {out_path.name}: X={X.shape} {X.dtype} ({mb:.1f} MB)  nan={nan}  "
          f"n={len(mode_l)}  topics={len(set(topic_l))}  err={n_err}", flush=True)
    for f in fam_names:
        s, e = slices[f]
        print(f"    {f:16s} [{s:4d}:{e:4d}]  ({e - s} feats)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="3b,8b")
    ap.add_argument("--workers", type=int, default=48,
                    help="process-pool size (pin OMP/OPENBLAS/MKL=1 in the launch env)")
    ap.add_argument("--limit", type=int, default=None, help="cap gens per model (smoke test)")
    args = ap.parse_args()
    print(f"RUNS={RUNS}  OUT={OUT.resolve()}  families={FAMILIES}  workers={args.workers}", flush=True)
    for model in args.models.split(","):
        model = model.strip()
        if model not in GROUPS:
            print(f"[skip] unknown model {model}", flush=True)
            continue
        print(f"\n===== {model} =====", flush=True)
        build_model(model, workers=args.workers, limit=args.limit)


if __name__ == "__main__":
    main()
