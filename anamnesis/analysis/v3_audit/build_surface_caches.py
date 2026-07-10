"""Build per-surface encoder caches from v3 raw tensors — the data layer for the
ALL-SURFACE encoder floor (Track B, redone honestly).

WHY: the Gate-B encoder (`encoder_floor.py`) — and the whole learned-representation lineage
(contrastive projection, "baby-encoder") — only ever saw the RESIDUAL stream
(`gate_a_lf_cache` H = 5 layers × 5 positions of hidden, flattened). "Encoder discovers over
ALL sources" was never built. This extracts each banked surface into an encoder-ready cache so
we can measure a per-surface learned FLOOR (decompose-by-source at the encoder level) and a
fused all-surface ceiling. Top question: does any non-residual surface — especially **values,
never featurized at all** — carry signal the residual encoder missed?

Surfaces (from `raw_saver.save_raw_tensors_v3`; verified on node1 2026-06-15):
  residual  hidden_states     (T, L+1, hidden)     ALL layers (idx0=embedding)        flatten
  keys      pre_rope_keys      (T, L, 8, 128)       ALL layers, per-KV-head            flatten
  values    v_proj_values      (T, L, 8, 128)       ALL layers, per-KV-head            flatten  [never featurized]
  queries   queries            (T, 7, qH, 128)      SAMPLED 7 layers, pre-RoPE         flatten
  gate      gate_activations   (T, 7, intermediate) SAMPLED 7 layers                   flatten
  attention attentions         (T, L, H, max_seq)   ALL layers, per-head, var-length   REGION-RESAMPLE

Simple surfaces are fixed-dim per token → flatten 5 sampled positions directly. **Attention rows
are variable-length** (step t attends over prompt_len+t positions), so they need a fixed-size
reduction before they can be an encoder input. We use the least-arbitrary one:
**region-aware resampling** — split each per-head row at the prompt/gen boundary, mean-pool the
prompt part → PROMPT_BINS and the gen-so-far part → GEN_BINS. Keeps per-head + per-layer
structure, keeps the prompt-vs-gen split explicit (clean length-residualization), preserves
coarse recency shape; it's "resizing", not hand-engineering (hand-features cover attention the
engineered way). attention is the TOP source, so the all-surface encoder must have it.

Each surface sampled at N_POS=5 evenly-spaced generation positions, flattened to fixed dim,
float16, one npz per (model, surface). Pure numpy, CPU. `np.load` is lazy → we touch only the
keys for the requested surfaces (attention-only runs don't load the big tensor for other passes).

Parallel over gens (process pool) — the work is I/O-bound (disk reads + npz zlib decompress), so a
single process barely loads the node (~1.2). Pin BLAS to 1 thread/worker (shared-node rule) and let
`--workers` scale the load. Run on node1 (BLAS pinned in the launch env so children inherit):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python build_surface_caches.py --models 3b,8b --workers 16
    python build_surface_caches.py --models 8b --surfaces values --limit 20 --workers 1   # smoke
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HARD = {"linear", "socratic", "contrastive", "dialectical", "analogical"}
RUNS = Path(os.environ.get("ANAMNESIS_RUNS", "/models/anamnesis-extract/runs"))
OUT = Path(os.environ.get("SURFACE_CACHE_DIR", "/dev/shm/anamnesis_surface_caches"))
GROUPS: dict[str, list[str]] = {"3b": ["3b_fat_01", "3b_fat_ext"], "8b": ["8b_fat_01", "8b_fat_ext"]}
# simple (fixed-dim per token) surfaces: name -> raw npz key. Flatten sampled positions directly.
SIMPLE: dict[str, str] = {
    "residual": "hidden_states",
    "keys": "pre_rope_keys",
    "values": "v_proj_values",
    "queries": "queries",
    "gate": "gate_activations",
}
ALL_SURFACES = list(SIMPLE) + ["attention"]
N_POS = 5
PROMPT_BINS, GEN_BINS = 8, 24          # region-aware attention reduction
ATTN_BINS = PROMPT_BINS + GEN_BINS


def sample_positions(T: int, n: int = N_POS) -> np.ndarray:
    """n evenly-spaced generation-step indices in [0, T). Fixed count regardless of T,
    so the flattened per-gen vector has constant dim across gens."""
    if T <= 0:
        return np.zeros(n, dtype=int)
    return np.linspace(0, T - 1, n).round().astype(int)


def load_metadata(run_dir: Path) -> dict[int, dict]:
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text())
    gens = meta["generations"] if isinstance(meta, dict) and "generations" in meta else meta
    return {int(g["generation_id"]): g for g in gens}


def _resample_rows(X: np.ndarray, B: int) -> np.ndarray:
    """(R, n) nonneg rows -> (R, B). Mean-pool into B contiguous bins (n>=B), linear-interp
    up (n<B), passthrough (n==B). The basis for the region-aware attention reduction."""
    R, n = X.shape
    if n == 0:
        return np.zeros((R, B), dtype=np.float64)
    Xf = X.astype(np.float64)
    if n == B:
        return Xf
    if n < B:
        xp = np.linspace(0.0, 1.0, n)
        xq = np.linspace(0.0, 1.0, B)
        out = np.empty((R, B), dtype=np.float64)
        for r in range(R):
            out[r] = np.interp(xq, xp, Xf[r])
        return out
    idx = (np.arange(n) * B) // n                     # bin id per source column
    out = np.zeros((R, B), dtype=np.float64)
    np.add.at(out, (slice(None), idx), Xf)            # sum columns into their bin (all rows)
    cnt = np.zeros(B, dtype=np.float64)
    np.add.at(cnt, idx, 1.0)
    return out / np.maximum(cnt, 1.0)[None, :]


def attention_vector(z: np.lib.npyio.NpzFile, pos: np.ndarray) -> np.ndarray | None:
    """Region-aware resample of per-head attention rows at the sampled positions.
    Returns flat (N_POS * L * H * ATTN_BINS,) float16, or None if attention absent."""
    if "attentions" not in z.files or "actual_lengths" not in z.files:
        return None
    attn = z["attentions"]                            # (T, L, H, max_seq) f16
    if attn.size == 0:
        return None
    alen = z["actual_lengths"]
    plen = int(z["prompt_length"])
    _, L, H, _ = attn.shape
    blocks = []
    for t in pos:
        al = int(alen[t])
        if al <= 0:
            blocks.append(np.zeros((L * H, ATTN_BINS), dtype=np.float64))
            continue
        pe = min(max(plen, 0), al)                     # prompt-end clamped into the row
        rows = attn[t].reshape(L * H, -1)[:, :al]      # (L*H, al) valid (unpadded) attention
        prm = _resample_rows(rows[:, :pe], PROMPT_BINS)
        gen = _resample_rows(rows[:, pe:al], GEN_BINS)
        blocks.append(np.concatenate([prm, gen], axis=1))   # (L*H, ATTN_BINS)
    return np.stack(blocks).reshape(-1).astype(np.float16)   # (N_POS*L*H*ATTN_BINS,)


def surface_vector(z: np.lib.npyio.NpzFile, surface: str, pos: np.ndarray, T: int) -> np.ndarray | None:
    """Fixed-dim per-gen vector for one surface, or None to skip this gen."""
    if surface == "attention":
        return attention_vector(z, pos)
    key = SIMPLE[surface]
    if key not in z.files:
        return None
    arr = z[key]
    if arr.size == 0 or arr.shape[0] < T:
        return None
    return arr[pos].reshape(-1).astype(np.float16)


def _process_gen(task: tuple) -> tuple | None:
    """Worker: load one gen's npz, extract requested surfaces. Returns (uid, mode, topic, plen,
    glen, {surface: vec}) or None. Module-level so it's picklable for the process pool."""
    run, path, gid, mode, topic, plen, glen, surfaces = task
    try:
        z = np.load(path, allow_pickle=True)  # lazy: only requested surfaces' keys are read
    except Exception:
        return None
    T = int(z["actual_lengths"].shape[0]) if "actual_lengths" in z.files else 0
    if T <= 0:
        return None
    pos = sample_positions(T)
    out: dict[str, np.ndarray] = {}
    for s in surfaces:
        vec = surface_vector(z, s, pos, T)
        if vec is None:
            return None
        out[s] = vec
    return (f"{run}:{gid}", mode, topic, float(plen), float(glen), out)


def build_model(model: str, surfaces: list[str], workers: int, limit: int | None = None) -> None:
    runs = GROUPS[model]
    # ── gather tasks (cheap: metadata only) ──
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
            tasks.append((run, str(p), gid, g["mode"], int(g["topic_idx"]),
                          g["prompt_length"], g["num_generated_tokens"], surfaces))
    if limit is not None:
        tasks = tasks[:limit]
    print(f"  {len(tasks)} hard-mode gens queued; {workers} workers", flush=True)

    # ── extract in parallel (I/O + decompress bound) ──
    acc: dict[str, list[np.ndarray]] = {s: [] for s in surfaces}
    dims: dict[str, int] = {}
    mode_l, topic_l, plen_l, glen_l, uid_l = [], [], [], [], []
    chunk = max(1, len(tasks) // (max(workers, 1) * 8)) if tasks else 1
    runner = (ProcessPoolExecutor(max_workers=workers) if workers > 1 else None)
    try:
        results = (runner.map(_process_gen, tasks, chunksize=chunk) if runner
                   else map(_process_gen, tasks))
        done = 0
        for r in results:
            done += 1
            if r is None:
                continue
            uid, mode, topic, plen, glen, surf = r
            ok = True
            for s in surfaces:
                d = int(surf[s].shape[0])
                if s in dims and dims[s] != d:
                    print(f"    [warn] {uid} {s} dim {d} != {dims[s]}; skip gen", flush=True)
                    ok = False
                    break
                dims.setdefault(s, d)
            if not ok:
                continue
            for s in surfaces:
                acc[s].append(surf[s])
            mode_l.append(mode); topic_l.append(topic)
            plen_l.append(plen); glen_l.append(glen); uid_l.append(uid)
            if done % 500 == 0:
                print(f"    ...{done}/{len(tasks)} processed, {len(uid_l)} kept", flush=True)
    finally:
        if runner is not None:
            runner.shutdown()

    if not mode_l:
        print(f"  [{model}] no gens accumulated — nothing written", flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    mode = np.array(mode_l); topic = np.array(topic_l, dtype=int)
    plen = np.array(plen_l, dtype=np.float32); glen = np.array(glen_l, dtype=np.float32)
    uid = np.array(uid_l)
    for s in surfaces:
        X = np.stack(acc[s]).astype(np.float16)
        out_path = OUT / f"surface_cache_{model}_{s}.npz"
        np.savez_compressed(
            out_path, X=X, mode=mode, topic=topic, plen=plen, glen=glen, gen_uid=uid,
            n_pos=np.array(N_POS), surface=np.array(s),
        )
        mb = out_path.stat().st_size / 1e6
        print(f"  wrote {out_path.name}: X={X.shape} {X.dtype} ({mb:.0f} MB)", flush=True)
    print(f"  [{model}] n={len(mode_l)}  topics={len(set(topic_l))}  dims={dims}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="3b,8b")
    ap.add_argument("--surfaces", default=",".join(ALL_SURFACES),
                    help=f"comma list from {ALL_SURFACES} (default = all 6)")
    ap.add_argument("--workers", type=int, default=16,
                    help="process-pool size (pin OMP/OPENBLAS=1 in the launch env). I/O-bound → scale up.")
    ap.add_argument("--limit", type=int, default=None, help="cap gens per model (smoke test)")
    args = ap.parse_args()
    surfaces = [s.strip() for s in args.surfaces.split(",") if s.strip()]
    bad = [s for s in surfaces if s not in ALL_SURFACES]
    if bad:
        raise SystemExit(f"unknown surfaces {bad}; choose from {ALL_SURFACES}")
    print(f"RUNS={RUNS}  OUT={OUT}  surfaces={surfaces}  n_pos={N_POS}  workers={args.workers}", flush=True)
    for model in args.models.split(","):
        print(f"\n===== {model} =====", flush=True)
        build_model(model, surfaces, workers=args.workers, limit=args.limit)


if __name__ == "__main__":
    main()
