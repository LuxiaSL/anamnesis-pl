"""ARM A4 (state surgery): surgically-modified-cache replays over banked Stage-0
continuations (RATIFIED block 12h; exp11 machinery ported to the battery surface).

Per banked continuation (2 per prompt class × 80 classes, seed-idx {0,1}):
  context  = prompt + generated[:mid]   (mid = generated_len // 2)
  forced   = generated[mid:]            (T = len(forced) - 1 per-step states)
  FULL     = unmodified-cache bridge replay (the reference; SAME code path as
             surgered conditions so path effects cancel; bitwise determinism
             makes deltas pure effect)
  NAIVE_f  = evict middle f·C tokens (sinks+recent protected), NO re-rotation,
             survivor positions unchanged, continuation resumes at position C
  ROTATE_f = evict + exact re-rotation to compacted positions 0..S'-1
  REC_f    = fresh prefill of the shortened context (exact recomputation)

Signatures land in <out-run>/signatures_v3/<condition>/gen_XXX.json with the
dissociation columns in metadata: teacher-forced NLL (likelihood rung), token-KL
vs FULL + top-1 agreement (P3's token-space rung). Content rung = text identical
by construction (AUC 0.5 structurally; analyzer states it).

Instrument rider: each worker's first gen runs FULL twice → asserts bitwise-equal
features (the 12b faithfulness-floor-is-zero check, per worker per model).

Launcher mode (--launch) partitions gen ids across (gpus × workers-per-gpu)
subprocesses, parallel_replay-style. Submit via Heimdall with slot-relative GPUs.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from anamnesis.config import MODEL_PRESETS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FRACTIONS = [0.0625, 0.125, 0.25, 0.5]
KINDS = ["naive", "rotate", "recompute"]
NUM_SINKS = 4
RECENT_PROTECT = 32
MIN_GEN_TOKENS = 24


def conditions() -> list[str]:
    return ["full"] + [f"{k}_f{f}" for k in KINDS for f in FRACTIONS]


def _log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    s = x - m
    return s - np.log(np.exp(s).sum(axis=-1, keepdims=True))


def _dissoc_columns(cond_logits: list[np.ndarray], full_lp: np.ndarray | None,
                    chosen: np.ndarray) -> dict:
    """Likelihood + token-space rungs from bridge logits (float32, T × vocab)."""
    lg = np.stack(cond_logits).astype(np.float32)
    lp = _log_softmax(lg)
    idx = chosen.astype(np.int64)
    nll = -lp[np.arange(len(idx)), idx]
    out = {"tf_nll_mean": float(nll.mean()), "tf_nll_sum": float(nll.sum())}
    if full_lp is not None:
        kl = (np.exp(full_lp) * (full_lp - lp)).sum(axis=-1)  # KL(FULL || cond) per step
        out["token_kl_vs_full_mean"] = float(kl.mean())
        out["token_kl_vs_full_max"] = float(kl.max())
        out["top1_agree_vs_full"] = float((lp.argmax(-1) == full_lp.argmax(-1)).mean())
    return out, lp


def run_worker(args) -> None:
    import torch

    from anamnesis.config import ExtractionConfig, FeaturePipelineConfig, ModelConfig
    from anamnesis.extraction.cache_surgery import (
        evict,
        from_hf_cache,
        inv_freq_from_config,
        middle_region_keep,
        reindex,
        to_hf_dynamic_cache,
    )
    from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data, save_features
    from anamnesis.extraction.model_loader import load_model
    from anamnesis.extraction.replay_cached import replay_extract_cached

    preset = MODEL_PRESETS[args.model]
    all_layers = list(range(preset.num_layers))
    extraction_config = ExtractionConfig(
        sampled_layers=preset.sampled_layers, pca_layers=preset.pca_layers,
        early_layer_cutoff=preset.early_layer_cutoff, late_layer_cutoff=preset.late_layer_cutoff,
        enable_tier3=True,
    )
    family_config = FeaturePipelineConfig(
        include_baseline_tiers=True, enable_residual_trajectory=True,
        enable_attention_flow=True, enable_gate_features=True,
        enable_temporal_dynamics=False, enable_per_head=True, enable_stft=True,
        enable_contrastive_projection=False, enable_value_geometry=True,
        enable_qk_geometry=True, enable_kv_cka=True,
        trajectory_layers=preset.trajectory_layers, contrastive_layers=preset.contrastive_layers,
    )
    model_config = ModelConfig(
        model_id=args.model_path, torch_dtype=preset.torch_dtype,
        num_layers=preset.num_layers, hidden_dim=preset.hidden_dim,
        num_attention_heads=preset.num_attention_heads, num_kv_heads=preset.num_kv_heads,
        head_dim=preset.head_dim,
    )
    loaded = load_model(
        model_config, sampled_layers=preset.sampled_layers, register_gate_hooks=True,
        key_layers=all_layers, value_layers=all_layers,
        query_layers=all_layers, attn_output_layers=all_layers,
    )
    device = next(loaded.model.parameters()).device

    # 12h RoPE gate: per-model config, asserted before any surgery.
    inv_freq = inv_freq_from_config(loaded.model.config).to(device)

    pm_path = args.calib_dir / "positional_means.npz"
    positional_means = None
    if pm_path.exists():
        positional_means = np.load(pm_path)["positional_means"].astype(np.float32)
    else:
        logger.warning(f"NO positional_means at {pm_path}")
    pca_components = pca_mean = None
    pca_path = args.calib_dir / "pca_model.pkl"
    if pca_path.exists():
        import pickle

        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
        if isinstance(pca, dict):
            pca_components = np.asarray(pca["components"], dtype=np.float32)
            pca_mean = np.asarray(pca["mean"], dtype=np.float32)
        else:
            pca_components = np.asarray(pca.components_, dtype=np.float32)
            pca_mean = np.asarray(pca.mean_, dtype=np.float32)

    with open(args.manifest) as f:
        entries = json.load(f)["entries"]
    src_meta: dict[int, dict] = {}
    meta_path = args.floor_run_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            md = json.load(f)
        gens = md["generations"] if isinstance(md, dict) and "generations" in md else md
        src_meta = {int(g["generation_id"]): g for g in gens}

    conds = conditions()
    sig_root = args.out_run_dir / "signatures_v3"
    for c in conds:
        (sig_root / c).mkdir(parents=True, exist_ok=True)

    todo = []
    for g in args.gen_ids:
        if all((sig_root / c / f"gen_{g:03d}.json").exists() for c in conds):
            continue
        todo.append(g)
    logger.info(f"[{args.label}] {len(todo)}/{len(args.gen_ids)} gens to process, "
                f"{len(conds)} conditions each")

    @torch.no_grad()
    def prefill(ids_list: list[int]):
        ids = torch.tensor([ids_list], dtype=torch.long, device=device)
        loaded.disable_hooks()
        out = loaded.model(ids, use_cache=True, return_dict=True)
        loaded.enable_hooks()
        loaded.clear_hook_state()
        return out.past_key_values

    rider_done = False
    n_done = n_fail = 0
    t0 = time.time()
    for i, gid in enumerate(todo):
        try:
            e = entries[str(gid)]
            input_ids = list(e["input_ids"])
            P = int(e["prompt_length"])
            n_gen = len(input_ids) - P
            if n_gen < MIN_GEN_TOKENS:
                logger.warning(f"[{args.label}] gen_{gid:03d} EXCLUDED: only {n_gen} generated tokens")
                continue
            mid = n_gen // 2
            C = P + mid
            context_ids = input_ids[:C]
            cont_ids = input_ids[C:]
            base_meta = dict(src_meta.get(gid, {"generation_id": gid}))
            base_meta.update({
                "a4_context_len": C, "a4_prompt_len": P, "a4_cont_len": len(cont_ids),
                "a4_num_sinks": NUM_SINKS, "a4_recent_protect": RECENT_PROTECT,
            })

            base_cache = prefill(context_ids)
            snapshot = from_hf_cache(base_cache, positions=torch.arange(C, device=device))

            full_lp = None
            chosen = np.asarray(cont_ids[1:], dtype=np.int64)

            for cond in conds:
                sig_path = sig_root / cond / f"gen_{gid:03d}.json"
                if sig_path.exists() and cond != "full":
                    continue
                if cond == "full":
                    snap = snapshot
                    offset = C
                    surg_meta = {"kind": "full", "evict_frac": 0.0, "n_evicted": 0}
                else:
                    kind, ftag = cond.split("_f")
                    f = float(ftag)
                    keep = middle_region_keep(C, f, num_sinks=NUM_SINKS,
                                              recent_protect=RECENT_PROTECT).to(device)
                    n_evicted = C - int(keep.shape[0])
                    if kind == "naive":
                        snap = evict(snapshot, keep)
                        offset = C  # survivors keep original positions; next = C
                    elif kind == "rotate":
                        snap = evict(snapshot, keep)
                        s_prime = int(keep.shape[0])
                        snap = reindex(snap, torch.arange(s_prime, device=device), inv_freq)
                        offset = s_prime
                    elif kind == "recompute":
                        keep_cpu = keep.cpu().tolist()
                        short_ids = [context_ids[j] for j in keep_cpu]
                        cache = prefill(short_ids)
                        snap = from_hf_cache(
                            cache, positions=torch.arange(len(short_ids), device=device))
                        offset = len(short_ids)
                    else:
                        raise ValueError(kind)
                    surg_meta = {"kind": kind, "evict_frac": f, "n_evicted": n_evicted}

                raw = replay_extract_cached(
                    loaded, to_hf_dynamic_cache(snap), cont_ids, offset,
                    positional_means=positional_means,
                )
                dis, lp = _dissoc_columns(raw.logits, full_lp, chosen)
                if cond == "full":
                    full_lp = lp
                    # top1 self-agreement trivially 1.0; recompute vs self for the record
                    dis["token_kl_vs_full_mean"] = 0.0
                    dis["token_kl_vs_full_max"] = 0.0
                    dis["top1_agree_vs_full"] = 1.0

                result = compute_features_v2_from_data(
                    raw, extraction_config, family_config, pca_components, pca_mean)

                # Instrument rider: first gen, FULL replayed twice → bitwise equal.
                if cond == "full" and not rider_done:
                    raw2 = replay_extract_cached(
                        loaded, to_hf_dynamic_cache(snap), cont_ids, offset,
                        positional_means=positional_means)
                    r2 = compute_features_v2_from_data(
                        raw2, extraction_config, family_config, pca_components, pca_mean)
                    if not np.array_equal(np.asarray(result.features),
                                          np.asarray(r2.features)):
                        raise RuntimeError(
                            "A4 rider FAILED: FULL bridge replay not bitwise-deterministic")
                    logger.info(f"[{args.label}] rider OK: FULL bridge bitwise-deterministic")
                    rider_done = True

                metadata = dict(base_meta)
                metadata["a4_condition"] = cond
                metadata.update({f"a4_{k}": v for k, v in surg_meta.items()})
                metadata["a4_position_offset"] = offset
                metadata["a4_cache_len_after"] = snap.seq_len()
                metadata["dissociation"] = dis
                metadata["num_features"] = int(len(result.features))
                metadata["extraction_version"] = 3
                save_features(gid, result, metadata, sig_root / cond)

            n_done += 1
            if (i + 1) % 2 == 0 or i == 0:
                el = time.time() - t0
                rate = (i + 1) / el if el else 0
                logger.info(f"[{args.label}] {i+1}/{len(todo)} gen_{gid:03d} done "
                            f"({rate*60:.1f} gens/min, ETA {(len(todo)-i-1)/max(rate,1e-9):.0f}s)")
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            logger.error(f"[{args.label}] gen_{gid:03d} FAILED: {exc}", exc_info=True)

    logger.info(f"[{args.label}] done: {n_done} ok, {n_fail} failed in {time.time()-t0:.0f}s")
    if n_fail:
        sys.exit(1)


def run_launcher(args) -> None:
    from anamnesis.scripts._gpu import resolve_physical_gpus

    gpu_ids = resolve_physical_gpus(
        [g.strip() for g in args.gpus.split(",") if g.strip() != ""])
    n_workers = len(gpu_ids) * args.workers_per_gpu

    conds = conditions()
    sig_root = args.out_run_dir / "signatures_v3"
    todo = [g for g in args.gen_ids
            if not all((sig_root / c / f"gen_{g:03d}.json").exists() for c in conds)]
    if not todo:
        logger.info("all gens complete — nothing to do")
        return
    worker_ids: list[list[int]] = [[] for _ in range(n_workers)]
    for i, gid in enumerate(todo):
        worker_ids[i % n_workers].append(gid)

    log_dir = args.out_run_dir / "a4_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for w, ids in enumerate(worker_ids):
        if not ids:
            continue
        gpu = gpu_ids[w % len(gpu_ids)]
        cmd = [sys.executable, "-m", "anamnesis.scripts.vmb_a4_surgery_replay",
               "--model", args.model, "--model-path", args.model_path,
               "--floor-run-dir", str(args.floor_run_dir), "--manifest", str(args.manifest),
               "--calib-dir", str(args.calib_dir), "--out-run-dir", str(args.out_run_dir),
               "--label", f"w{w}g{gpu}",
               "--gen-ids", *[str(g) for g in ids]]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu,
               "PYTHONPATH": os.environ.get("PYTHONPATH", "."),
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        lf = open(log_dir / f"worker_{w}.log", "w")
        procs.append((w, subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT), lf))
        logger.info(f"worker {w} (GPU {gpu}): {len(ids)} gens")

    fails = 0
    for w, p, lf in procs:
        rc = p.wait()
        lf.close()
        if rc != 0:
            fails += 1
            logger.error(f"worker {w} exited rc={rc}")
    logger.info(f"A4 launcher done: {len(procs) - fails}/{len(procs)} workers clean")
    sys.exit(1 if fails else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description="ARM A4 cache-surgery replay")
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--floor-run-dir", type=Path, required=True,
                    help="Stage-0 floor run (metadata.json for class info)")
    ap.add_argument("--manifest", type=Path, required=True,
                    help="Stage-0 replay manifest (input_ids + prompt_length per gid)")
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--out-run-dir", type=Path, required=True)
    ap.add_argument("--gen-ids", type=int, nargs="+", required=True,
                    help="Banked continuation gids (2 per prompt class; submit script computes)")
    ap.add_argument("--label", default="w")
    ap.add_argument("--launch", action="store_true", help="Launcher mode (spawn workers)")
    ap.add_argument("--gpus", default="0", help="Launcher: logical GPU slots")
    ap.add_argument("--workers-per-gpu", type=int, default=4)
    args = ap.parse_args()

    if args.launch:
        run_launcher(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
