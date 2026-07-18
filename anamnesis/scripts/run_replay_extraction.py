"""Per-worker replay extraction: re-process a banked run to v3 signatures + full raw.

Loads the model ONCE with the v3 capture surface (all-layer k_proj + v_proj hooks,
sampled-layer q_proj + gate_proj hooks), the run's calibration (positional_means + PCA),
and the replay manifest; then for each assigned gen_id:
    replay_extract -> save_raw_tensors_v3 -> compute_features_v2_from_data -> save signature.
Resume-aware: skips gens whose signature json already exists. Designed to be spawned
N-per-GPU by parallel_replay.py with CUDA_VISIBLE_DEVICES set per worker.

Usage (one worker, all gens, resume):
    PYTHONPATH=. python -m anamnesis.scripts.run_replay_extraction \
        --model 8b --model-path /models/llama-3.1-8b-instruct \
        --run-dir /models/anamnesis-extract/runs/8b_fat_01 \
        --calib-dir /models/anamnesis-extract/calibration/llama31_8b \
        --manifest /models/anamnesis-extract/runs/8b_fat_01/replay_manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from anamnesis.config import MODEL_PRESETS

F32 = NDArray[np.float32]
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_calibration(calib_dir: Path, enable_tier3: bool) -> tuple[F32 | None, F32 | None, F32 | None]:
    """Load positional_means + PCA (components, mean) from a calibration dir."""
    positional_means = pca_components = pca_mean = None
    pm_path = calib_dir / "positional_means.npz"
    if pm_path.exists():
        positional_means = np.load(pm_path)["positional_means"].astype(np.float32)
        logger.info(f"positional_means {positional_means.shape}")
    else:
        logger.warning(f"NO positional_means at {pm_path} — corrected features will be wrong")
    pca_path = calib_dir / "pca_model.pkl"
    if enable_tier3 and pca_path.exists():
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
        if isinstance(pca, dict):
            pca_components = np.asarray(pca["components"], dtype=np.float32)
            pca_mean = np.asarray(pca["mean"], dtype=np.float32)
        else:
            pca_components = np.asarray(pca.components_, dtype=np.float32)
            pca_mean = np.asarray(pca.mean_, dtype=np.float32)
        logger.info(f"pca components {pca_components.shape}")
    return positional_means, pca_components, pca_mean


def _resolve_injection(run_dir: Path, inject_from_metadata: bool, inject_npz,
                       inject_key, inject_layer, inject_alpha, inject_alpha_frac):
    """Return (npz, key, layer, alpha, frac) — from run-dir/metadata.json when
    inject_from_metadata, else the explicit args. (None,...) = no injection."""
    if inject_from_metadata:
        inj = json.loads((run_dir / "metadata.json").read_text()).get("a5_injection")
        if not inj:
            raise SystemExit(f"--inject-from-metadata: no a5_injection in {run_dir}/metadata.json")
        return (Path(inj["inject_npz"]), inj["inject_key"], int(inj["inject_layer"]),
                float(inj["inject_alpha"]), inj.get("inject_alpha_frac"))
    if inject_npz is None:
        return None, None, None, None, None
    return Path(inject_npz), inject_key, inject_layer, inject_alpha, inject_alpha_frac


def _setup_replay_injection(loaded, npz, key, layer, alpha, frac, label):
    """Attach the residual-write hook for one cell; return (write_handle, inject_meta).
    Factored so the single-cell path is unchanged and multi-cell can re-attach per cell."""
    if npz is None:
        return None, None
    if key is None or layer is None or alpha is None:
        raise SystemExit("injection requires key, layer, alpha")
    import torch as _torch

    from anamnesis.extraction.model_loader import ResidualWriteSpec
    vec_bank = np.load(npz)
    if key not in vec_bank:
        raise SystemExit(f"vector key {key!r} not in {npz} (has {list(vec_bank.keys())})")
    spec = ResidualWriteSpec(layer_idx=int(layer),
                             vector=_torch.from_numpy(vec_bank[key].astype(np.float32)),
                             alpha=float(alpha), start_pos=None, normalize=True)
    write_handle = loaded.add_residual_write(spec)
    inject_meta = {"inject_npz": str(npz), "inject_key": key, "inject_layer": int(layer),
                   "inject_alpha": float(alpha), "inject_alpha_frac": frac}
    logger.info(f"[{label}] replay injection active: {inject_meta}")
    return write_handle, inject_meta


def _replay_cell(loaded, ec, fc, calib, run_dir: Path, manifest_path: Path, gen_ids,
                 sig_subdir, raw_dir_arg, raw_subdir, no_raw, no_resume, logits_top_k,
                 write_handle, inject_meta, label) -> tuple[int, int]:
    """Replay one cell's gens to signatures. Exact transcription of the original loop."""
    from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data, save_features
    from anamnesis.extraction.raw_saver import save_raw_tensors_v3
    from anamnesis.extraction.replay_extract import replay_extract
    positional_means, pca_components, pca_mean = calib

    with open(manifest_path) as f:
        manifest = json.load(f)
    entries = manifest["entries"]
    src_meta: dict[int, dict] = {}
    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        md = json.load(open(meta_path))
        gens = md["generations"] if isinstance(md, dict) and "generations" in md else md
        src_meta = {int(g["generation_id"]): g for g in gens}

    raw_dir = raw_dir_arg if raw_dir_arg is not None else (run_dir / raw_subdir)
    sig_dir = run_dir / sig_subdir
    sig_dir.mkdir(parents=True, exist_ok=True)
    if not no_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    avail = sorted(int(k) for k in entries)
    todo = [g for g in avail if (gen_ids is None or g in set(gen_ids))]
    if not no_resume:
        todo = [g for g in todo if not (sig_dir / f"gen_{g:03d}.json").exists()]
    logger.info(f"[{label}] {len(todo)} gens to process -> {sig_dir} ({len(avail)} in manifest)")

    n_done = n_fail = 0
    t0 = time.time()
    for i, gid in enumerate(todo):
        try:
            e = entries[str(gid)]
            input_ids = e["input_ids"]
            plen = int(e["prompt_length"])
            if write_handle is not None:
                write_handle.spec.start_pos = plen
                write_handle.reset_stats()
            raw_data = replay_extract(loaded, input_ids, plen, positional_means=positional_means)
            if write_handle is not None and inject_meta["inject_alpha"] != 0.0:
                st = write_handle.stats
                expected = len(input_ids) - plen
                got = int(st.get("positions", 0))
                if not st.get("saw_cache_position", False) or got != expected:
                    raise RuntimeError(
                        f"gen_{gid:03d}: replay injection gating broken "
                        f"(saw_cache_position={st.get('saw_cache_position')}, "
                        f"positions={got}, expected={expected})")
            if not no_raw:
                save_raw_tensors_v3(raw_data, gid, raw_dir, prompt_length=plen,
                                    input_ids=input_ids, top_k_logits=logits_top_k)
            result = compute_features_v2_from_data(
                raw_data, ec, fc, pca_components, pca_mean)
            metadata = dict(src_meta.get(gid, {"generation_id": gid}))
            if inject_meta is not None:
                metadata["injection"] = dict(inject_meta)
            metadata["num_features"] = int(len(result.features))
            metadata["tier_slices"] = {k: list(v) for k, v in result.tier_slices.items()}
            metadata["extraction_version"] = 3
            # xrt_version rider (48870af4): 2 = v2.1 enriched routing family (120 feats/M6), 0 = no
            # routing surface (dense model). v1-era sigs lack this key entirely → never silently mixed;
            # whole-vector dim also differs (2867 v2 vs 2807 v1), caught by the modal-length loader guard.
            metadata["xrt_version"] = 2 if "expert_routing" in result.tier_slices else 0
            save_features(gid, result, metadata, sig_dir)
            n_done += 1
            if (i + 1) % 10 == 0 or i == 0:
                el = time.time() - t0
                rate = (i + 1) / el if el > 0 else 0
                eta = (len(todo) - i - 1) / rate if rate > 0 else 0
                logger.info(f"[{label}] {i+1}/{len(todo)} gen_{gid:03d}: "
                            f"{len(result.features)} feats, {el:.0f}s, ETA {eta:.0f}s")
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            logger.error(f"[{label}] gen_{gid:03d} FAILED: {exc}", exc_info=True)
    logger.info(f"[{label}] cell done: {n_done} ok, {n_fail} failed in {time.time()-t0:.0f}s")
    return n_done, n_fail


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay-extract a banked run to v3 sigs + raw")
    parser.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    parser.add_argument("--model-path", type=str, required=True, help="Local model dir (overrides preset HF id)")
    parser.add_argument("--run-dir", type=Path, default=None, help="Output run dir (holds metadata.json + outputs); single-cell mode")
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None, help="single-cell mode")
    parser.add_argument("--gen-ids", type=int, nargs="+", default=None, help="Subset of gen ids (default: all in manifest)")
    parser.add_argument("--raw-subdir", default="raw_tensors_v3")
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="Absolute raw output dir (overrides run-dir/raw-subdir; e.g. /dev/shm scratch)")
    parser.add_argument("--sig-subdir", default="signatures_v3")
    parser.add_argument("--no-raw", action="store_true", help="Skip raw banking (signatures only)")
    parser.add_argument("--logits-top-k", type=int, default=50,
                        help="Per-position logits retained in raw (default 50). Raise (e.g. "
                             "2000 or vocab size) for the §1.5 token-KL / §4.3 entropy cells, "
                             "which need enough logit mass for KL(unsteered‖steered).")
    parser.add_argument("--no-tier3", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="Recompute even if a signature exists")
    parser.add_argument("--label", default="w", help="Worker label for logs")
    parser.add_argument("--inject-npz", type=Path, default=None,
                    help="A5: apply a residual write during replay (steered free-gen "
                         "signature extraction + matched-token cells). Same semantics "
                         "as run_gen_tokens: injection at absolute positions >= each "
                         "gen's prompt_length, cache_position-gated")
    parser.add_argument("--inject-key", default=None)
    parser.add_argument("--inject-layer", type=int, default=None)
    parser.add_argument("--inject-alpha", type=float, default=None)
    parser.add_argument("--inject-alpha-frac", type=float, default=None,
                    help="Bookkeeping only; recorded in each signature's metadata")
    parser.add_argument("--adapter-path", default=None,
                    help="A6: PEFT adapter dir merged before hooks (checkpoint replay)")
    parser.add_argument("--inject-from-metadata", action="store_true",
                    help="Read the injection spec from run-dir/metadata.json "
                         "['a5_injection'] (written by the steered gen job) — "
                         "guarantees gen and replay use the identical spec")
    parser.add_argument("--jobs-file", type=Path, default=None,
                    help="MULTI-CELL mode: JSON list of {run_dir, manifest, gen_ids?, "
                         "inject_from_metadata?} — model + calibration load ONCE, the "
                         "injection hook re-attaches per cell. Mutually exclusive with "
                         "--run-dir/--manifest.")
    args = parser.parse_args()

    preset = MODEL_PRESETS[args.model]
    all_layers = list(range(preset.num_layers))

    # ── Configs (v3 capture surface) ──
    from anamnesis.config import ExtractionConfig, FeaturePipelineConfig, ModelConfig

    extraction_config = ExtractionConfig(
        sampled_layers=preset.sampled_layers,
        pca_layers=preset.pca_layers,
        early_layer_cutoff=preset.early_layer_cutoff,
        late_layer_cutoff=preset.late_layer_cutoff,
        enable_tier3=not args.no_tier3,
    )
    family_config = FeaturePipelineConfig(
        include_baseline_tiers=True,
        enable_residual_trajectory=True,
        enable_attention_flow=True,
        enable_gate_features=True,
        enable_temporal_dynamics=False,   # v3: temporal_dynamics ignored
        enable_per_head=True,             # v3: new surface
        enable_stft=True,
        enable_contrastive_projection=False,  # contrastive is a separate addon
        # vmb matrix completion pass 1 (prereg Stage A(ii), census 2026-07-12): the
        # deployed 2,713-dim v3 vector carried ZERO value/qk/cka features — the
        # families existed but were never enabled here. Floors must cover every
        # featurized cell natively (ordering rule), so the battery vector is the
        # v3 superset. Old fat_01 signatures remain the frozen 2,713 baseline.
        enable_value_geometry=True,
        enable_qk_geometry=True,
        enable_kv_cka=True,
        # MoE expert routing (vmb arm A7, M6): None-guarded — the xrt family returns empty
        # for dense models (router_dist is None), so enabling it here is a no-op everywhere
        # except DeepSeek-V2-Lite, where it adds the 60 xrt features to the battery vector.
        enable_expert_routing=True,
        trajectory_layers=preset.trajectory_layers,
        contrastive_layers=preset.contrastive_layers,
    )
    model_config = ModelConfig(
        model_id=args.model_path,
        torch_dtype=preset.torch_dtype,
        num_layers=preset.num_layers,
        hidden_dim=preset.hidden_dim,
        num_attention_heads=preset.num_attention_heads,
        num_kv_heads=preset.num_kv_heads,
        head_dim=preset.head_dim,
    )

    # ── Load model with the full v3 capture surface ──
    from anamnesis.extraction.model_loader import load_model
    from anamnesis.extraction.replay_extract import replay_extract
    from anamnesis.extraction.raw_saver import save_raw_tensors_v3
    from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data, save_features

    loaded = load_model(
        model_config,
        sampled_layers=preset.sampled_layers,
        register_gate_hooks=True,
        key_layers=all_layers,        # all-layer keys
        value_layers=all_layers,      # all-layer values (v_proj)
        # vmb battery capture surface (prereg addendum 2026-07-12a §4): queries and
        # attention outputs banked at ALL layers. Feature families still consume
        # explicit sampled_layers lists, so the v3 feature vector is unchanged —
        # the extra layers are banked raw only (depth = a measured axis later).
        query_layers=all_layers,
        attn_output_layers=all_layers,
        adapter_path=args.adapter_path,
    )

    positional_means, pca_components, pca_mean = _load_calibration(
        args.calib_dir, enable_tier3=not args.no_tier3,
    )

    calib = (positional_means, pca_components, pca_mean)
    ec, fc = extraction_config, family_config

    if args.jobs_file is not None:
        # MULTI-CELL: model + calibration loaded ONCE above; loop cells, re-attaching
        # the injection hook per cell (removing the previous first). Sig output is
        # byte-identical to the per-cell path (bitwise smoke: vmb_a5_replay_multicell).
        jobs = json.loads(args.jobs_file.read_text())
        logger.info(f"[{args.label}] multi-cell replay: {len(jobs)} cells, one model+calib load")
        handle = None
        for ji, job in enumerate(jobs):
            if handle is not None:
                handle.remove()
                handle = None
            run_dir = Path(job["run_dir"])
            npz, key, layer, alpha, frac = _resolve_injection(
                run_dir, job.get("inject_from_metadata", False),
                job.get("inject_npz"), job.get("inject_key"), job.get("inject_layer"),
                job.get("inject_alpha"), job.get("inject_alpha_frac"))
            handle, meta = _setup_replay_injection(loaded, npz, key, layer, alpha, frac,
                                                   args.label)
            _replay_cell(loaded, ec, fc, calib, run_dir, Path(job["manifest"]),
                         job.get("gen_ids"), args.sig_subdir, args.raw_dir, args.raw_subdir,
                         args.no_raw, args.no_resume, args.logits_top_k, handle, meta,
                         f"{args.label}c{ji}")
        if handle is not None:
            handle.remove()
        return

    if args.run_dir is None or args.manifest is None:
        raise SystemExit("single-cell mode requires --run-dir and --manifest (or --jobs-file)")
    npz, key, layer, alpha, frac = _resolve_injection(
        args.run_dir, args.inject_from_metadata, args.inject_npz, args.inject_key,
        args.inject_layer, args.inject_alpha, args.inject_alpha_frac)
    handle, meta = _setup_replay_injection(loaded, npz, key, layer, alpha, frac, args.label)
    _replay_cell(loaded, ec, fc, calib, args.run_dir, args.manifest, args.gen_ids,
                 args.sig_subdir, args.raw_dir, args.raw_subdir, args.no_raw,
                 args.no_resume, args.logits_top_k, handle, meta, args.label)
    if handle is not None:
        handle.remove()


if __name__ == "__main__":
    main()
