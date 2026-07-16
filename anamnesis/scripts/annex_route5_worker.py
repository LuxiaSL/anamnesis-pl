"""ANNEX route-5: PERSISTENT replay-eval worker (Luxia directive 2026-07-16:
the working loop must NOT reload models).

Loads model + calibration + manifest + stage0 reference ONCE, then polls a
file-based job queue until STOP. Each job = a batch of (candidate, gen_id)
pairs at one dose; per pair the worker attaches the candidate as a residual
write at the map site, replays the banked tokens (bitwise-deterministic MT
replay), extracts features (per-head DROPPED — pricing doc §1, declared), and
returns the z-space delta vs the stage0 signature of the SAME gen id.

Queue protocol (work_dir on node-local disk):
  work_dir/candidates/gen_<G>.npy      float32 (lambda, hidden) — written by driver
  work_dir/jobs/w<ID>/job_<G>.json     {"gen": G, "pairs": [[cand_idx, gid], ...],
                                        "alpha_frac": f}
  work_dir/results/w<ID>_job_<G>.npz   {deltas (m, n_feat_kept) f32, cand_idx, gids}
  work_dir/STOP                        exit marker

SMOKE mode (--smoke): replays 2 gens of the banked V3_L14_a0.1 MT cell with the
banked V3_L14 vector at frac .1, FULL feature set, and asserts the feature
vectors match the banked signatures EXACTLY (bitwise replay determinism is a
banked 4/4-family result — any mismatch means this worker's path diverges from
the path of record and NOTHING may launch).

Spawned N-per-GPU by the driver with CUDA_VISIBLE_DEVICES set per worker.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from anamnesis.config import MODEL_PRESETS

F32 = NDArray[np.float32]
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

POLL_S = 0.05


def build_loaded(model: str, model_path: str, calib_dir: Path, per_head: bool):
    """Model + configs, identical assembly to run_replay_extraction.main (v3 surface)."""
    from anamnesis.config import ExtractionConfig, FeaturePipelineConfig, ModelConfig
    from anamnesis.extraction.model_loader import load_model
    from anamnesis.scripts.run_replay_extraction import _load_calibration

    preset = MODEL_PRESETS[model]
    all_layers = list(range(preset.num_layers))
    ec = ExtractionConfig(
        sampled_layers=preset.sampled_layers, pca_layers=preset.pca_layers,
        early_layer_cutoff=preset.early_layer_cutoff,
        late_layer_cutoff=preset.late_layer_cutoff, enable_tier3=True)
    fc = FeaturePipelineConfig(
        include_baseline_tiers=True, enable_residual_trajectory=True,
        enable_attention_flow=True, enable_gate_features=True,
        enable_temporal_dynamics=False, enable_per_head=per_head, enable_stft=True,
        enable_contrastive_projection=False, enable_value_geometry=True,
        enable_qk_geometry=True, enable_kv_cka=True,
        trajectory_layers=preset.trajectory_layers,
        contrastive_layers=preset.contrastive_layers)
    mc = ModelConfig(
        model_id=model_path, torch_dtype=preset.torch_dtype,
        num_layers=preset.num_layers, hidden_dim=preset.hidden_dim,
        num_attention_heads=preset.num_attention_heads,
        num_kv_heads=preset.num_kv_heads, head_dim=preset.head_dim)
    loaded = load_model(mc, sampled_layers=preset.sampled_layers,
                        register_gate_hooks=True, key_layers=all_layers,
                        value_layers=all_layers, query_layers=all_layers,
                        attn_output_layers=all_layers)
    calib = _load_calibration(calib_dir, enable_tier3=True)
    return loaded, ec, fc, calib, preset


def replay_features(loaded, ec, fc, calib, input_ids, plen, write_handle) -> tuple[F32, list[str]]:
    """One MT replay under the currently-attached write; returns (features, names)."""
    from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data
    from anamnesis.extraction.replay_extract import replay_extract

    positional_means, pca_components, pca_mean = calib
    if write_handle is not None:
        write_handle.spec.start_pos = plen
        write_handle.reset_stats()
    raw = replay_extract(loaded, input_ids, plen, positional_means=positional_means)
    if write_handle is not None and write_handle.spec.alpha != 0.0:
        st = write_handle.stats
        expected = len(input_ids) - plen
        if not st.get("saw_cache_position", False) or int(st.get("positions", 0)) != expected:
            raise RuntimeError(f"injection gating broken: {st} expected {expected}")
    res = compute_features_v2_from_data(raw, ec, fc, pca_components, pca_mean)
    return np.asarray(res.features, dtype=np.float32), list(res.feature_names)


def attach(loaded, vector: F32, layer: int, alpha_abs: float):
    import torch
    from anamnesis.extraction.model_loader import ResidualWriteSpec
    spec = ResidualWriteSpec(layer_idx=int(layer),
                             vector=torch.from_numpy(vector.astype(np.float32)),
                             alpha=float(alpha_abs), start_pos=None, normalize=True)
    return loaded.add_residual_write(spec)


def smoke(args) -> None:
    """Bitwise gate vs the banked path of record.

    MT cells replay STAGE0's tokens by gen-id (load_mt_cells pairs by gid), so the
    manifest of record is stage0's; the banked MT signatures are the expected output."""
    cell = args.battery_root / "vmb_a5_mt_3b/V3_L14_a0.1"
    stage0 = args.battery_root / "vmb_stage0_3b"
    manifest = json.loads((stage0 / "replay_manifest.json").read_text())["entries"]
    bank = np.load(args.battery_root / "a5_vectors_3b/a5_vectors.npz")
    stamps = json.loads((args.battery_root / "a5_vectors_3b/a5_vectors_stamps.json").read_text())
    alpha = 0.1 * float(stamps["median_resid_norms"]["L14"])
    loaded, ec, fc, calib, _ = build_loaded(args.model, args.model_path, args.calib_dir,
                                            per_head=True)   # FULL set for the bitwise gate
    wh = attach(loaded, bank["V3_L14"], 14, alpha)
    n_ok = 0
    for gid in (0, 1):
        e = manifest[str(gid)]
        feats, _names = replay_features(loaded, ec, fc, calib,
                                        e["input_ids"], int(e["prompt_length"]), wh)
        banked = np.load(cell / f"signatures_v3/gen_{gid:03d}.npz")["features"].astype(np.float32)
        if not np.array_equal(feats, banked):
            diff = int((feats != banked).sum())
            raise SystemExit(f"SMOKE FAIL gen {gid}: {diff}/{len(feats)} features differ "
                             "— worker path diverges from the path of record. DO NOT LAUNCH.")
        n_ok += 1
        logger.info(f"smoke gen {gid}: EXACT match ({len(feats)} features)")
    wh.remove()
    print(f"SMOKE PASSED: {n_ok}/2 gens bitwise-identical to the banked record")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=list(MODEL_PRESETS.keys()))
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--site", type=int, default=14)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        smoke(args)
        return
    if args.work_dir is None:
        raise SystemExit("--work-dir required outside --smoke")

    cfg = json.loads((args.work_dir / "worker_config.json").read_text())
    # cfg: {manifest, stage0_sig_dir, gen_ids, feature_names_kept, med, scale paths...}
    manifest = json.loads(Path(cfg["manifest"]).read_text())["entries"]
    stamps = json.loads((args.battery_root / "a5_vectors_3b/a5_vectors_stamps.json").read_text())
    site_norm = float(stamps["median_resid_norms"][f"L{args.site}"])

    loaded, ec, fc, calib, _ = build_loaded(args.model, args.model_path, args.calib_dir,
                                            per_head=False)  # pricing §1: per-head dropped

    # feature alignment: names with per-head OFF vs stage0's stored (full) names.
    # Probe once with gen 0, unsteered, to learn this worker's name list.
    e0 = manifest[str(cfg["gen_ids"][0])]
    _probe, kept_names = replay_features(loaded, ec, fc, calib, e0["input_ids"],
                                         int(e0["prompt_length"]), None)
    ref = json.loads((args.work_dir / "reference.json").read_text())
    if kept_names != ref["feature_names_kept"]:
        raise SystemExit("worker feature names != driver reference — feature fork, aborting")
    ref_npz = np.load(args.work_dir / "reference.npz")
    med, scale = ref_npz["med"], ref_npz["scale"]          # already subset to kept
    s0Z = ref_npz["stage0_z"]                              # (n_gens_all, kept) z-space
    s0_gid_row = {int(g): i for i, g in enumerate(ref_npz["stage0_gids"])}

    jobs_dir = args.work_dir / "jobs" / f"w{args.worker_id}"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    results_dir = args.work_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "ready" ).mkdir(exist_ok=True)
    (args.work_dir / "ready" / f"w{args.worker_id}").touch()
    logger.info(f"worker {args.worker_id} READY (model resident; per-head OFF; "
                f"{len(kept_names)} features)")

    wh = None
    cand_cache: tuple[str, F32] | None = None
    while not (args.work_dir / "STOP").exists():
        jobs = sorted(jobs_dir.glob("job_*.json"))
        if not jobs:
            time.sleep(POLL_S)
            continue
        jf = jobs[0]
        job = json.loads(jf.read_text())
        cand_path = args.work_dir / "candidates" / f"gen_{job['gen']:05d}.npy"
        if cand_cache is None or cand_cache[0] != str(cand_path):
            cand_cache = (str(cand_path), np.load(cand_path))
        C = cand_cache[1]
        alpha_abs = float(job["alpha_frac"]) * site_norm
        deltas, cidx, gids = [], [], []
        for ci, gid in job["pairs"]:
            if wh is not None:
                wh.remove()
            wh = attach(loaded, C[ci], args.site, alpha_abs)
            e = manifest[str(gid)]
            feats, _ = replay_features(loaded, ec, fc, calib, e["input_ids"],
                                       int(e["prompt_length"]), wh)
            z = (feats - med) / scale
            deltas.append((z - s0Z[s0_gid_row[gid]]).astype(np.float32))
            cidx.append(ci)
            gids.append(gid)
        out = results_dir / f"w{args.worker_id}_job_{job['gen']:05d}.npz"
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, deltas=np.stack(deltas), cand_idx=np.array(cidx),
                 gids=np.array(gids))
        tmp.rename(out)
        jf.unlink()
    if wh is not None:
        wh.remove()
    logger.info(f"worker {args.worker_id} STOP")


if __name__ == "__main__":
    main()
