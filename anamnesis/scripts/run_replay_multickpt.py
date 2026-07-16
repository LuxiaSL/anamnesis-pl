"""Multi-checkpoint replay — load the base model ONCE per worker, swap LoRA adapters.

The A6 cohort replays the SAME fixed manifest through many PEFT checkpoints. The naive
path (one job/checkpoint, 8 workers each `merge_and_unload`) pays the 17 GB base-load tax
per worker per checkpoint. Here each worker loads the base ONCE, wraps it with PEFT, and
loops its checkpoint list swapping adapters:

    load_adapter(ckpt) → set_adapter → [restore pristine] → merge_adapter → replay slice
    → unmerge_adapter → delete_adapter → next

`merge_adapter` bakes W' = W + s·B·A into `base_layer.weight` (same math as the validated
`merge_and_unload`, so bitwise-equivalent) while keeping the module objects alive, so the
k_proj/etc. hooks registered by load_model stay attached. `--pristine-restore` snapshots
the wrapped modules' weights and restores before each merge (drift-free across many
checkpoints); default OFF — validate bitwise against a known-good cell first and enable
only if drift appears.

Launcher (default) partitions the manifest gen-ids across (gpus × workers/gpu) workers and
hands EACH worker the full checkpoint list + its gen slice. Resume-aware per checkpoint
(skips gens whose signature exists). One Heimdall job → easy teardown.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from anamnesis.scripts._gpu import resolve_physical_gpus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ───────────────────────────── worker ─────────────────────────────
def _build_configs(model: str):
    from anamnesis.config import MODEL_PRESETS
    from anamnesis.extraction.feature_pipeline import FeaturePipelineConfig
    from anamnesis.extraction.state_extractor import ExtractionConfig
    from anamnesis.extraction.model_loader import ModelConfig
    preset = MODEL_PRESETS[model]
    all_layers = list(range(preset.num_layers))
    ec = ExtractionConfig(
        sampled_layers=preset.sampled_layers, pca_layers=preset.pca_layers,
        early_layer_cutoff=preset.early_layer_cutoff, late_layer_cutoff=preset.late_layer_cutoff,
        enable_tier3=True)
    fc = FeaturePipelineConfig(
        include_baseline_tiers=True, enable_residual_trajectory=True, enable_attention_flow=True,
        enable_gate_features=True, enable_temporal_dynamics=False, enable_per_head=True,
        enable_stft=True, enable_contrastive_projection=False, enable_value_geometry=True,
        enable_qk_geometry=True, enable_kv_cka=True,
        trajectory_layers=preset.trajectory_layers, contrastive_layers=preset.contrastive_layers)
    mc = ModelConfig(model_id="", torch_dtype=preset.torch_dtype, num_layers=preset.num_layers,
                     hidden_dim=preset.hidden_dim, num_attention_heads=preset.num_attention_heads,
                     num_kv_heads=preset.num_kv_heads, head_dim=preset.head_dim)
    return preset, all_layers, ec, fc, mc


def run_worker(args) -> None:
    from anamnesis.config import MODEL_PRESETS
    from anamnesis.extraction.model_loader import load_model
    from anamnesis.scripts.run_replay_extraction import _replay_cell, _load_calibration
    from peft import PeftModel

    preset, all_layers, ec, fc, mc = _build_configs(args.model)
    mc.model_id = args.model_path
    checkpoints = json.loads(Path(args.checkpoints_json).read_text())["checkpoints"]
    gen_ids = args.gen_ids

    logger.info(f"[{args.label}] loading base once: {args.model_path}")
    loaded = load_model(
        mc, sampled_layers=preset.sampled_layers, register_gate_hooks=True,
        key_layers=all_layers, value_layers=all_layers, query_layers=all_layers,
        attn_output_layers=all_layers, adapter_path=None)
    calib = _load_calibration(args.calib_dir, enable_tier3=True)

    # wrap once; hooks (on plain modules) become hooks on the surviving base_layers
    base = loaded.model
    pm = PeftModel.from_pretrained(base, checkpoints[0]["adapter_path"], adapter_name="ck0")
    loaded.model = pm
    lora = pm.base_model  # LoraModel: merge_adapter / unmerge_adapter live here

    pristine = None
    if args.pristine_restore:
        pristine = {n: m.base_layer.weight.detach().clone()
                    for n, m in pm.named_modules() if hasattr(m, "base_layer")}
        logger.info(f"[{args.label}] pristine snapshot: {len(pristine)} wrapped modules")

    t0 = time.time()
    for ci, ck in enumerate(checkpoints):
        name = f"ck{ci}"
        if ci > 0:
            pm.load_adapter(ck["adapter_path"], adapter_name=name)
        pm.set_adapter(name)
        if pristine is not None:
            with_no_grad_restore(pm, pristine)
        lora.merge_adapter()
        run_dir = Path(ck["run_dir"])
        _replay_cell(loaded, ec, fc, calib, run_dir, Path(args.manifest), gen_ids,
                     args.sig_subdir, None, "raw_tensors_v3", True, args.no_resume, 50,
                     None, None, f"{args.label}-{ck['label']}")
        lora.unmerge_adapter()
        if ci > 0 or len(checkpoints) > 1:
            try:
                pm.delete_adapter(name)
            except Exception:  # keep going; a stale adapter slot is harmless
                pass
        logger.info(f"[{args.label}] checkpoint {ci+1}/{len(checkpoints)} "
                    f"({ck['label']}) done, {time.time()-t0:.0f}s elapsed")
    logger.info(f"[{args.label}] ALL {len(checkpoints)} checkpoints done in {time.time()-t0:.0f}s")


def with_no_grad_restore(pm, pristine) -> None:
    import torch
    with torch.no_grad():
        for n, m in pm.named_modules():
            if hasattr(m, "base_layer") and n in pristine:
                m.base_layer.weight.copy_(pristine[n])


# ───────────────────────────── launcher ─────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help="internal worker mode")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--checkpoints-json", type=Path, required=True,
                    help='{"checkpoints":[{"label","adapter_path","run_dir"}, ...]}')
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--workers-per-gpu", type=int, default=8)
    ap.add_argument("--sig-subdir", default="signatures_v3")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--pristine-restore", action="store_true",
                    help="restore wrapped-module weights before each merge (drift-free)")
    ap.add_argument("--gen-ids", type=int, nargs="+", default=None)  # worker slice
    ap.add_argument("--label", default="w")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.worker:
        run_worker(args)
        return

    gpu_ids = resolve_physical_gpus([g.strip() for g in args.gpus.split(",") if g.strip()])
    n_workers = len(gpu_ids) * args.workers_per_gpu
    all_ids = sorted(int(k) for k in json.loads(Path(args.manifest).read_text())["entries"])
    worker_ids: list[list[int]] = [[] for _ in range(n_workers)]
    for i, gid in enumerate(all_ids):
        worker_ids[i % n_workers].append(gid)

    n_ck = len(json.loads(Path(args.checkpoints_json).read_text())["checkpoints"])
    logger.info(f"multickpt: {n_ck} checkpoints × {len(all_ids)} gens across {n_workers} workers "
                f"({len(gpu_ids)} GPUs × {args.workers_per_gpu}); base load ×{n_workers}")
    if args.dry_run:
        logger.info(f"  worker0 slice: {len(worker_ids[0])} gens {worker_ids[0][:5]}")
        return

    log_dir = Path(f"/tmp/claude-output/multickpt_{args.label}")
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for w, ids in enumerate(worker_ids):
        if not ids:
            continue
        gpu = gpu_ids[w % len(gpu_ids)]
        cmd = [sys.executable, "-m", "anamnesis.scripts.run_replay_multickpt", "--worker",
               "--model", args.model, "--model-path", args.model_path,
               "--calib-dir", str(args.calib_dir), "--manifest", str(args.manifest),
               "--checkpoints-json", str(args.checkpoints_json), "--sig-subdir", args.sig_subdir,
               "--label", f"w{w}g{gpu}", "--gen-ids", *[str(g) for g in ids]]
        if args.no_resume:
            cmd.append("--no-resume")
        if args.pristine_restore:
            cmd.append("--pristine-restore")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu,
               "PYTHONPATH": os.environ.get("PYTHONPATH", "."),
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
        fh = open(log_dir / f"w{w}_gpu{gpu}.log", "w")
        procs.append((w, gpu, subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT), fh))
        logger.info(f"  worker {w} on GPU {gpu}: {len(ids)} gens × {n_ck} ckpts")

    t0 = time.time()
    failed = 0
    for w, gpu, proc, fh in procs:
        proc.wait(); fh.close()
        if proc.returncode != 0:
            failed += 1
            logger.error(f"  worker {w} (GPU {gpu}): FAILED rc={proc.returncode}")
    logger.info(f"Done in {time.time()-t0:.0f}s — {len(procs)-failed}/{len(procs)} workers OK")


if __name__ == "__main__":
    main()
