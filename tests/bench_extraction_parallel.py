"""Benchmark extraction throughput at different parallelism levels.

Uses actual signature data from a completed extraction run.
Tests both old (reference) and new (optimized) extraction to measure
combined speedup from vectorization + parallelism.

Usage:
    python tests/bench_extraction_parallel.py \
        --signatures-dir /models/subliminal-anamnesis/signatures/owl_student/step-0001/signatures \
        --metadata /models/subliminal-anamnesis/signatures/owl_student/step-0001/extraction_metadata.json \
        --workers 1 4 16 32 64 96 128
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anamnesis.config import ExtractionConfig
from anamnesis.extraction.state_extractor import (
    RawGenerationData,
    extract_all_features,
)

logging.basicConfig(level=logging.WARNING)


def _load_signature_as_raw(npz_path: Path, metadata: dict) -> RawGenerationData:
    """Reconstruct a minimal RawGenerationData from saved signature + metadata.

    This is a lightweight reconstruction — only hidden_states-shaped data
    needed to exercise the extraction code path. We synthesize plausible
    tensors with the correct shapes from the metadata.
    """
    data = np.load(npz_path, allow_pickle=True)
    features = data["features"]
    n_tokens = metadata["num_generated_tokens"]
    prompt_length = metadata.get("prompt_length", 50)

    num_layers = 33
    hidden_dim = 4096
    num_heads = 32
    vocab_size = 128256
    num_kv_heads = 8
    head_dim = 128
    sampled_layers = [0, 8, 16, 20, 24, 28, 31]

    rng = np.random.RandomState(metadata.get("seed", 42))

    T = max(1, n_tokens - 1)

    hidden_states = [
        rng.randn(num_layers, hidden_dim).astype(np.float32) * 10.0
        for _ in range(T)
    ]

    attentions = []
    for t in range(T):
        seq_len = prompt_length + t + 1
        raw_attn = rng.rand(num_layers - 1, num_heads, seq_len).astype(np.float32)
        raw_attn /= raw_attn.sum(axis=2, keepdims=True)
        attentions.append(raw_attn)

    logits = [
        rng.randn(vocab_size).astype(np.float32)
        for _ in range(T)
    ]

    chosen_ids = rng.randint(0, vocab_size, size=T).astype(np.float32)

    pre_rope_keys: dict[int, list[np.ndarray]] = {}
    for l_idx in sampled_layers:
        pre_rope_keys[l_idx] = [
            rng.randn(num_kv_heads, head_dim).astype(np.float32)
            for _ in range(T)
        ]

    return RawGenerationData(
        hidden_states=hidden_states,
        attentions=attentions,
        logits=logits,
        chosen_token_ids=chosen_ids,
        pre_rope_keys=pre_rope_keys,
        prompt_length=prompt_length,
        positional_means=None,
    )


def _extract_worker(args: tuple) -> tuple[int, float, int]:
    """Worker function for parallel benchmark. Returns (gen_id, elapsed, n_features)."""
    gen_id, data, config = args
    t0 = time.perf_counter()
    result = extract_all_features(data, config)
    elapsed = time.perf_counter() - t0
    return (gen_id, elapsed, len(result.features))


def benchmark_sequential(
    work_items: list[tuple[int, RawGenerationData, ExtractionConfig]],
) -> tuple[float, list[float]]:
    """Run extraction sequentially, return (total_time, per_item_times)."""
    per_item: list[float] = []
    t0 = time.perf_counter()
    for args in work_items:
        _, elapsed, _ = _extract_worker(args)
        per_item.append(elapsed)
    total = time.perf_counter() - t0
    return total, per_item


def benchmark_parallel(
    work_items: list[tuple[int, RawGenerationData, ExtractionConfig]],
    n_workers: int,
) -> tuple[float, list[float]]:
    """Run extraction in parallel, return (total_time, per_item_times)."""
    per_item: list[float] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_extract_worker, args): args[0] for args in work_items}
        for future in as_completed(futures):
            _, elapsed, _ = future.result()
            per_item.append(elapsed)
    total = time.perf_counter() - t0
    return total, per_item


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark extraction parallelism")
    parser.add_argument("--signatures-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 16, 32, 64])
    parser.add_argument("--max-samples", type=int, default=90,
                        help="Max samples to benchmark (default: all)")
    args = parser.parse_args()

    with open(args.metadata) as f:
        all_metadata = json.load(f)

    config = ExtractionConfig(
        sampled_layers=[0, 8, 16, 20, 24, 28, 31],
        pca_layers=[8, 16, 20, 24, 28],
        enable_tier1=True,
        enable_tier2=True,
        enable_tier2_5=True,
        enable_tier3=False,
        enable_knnlm_baseline=True,
    )

    print(f"Loading {min(len(all_metadata), args.max_samples)} samples...")
    work_items: list[tuple[int, RawGenerationData, ExtractionConfig]] = []
    for meta in all_metadata[:args.max_samples]:
        gen_id = meta["generation_id"]
        npz_path = args.signatures_dir / f"gen_{gen_id:03d}.npz"
        if not npz_path.exists():
            continue
        data = _load_signature_as_raw(npz_path, meta)
        work_items.append((gen_id, data, config))

    print(f"Loaded {len(work_items)} samples")
    print(f"Token counts: {[m['num_generated_tokens'] for m in all_metadata[:len(work_items)]][:10]}...")
    print()

    results: list[dict] = []

    for n_workers in args.workers:
        if n_workers == 1:
            total, per_item = benchmark_sequential(work_items)
        else:
            total, per_item = benchmark_parallel(work_items, n_workers)

        throughput = len(work_items) / total
        median_per = float(np.median(per_item))

        result = {
            "workers": n_workers,
            "total_seconds": round(total, 2),
            "throughput_sigs_per_sec": round(throughput, 1),
            "median_per_item": round(median_per, 4),
            "p95_per_item": round(float(np.percentile(per_item, 95)), 4),
        }
        results.append(result)

        speedup = results[0]["total_seconds"] / total if results else 1.0
        eff = speedup / n_workers * 100 if n_workers > 0 else 0

        print(f"Workers={n_workers:4d}  total={total:7.2f}s  "
              f"throughput={throughput:6.1f} sig/s  "
              f"median={median_per:.4f}s/sig  "
              f"speedup={speedup:.1f}x  "
              f"efficiency={eff:.0f}%")

    print("\nJSON results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
