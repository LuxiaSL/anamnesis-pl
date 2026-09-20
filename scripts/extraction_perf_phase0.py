"""Pinned, restartable capture-path baseline. No optimized extraction code.

Run from an isolated checkout with the frozen Palimpsest bank on PYTHONPATH.
Every row is written atomically; a new attempt requires a new output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data: Any) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_pin(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*.py'))
            if not any(x.startswith('.') or x == '__pycache__' for x in p.relative_to(root).parts)}


def rows_from_bank(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    names: list[str] | None = None
    seen: set[tuple[str, str, int, str]] = set()
    for shard in sorted(root.glob('run*/own_content_shard*.npz')):
        suffix = shard.stem.removeprefix('own_content_shard')
        docs = shard.with_name(f'branch_documents_shard{suffix}.jsonl')
        with docs.open() as f:
            records = [json.loads(line) for line in f if line.strip()]
        by_key = {(r['doc_id'], r['cell'], int(r['branch_id'])): r for r in records}
        if len(by_key) != len(records):
            raise ValueError(f'duplicate document key in {docs}')
        with np.load(shard, allow_pickle=False) as archive:
            z = {k: archive[k] for k in ('feature_names', 'cell', 'doc_id', 'branch_id',
                                        'excluded', 'cluster', 'sig_ancestor_h1_raw')}
            current = z['feature_names'].astype(str).tolist()
            if names is not None and names != current:
                raise ValueError('feature schema drift across shards')
            names = current
            for i in range(len(z['cell'])):
                key = (str(z['doc_id'][i]), str(z['cell'][i]), int(z['branch_id'][i]))
                r = by_key[key]
                spans = [s for s in r['spans'] if s['role'] == 'holder' and s['index'] == 1]
                if len(spans) != 1:
                    raise ValueError(f'need exactly one h1: {key}')
                span = spans[0]
                full_key = (*key, 'h1')
                if full_key in seen:
                    raise ValueError(f'duplicate row across shards: {full_key}')
                seen.add(full_key)
                rows.append(dict(key=list(full_key), tokens=r['tokens'], start=span['start'],
                                 end=span['end'], shard=str(shard), index=i,
                                 excluded=bool(z['excluded'][i]),
                                 cluster=str(z['cluster'][i]),
                                 bank=z['sig_ancestor_h1_raw'][i].copy()))
    if not rows or names is None or len(names) != 4086:
        raise ValueError('missing 4086-dimensional reference bank')
    return rows, names


def select_pilot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic length extremes and median of every cell (including nulls)."""
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for cell in sorted({r['key'][1] for r in rows}):
        group = sorted((r for r in rows if r['key'][1] == cell),
                       key=lambda r: (r['start'], r['end'] - r['start'], r['key']))
        for r in (group[0], group[len(group)//2], group[-1],
                  min(group, key=lambda r: r['end']-r['start'])):
            selected[tuple(r['key'])] = r
    return list(selected.values())


def incremental_raw(loaded: Any, row: dict[str, Any], pm: np.ndarray) -> Any:
    """Independent prefix prefill, then one-token teacher forcing; T=N-1.

    Uses the same alignment, layer indexing and eager backend as cached replay.
    Captures every surface used by the CPU reference without sampling new tokens.
    """
    import torch
    from anamnesis.extraction.state_extractor import RawGenerationData
    from anamnesis.extraction.replay_cached import _slice_head_hook, _slice_seq_hook

    device = next(loaded.model.parameters()).device
    ids = torch.tensor(row['tokens'][:row['end']], device=device).unsqueeze(0)
    loaded.clear_hook_state()
    loaded.disable_hooks()
    with torch.no_grad():
        pre = loaded.model(ids[:, :row['start']], use_cache=True, return_dict=True)
    cache = pre.past_key_values
    del pre
    hs, att, logits = [], [], []
    stores: dict[str, dict[int, list[np.ndarray]]] = {k: {} for k in (
        'pre_rope_keys', 'v_proj_values', 'queries', 'gate_activations', 'attn_outputs')}
    try:
        for pos in range(row['start'], row['end']-1):
            loaded.clear_hook_state()
            loaded.enable_hooks()
            with torch.no_grad():
                out = loaded.model(ids[:, pos:pos+1], past_key_values=cache, use_cache=True,
                                   output_hidden_states=True, output_attentions=True,
                                   return_dict=True, position_ids=torch.tensor([[pos]], device=device),
                                   cache_position=torch.tensor([pos], device=device))
            cache = out.past_key_values
            hs.append(np.stack([h[0, 0].float().cpu().numpy() for h in out.hidden_states]))
            att.append(np.stack([a[0, :, 0, :].float().cpu().numpy() for a in out.attentions]))
            logits.append(out.logits[0, 0].float().cpu().numpy())
            for name, dest in stores.items():
                fn = _slice_seq_hook if name in ('gate_activations', 'attn_outputs') else _slice_head_hook
                for layer, values in (fn(getattr(loaded.hook_state, name), 1) or {}).items():
                    dest.setdefault(layer, []).extend(values)
            del out
    finally:
        loaded.clear_hook_state()
    return RawGenerationData(hidden_states=hs, attentions=att, logits=logits,
                             chosen_token_ids=np.asarray(row['tokens'][row['start']+1:row['end']],
                                                         dtype=np.float32),
                             prompt_length=row['start'], positional_means=pm, **stores)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', type=Path, required=True)
    ap.add_argument('--calibration', type=Path, required=True)
    ap.add_argument('--sigma', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--model', default='/models/Llama-3.1-70B')
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--all-rows', action='store_true')
    ap.add_argument('--floors', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--part', type=int, default=0)
    ap.add_argument('--skip-repeat', action='store_true',
                    help='Floor survey only: pilot already priced CPU repeat; GPU G0 still requires repeats')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    print('START', stamp(), flush=True)
    import psutil
    import torch
    import transformers
    import anamnesis
    import palimpsest.sigbridge as bridge
    from anamnesis.config import ModelConfig
    from anamnesis.extraction.model_loader import load_model
    from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data
    from palimpsest.featside import FeaturePool, describe_raw

    rows, names = rows_from_bank(args.bank)
    selected = rows if args.all_rows else select_pilot(rows)
    if args.limit:
        selected = selected[:args.limit]
    if args.parts < 1 or not 0 <= args.part < args.parts or args.workers < 0:
        raise ValueError('invalid part/worker count')
    selected = selected[args.part::args.parts]
    preset = bridge.resolve_preset('70b-base')
    extraction, families = bridge.battery_configs(preset)
    pin = dict(created_utc=stamp(), scope='probe-free',
               anamnesis_root=str(Path(anamnesis.__file__).parent),
               anamnesis=source_pin(Path(anamnesis.__file__).parent),
               caller_root=str(Path(bridge.__file__).parent),
               caller=source_pin(Path(bridge.__file__).parent),
               runner_sha256=sha(Path(__file__)),
               arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
               torch=torch.__version__, transformers=transformers.__version__,
               numpy=np.__version__, python=platform.python_version(),
               extraction=extraction.model_dump(mode='json'), families=families.model_dump(mode='json'),
               env={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
                    'MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG')},
               deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
               tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
               files={str(p):sha(p) for p in [args.sigma, *sorted(args.calibration.glob('*'))]
                      if p.is_file()},
               bank_files={str(p):sha(p) for p in sorted(args.bank.glob('run*/*'))
                           if p.is_file() and p.suffix in ('.npz','.jsonl')},
               rows=len(rows), selected=len(selected),
               selection=[{k:v for k,v in r.items() if k not in ('tokens','bank')} for r in selected],
               deployment_reason='isolated current upstream; path-signature disabled in frozen battery')
    write_json(args.out/'deployment.json', pin)
    print('PINNED', len(rows), len(selected), flush=True)
    if args.prepare_only:
        return
    # Verify the expected sigma before any costly execution.
    if sha(args.sigma) != '866d6aca25a716e726c12f8c6414d4292de0bf26b19dd8a0b211db5ddef3919b':
        raise ValueError('unexpected standardization artifact')
    z = np.load(args.sigma, allow_pickle=False)
    print('SIGMA_FIELDS', z.files, flush=True)
    sigma = np.asarray(z['sigma_cal'], dtype=np.float64)
    weights = np.asarray(z['weights'], dtype=np.float64)
    if z['feature_names'].astype(str).tolist() != names:
        raise ValueError('standardizer feature order mismatch')
    t0 = time.perf_counter()
    calibration = bridge.CalibrationBundle.load(args.calibration, expect_model_path=args.model,
                                                 expect_preset='70b-base')
    calibration_s = time.perf_counter()-t0
    cfg = ModelConfig(model_id=args.model, torch_dtype=preset.torch_dtype,
                      num_layers=preset.num_layers, hidden_dim=preset.hidden_dim,
                      num_attention_heads=preset.num_attention_heads,
                      num_kv_heads=preset.num_kv_heads, head_dim=preset.head_dim,
                      device_map='cuda:0', attn_implementation='eager')
    t0 = time.perf_counter()
    layers = list(range(preset.num_layers))
    loaded = load_model(cfg, sampled_layers=preset.sampled_layers, register_gate_hooks=True,
                        key_layers=layers, value_layers=layers, query_layers=layers,
                        attn_output_layers=layers)
    model_s = time.perf_counter()-t0

    def capture(r: dict[str, Any]) -> Any:
        calibration.assert_covers(r['end']-1)
        return bridge.replay_span(loaded, r['tokens'], bridge.SpanRef(
            span_id='h1', start=r['start'], length=r['end']-r['start']),
            positional_means=calibration.positional_means)

    def features(raw: Any) -> np.ndarray:
        f = compute_features_v2_from_data(raw, extraction, families,
                                         calibration.pca_components, calibration.pca_mean)
        if f.feature_names != names or not np.isfinite(f.features).all():
            raise ValueError('reference schema/nonfinite failure')
        return f.features

    def timed(fn: Any, *fn_args: Any) -> tuple[Any, float]:
        torch.cuda.synchronize()
        t = time.perf_counter()
        result = fn(*fn_args)
        torch.cuda.synchronize()
        return result, time.perf_counter()-t

    # Warm-up includes longest selected prefix. Size shm from its actual raw.
    longest = max(selected, key=lambda r:r['end'])
    raw, warmup_s = timed(lambda:capture(longest))
    slot_bytes = int(describe_raw(raw).total_bytes * 1.2)
    _, feature_warmup_s = timed(features, raw)
    del raw
    pool = FeaturePool(args.workers, slot_bytes, str(args.calibration), '70b-base') if args.workers else None
    try:
        startup_s = 0.0
        if pool is not None:
            _, startup_s = timed(pool.start)
        write_json(args.out/'startup.json', dict(calibration_s=calibration_s, model_s=model_s,
                   warmup_s=warmup_s, feature_warmup_s=feature_warmup_s, pool_s=startup_s,
                   workers=args.workers, slot_bytes=slot_bytes))
        for idx, r in enumerate(selected):
            torch.cuda.reset_peak_memory_stats()
            raw, capture_s = timed(lambda:capture(r))
            wire_bytes = describe_raw(raw).total_bytes
            def pooled(value: Any) -> np.ndarray:
                if pool is None:
                    return features(value)
                pool.submit(value)
                _, f, fnames = pool.drain_one()
                if fnames != names:
                    raise ValueError('pool feature schema mismatch')
                return f
            f, feature_s = timed(pooled, raw)
            del raw
            peak = dict(allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            record = dict(utc=stamp(), key=r['key'], start=r['start'], end=r['end'],
                          scope=('prefill+cached_replay+D2H+pool_submit_drain' if pool is not None
                                 else 'prefill+cached_replay+D2H+inprocess_features'),
                          lane='cpu-reference', capture_s=capture_s, features_s=feature_s,
                          seconds_per_row=capture_s+feature_s, raw_wire_bytes=wire_bytes,
                          peak=peak, bank_max_abs=float(np.max(np.abs(f-r['bank']))),
                          bank_l2=float(np.linalg.norm((f.astype(float)-r['bank'])*weights/sigma)),
                          worker_rss_bytes=(sum(psutil.Process(p.pid).memory_info().rss for p in pool._procs)
                                            if pool is not None else 0),
                          parent_rss_bytes=psutil.Process().memory_info().rss)
            arrays = dict(cpu=f, bank=r['bank'])
            if args.floors and not args.skip_repeat:
                raw2, repeat_s = timed(lambda:capture(r))
                repeat, repeat_features_s = timed(lambda:features(raw2))
                del raw2
                arrays['repeat'] = repeat
                record['cpu_repeat_bit_exact'] = f.tobytes() == repeat.tobytes()
                record['repeat_s'] = repeat_s+repeat_features_s
            if args.floors:
                iraw, incremental_s = timed(lambda:incremental_raw(loaded,r,calibration.positional_means))
                inc, inc_features_s = timed(lambda:features(iraw))
                del iraw
                arrays['incremental'] = inc
                record['floor_b'] = float(np.linalg.norm((inc.astype(float)-f)*weights/sigma))
                record['incremental_s'] = incremental_s+inc_features_s
            np.savez(args.out/f'row-{idx:05d}.npz', **arrays)
            write_json(args.out/f'row-{idx:05d}.json', record)
            print('ROW',idx,json.dumps(record),flush=True)
            if record.get('cpu_repeat_bit_exact') is False:
                raise RuntimeError('CPU independent-repeat/pool parity failed; stop baseline')
            if record['bank_max_abs'] != 0:
                raise RuntimeError('CPU differs from frozen bank; stop floor survey and diagnose')
        if pin['anamnesis'] != source_pin(Path(anamnesis.__file__).parent):
            raise RuntimeError('source changed during run')
        if pin['caller'] != source_pin(Path(bridge.__file__).parent):
            raise RuntimeError('caller source changed during run')
        write_json(args.out/'complete.json', dict(utc=stamp(), rows=len(selected),
                   certification=False, note='Phase 0 measurements only; no GPU lane certified'))
    finally:
        if pool is not None:
            pool.close()


if __name__ == '__main__':
    main()
