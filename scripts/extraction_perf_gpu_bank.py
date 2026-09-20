"""Bank two independent GPU replays for every declared reference row.

The CPU bank was pinned/reproduced in Phase 0. Full per-row path-floor and
downstream gates are evaluated by the subsequent join, never assumed here.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from extraction_perf_phase0 import rows_from_bank,sha,source_pin,stamp,write_json
from extraction_perf_memory import ProcessMemorySampler


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bank',type=Path,required=True)
    ap.add_argument('--calibration',type=Path,required=True)
    ap.add_argument('--sigma',type=Path,required=True)
    ap.add_argument('--pilot',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--limit',type=int)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':16:8':
        raise ValueError('requires the verified :16:8 arithmetic workspace')
    import psutil
    import torch
    import anamnesis
    import palimpsest.sigbridge as bridge
    from anamnesis.config import ModelConfig
    from anamnesis.extraction.model_loader import load_model
    from anamnesis.extraction.gpu_features import GpuFeatureLane

    print('START',stamp(),flush=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    pilot=json.loads((args.pilot/'deployment.json').read_text())
    if source_pin(Path(anamnesis.__file__).parent)!=pilot['source']:
        raise ValueError('GPU feature sources differ from successful pilot')
    # Pilot itself is incomplete certification; only a working numeric precursor.
    for row in sorted(args.pilot.glob('row-*.json')):
        r=json.loads(row.read_text())
        if not r['G0'] or r['G1'] is False or not r['G5_historical_cap_check']:
            raise ValueError('pilot contains a failed evaluated gate')
    baseline_pin=json.loads((args.pilot.parent/'baseline-001/deployment.json').read_text())
    for path,digest in baseline_pin['files'].items():
        if sha(Path(path))!=digest:
            raise ValueError(f'calibration/standardizer changed: {path}')
    rows,names=rows_from_bank(args.bank)
    if args.limit:
        rows=rows[:args.limit]
    with np.load(args.sigma) as z:
        sigma=z['sigma_cal'].astype(float)
        weights=z['weights'].astype(float)
        if z['feature_names'].astype(str).tolist()!=names:
            raise ValueError('standardizer schema mismatch')
    for path,digest in baseline_pin['bank_files'].items():
        if sha(Path(path))!=digest:
            raise ValueError(f'reference bank changed: {path}')
    preset=bridge.resolve_preset('70b-base')
    extraction,families=bridge.battery_configs(preset)
    calibration=bridge.CalibrationBundle.load(args.calibration,expect_model_path='/models/Llama-3.1-70B')
    pin=dict(utc=stamp(),source=pilot['source'],caller=source_pin(Path(bridge.__file__).parent),
             runner_sha256=sha(Path(__file__)),
             feature_schema_sha256=hashlib.sha256(json.dumps(names).encode()).hexdigest(),
             helper_sha256={name:sha(Path(__file__).with_name(name)) for name in
                            ('extraction_perf_phase0.py','extraction_perf_memory.py')},
             pilot_sha256=sha(args.pilot/'deployment.json'),
             calibration_sha256=pilot['calibration_sha256'],standardizer_sha256=sha(args.sigma),
             reference_files=baseline_pin['bank_files'],
             selected=[{k:v for k,v in r.items() if k not in ('tokens','bank')} for r in rows],
             cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),certification=False)
    write_json(args.out/'deployment.json',pin)
    cfg=ModelConfig(model_id='/models/Llama-3.1-70B',torch_dtype=preset.torch_dtype,
                    num_layers=preset.num_layers,hidden_dim=preset.hidden_dim,
                    num_attention_heads=preset.num_attention_heads,num_kv_heads=preset.num_kv_heads,
                    head_dim=preset.head_dim,device_map='cuda:0',attn_implementation='eager')
    memory=ProcessMemorySampler()
    atexit.register(lambda:write_json(args.out/'process-vram.json',memory.close()))
    layers=list(range(preset.num_layers))
    with memory.phase('model_load'):
        loaded=load_model(cfg,sampled_layers=preset.sampled_layers,register_gate_hooks=True,
                          key_layers=layers,value_layers=layers,query_layers=layers,attn_output_layers=layers)
    lane=GpuFeatureLane(extraction,families,names,calibration.positional_means,
                        calibration.pca_components,calibration.pca_mean,device='cuda:0',
                        calibration_sha256=pilot['calibration_sha256'])
    if lane.identity!=json.loads((args.pilot/'lane.json').read_text()):
        raise ValueError('lane identity differs from pilot')
    write_json(args.out/'lane.json',lane.identity)
    def replay(row,phase):
        calibration.assert_covers(row['end']-1)
        torch.cuda.synchronize()
        start=time.perf_counter()
        with memory.phase(phase):
            result=lane.replay_span(loaded,row['tokens'],row['start'],row['end'])
            torch.cuda.synchronize()
        return result,time.perf_counter()-start
    warm,elapsed=replay(rows[0],'warmup')
    del warm
    write_json(args.out/'startup.json',dict(gpu_warmup_s=elapsed,pool_s=0.,workers=0))
    for i,row in enumerate(rows):
        torch.cuda.reset_peak_memory_stats()
        first,first_s=replay(row,'gpu_candidate')
        second,second_s=replay(row,'gpu_repeat')
        exact=first.features.tobytes()==second.features.tobytes()
        record=dict(utc=stamp(),key=row['key'],lane=lane.lane_id,
                    scope='prefill+gpu_features+feature_D2H',seconds_per_row=first_s,
                    gpu_repeat_s=second_s,G0=exact,G1=None,fidelity_status='awaiting_full_gates',
                    telemetry=first.metadata,repeat_telemetry=second.metadata,
                    peak=dict(allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                              reserved_gib=torch.cuda.max_memory_reserved()/2**30),
                    worker_rss_bytes=0,parent_rss_bytes=psutil.Process().memory_info().rss)
        np.savez(args.out/f'row-{i:05d}.npz',gpu=first.features,repeat=second.features,bank=row['bank'])
        if not exact:
            record['fidelity_status']='failed'
            write_json(args.out/f'row-{i:05d}.json',record)
            raise RuntimeError('G0 failed: stop before cross-lane evaluation')
        delta=first.features.astype(float)-row['bank'].astype(float)
        error=np.abs(delta)/sigma
        record.update(floor_c=float(np.linalg.norm(delta*weights/sigma)),
                      max_feature_sigma_error=float(error.max()),worst_feature=names[int(error.argmax())],
                      G1_not_evaluated_reason='join to completed row-floor/condition manifest required')
        mem=memory.snapshot()
        required=('model_load','warmup','gpu_candidate','gpu_repeat')
        if mem['errors'] or any(not mem['phases'].get(p,{}).get('samples') for p in required):
            write_json(args.out/'process-vram.json',mem)
            raise RuntimeError('VRAM evidence missing; no gate pass')
        process_peak=max(mem['phases'][p]['peak_bytes'] for p in required)/2**30
        record['peak']['process_gib']=process_peak
        record['G5_historical_cap_check']=process_peak<=135.29
        if error.max()>.1 or process_peak>135.29:
            record['fidelity_status']='failed'
        write_json(args.out/f'row-{i:05d}.json',record)
        write_json(args.out/'process-vram.json',mem)
        print('ROW',i,json.dumps({k:v for k,v in record.items() if k not in ('telemetry','repeat_telemetry')}),flush=True)
        if record['fidelity_status']=='failed':
            raise RuntimeError('feature/resource gate failed; stop and retain evidence')
    if source_pin(Path(anamnesis.__file__).parent)!=pin['source']:
        raise RuntimeError('source changed during capture')
    write_json(args.out/'complete.json',dict(utc=stamp(),rows=len(rows),certification=False,
               note='G0 and feature/resource checks only; full joined G1/G2/G3 remain required'))


if __name__=='__main__':
    main()
