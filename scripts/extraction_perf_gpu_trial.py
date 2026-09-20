"""Paired, independently repeated real-model trial; never self-certifies a lane."""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from extraction_perf_phase0 import rows_from_bank,sha,source_pin,stamp,write_json


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--bank',type=Path,required=True)
    ap.add_argument('--calibration',type=Path,required=True)
    ap.add_argument('--sigma',type=Path,required=True)
    ap.add_argument('--floors',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--indices',default='0,6,17,3')
    ap.add_argument('--model',default='/models/Llama-3.1-70B')
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':16:8')
    import psutil
    import torch
    import transformers
    import anamnesis
    import palimpsest.sigbridge as bridge
    from anamnesis.config import ModelConfig
    from anamnesis.extraction.model_loader import load_model
    from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data
    from anamnesis.extraction.gpu_features import GpuFeatureLane
    from extraction_perf_memory import ProcessMemorySampler

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    print('START',stamp(),flush=True)
    rows,names=rows_from_bank(args.bank)
    by_key={tuple(r['key']):r for r in rows}
    floor_pin=json.loads((args.floors/'deployment.json').read_text())
    for path,digest in floor_pin['files'].items():
        if sha(Path(path))!=digest:
            raise ValueError(f'calibration/standardizer file changed: {path}')
    selection=floor_pin['selection']
    indices=[int(i) for i in args.indices.split(',')]
    selected=[by_key[tuple(selection[i]['key'])] for i in indices]
    floor_records={}
    for i in indices:
        record=json.loads((args.floors/f'row-{i:05d}.json').read_text())
        floor_records[tuple(record['key'])]=record
    if sha(args.sigma)!='866d6aca25a716e726c12f8c6414d4292de0bf26b19dd8a0b211db5ddef3919b':
        raise ValueError('standardizer changed')
    with np.load(args.sigma) as z:
        sigma=z['sigma_cal'].astype(float)
        weights=z['weights'].astype(float)
        if z['feature_names'].astype(str).tolist()!=names:
            raise ValueError('ruler feature order mismatch')
    calib_hash=hashlib.sha256(json.dumps(floor_pin['files'],sort_keys=True).encode()).hexdigest()
    preset=bridge.resolve_preset('70b-base')
    extraction,families=bridge.battery_configs(preset)
    pin=dict(utc=stamp(),source=source_pin(Path(anamnesis.__file__).parent),
             caller=source_pin(Path(bridge.__file__).parent),runner_sha256=sha(Path(__file__)),
             floor_receipt_sha256=sha(args.floors/'deployment.json'),
             reference_rows_sha256={str(i):sha(args.floors/f'row-{i:05d}.npz') for i in indices},
             selected=[{k:v for k,v in r.items() if k not in ('tokens','bank')} for r in selected],
             torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,
             deterministic_algorithms=True,tf32=False,scope='probe-free candidate pilot',
             cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
             cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
             calibration_sha256=calib_hash,standardizer_sha256=sha(args.sigma))
    write_json(args.out/'deployment.json',pin)
    calibration=bridge.CalibrationBundle.load(args.calibration,expect_model_path=args.model,expect_preset='70b-base')
    cfg=ModelConfig(model_id=args.model,torch_dtype=preset.torch_dtype,
                    num_layers=preset.num_layers,hidden_dim=preset.hidden_dim,
                    num_attention_heads=preset.num_attention_heads,num_kv_heads=preset.num_kv_heads,
                    head_dim=preset.head_dim,device_map='cuda:0',attn_implementation='eager')
    all_layers=list(range(preset.num_layers))
    memory=ProcessMemorySampler()
    atexit.register(lambda:write_json(args.out/'process-vram.json',memory.close()))
    with memory.phase('model_load'):
        loaded=load_model(cfg,sampled_layers=preset.sampled_layers,register_gate_hooks=True,
                          key_layers=all_layers,value_layers=all_layers,query_layers=all_layers,
                          attn_output_layers=all_layers)
    lane=GpuFeatureLane(extraction,families,names,calibration.positional_means,
                        calibration.pca_components,calibration.pca_mean,
                        device='cuda:0',calibration_sha256=calib_hash)
    write_json(args.out/'lane.json',lane.identity)

    def timed(fn,phase):
        torch.cuda.synchronize()
        t=time.perf_counter()
        with memory.phase(phase):
            result=fn()
            torch.cuda.synchronize()
        write_json(args.out/'process-vram.json',memory.snapshot())
        return result,time.perf_counter()-t

    def cpu(r):
        raw=bridge.replay_span(loaded,r['tokens'],bridge.SpanRef(span_id='h1',start=r['start'],
                              length=r['end']-r['start']),positional_means=calibration.positional_means)
        result=compute_features_v2_from_data(raw,extraction,families,calibration.pca_components,calibration.pca_mean)
        logprob=bridge.teacher_forced_mean_logprob(raw)
        if result.feature_names!=names:
            raise ValueError('CPU feature order mismatch')
        return result.features,logprob

    def gpu(r):
        return lane.replay_span(loaded,r['tokens'],r['start'],r['end'])

    # Warm both execution paths before timed paired rows. Record, don't hide it.
    warm_cpu,cpu_warmup_s=timed(lambda:cpu(selected[0]),'cpu_warmup')
    warm_delta=warm_cpu[0].astype(float)-selected[0]['bank'].astype(float)
    np.savez(args.out/'reference-before-gpu.npz',cpu=warm_cpu[0],bank=selected[0]['bank'])
    write_json(args.out/'reference-before-gpu.json',dict(
        byte_exact=warm_cpu[0].tobytes()==selected[0]['bank'].tobytes(),
        max_abs=float(np.abs(warm_delta).max()),
        standardized_l2=float(np.linalg.norm(warm_delta*weights/sigma)),
        differing_features=int(np.count_nonzero(warm_delta))))
    print('REFERENCE_BEFORE_GPU',json.dumps(json.loads((args.out/'reference-before-gpu.json').read_text())),flush=True)
    warm_gpu,gpu_warmup_s=timed(lambda:gpu(selected[0]),'gpu_warmup')
    del warm_cpu,warm_gpu
    write_json(args.out/'startup.json',dict(cpu_warmup_s=cpu_warmup_s,gpu_warmup_s=gpu_warmup_s,pool_s=0.))
    for i,r in enumerate(selected):
        key=tuple(r['key'])
        reference,cpu_s=timed(lambda:cpu(r),'cpu_reference')
        f_cpu,lp_cpu=reference
        if f_cpu.tobytes()!=r['bank'].tobytes():
            delta=f_cpu.astype(float)-r['bank'].astype(float)
            worst=np.argsort(np.abs(delta)/sigma)[-20:][::-1]
            np.savez(args.out/'reference-failure.npz',cpu=f_cpu,bank=r['bank'])
            write_json(args.out/'reference-failure.json',dict(
                utc=stamp(),key=r['key'],max_abs=float(np.abs(delta).max()),
                standardized_l2=float(np.linalg.norm(delta*weights/sigma)),
                differing_features=int(np.count_nonzero(delta)),
                worst=[dict(name=names[j],cpu=float(f_cpu[j]),bank=float(r['bank'][j]),
                            sigma_error=float(abs(delta[j])/sigma[j])) for j in worst]))
            raise RuntimeError('deterministic CPU reference differs from frozen bank; floors need revalidation')
        # Empty the allocator cache between paths, so the candidate's peak is
        # not the CPU reference's retained allocator blocks. Weights/PCA remain.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        first,gpu_s=timed(lambda:gpu(r),'gpu_candidate')
        second,repeat_s=timed(lambda:gpu(r),'gpu_repeat')
        allocated=torch.cuda.max_memory_allocated()/2**30
        reserved=torch.cuda.max_memory_reserved()/2**30
        mem=memory.snapshot()
        phases=('gpu_warmup','gpu_candidate','gpu_repeat')
        if mem['errors'] or any(not mem['phases'].get(p,{}).get('samples') for p in phases):
            raise RuntimeError('process VRAM measurements missing; resource gate cannot be evaluated')
        process_peak=max(mem['phases'][p]['peak_bytes'] for p in phases)/2**30
        exact=first.features.tobytes()==second.features.tobytes()
        np.savez(args.out/f'row-{i:05d}.npz',cpu=f_cpu,gpu=first.features,repeat=second.features)
        record=dict(utc=stamp(),key=r['key'],lane=lane.lane_id,
                    scope='prefill+gpu_features+feature_D2H',seconds_per_row=gpu_s,
                    cpu_matched_s=cpu_s,gpu_repeat_s=repeat_s,speedup_vs_matched_cpu=cpu_s/gpu_s,
                    peak=dict(allocated_gib=allocated,reserved_gib=reserved,process_gib=process_peak),
                    parent_rss_bytes=psutil.Process().memory_info().rss,worker_rss_bytes=0,
                    G0=exact,G1=None,G5_historical_cap_check=process_peak<=135.29,
                    fidelity_status='pilot_incomplete',telemetry=first.metadata)
        if not exact:
            record['fidelity_status']='failed'
            write_json(args.out/f'row-{i:05d}.json',record)
            raise RuntimeError('G0 failed: independent full GPU replays differ; stop before G1')
        delta=first.features.astype(float)-f_cpu.astype(float)
        floor_c=float(np.linalg.norm(delta*weights/sigma))
        per_feature=float(np.max(np.abs(delta)/sigma))
        record.update(floor_c=floor_c,floor_b=floor_records[key]['floor_b'],
                      max_feature_sigma_error=per_feature,
                      worst_feature=names[int(np.argmax(np.abs(delta)/sigma))],
                      mean_logprob_delta=first.mean_logprob-lp_cpu)
        if key[1]!='own_cont':
            anchor=by_key[(key[0],'own_cont',0,'h1')]['bank']
            condition=float(np.linalg.norm((f_cpu.astype(float)-anchor.astype(float))*weights/sigma))
            bound=min(floor_records[key]['floor_b'],.1*condition)
            record.update(condition_distance=condition,bound=bound,G1=floor_c<=bound and per_feature<=.1)
        else:
            record['G1_not_evaluated_reason']='anchor condition comparator pending governing ruling'
        if record['G1'] is False or per_feature>.1 or not record['G5_historical_cap_check']:
            record['fidelity_status']='failed'
        write_json(args.out/f'row-{i:05d}.json',record)
        print('ROW',i,json.dumps(record),flush=True)
        if record['fidelity_status']=='failed':
            raise RuntimeError('candidate fidelity/resource gate failed; retain receipt and stop')
    if pin['source']!=source_pin(Path(anamnesis.__file__).parent):
        raise RuntimeError('candidate code changed during run')
    write_json(args.out/'complete.json',dict(utc=stamp(),rows=len(selected),certification=False,
               note='pilot only; full G0-G5 and pending scope/comparator rulings remain'))


if __name__=='__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
