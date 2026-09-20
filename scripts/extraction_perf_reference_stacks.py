"""Diagnose the reference drift without invoking any GPU feature reducer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from extraction_perf_phase0 import rows_from_bank,sha,source_pin,stamp,write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bank',type=Path,required=True)
    ap.add_argument('--calibration',type=Path,required=True)
    ap.add_argument('--sigma',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    import torch
    import anamnesis
    import palimpsest.sigbridge as bridge
    from anamnesis.config import ModelConfig
    from anamnesis.extraction.model_loader import load_model
    from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data
    print('START',stamp(),flush=True)
    rows,names=rows_from_bank(args.bank)
    row=next(r for r in rows if r['key']==['s000','null::own_cont',0,'h1'])
    z=np.load(args.sigma)
    scale=z['weights'].astype(float)/z['sigma_cal'].astype(float)
    preset=bridge.resolve_preset('70b-base')
    extraction,families=bridge.battery_configs(preset)
    calibration=bridge.CalibrationBundle.load(args.calibration,expect_model_path='/models/Llama-3.1-70B')
    cfg=ModelConfig(model_id='/models/Llama-3.1-70B',torch_dtype=preset.torch_dtype,
                    num_layers=preset.num_layers,hidden_dim=preset.hidden_dim,
                    num_attention_heads=preset.num_attention_heads,num_kv_heads=preset.num_kv_heads,
                    head_dim=preset.head_dim,device_map='cuda:0',attn_implementation='eager')
    all_layers=list(range(preset.num_layers))
    loaded=load_model(cfg,sampled_layers=preset.sampled_layers,register_gate_hooks=True,
                      key_layers=all_layers,value_layers=all_layers,query_layers=all_layers,
                      attn_output_layers=all_layers)
    write_json(args.out/'deployment.json',dict(source=source_pin(Path(anamnesis.__file__).parent),
        runner_sha256=sha(Path(__file__)),workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        defaults=dict(deterministic=torch.are_deterministic_algorithms_enabled(),
                      tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                      tf32_cudnn=torch.backends.cudnn.allow_tf32),key=row['key']))
    summaries=[]
    for name,det,cudnn in [('legacy-flags',False,True),('deterministic-only',True,True),
                           ('deterministic-no-tf32',True,False),('legacy-restored',False,True)]:
        torch.use_deterministic_algorithms(det)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=cudnn
        try:
            raw=bridge.replay_span(loaded,row['tokens'],bridge.SpanRef(span_id='h1',start=row['start'],
                                  length=row['end']-row['start']),positional_means=calibration.positional_means)
        except RuntimeError as error:
            summary=dict(name=name,utc=stamp(),error=str(error),byte_exact=None)
            summaries.append(summary)
            write_json(args.out/'results.json',summaries)
            print(json.dumps(summary),flush=True)
            continue
        f=compute_features_v2_from_data(raw,extraction,families,calibration.pca_components,calibration.pca_mean)
        del raw
        delta=f.features.astype(float)-row['bank'].astype(float)
        summary=dict(name=name,utc=stamp(),byte_exact=f.features.tobytes()==row['bank'].tobytes(),
                     l2=float(np.linalg.norm(delta*scale)),max_abs=float(np.abs(delta).max()),
                     differing_features=int(np.count_nonzero(delta)))
        np.savez(args.out/f'{name}.npz',features=f.features,bank=row['bank'],feature_names=names)
        summaries.append(summary)
        write_json(args.out/'results.json',summaries)
        print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
