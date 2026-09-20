"""Reproduce the pinned downstream anchor with its owner's frozen estimator."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from extraction_perf_phase0 import rows_from_bank, sha, write_json
from anamnesis.extraction.fidelity import DownstreamAnchor, verify_downstream


def main() -> None:
    import json
    import palimpsest.genpair as genpair

    ap=argparse.ArgumentParser()
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    anchor_path=args.reference/'run/fb-definitive-final.json'
    report=json.loads(anchor_path.read_text())
    rows,names=rows_from_bank(args.reference)
    sigma_path=args.reference/'standardization_floor_p05.npz'
    if sha(sigma_path)!=report['standardization']['sha256']:
        raise ValueError('sigma differs from pinned anchor')
    with np.load(sigma_path) as z:
        if z['feature_names'].astype(str).tolist()!=names:
            raise ValueError('feature names differ from ruler')
        scale=z['weights'].astype(float)/z['sigma_cal'].astype(float)
    by_key={tuple(r['key']):r for r in rows}
    docs=[]
    clusters=[]
    deltas=[]
    for doc in sorted({r['key'][0] for r in rows}):
        keys=[(doc,c,0,'h1') for c in ('own_cont','xfam_paraphrase','paraphrase_own')]
        if not all(k in by_key and not by_key[k]['excluded'] for k in keys):
            continue
        a,f,b=[by_key[k] for k in keys]
        za,zf,zb=[r['bank'].astype(float)*scale for r in (a,f,b)]
        deltas.append((zf-za)-(zb-za))
        docs.append(doc)
        clusters.append(a['cluster'])
    ref=report['fb_contrast']
    source_sha=sha(Path(genpair.__file__))
    anchor=DownstreamAnchor(ref['ratio'],ref['se_corrected'],tuple(docs),tuple(clusters),
                            sha(anchor_path),source_sha)
    if len(docs)!=ref['n_documents']:
        raise ValueError('matched-n reproduction failed')
    data=np.stack(deltas)
    result=verify_downstream(data,data,docs,clusters,anchor,genpair.jackknife_coherence_ratio,source_sha)
    result['cohort']={'documents':docs,'clusters':clusters}
    result['statistic_source_sha256']=source_sha
    result['scope']='CPU-self downstream reproduction only; not GPU certification'
    write_json(args.out,result)
    print(json.dumps({k:v for k,v in result.items() if k!='cohort'}))


if __name__=='__main__':
    main()
