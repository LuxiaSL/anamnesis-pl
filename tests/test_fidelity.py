from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from anamnesis.extraction.fidelity import (
    DownstreamAnchor, FidelityError, ReplayBatch, Ruler, verify_coverage,
    verify_downstream, verify_memory, verify_vectors, verify_within_tol,
)


@pytest.fixture
def evidence():
    keys = tuple(('bank', f'd{i}', 'F', 0, 'h1') for i in range(6))
    names = ('a','b','c')
    cpu = ReplayBatch(np.arange(18,dtype=np.float32).reshape(6,3),keys,names,
                      ('cpu',)*6,('a'*64,)*6,tuple(f'cpu{i}' for i in range(6)),
                      'b'*64,'c'*64,'d'*64)
    gpu = replace(cpu,lane_ids=('gpu',)*6,replay_ids=tuple(f'gpu{i}' for i in range(6)))
    repeat = replace(gpu,replay_ids=tuple(f'repeat{i}' for i in range(6)))
    distances = {k:float(np.linalg.norm(cpu.features[i].astype(float)-cpu.features[(i+1)%6]))
                 for i,k in enumerate(keys)}
    ruler = Ruler(np.ones(3),np.ones(3),'e'*64,names,
                  dict.fromkeys(keys,.05),distances,
                  {k:(k,keys[(i+1)%6]) for i,k in enumerate(keys)},
                  {k:i+1 for i,k in enumerate(keys)},keys,'c'*64,'d'*64,'f'*64)
    return cpu,gpu,repeat,ruler


def test_cpu_self_passes_all_vector_gates(evidence):
    cpu,_,_,ruler=evidence
    other=replace(cpu,replay_ids=tuple(f'repeat{i}' for i in range(6)))
    result=verify_vectors(cpu,cpu,other,ruler)
    assert result['G0'] and result['G1'] and result['G4']
    assert [t['n'] for t in result['terciles']] == [2,2,2]


def test_known_bad_path_distance_fails_despite_per_feature_pass(evidence):
    cpu,gpu,repeat,ruler=evidence
    changed=gpu.features.copy()
    changed[5,0]+=.06
    result=verify_vectors(cpu,replace(gpu,features=changed),replace(repeat,features=changed),ruler)
    assert result['G0'] and not result['G1']
    assert result['max_feature_sigma_error'] < .1
    assert result['terciles'][2]['passed'] is False


def test_known_bad_coordinate_fails_even_with_large_path_bound(evidence):
    cpu,gpu,repeat,ruler=evidence
    changed=gpu.features.copy()
    changed[0,0]+=.11
    ruler=replace(ruler,floor_b=dict.fromkeys(ruler.reference_keys,1.))
    result=verify_vectors(cpu,replace(gpu,features=changed),replace(repeat,features=changed),ruler)
    assert result['G0'] and not result['G1']
    assert result['rows'][0]['floor_c'] < result['rows'][0]['bound']


def test_repeat_failure_stops_before_cross_gate_including_signed_zero(evidence):
    cpu,gpu,repeat,ruler=evidence
    changed=repeat.features.copy()
    changed[0,0]=-0.0
    result=verify_vectors(cpu,gpu,replace(repeat,features=changed),ruler)
    assert result['G0'] is False and result['G1'] is None


@pytest.mark.parametrize('change,match',[
    ({'feature_names':('b','a','c')},'feature order'),
    ({'lane_ids':('gpu','other','gpu','gpu','gpu','gpu')},'mixed'),
    ({'calibration_sha256':'0'*64},'calibration'),
    ({'fresh_cache':False},'fresh caches'),
    ({'channel':'excess'},'not certified'),
])
def test_bad_metadata_refused(evidence,change,match):
    cpu,gpu,repeat,ruler=evidence
    with pytest.raises(FidelityError,match=match):
        verify_vectors(cpu,replace(gpu,**change),repeat,ruler)


def test_same_capture_cannot_satisfy_g0(evidence):
    cpu,gpu,_,ruler=evidence
    with pytest.raises(FidelityError,match='same replay'):
        verify_vectors(cpu,gpu,gpu,ruler)


def test_missing_row_and_nonfinite_refused(evidence):
    cpu,gpu,repeat,ruler=evidence
    with pytest.raises(FidelityError,match='missing/extra'):
        verify_vectors(cpu,gpu,repeat,replace(ruler, floor_b={}))
    bad=gpu.features.copy()
    bad[0,0]=np.nan
    with pytest.raises(FidelityError,match='nonfinite'):
        verify_vectors(cpu,replace(gpu,features=bad),repeat,ruler)


def test_reordering_rows_joins_by_full_key(evidence):
    cpu,gpu,repeat,ruler=evidence
    fields={name:getattr(gpu,name)[::-1] for name in ('features','keys','lane_ids','input_sha256','replay_ids')}
    assert verify_vectors(cpu,replace(gpu,**fields),repeat,ruler)['G1']


def test_zero_condition_bound_is_not_silently_relaxed(evidence):
    cpu,gpu,repeat,ruler=evidence
    ruler=replace(ruler,condition_distance=dict.fromkeys(ruler.reference_keys,0.),
                  condition_pair={k:(k,k) for k in ruler.reference_keys})
    changed=gpu.features.copy()
    changed[0,0]=1e-8
    assert not verify_vectors(cpu,replace(gpu,features=changed),replace(repeat,features=changed),ruler)['G1']


def test_coverage_and_memory_are_mandatory(evidence):
    cpu,gpu,repeat,ruler=evidence
    vec=verify_vectors(cpu,gpu,repeat,ruler)
    cover=verify_coverage(gpu,ruler.reference_keys,{'near-floor':[gpu.keys[0]],'short':[]})
    assert not cover['G3']
    mem=verify_memory({'replay':[101]},{'replay':[100]},cap_bytes=100,
                      metric='reserved',baseline_sha256='a'*64)
    assert not mem['G5']
    assert not verify_within_tol(vec,{'G2':True},cover,mem)['passed']
    assert not verify_within_tol(vec,{}, {'G3':True},{'G5':True})['passed']


def test_g2_pinned_anchor_and_matched_cohort():
    # Controlled statistic isolates gate behavior from the owner's tested estimator.
    def stat(d,c):
        return SimpleNamespace(point=float(d.mean()),se=float(d.std()))
    d=np.asarray([[1.,2.],[4.,8.],[3.,2.]])
    docs=('d0','d1','d2')
    clusters=('c0','c1','c2')
    anchor=DownstreamAnchor(float(d.mean()),float(d.std()),docs,clusters,'a'*64,'b'*64)
    assert verify_downstream(d,d,docs,clusters,anchor,stat,'b'*64)['G2']
    assert not verify_downstream(d,d+anchor.se,docs,clusters,anchor,stat,'b'*64)['G2']
    assert not verify_downstream(d,d*2,docs,clusters,anchor,stat,'b'*64)['G2']
    with pytest.raises(FidelityError,match='cohort'):
        verify_downstream(d,d,docs[::-1],clusters,anchor,stat,'b'*64)
    with pytest.raises(FidelityError,match='reproduce'):
        verify_downstream(d,d,docs,clusters,replace(anchor,point=99.),stat,'b'*64)
