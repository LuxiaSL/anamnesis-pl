import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def sampler_class():
    path=Path(__file__).parents[1]/'scripts/extraction_perf_memory.py'
    spec=importlib.util.spec_from_file_location('memory_sampler',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ProcessMemorySampler


def install_backend(monkeypatch,reader):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','7')
    backend=SimpleNamespace(nvmlInit=lambda:None,nvmlDeviceGetHandleByIndex=lambda index:index,
                            nvmlDeviceGetUUID=lambda handle:b'GPU-TEST',
                            nvmlDeviceGetComputeRunningProcesses=reader)
    monkeypatch.setitem(sys.modules,'pynvml',backend)


def test_sampler_counts_only_our_process_and_serializes_uuid(monkeypatch):
    install_backend(monkeypatch,lambda handle:[SimpleNamespace(pid=os.getpid(),usedGpuMemory=123),
                                               SimpleNamespace(pid=os.getpid()+1,usedGpuMemory=999999)])
    sampler=sampler_class()()
    try:
        with sampler.phase('gpu'):
            pass
    finally:
        receipt=sampler.close()
    assert receipt['phases']['gpu']['samples']>=2
    assert receipt['phases']['gpu']['peak_bytes']==123
    assert not receipt['errors']
    assert json.loads(json.dumps(receipt))['gpu_uuid']=='GPU-TEST'


def test_missing_process_is_unknown_not_zero_peak(monkeypatch):
    install_backend(monkeypatch,lambda handle:[])
    sampler=sampler_class()()
    try:
        with sampler.phase('gpu'):
            pass
    finally:
        receipt=sampler.close()
    assert receipt['phases']['gpu']=={'samples':0,'peak_bytes':None}


def test_nvml_failure_is_reported_without_crashing_model_thread(monkeypatch):
    def reader(handle):
        raise RuntimeError('driver unavailable')
    install_backend(monkeypatch,reader)
    sampler=sampler_class()()
    try:
        with sampler.phase('gpu'):
            pass
    finally:
        receipt=sampler.close()
    assert receipt['errors']
    assert 'driver unavailable' in receipt['errors'][0]
