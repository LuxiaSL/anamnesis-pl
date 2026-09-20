"""One persistent NVML sampler for this process, with explicitly named phases."""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager


class ProcessMemorySampler:
    def __init__(self, interval_s: float = .02):
        import pynvml
        self.nvml=pynvml
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
        if not visible.isdigit():
            raise ValueError('memory sampler requires one Heimdall-assigned physical GPU index')
        pynvml.nvmlInit()
        self.handle=pynvml.nvmlDeviceGetHandleByIndex(int(visible))
        self.gpu_uuid=pynvml.nvmlDeviceGetUUID(self.handle)
        if isinstance(self.gpu_uuid,bytes):
            self.gpu_uuid=self.gpu_uuid.decode('ascii')
        self.pid=os.getpid()
        self.interval_s=interval_s
        self._phase='setup'
        self._lock=threading.Lock()
        self._stop=threading.Event()
        self._phases: dict[str,dict]={}
        self._errors: list[str]=[]
        self._thread=threading.Thread(target=self._loop,daemon=True)
        self._thread.start()

    def _sample(self):
        try:
            processes=self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
            found=[p for p in processes if p.pid==self.pid]
            used=sum(int(p.usedGpuMemory) for p in found)
            if used>=2**60:
                raise RuntimeError('NVML process memory unavailable')
            with self._lock:
                phase=self._phases.setdefault(self._phase,dict(samples=0,peak_bytes=None))
                if found:
                    phase['samples']+=1
                    phase['peak_bytes']=max(used,phase['peak_bytes'] or 0)
        except Exception as error:
            with self._lock:
                if len(self._errors)<20:
                    self._errors.append(repr(error))

    def _loop(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_s)

    @contextmanager
    def phase(self,name: str):
        with self._lock:
            previous=self._phase
            self._phase=name
        self._sample()
        try:
            yield
        finally:
            self._sample()
            with self._lock:
                self._phase=previous

    def snapshot(self) -> dict:
        with self._lock:
            return dict(metric='NVML per-process usedGpuMemory bytes',pid=self.pid,
                        gpu_uuid=self.gpu_uuid,interval_s=self.interval_s,
                        phases={k:dict(v) for k,v in self._phases.items()},
                        errors=list(self._errors),observed_at_unix=time.time())

    def close(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=2.)
        if self._thread.is_alive():
            with self._lock:
                self._errors.append('NVML sampler did not stop within 2 seconds')
        return self.snapshot()
