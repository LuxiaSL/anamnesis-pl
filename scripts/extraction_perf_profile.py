"""One diagnostic trace around a frozen bank runner; never a new certificate.

Pass --runner PATH followed by the bank runner's normal arguments, including
--limit 1 and a fresh --out directory. Calls 1/2 are warmup/unprofiled timing;
call 3 is the independently repeated traced replay, checked by the bank runner.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
import runpy
import sys
from contextlib import ExitStack
from unittest.mock import patch


def main():
    if sys.argv[1] != "--runner":
        raise ValueError("first argument must be --runner PATH")
    runner = Path(sys.argv[2])
    sys.path.insert(0, str(runner.parent))
    sys.argv = [str(runner), *sys.argv[3:]]
    if sys.argv[sys.argv.index("--limit") + 1] != "1":
        raise ValueError("lightweight profile requires --limit 1")
    output = Path(sys.argv[sys.argv.index("--out") + 1])
    import numpy as np
    import torch
    from transformers.models.llama.modeling_llama import LlamaForCausalLM
    from anamnesis.extraction.gpu_features import GpuFeatureLane
    from anamnesis.extraction.gpu_attention import AttentionReducer
    from anamnesis.extraction.gpu_families import FamilyReducer

    original = GpuFeatureLane.replay_span
    calls = 0

    def scoped(function, name):
        @functools.wraps(function)
        def call(*args, **kwargs):
            with torch.profiler.record_function("extraction::" + name):
                return function(*args, **kwargs)

        return call

    def replay(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls != 3:
            return original(self, *args, **kwargs)
        # Wrappers annotate only this replay; no tensor math/settings changed.
        with ExitStack() as stack:
            targets = [
                (LlamaForCausalLM, "forward"),
                (AttentionReducer, "consume"),
                (AttentionReducer, "smoothness"),
                (AttentionReducer, "head_features"),
            ]
            targets += [
                (FamilyReducer, name)
                for name in (
                    "key",
                    "value",
                    "query",
                    "gate",
                    "residual_trajectory",
                    "finish",
                )
            ]
            targets += [(np, "argsort"), (torch, "as_tensor")]
            for owner, name in targets:
                label = getattr(owner, "__name__", str(owner)) + "." + name
                stack.enter_context(
                    patch.object(owner, name, scoped(getattr(owner, name), label))
                )
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            ) as prof:
                result = original(self, *args, **kwargs)
                torch.cuda.synchronize()
        prof.export_chrome_trace(str(output / "trace.json"))
        events = prof.key_averages()
        summary = [
            dict(
                name=e.key,
                count=e.count,
                self_cpu_us=e.self_cpu_time_total,
                cpu_total_us=e.cpu_time_total,
                self_device_us=e.self_device_time_total,
                device_total_us=e.device_time_total,
            )
            for e in events
        ]
        (output / "profile.json").write_text(
            json.dumps(
                dict(
                    diagnostic_only=True,
                    note="Nested inclusive scopes overlap; do not add them. Profiled repeat time includes trace export; use first-call timing as unprofiled baseline.",
                    events=summary,
                ),
                indent=2,
            )
            + "\n"
        )
        print(events.table(sort_by="self_device_time_total", row_limit=30), flush=True)
        return result

    with patch.object(GpuFeatureLane, "replay_span", replay):
        runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()
