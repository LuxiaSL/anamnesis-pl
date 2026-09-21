"""Balanced warm A/B of bounded cache versus frozen accepted single lane.

Wraps the existing four-row pilot for loading and fidelity/resource checks.
Its outer timing fields include this diagnostic and are NOT throughput evidence;
only cache-ab-*.json contains synchronized unprofiled call timing.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch


def main():
    if sys.argv[1] != "--reference-module" or sys.argv[3] != "--runner":
        raise ValueError(
            "expected --reference-module PATH --runner PATH then pilot arguments"
        )
    reference_path, runner = Path(sys.argv[2]), Path(sys.argv[4])
    sys.argv = [str(runner), *sys.argv[5:]]
    sys.path.insert(0, str(runner.parent))
    output = Path(sys.argv[sys.argv.index("--out") + 1])
    import runpy
    import torch
    from anamnesis.extraction.gpu_features import GpuFeatureLane
    from extraction_perf_phase0 import sha, write_json

    # Reference module imports these shared numerical dependencies: check bytes.
    import anamnesis.extraction.gpu_features as candidate_module

    for name in ("gpu_ops.py", "gpu_attention.py", "gpu_families.py"):
        if sha(reference_path.with_name(name)) != sha(
            Path(candidate_module.__file__).with_name(name)
        ):
            raise ValueError("reference dependency differs: " + name)
    spec = importlib.util.spec_from_file_location(
        "accepted_gpu_features", reference_path
    )
    reference_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference_module
    spec.loader.exec_module(reference_module)
    init, replay = GpuFeatureLane.__init__, GpuFeatureLane.replay_span
    references = {}
    visited = set()

    def construct(self, *args, **kwargs):
        references[id(self)] = reference_module.GpuFeatureLane(*args, **kwargs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        init(self, *args, **kwargs, positional_cache_range=(263, 426))
        torch.cuda.synchronize()
        write_json(
            output / "cache-construction.json",
            dict(
                seconds=time.perf_counter() - start,
                bytes=self.positional_cache_bytes,
                candidate_lane=self.lane_id,
                reference_lane=references[id(self)].lane_id,
                reference_source_sha256=sha(reference_path),
                wrapper_sha256=sha(Path(__file__)),
                outer_pilot_timings_valid=False,
            ),
        )

    def compare(self, loaded, tokens, start, end):
        reference = references[id(self)]
        key = (tuple(tokens), start, end)
        first = replay(self, loaded, tokens, start, end)
        if key in visited:
            return first
        visited.add(key)
        warm = reference.replay_span(loaded, tokens, start, end)
        if warm.features.tobytes() != first.features.tobytes():
            raise RuntimeError(
                "cache candidate differs from accepted lane before timing"
            )
        samples = []
        for round_index in range(6):
            times = {}
            order = (
                ("reference", "candidate")
                if round_index % 2 == 0
                else ("candidate", "reference")
            )
            for name in order:
                torch.cuda.synchronize()
                begin = time.perf_counter()
                result = (
                    reference.replay_span(loaded, tokens, start, end)
                    if name == "reference"
                    else replay(self, loaded, tokens, start, end)
                )
                torch.cuda.synchronize()
                times[name] = time.perf_counter() - begin
                if result.features.tobytes() != first.features.tobytes():
                    raise RuntimeError(
                        "A/B repeatability or cross-candidate equality failed"
                    )
            samples.append(dict(order=order, **times))
        write_json(
            output / f"cache-ab-{len(visited):03d}.json",
            dict(
                start=start,
                end=end,
                input_sha256=first.metadata["input_tokens_sha256"],
                candidate_lane=self.lane_id,
                reference_lane=reference.lane_id,
                samples=samples,
                all_vectors_byte_exact=True,
                certification=False,
            ),
        )
        print("CACHE_AB", len(visited), json.dumps(samples), flush=True)
        return first

    with (
        patch.object(GpuFeatureLane, "__init__", construct),
        patch.object(GpuFeatureLane, "replay_span", compare),
    ):
        runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()
