"""Graph-replayed vs eager ExternalSpan on tsumugi rows: byte identity, wall per row, capture cost.

Rows are windows of real token text (the momentum v2 replay rows' input_ids, concatenated in
file order), cut to ``prompt + completion`` tokens with the prompt boundary at ``prompt``; the
span has ``completion - 1`` steps, as the batched signing bench lays them out.

Per cell (``--cells p1024c12,p1024c24,...``):
  identity: every row's forward feeds BOTH an eager ``ExternalSpan`` and a
    ``GraphedExternalSpan`` (one forward, attention teed to both); their vectors, names,
    mean log-probabilities and lane ids are compared byte for byte. The first row of the
    shape records the graphs (and is itself verified inside the graphed span).
  timing: ``replay_features_gpu`` per row, eager then graphed (graphs already warm), wall
    per row with the device synchronised; also the time inside ``finish`` alone.
  capture: seconds to record a shape, graphs and solver steps, input-replica and pool bytes.

Environment as the tsumugi GPU lane job (KOTODAMA_ATTNRES_EPS_FP64_COMPAT=1,
CUBLAS_WORKSPACE_CONFIG=:16:8, ANAMNESIS_MODELS=pleroma's profile), PYTHONPATH with the
anamnesis tree under test FIRST, then pleroma and its vendored kotodama src.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def _cells(spec: str) -> list[tuple[int, int]]:
    out = []
    for cell in spec.split(","):
        p, c = cell.strip().lstrip("p").split("c")
        out.append((int(p), int(c)))
    return out


def _stream(rows_path: Path, need: int) -> list[int]:
    stream: list[int] = []
    with rows_path.open() as fh:
        for i, line in enumerate(fh):
            rec = json.loads(line)
            if i == 0 and "input_ids" not in rec:
                continue
            stream += list(rec["input_ids"])
            if len(stream) >= need:
                break
    return stream


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, type=Path)
    ap.add_argument("--checkpoint-path", required=True)
    ap.add_argument("--calib-dir", required=True, type=Path)
    ap.add_argument("--preset", default="tsumugi-1b-release-s52450")
    ap.add_argument("--cells", default="p1024c12,p1024c24,p1024c48,p256c512")
    ap.add_argument("--identity-rows", type=int, default=8)
    ap.add_argument("--timing-rows", type=int, default=16)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--profile", action="store_true",
                    help="also profile one warm eager row and one warm graphed row per cell")
    args = ap.parse_args(argv)

    import torch

    import anamnesis
    import anamnesis.extraction.fast.external as external_mod
    from anamnesis.extraction.fast.external import ExternalSpan
    from anamnesis.extraction.fast.graphed import GraphedExternalSpan, SpanGraphCache
    from kotodama.model.capture import replay_features_gpu
    from pleroma.config.anamnesis_registry import ensure_registry
    from pleroma.harvest.koto_gpu_lane import KotoGpuLane
    from pleroma.model import koto

    print(f"[bench] anamnesis from {anamnesis.__file__}", flush=True)
    ensure_registry()
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = koto.load_koto(args.checkpoint_path, size="1b-tsumugi", boundaries=koto.boundaries_for("tsumugi_loop"),
                           device="cuda:0", rope_fp32=True, eps_fp64_compat=True)
    lanes = KotoGpuLane(args.preset, args.calib_dir, [], "cuda:0")
    cells = _cells(args.cells)
    n_rows = args.identity_rows + args.timing_rows + 2
    stream = _stream(args.rows, max(p + c for p, c in cells) * (n_rows + 1))
    cache = SpanGraphCache(max_shapes=8)
    finish_seconds: list[float] = []

    def timed(cls):
        class Timed(cls):
            def finish(self, **kw):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                result = super().finish(**kw)
                torch.cuda.synchronize()
                finish_seconds.append(time.perf_counter() - t0)
                return result
        return Timed

    pairs: list[tuple] = []

    class Tee:
        def __init__(self, lane, *, start, end, num_layers):
            self.eager = ExternalSpan(lane, start=start, end=end, num_layers=num_layers)
            self.graphed = GraphedExternalSpan(lane, start=start, end=end, num_layers=num_layers, cache=cache)

        def attention(self, layer, weights):
            self.eager.attention(layer, weights)
            self.graphed.attention(layer, weights)

        def finish(self, **kw):
            eager = self.eager.finish(**kw)
            graphed = self.graphed.finish(**kw)
            pairs.append((eager, graphed))
            return graphed

    graphed_cls = timed(GraphedExternalSpan)

    class GraphedWithCache(graphed_cls):
        def __init__(self, lane, *, start, end, num_layers):
            super().__init__(lane, start=start, end=end, num_layers=num_layers, cache=cache)

    report: dict = {"anamnesis": anamnesis.__file__, "torch": torch.__version__, "cells": {}}
    for p, c in cells:
        name = f"p{p}c{c}"
        L = p + c
        windows = [stream[i * L:(i + 1) * L] for i in range(n_rows)]
        lane = lanes.lane(c - 1)
        cell: dict = {"prompt": p, "completion": c, "steps": c - 1}
        # identity
        pairs.clear()
        external_mod.ExternalSpan = Tee
        try:
            for ids in windows[:args.identity_rows]:
                replay_features_gpu(model, ids, p, lane)
        finally:
            external_mod.ExternalSpan = ExternalSpan
        mismatched, coords = [], 0
        replayed = 0
        for k, (eager, graphed) in enumerate(pairs):
            coords += len(eager.features)
            same = (eager.feature_names == graphed.feature_names
                    and eager.features.tobytes() == graphed.features.tobytes()
                    and np.float64(eager.mean_logprob).tobytes() == np.float64(graphed.mean_logprob).tobytes()
                    and eager.metadata["lane_id"] == graphed.metadata["lane_id"])
            replayed += bool(graphed.metadata.get("graph_replayed")) and bool(
                graphed.metadata.get("attention_replayed"))
            if not same:
                diff = [n for n, a, b in zip(eager.feature_names, eager.features, graphed.features)
                        if np.float32(a).tobytes() != np.float32(b).tobytes()]
                mismatched.append({"row": k, "n_diff": len(diff), "first": diff[:10]})
        cell["identity"] = {"rows": len(pairs), "replayed_rows": replayed, "coordinates": coords,
                            "features_per_row": len(pairs[0][0].features) if pairs else 0,
                            "mismatched_rows": mismatched,
                            "lane_ids": sorted({e.metadata["lane_id"] for e, _ in pairs})}
        print(f"[bench] {name} identity: {cell['identity']['rows']} rows, {replayed} replayed, "
              f"{coords} coordinates, {len(mismatched)} mismatched", flush=True)
        # timing
        timing_windows = windows[args.identity_rows:args.identity_rows + args.timing_rows]
        for label, cls in (("eager", timed(ExternalSpan)), ("graphed", GraphedWithCache)):
            external_mod.ExternalSpan = cls
            try:
                replay_features_gpu(model, windows[-1], p, lane)  # warm
                walls, finish_seconds[:] = [], []
                for ids in timing_windows:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    result = replay_features_gpu(model, ids, p, lane)
                    torch.cuda.synchronize()
                    walls.append(time.perf_counter() - t0)
                    if label == "graphed" and not (result.metadata.get("graph_replayed")
                                                   and result.metadata.get("attention_replayed")):
                        print(f"[bench] {name}: a graphed row was reduced eagerly", flush=True)
                cell[label] = {"rows": len(walls), "wall_ms_median": round(1e3 * statistics.median(walls), 2),
                               "wall_ms_mean": round(1e3 * statistics.fmean(walls), 2),
                               "wall_ms_min": round(1e3 * min(walls), 2),
                               "finish_ms_median": round(1e3 * statistics.median(finish_seconds), 2)}
            finally:
                external_mod.ExternalSpan = ExternalSpan
            print(f"[bench] {name} {label}: {cell[label]}", flush=True)
        if args.profile:
            class NullSpan:
                """The forward alone: attention is dropped, finish reduces nothing."""

                def __init__(self, lane, *, start, end, num_layers):
                    pass

                def attention(self, layer, weights):
                    pass

                def finish(self, **kw):
                    return None

            class ConsumeOnly(ExternalSpan):
                """The forward and the per-layer attention reduction, no finish."""

                def finish(self, **kw):
                    torch.cuda.synchronize()
                    return None

            for label, cls in (("forward_only", NullSpan), ("forward_consume", ConsumeOnly)):
                external_mod.ExternalSpan = cls
                try:
                    replay_features_gpu(model, windows[-1], p, lane)
                    walls = []
                    for ids in timing_windows:
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        replay_features_gpu(model, ids, p, lane)
                        torch.cuda.synchronize()
                        walls.append(time.perf_counter() - t0)
                finally:
                    external_mod.ExternalSpan = ExternalSpan
                cell[label] = {"wall_ms_median": round(1e3 * statistics.median(walls), 2)}
                print(f"[bench] {name} {label}: {cell[label]}", flush=True)
            acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            for label, cls in (("eager", ExternalSpan), ("graphed", GraphedWithCache)):
                external_mod.ExternalSpan = cls
                try:
                    replay_features_gpu(model, windows[-1], p, lane)
                    torch.cuda.synchronize()
                    with torch.profiler.profile(activities=acts) as prof:
                        with torch.profiler.record_function("ROW"):
                            replay_features_gpu(model, windows[-2], p, lane)
                        torch.cuda.synchronize()
                finally:
                    external_mod.ExternalSpan = ExternalSpan
                kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                cell[f"profile_{label}"] = {"kernels": len(kernels),
                                            "device_ms": round(sum(e.device_time for e in kernels) / 1e3, 2)}
                table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=45)
                (args.out.parent / f"{args.out.stem}-{name}-{label}-profile.txt").write_text(table)
                print(f"[bench] {name} profile {label}: {cell[f'profile_{label}']}", flush=True)
        report["cells"][name] = cell
    stats = cache.stats()
    report["cache"] = {k: v for k, v in vars(stats).items()}
    report["finish_shapes"] = [
        {"start": key[1], "end": key[2], "capture_seconds": round(e.capture_seconds, 3),
         "graphs": len(e.tape.graphs), "solver_steps": e.tape.solver_steps,
         "input_mib": round(e.input_bytes / 2**20, 1), "pool_mib": round(e.pool_bytes / 2**20, 1),
         "replays": e.replays}
        for key, e in cache.finish.items()]
    report["consume_shapes"] = [
        {"start": key[1], "end": key[2], "capture_seconds": round(e.capture_seconds, 3),
         "layers": len(e.layers), "graphs": sum(len(lt.tape.graphs) for lt in e.layers),
         "solver_steps": sum(lt.tape.solver_steps for lt in e.layers),
         "input_mib": round(e.input_bytes / 2**20, 1), "pool_mib": round(e.pool_bytes / 2**20, 1),
         "replays": e.replays}
        for key, e in cache.consume.items()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
