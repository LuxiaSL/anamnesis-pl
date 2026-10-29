"""EngineReducer cost per row at tsumugi's shapes: where a signed row's reduction spends its time.

Signed rows are synthetic, laid out exactly as kotodama's decode engine exports them
(``kotodama.model.signing.SignedRow``): every surface at tsumugi's dimensions (72 layers,
hidden 1536, 12 query / 6 key-value heads of 128, gate width 4096, the preset's 10 sampled
layers, AttnRes routing over up to 9 sources). Their values are random but well-formed
(row-normalised routing and mean rows, masses in [0, 1]). Cost is shape-driven, not
value-driven, so the timing stands for real rows. The lane is real: the release preset and
its ext4096 calibration, as the drain job builds it.

Per cell (``--cells p1024c12,p1024c24,...``: prompt length P and completion length c, so
the span has c - 1 steps), after ``--warmup`` rows:
  total     wall per row of ``EngineReducer.reduce`` on one row, device synchronised;
  phases    attention_stats (72 layers of statistics) vs finish (block outputs, PCA, outputs,
            key/value/query/gate families, routing) per row;
  profile   with ``--profile``, the top operators by self CPU time and by self CUDA time for
            one row, to tell launch-bound from device-bound.

Environment: PYTHONPATH with an anamnesis tree that has ``engine_reducer`` (main 7e4a410 or
later) and ANAMNESIS_MODELS pointing at pleroma's registry extension (the tsumugi preset).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LAYERS, HIDDEN, HEADS, KV_HEADS, HEAD_DIM, GATE = 72, 1536, 12, 6, 128, 4096
BOUNDARIES = (0, 1, 7, 9, 17, 19, 39, 59)   # tsumugi's AttnRes blocks (moonshot_loop)


@dataclass
class SignedRow:
    positions: int
    kept: int
    start: int
    lever_slot: int
    steered: bool
    route_sources: tuple[tuple[int, int], ...]
    final_sources: int
    weights: str = ""
    surfaces: dict[str, Any] = field(default_factory=dict)


def _cells(spec: str) -> list[tuple[int, int]]:
    out = []
    for cell in spec.split(","):
        p, c = cell.strip().lstrip("p").split("c")
        out.append((int(p), int(c)))
    return out


def route_plan() -> tuple[tuple[tuple[int, int], ...], int]:
    """Per layer (attention-input sources, MLP-input sources), and the final read's."""
    plan = []
    committed = 0
    for layer in range(LAYERS):
        if layer in BOUNDARIES and layer > 0:
            committed += 1
        plan.append((committed + 1 if layer > 0 else 0, committed + 2))
    return tuple(plan), committed + 2


def synthetic_row(torch, prompt: int, completion: int, sampled: list[int], device, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=g).to(device)

    def randn(*shape):
        return torch.randn(*shape, generator=g).to(device)

    steps = completion - 1
    n_s = len(sampled)
    width = prompt + completion
    plan, final = route_plan()
    route_width = max(final, max(max(a, m) for a, m in plan))
    route = torch.zeros(steps, LAYERS, 2, route_width, device=device)
    for layer, (n_attn, n_mlp) in enumerate(plan):
        for point, n in ((0, n_attn), (1, n_mlp)):
            if n:
                w = rand(steps, n) + 1e-3
                route[:, layer, point, :n] = w / w.sum(1, keepdim=True)
    route_final = rand(steps, route_width) + 1e-3
    route_final = route_final / route_final.sum(1, keepdim=True)
    rows = torch.zeros(steps, n_s, width, device=device)
    for j in range(steps):
        live = prompt + j + 1
        w = rand(n_s, live) + 1e-4
        rows[j, :, :live] = w / w.sum(1, keepdim=True)
    import math
    logit_stats = torch.stack([rand(steps) * 5, rand(steps), rand(steps), (rand(steps) * 50).floor(),
                               rand(steps), -rand(steps) * 5], dim=1).double()
    n_blocks = len(BOUNDARIES)
    surfaces = dict(
        depth=randn(steps, LAYERS, HIDDEN),
        q=randn(steps, n_s, HEADS, HEAD_DIM), k=randn(steps, n_s, KV_HEADS, HEAD_DIM),
        v=randn(steps, n_s, KV_HEADS, HEAD_DIM), gate=randn(steps, n_s, GATE),
        head_entropy=rand(steps, LAYERS, HEADS) * math.log(width),
        mean_entropy=rand(steps, LAYERS) * math.log(width),
        head_scalars=rand(steps, n_s, 3, HEADS),
        mean_rows=rows, logit_stats=logit_stats, route=route, route_final=route_final,
        committed=[randn(steps, HIDDEN) for _ in range(n_blocks)])
    token_ids = [int(t) for t in torch.randint(3, 40000, (prompt + completion,), generator=g)]
    row = SignedRow(positions=steps, kept=steps, start=prompt, lever_slot=0, steered=False,
                    route_sources=plan, final_sources=final, surfaces=surfaces)
    return row, token_ids


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib-dir", required=True, type=Path)
    ap.add_argument("--preset", default="tsumugi-1b-release-s52450")
    ap.add_argument("--cells", default="p1024c12,p1024c24,p1024c48,p1024c128")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    import torch

    from anamnesis.extraction.fast import engine_reducer as er
    from anamnesis.extraction.fast.external_stats import ExternalStatsSpan

    device = torch.device("cuda")
    reducer = er.EngineReducer(preset=args.preset, calib_dir=args.calib_dir, device="cuda")
    engine = {"producer": "bench-synthetic", "width": 128}
    report: dict[str, Any] = dict(preset=args.preset, calib_dir=str(args.calib_dir),
                                  reducer_identity=reducer.reducer_identity, cells={})

    phase: dict[str, float] = {}
    orig_stats, orig_finish = ExternalStatsSpan.attention_stats, ExternalStatsSpan.finish

    def timed(name, fn):
        def wrapper(self, *a, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(self, *a, **kw)
            torch.cuda.synchronize()
            phase[name] = time.perf_counter() - t0
            return out
        return wrapper

    ExternalStatsSpan.attention_stats = timed("attention_stats", orig_stats)
    ExternalStatsSpan.finish = timed("finish", orig_finish)

    for prompt, completion in _cells(args.cells):
        sampled = list(reducer.lane(completion - 1).config.sampled_layers)
        totals, stats_t, finish_t = [], [], []
        n_features = None
        for i in range(args.warmup + args.rows):
            row, ids = synthetic_row(torch, prompt, completion, sampled, device, seed=i)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = reducer.reduce([row], token_ids=[ids], engine=engine)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            n_features = int(out[0].features.size)
            if i >= args.warmup:
                totals.append(dt)
                stats_t.append(phase["attention_stats"])
                finish_t.append(phase["finish"])
        cell = dict(prompt=prompt, completion=completion, steps=completion - 1, rows=args.rows,
                    features=n_features,
                    ms_per_row=round(1e3 * statistics.median(totals), 2),
                    ms_attention_stats=round(1e3 * statistics.median(stats_t), 2),
                    ms_finish=round(1e3 * statistics.median(finish_t), 2))
        if args.profile:
            from torch.profiler import ProfilerActivity, profile
            row, ids = synthetic_row(torch, prompt, completion, sampled, device, seed=999)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                reducer.reduce([row], token_ids=[ids], engine=engine)
                torch.cuda.synchronize()
            events = prof.key_averages()
            cell["kernel_launches"] = int(sum(e.count for e in events if e.key.startswith("cudaLaunchKernel")))
            cell["self_cuda_ms_total"] = round(sum(e.self_device_time_total for e in events) / 1e3, 2)
            cell["top_cpu"] = [(e.key, round(e.self_cpu_time_total / 1e3, 2), e.count) for e in
                               sorted(events, key=lambda e: -e.self_cpu_time_total)[:12]]
            cell["top_cuda"] = [(e.key, round(e.self_device_time_total / 1e3, 2), e.count) for e in
                                sorted(events, key=lambda e: -e.self_device_time_total)[:12]]
        report["cells"][f"p{prompt}c{completion}"] = cell
        print(json.dumps(cell), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
