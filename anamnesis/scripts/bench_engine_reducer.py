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

With ``--batched`` (an anamnesis tree with ``batched_stats``), per cell and per batch size B
in ``--batch-sizes``, the hybrid reduction (block-output family batched across B rows of one
step count, every other family per row) against the per-row reduction:
  per_row         ms/row of the per-row reduce, its finish, and the hybrid finish of the same
                  row without the block-output loop; their difference is the per-row loop;
  batched[B]      ms/row of the whole hybrid reduce of B rows, of block_output_features alone,
                  and of attention_stats and the remaining finish;
  invariance[B]   row 0 reduced alone vs inside the batch of B: coordinates that differ (0 is
                  the lane's definition);
  agreement       hybrid vs per-row on ``--rows`` rows: per name group, max abs and rel
                  difference; and whether every non-ported coordinate is byte-identical.

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


PORTED = ("activation_norm_", "delta_norm_", "delta_cosine_", "spectral_smoothness_",
          "res_traj_", "pca_",
          # the key/value/query/gate, per-head, output and routing families
          "kv_", "value_", "qk_", "q_crosslayer", "gate_", "ph_", "epoch_",
          "logit_entropy", "top1_prob", "top5_mass", "mean_chosen_rank", "std_chosen_rank",
          "mean_surprise", "std_surprise", "surprise_", "attnres_")
"""Name prefixes of the families a hybrid tree may compute batched. A family the tree
still reduces per row shows as an exact zero difference in the agreement table."""

EFF_DIM = ("_eff_dim",)
"""Names containing these are the effective dimensions: SVD per row, Frobenius norms
batched. Reported as their own group."""

BATCHED_GROUP_FUNCTIONS = ("block_output_features", "kvqg_features", "outputs_features",
                           "routing_features", "validate_batch")
"""The batched reducer's per-group functions, timed where the tree has them."""


def _group(name: str) -> str:
    if "_eff_dim" in name and not name.startswith("gate_"):
        return "eff_dim(norms-vs-svd)"
    for prefix in PORTED:
        if name.startswith(prefix):
            return prefix
    return "other"


def profile_batched(torch, hybrid, er, made, size: int, engine: dict) -> dict:
    """One hybrid reduce of ``size`` rows under the profiler: kernel launches and host time
    inside the batched block-output call against the whole reduce, and the top host ops of
    the block-output call."""
    from torch.profiler import ProfilerActivity, profile, record_function

    inner = er.block_output_features

    def marked(*a, **kw):
        with record_function("block_outputs"):
            return inner(*a, **kw)

    er.block_output_features = marked
    rows, ids = [r for r, _ in made[:size]], [x for _, x in made[:size]]
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            hybrid.reduce(rows, token_ids=ids, engine=engine)
            torch.cuda.synchronize()
    finally:
        er.block_output_features = inner
    events = list(prof.events())
    spans = [(e.time_range.start, e.time_range.end) for e in events if e.name == "block_outputs"]

    def inside(e) -> bool:
        return any(a <= e.time_range.start and e.time_range.end <= b for a, b in spans)

    launches = [e for e in events if e.name == "cudaLaunchKernel"]
    block_ops: dict[str, list[float]] = {}
    for e in events:
        if e.name.startswith("aten::") and inside(e):
            acc = block_ops.setdefault(e.name, [0.0, 0])
            acc[0] += e.self_cpu_time_total
            acc[1] += 1
    return dict(
        batch=size,
        launches_total=len(launches),
        launches_block_outputs=sum(1 for e in launches if inside(e)),
        host_ms_block_outputs=round(sum(b - a for a, b in spans) / 1e3, 2),
        top_block_output_ops=[(k, round(v[0] / 1e3, 2), int(v[1])) for k, v in
                              sorted(block_ops.items(), key=lambda kv: -kv[1][0])[:12]])


def batched_bench(args, torch, reducer, device, report) -> None:
    """The hybrid against the per-row reduction (``--batched``)."""
    import numpy as np

    from anamnesis.extraction.fast import batched_stats as bs
    from anamnesis.extraction.fast import engine_reducer as er
    from anamnesis.extraction.fast.external_stats import ExternalStatsSpan

    hybrid = er.EngineReducer(preset=args.preset, calib_dir=args.calib_dir, device="cuda",
                              batched=True)
    report["hybrid_reducer_identity"] = hybrid.reducer_identity
    import os
    report["cpus_available"] = len(os.sched_getaffinity(0))
    try:
        from anamnesis.extraction.fast import batched_families
        report["topk_workers"] = batched_families.TOPK_WORKERS
    except ImportError:
        pass
    phase: dict[str, float] = {}

    def timed(name, fn):
        def wrapper(*a, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            torch.cuda.synchronize()
            phase[name] = phase.get(name, 0.0) + time.perf_counter() - t0
            return out
        return wrapper

    ExternalStatsSpan.attention_stats = timed("attention_stats", ExternalStatsSpan.attention_stats)
    ExternalStatsSpan.finish = timed("finish", ExternalStatsSpan.finish)
    bs.HybridStatsSpan.finish = timed("hybrid_finish", bs.HybridStatsSpan.finish)
    er.block_output_features = timed("block_outputs", er.block_output_features)
    # Per family group: every function the hybrid finish (or the batched reducer around it)
    # spends time in, whichever of them the tree has. Timers synchronise, so each group's
    # time is its own host and device time.
    from anamnesis.extraction.fast import external as ext
    from anamnesis.extraction.fast import external_stats as es
    from anamnesis.extraction.fast import families as fam
    from anamnesis.extraction.fast.attention import AttentionReducer
    from anamnesis.extraction.fast.ops import FeatureCollector
    targets = [(ext.ExternalSpan, "_validate", "g_validate"),
               (fam.FamilyReducer, "key", "g_key"), (fam.FamilyReducer, "value", "g_value"),
               (fam.FamilyReducer, "query", "g_query"), (fam.FamilyReducer, "gate", "g_gate"),
               (fam.FamilyReducer, "finish", "g_family_finish"),
               (AttentionReducer, "head_features", "g_head_features"),
               (FeatureCollector, "finish", "g_collector_finish"),
               (bs, "_outputs_from_stats", "g_outputs"), (bs, "_routing_from_stats", "g_routing"),
               (es, "_outputs_from_stats", "g_outputs"), (es, "_routing_from_stats", "g_routing"),
               (er, "_finish_arguments", "g_finish_arguments")]
    for name in BATCHED_GROUP_FUNCTIONS:
        targets.append((er, name, "g_" + name))
    for owner, attr, label in targets:
        if hasattr(owner, attr):
            setattr(owner, attr, timed(label, getattr(owner, attr)))
    # The gate's top-k Jaccard runs on host threads: sum each task's wall time.
    import threading
    topk_lock = threading.Lock()
    try:
        from anamnesis.extraction.fast import batched_families as bf
    except ImportError:
        bf = None
    if bf is not None:
        topk_inner = bf.gate_topk_overlap

        def topk_timed(host):
            t0 = time.perf_counter()
            value = topk_inner(host)
            with topk_lock:
                phase["g_topk_tasks_summed"] = phase.get("g_topk_tasks_summed", 0.0) + time.perf_counter() - t0
            return value

        bf.gate_topk_overlap = topk_timed
    engine = {"producer": "bench-synthetic", "width": 128}
    sizes = [int(b) for b in args.batch_sizes.split(",")]

    def run(red, rows, ids):
        phase.clear()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = red.reduce(rows, token_ids=ids, engine=engine)
        torch.cuda.synchronize()
        return out, time.perf_counter() - t0, dict(phase)

    for prompt, completion in _cells(args.cells):
        sampled = list(reducer.lane(completion - 1).config.sampled_layers)
        made = [synthetic_row(torch, prompt, completion, sampled, device, seed=i)
                for i in range(max(sizes + [args.rows]))]
        for i in range(args.warmup):   # compile Triton, warm the allocator
            run(reducer, [made[i][0]], [made[i][1]])
            run(hybrid, [made[i][0]], [made[i][1]])
            run(hybrid, [r for r, _ in made[:8]], [x for _, x in made[:8]])

        per_row, alone, totals, fin, hfin = [], [], [], [], []
        per_row_groups: list[dict[str, float]] = []
        for i in range(args.rows):
            out, dt, ph = run(reducer, [made[i][0]], [made[i][1]])
            per_row.append(out[0])
            totals.append(dt)
            fin.append(ph["finish"])
            per_row_groups.append({k: v for k, v in ph.items() if k.startswith("g_")})
            out, _, ph = run(hybrid, [made[i][0]], [made[i][1]])
            alone.append(out[0])
            hfin.append(ph["hybrid_finish"])
        med = statistics.median
        labels = sorted({k for g in per_row_groups for k in g})
        cell: dict = dict(prompt=prompt, completion=completion, steps=completion - 1,
                          rows=args.rows, features=int(per_row[0].features.size),
                          per_row=dict(ms_per_row=round(1e3 * med(totals), 2),
                                       ms_finish=round(1e3 * med(fin), 2),
                                       ms_finish_without_block_loop=round(1e3 * med(hfin), 2),
                                       ms_block_loop=round(1e3 * (med(fin) - med(hfin)), 2),
                                       groups_ms={k[2:]: round(1e3 * med([g.get(k, 0.0) for g in
                                                                          per_row_groups]), 2)
                                                  for k in labels}),
                          batched={}, invariance={})

        names = per_row[0].meta["feature_names"]
        label = [_group(n) for n in names]
        groups: dict[str, list[float]] = {}
        exact = {g: True for g in set(label)}
        for h, p in zip(alone, per_row):
            assert h.meta["feature_names"] == names
            for g in exact:
                sel = np.array([x == g for x in label])
                a, b = h.features[sel].astype(np.float64), p.features[sel].astype(np.float64)
                exact[g] &= a.tobytes() == b.tobytes()
                d = np.abs(a - b)
                big = np.abs(b) > 1e-6
                acc = groups.setdefault(g, [0.0, 0.0])
                acc[0] = max(acc[0], float(d.max(initial=0)))
                acc[1] = max(acc[1], float((d[big] / np.abs(b[big])).max(initial=0)))
        exact_rest = exact.get("other", True)
        cell["agreement"] = dict(non_ported_bytes_identical=bool(exact_rest),
                                 bytes_identical_by_group=dict(sorted(exact.items())),
                                 count_by_group={g: label.count(g) for g in sorted(exact)},
                                 mean_logprob_max_abs=max(abs(h.meta["mean_logprob"]
                                                              - p.meta["mean_logprob"])
                                                          for h, p in zip(alone, per_row)),
                                 max_abs_rel={k: [float(f"{v[0]:.3g}"), float(f"{v[1]:.3g}")]
                                              for k, v in groups.items()},
                                 hybrid_lane_id=alone[0].lane_id,
                                 per_row_lane_id=per_row[0].lane_id)

        for size in sizes:
            rows, ids = [r for r, _ in made[:size]], [x for _, x in made[:size]]
            reps = []
            for _ in range(args.reps):
                out, dt, ph = run(hybrid, rows, ids)
                reps.append((dt, ph))
            dt, ph = sorted(reps, key=lambda r: r[0])[len(reps) // 2]
            cell["batched"][str(size)] = dict(
                ms_per_row=round(1e3 * dt / size, 2),
                ms_block_outputs_per_row=round(1e3 * ph["block_outputs"] / size, 2),
                ms_attention_stats_per_row=round(1e3 * ph["attention_stats"] / size, 2),
                ms_rest_of_finish_per_row=round(1e3 * ph["hybrid_finish"] / size, 2),
                groups_ms_per_row={k[2:]: round(1e3 * v / size, 3) for k, v in sorted(ph.items())
                                   if k.startswith("g_")})
            moved = 0
            for b in range(min(size, args.rows)):
                moved += int((out[b].features.view(np.uint32)
                              != alone[b].features.view(np.uint32)).sum())
            cell["invariance"][str(size)] = dict(rows_checked=min(size, args.rows),
                                                 coordinates_moved=moved)
        if args.profile:
            cell["batched_profile"] = profile_batched(torch, hybrid, er, made, max(sizes), engine)
            if bf is not None:
                # The device side alone: the host top-k replaced by a constant.
                size = max(sizes)
                bf.gate_topk_overlap = lambda host: 0.0
                try:
                    _, dt, ph = run(hybrid, [r for r, _ in made[:size]], [x for _, x in made[:size]])
                finally:
                    bf.gate_topk_overlap = topk_timed
                cell["batched_without_topk"] = dict(
                    batch=size, ms_per_row=round(1e3 * dt / size, 2),
                    ms_kvqg_per_row=round(1e3 * ph.get("g_kvqg_features", 0.0) / size, 3))
        report["cells"][f"p{prompt}c{completion}"] = cell
        print(json.dumps(cell), flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib-dir", required=True, type=Path)
    ap.add_argument("--preset", default="tsumugi-1b-release-s52450")
    ap.add_argument("--cells", default="p1024c12,p1024c24,p1024c48,p1024c128")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--batched", action="store_true",
                    help="bench the hybrid (batched block-output family) against the per-row reduction")
    ap.add_argument("--batch-sizes", default="1,8,32,128")
    ap.add_argument("--reps", type=int, default=3, help="--batched: timed repeats per batch size")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    import torch

    # the fast lane's reductions refuse to run without deterministic algorithms (lane safety)
    torch.use_deterministic_algorithms(True)

    from anamnesis.extraction.fast import engine_reducer as er
    from anamnesis.extraction.fast.external_stats import ExternalStatsSpan

    device = torch.device("cuda")
    reducer = er.EngineReducer(preset=args.preset, calib_dir=args.calib_dir, device="cuda")
    engine = {"producer": "bench-synthetic", "width": 128}
    report: dict[str, Any] = dict(preset=args.preset, calib_dir=str(args.calib_dir),
                                  reducer_identity=reducer.reducer_identity, cells={})

    if args.batched:
        batched_bench(args, torch, reducer, device, report)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}")
        return 0

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
