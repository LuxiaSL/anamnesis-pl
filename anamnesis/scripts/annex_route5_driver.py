"""ANNEX route-5: search driver — sep-CMA-ES over the full residual space at L14,
fitness = selectivity on the banked dir0 gauge (charter v1.1; pricing doc 2026-07-16).

Spawns PERSISTENT workers (annex_route5_worker: model loads ONCE — Luxia directive),
then per CMA generation: ask() -> bank candidates -> dispatch (candidate, gen) pairs
-> collect z-space deltas -> selectivity per candidate -> tell(). Prompt-disjoint
split is enforced structurally: search evals draw ONLY from the OPT topic split;
the final acceptance scoring runs ONLY on ACC.

Modes (--mode):
  cold      random init, gauge axis                      [primary, v1.1 §5]
  killrung  gauge -> SHUFFLED-null axis draw, budget/10  [runs FIRST; can kill cheap]
  null      shuffled-null axis draw k, FULL budget       [mandatory, k>=2 draws]
  refine    init at unit(V4_L14), 1*d budget             [variant (ii), diagnostic]

STOPPING RULE (frozen in the pricing doc BEFORE any GPU; do not edit mid-run):
  hard cap --budget-evals; early stop after 150 consecutive CMA generations
  without improvement of best held-in selectivity; no restarts inside a budget.

--dry-run: full driver loop against a synthetic replay map (planted needle) with
no GPU, no workers — validates plumbing, logging, stopping, and checkpointing.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from anamnesis.scripts.annex_sepcma import SepCMA

F32 = NDArray[np.float32]
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STAGNATION_GENS = 150
CKPT_EVERY = 50


def selectivity(mean_delta: F32, axis: F32) -> float:
    t = float(abs(mean_delta @ axis))
    tot = float(np.linalg.norm(mean_delta))
    off = float(np.sqrt(max(tot * tot - t * t, 0.0)))
    return t / max(off, 1e-9)


def load_axis(axis_npz: Path, null_draw: int | None, kept_names: list[str]) -> F32:
    z = np.load(axis_npz, allow_pickle=True)
    ax = z["axes"][null_draw] if null_draw is not None else z["axis"]
    names = [str(n) for n in z["feature_names"]]
    pos = {n: i for i, n in enumerate(names)}
    idx = [pos[n] for n in kept_names]            # KeyError = feature fork, fail loud
    sub = ax[idx].astype(np.float64)
    n = np.linalg.norm(sub)
    if n <= 1e-12:
        raise SystemExit("axis subset degenerate")
    return (sub / n).astype(np.float32)


def build_reference(battery_root: Path, work_dir: Path, model: str) -> tuple[list[str], dict]:
    """Bank med/scale + stage0 z-matrix subset to the per-head-dropped feature set."""
    from anamnesis.analysis.battery.deltas import load_floor_scale
    from anamnesis.analysis.battery.floors import load_signature_matrix
    from anamnesis.analysis.battery.manifest import MODEL_META

    stage0 = battery_root / MODEL_META[model].stage0_dir / "signatures_v3"
    med, scale = load_floor_scale(stage0)
    X, names, gids = load_signature_matrix(stage0)
    kept = [i for i, n in enumerate(names) if not n.startswith("ph_")]
    kept_names = [names[i] for i in kept]
    Z = ((X[:, kept] - med[kept]) / scale[kept]).astype(np.float32)
    np.savez(work_dir / "reference.npz", med=med[kept], scale=scale[kept],
             stage0_z=Z, stage0_gids=np.array(gids))
    (work_dir / "reference.json").write_text(json.dumps(
        {"feature_names_kept": kept_names, "n_dropped_ph": len(names) - len(kept)}))
    return kept_names, {"stage0_gids": gids}


class WorkerPool:
    def __init__(self, work_dir: Path, worker_ids: list[int]):
        self.work_dir = work_dir
        self.ids = worker_ids

    def wait_ready(self, timeout_s: float = 900.0) -> None:
        t0 = time.time()
        while True:
            ready = [(self.work_dir / "ready" / f"w{w}").exists() for w in self.ids]
            if all(ready):
                return
            if time.time() - t0 > timeout_s:
                raise SystemExit(f"workers not ready after {timeout_s}s: {ready}")
            time.sleep(0.5)

    def evaluate(self, gen: int, C: F32, pairs: list[tuple[int, int]],
                 alpha_frac: float, timeout_s: float = 3600.0) -> dict[int, list[F32]]:
        cdir = self.work_dir / "candidates"
        cdir.mkdir(exist_ok=True)
        np.save(cdir / f"gen_{gen:05d}.npy", C.astype(np.float32))
        chunks = {w: [] for w in self.ids}
        for i, p in enumerate(pairs):
            chunks[self.ids[i % len(self.ids)]].append(list(p))
        expected = []
        for w, ch in chunks.items():
            if not ch:
                continue
            jf = self.work_dir / "jobs" / f"w{w}" / f"job_{gen:05d}.json"
            jf.parent.mkdir(parents=True, exist_ok=True)
            tmp = jf.with_suffix(".tmp")
            tmp.write_text(json.dumps({"gen": gen, "pairs": ch, "alpha_frac": alpha_frac}))
            tmp.rename(jf)
            expected.append(self.work_dir / "results" / f"w{w}_job_{gen:05d}.npz")
        out: dict[int, list[F32]] = {}
        t0 = time.time()
        while expected:
            done = [p for p in expected if p.exists()]
            for p in done:
                z = np.load(p)
                for d, ci in zip(z["deltas"], z["cand_idx"]):
                    out.setdefault(int(ci), []).append(d)
                p.unlink()
                expected.remove(p)
            if expected:
                if time.time() - t0 > timeout_s:
                    raise SystemExit(f"eval gen {gen} timed out; missing {expected}")
                time.sleep(0.05)
        (cdir / f"gen_{gen:05d}.npy").unlink()
        return out


class DryRunMap:
    """Synthetic replay map with a planted needle: validates plumbing only."""

    def __init__(self, d_in: int, d_out: int, axis: F32, seed: int, gamma: float = 3.0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((d_out, d_in)).astype(np.float32) / np.sqrt(d_in)
        self.u = rng.standard_normal(d_in).astype(np.float32)
        self.u /= np.linalg.norm(self.u)
        self.axis = axis
        self.gamma = gamma

    def evaluate(self, gen: int, C: F32, pairs, alpha_frac, timeout_s=0) -> dict[int, list[F32]]:
        out: dict[int, list[F32]] = {}
        for ci, gid in pairs:
            x = C[ci] / max(np.linalg.norm(C[ci]), 1e-12)
            g_rng = np.random.default_rng(10_000 + gid)
            noise = 0.05 * g_rng.standard_normal(self.W.shape[0]).astype(np.float32)
            d = self.W @ x + self.gamma * float(x @ self.u) * self.axis + noise
            out.setdefault(int(ci), []).append(d.astype(np.float32))
        return out


def spawn_workers(args, n_workers: int, gpu_ids: list[str]) -> list[subprocess.Popen]:
    logs = args.work_dir / "logs"
    logs.mkdir(exist_ok=True)
    procs = []
    for w in range(n_workers):
        gpu = gpu_ids[w % len(gpu_ids)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu,
               "PYTHONPATH": os.environ.get("PYTHONPATH", "."),
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
        cmd = [sys.executable, "-m", "anamnesis.scripts.annex_route5_worker",
               "--model", args.model, "--model-path", args.model_path,
               "--calib-dir", str(args.calib_dir), "--battery-root", str(args.battery_root),
               "--work-dir", str(args.work_dir), "--worker-id", str(w),
               "--site", str(args.site)]
        fh = open(logs / f"worker_{w}.log", "w")
        procs.append(subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT))
    return procs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["cold", "killrung", "null", "refine"])
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--model", default="3b")
    ap.add_argument("--model-path", default="/models/llama-3.2-3b-instruct")
    ap.add_argument("--calib-dir", type=Path, default=Path("/models/anamnesis-extract/calibration/3b"))
    ap.add_argument("--site", type=int, default=14)
    ap.add_argument("--hidden-dim", type=int, default=3072)
    ap.add_argument("--budget-evals", type=int, default=30720)     # 10*d
    ap.add_argument("--n-eval", type=int, default=10)              # stability rung, n of record
    ap.add_argument("--alpha-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--null-draw", type=int, default=None,
                    help="killrung/null modes: which shuffled-axis draw (0..7)")
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="MT manifest whose tokens the search replays (V3_L14_a0.1 cell's)")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--workers-per-gpu", type=int, default=12)     # w96 overnight config
    ap.add_argument("--acc-n", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.mode in ("killrung", "null") and args.null_draw is None:
        raise SystemExit(f"--null-draw required for mode {args.mode}")
    if args.mode == "killrung":
        args.budget_evals = max(args.budget_evals // 10, 1)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    kept_names, _ = build_reference(args.battery_root, args.work_dir, args.model)

    gauge_path = args.battery_root / "annex/annex_dir0_axis_3b.npz"
    null_path = args.battery_root / "annex/annex_dir0_shufnull_axis_3b.npz"
    if args.mode in ("killrung", "null"):
        axis = load_axis(null_path, args.null_draw, kept_names)
        axis_src = f"shufnull draw {args.null_draw}"
    else:
        axis = load_axis(gauge_path, None, kept_names)
        axis_src = "dir0 gauge"

    split = json.loads(args.split_json.read_text())
    opt_gids, acc_gids = split["opt_gids"], split["acc_gids"]

    x0 = None
    if args.mode == "refine":
        bank = np.load(args.battery_root / "a5_vectors_3b/a5_vectors.npz")
        v4 = bank["V4_L14"].astype(np.float64)
        x0 = v4 / np.linalg.norm(v4)
        args.budget_evals = min(args.budget_evals, args.hidden_dim)   # 1*d, variant (ii)

    (args.work_dir / "run_config.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(args).items()},
         "axis_src": axis_src, "n_kept_features": len(kept_names),
         "stopping": {"hard_cap_evals": args.budget_evals,
                      "stagnation_gens": STAGNATION_GENS, "restarts": 0}}, indent=1))

    workers, procs = None, []
    if args.dry_run:
        evaluator = DryRunMap(args.hidden_dim, len(kept_names), axis, seed=args.seed)
        (args.work_dir / "worker_config.json").write_text("{}")
    else:
        (args.work_dir / "worker_config.json").write_text(json.dumps(
            {"manifest": str(args.manifest), "gen_ids": opt_gids + acc_gids}))
        gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
        from anamnesis.scripts._gpu import resolve_physical_gpus
        gpu_ids = resolve_physical_gpus(gpu_ids)
        n_workers = len(gpu_ids) * args.workers_per_gpu
        procs = spawn_workers(args, n_workers, gpu_ids)
        workers = WorkerPool(args.work_dir, list(range(n_workers)))
        workers.wait_ready()
        evaluator = workers
        logger.info(f"{n_workers} workers ready (persistent; no reloads)")

    opt = SepCMA(dim=args.hidden_dim, seed=args.seed, sigma0=1.0, x0=x0)
    ev_rng = np.random.default_rng(args.seed + 999)
    hist = open(args.work_dir / "history.jsonl", "a")
    best_sel, best_x, best_gen, stagnant = -1.0, None, -1, 0
    t_start = time.time()

    while opt.evals < args.budget_evals:
        X = opt.ask()
        gids = list(ev_rng.choice(opt_gids, size=args.n_eval, replace=False))
        pairs = [(ci, int(g)) for ci in range(opt.lam) for g in gids]
        got = evaluator.evaluate(opt.gen, X.astype(np.float32), pairs, args.alpha_frac)
        fit = np.empty(opt.lam)
        for ci in range(opt.lam):
            md = np.mean(np.stack(got[ci]), axis=0)
            fit[ci] = selectivity(md, axis)
        opt.tell(X, fit)
        gbest = float(fit.max())
        if gbest > best_sel:
            best_sel, best_x, best_gen, stagnant = gbest, X[int(np.argmax(fit))].copy(), opt.gen, 0
        else:
            stagnant += 1
        hist.write(json.dumps({"gen": opt.gen, "evals": opt.evals,
                               "best_this_gen": gbest, "best_ever": best_sel,
                               "median": float(np.median(fit)), "sigma": opt.sigma,
                               "elapsed_s": round(time.time() - t_start, 1)}) + "\n")
        hist.flush()
        if opt.gen % CKPT_EVERY == 0:
            np.savez(args.work_dir / "ckpt.npz", best_x=best_x, best_sel=best_sel,
                     best_gen=best_gen, **{f"cma_{k}": v for k, v in opt.state().items()
                                           if k != "rng_state"})
        if stagnant >= STAGNATION_GENS:
            logger.info(f"STAGNATION stop at gen {opt.gen} ({opt.evals} evals)")
            break

    # ── acceptance scoring on ACC (held-out), n=acc_n, gauge AND all null draws ──
    summary = {"mode": args.mode, "axis_src": axis_src, "evals": opt.evals,
               "gens": opt.gen, "best_heldin_sel": best_sel, "best_gen": best_gen,
               "wall_s": round(time.time() - t_start, 1),
               "stopped": "stagnation" if stagnant >= STAGNATION_GENS else "budget"}
    if best_x is not None:
        acc = list(ev_rng.choice(acc_gids, size=min(args.acc_n, len(acc_gids)),
                                 replace=False))
        pairs = [(0, int(g)) for g in acc]
        got = evaluator.evaluate(opt.gen + 1, best_x[None, :].astype(np.float32),
                                 pairs, args.alpha_frac)
        md = np.mean(np.stack(got[0]), axis=0)
        gauge_axis = load_axis(gauge_path, None, kept_names)
        summary["acc_sel_on_gauge"] = selectivity(md, gauge_axis)
        nulls = np.load(null_path, allow_pickle=True)
        names = [str(n) for n in nulls["feature_names"]]
        pos = {n: i for i, n in enumerate(names)}
        idx = [pos[n] for n in kept_names]
        null_sels = []
        for k in range(nulls["axes"].shape[0]):
            a = nulls["axes"][k][idx]
            a = (a / np.linalg.norm(a)).astype(np.float32)
            null_sels.append(selectivity(md, a))
        summary["acc_sel_null_draws"] = null_sels
        bank = np.load(args.battery_root / "a5_vectors_3b/a5_vectors.npz")
        for key in ("V3_L14", "V4_L14"):
            v = bank[key].astype(np.float64)
            summary[f"cos_to_{key}_DIAGNOSTIC_ONLY"] = float(
                best_x @ v / (np.linalg.norm(best_x) * np.linalg.norm(v)))
        np.save(args.work_dir / "best_candidate.npy", best_x.astype(np.float32))

    (args.work_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    logger.info(f"SUMMARY: {json.dumps(summary)}")

    if not args.dry_run:
        (args.work_dir / "STOP").touch()
        for p in procs:
            p.wait(timeout=120)


if __name__ == "__main__":
    main()
