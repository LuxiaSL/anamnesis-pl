"""Feature support table: the span length at which each fast-lane feature is COMPUTED.

For every feature name the anamnesis fast lane emits on a kotodama looped-AttnRes
model (tsumugi geometry), find the minimum span length (steps = generated tokens - 1)
at which the value depends on the input, rather than being a constant the reducer
writes for a span too short to define it (usually a literal 0.0).

Two independent readings, cross-checked:

1. EMPIRICAL. Run kotodama's ``replay_features_gpu`` (the GPU-lane path pleroma uses,
   here on CPU) over several random token rows on several randomly initialised models
   at each step count. A feature is constant at a length when it takes one exact value
   across every row and model there.
2. SOURCE. A name-driven predictor encodes every length guard found in
   ``anamnesis/extraction/fast/`` (literal placeholder writes, short-series fallbacks,
   degenerate one-point statistics, and arithmetic bounds) and predicts each name's
   threshold and the guard's file:line, resolved from the source text at run time.

Each feature is classified:
  computed       input-dependent at every tested length >= its threshold
  literal        constant below threshold because a guard writes a literal
  degenerate     constant below threshold because the statistic is undefined there
                 (population std / slope of one point, two-node spectrum) with no guard
  structural     constant at every tested length (genuinely constant on this
                 architecture, not a short-span artefact)

Read-only with respect to anamnesis and kotodama. Run with the anamnesis venv and
PYTHONPATH=<anamnesis checkout>:<kotodama>/src; the script asserts where both resolve.

Run BY PATH (not ``-m``), so this repo's own ``anamnesis`` package does not shadow
the public one the lane is read from:
  CUBLAS_WORKSPACE_CONFIG=:16:8 PYTHONPATH=~/projects/anamnesis-worktrees/main-ro:~/projects/kotodama/src \\
    ~/projects/anamnesis/.venv/bin/python pipeline/anamnesis/scripts/feature_support_table.py --scale full
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import numpy as np

DEFAULT_STEPS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 24, 32, 64, 128]
# extra lengths that pin the long guards (surprise window, epoch windows, epoch bound)
EXTRA_STEPS = [15, 19, 20, 31, 63, 74, 75, 99, 100, 130, 149, 150, 200]  # 15/31/63/130 = pleroma's 16/32/64/131 tokens
PLEROMA_CURVE = {4: 960, 8: 570, 9: 240, 12: 10, 16: 10, 32: 9, 64: 9, 131: 3}  # tokens -> zeros
# pleroma's banked tsumugi space (3,885 names) = the lane's names less the three v3 families the
# frozen fat_01 space predates (4,833 - 421 - 431 - 96 = 3,885); an inference from the counts.
NOT_BANKED = {"value_geometry", "qk_geometry", "kv_cka"}
# the canonical schema's four core blocks keep legacy slice keys; label them by the block each
# ExtractionConfig switch enables (enable_norms_and_output_stats, _attention_and_deltas,
# _cache_and_keys, _residual_pca), the project's source vocabulary
CORE_BLOCK_LABELS = {"tier1": "core_norms_outputs", "tier2": "core_attention_deltas",
                     "tier2_5": "core_cache_keys_spectral", "tier3": "core_residual_pca"}
Kind = Literal["computed", "literal", "degenerate", "bound", "structural", "absent"]


# --------------------------------------------------------------------------- source refs

@dataclass
class SourceIndex:
    """Resolve a guard's anchor text to file:line in the anamnesis checkout."""

    root: Path
    cache: dict[str, list[str]] = field(default_factory=dict)

    def ref(self, rel: str, anchor: str) -> str:
        if rel not in self.cache:
            self.cache[rel] = (self.root / rel).read_text().splitlines()
        text = "\n".join(self.cache[rel])
        hits = [m.start() for m in re.finditer(re.escape(anchor), text)]
        if len(hits) != 1:
            raise RuntimeError(f"anchor {anchor!r} matched {len(hits)} times in {rel}")
        return f"{rel}:{text[: hits[0]].count(chr(10)) + 1}"


@dataclass(frozen=True)
class Req:
    min_steps: int
    kind: Kind
    guard: str
    note: str


def guards(src: SourceIndex) -> dict[str, str]:
    f = "anamnesis/extraction/fast/"
    return {
        "key<2": src.ref(f + "families.py", "if len(x) < 2:\n            for name in (\n                \"key_spread\""),
        "value<2": src.ref(f + "families.py", "if len(x) < 2:\n            for name in _value_layer_names("),
        "query<2": src.ref(f + "families.py", "if len(q) < 2:"),
        "gate<2": src.ref(f + "families.py", "if len(pre_silu) < 2:\n            return"),
        "traj<3": src.ref(f + "families.py", "if len(corrected) < 3:"),
        "epoch": src.ref(f + "families.py", "if len(x) >= window + stride:"),
        "epoch_empty": src.ref(f + "families.py", "else torch.zeros(1, device=self.out.device, dtype=torch.float64)"),
        "epoch_threshold": src.ref(f + "families.py", "threshold = boundaries.mean().double() + 1.5 * std("),
        "xlayer<2": src.ref(f + "families.py", "std(torch.stack(values)) if len(values) >= 2 else 0.0"),
        "cka": src.ref(f + "families.py", "if not available:"),
        "gate_xlayer": src.ref(f + "families.py", "if len(self.gate_means) >= 2:"),
        "decay<10": src.ref(f + "attention.py", "if t < 10:"),
        "flow<2": src.ref(f + "attention.py", "if t < 2:"),
        "flow_third": src.ref(f + "attention.py", "third = ((lengths - c) // 3).clamp_min(1)"),
        "ph<4": src.ref(f + "attention.py", "if self.t < 4:"),
        "spec_nodes": src.ref(f + "attention.py", "if len(steps) < 3:"),
        "fiedler": src.ref(f + "attention.py", "ev[1] / radius.clamp_min(1e-30), ev.new_zeros(())"),
        "spec_entropy": src.ref(f + "attention.py", "-(p * p.log()).sum() / math.log(len(ev))"),
        "win<4": src.ref(f + "ops.py", "if n < 4:\n        windowed"),
        "stft<8": src.ref(f + "ops.py", "if n < 8:\n        spectral"),
        "slope<2": src.ref(f + "ops.py", "if len(y) < 2:\n        return y.new_zeros(())"),
        "corr<min": src.ref(f + "ops.py", "if n < min_n:"),
        "std": src.ref(f + "ops.py", "return x.std(dim=dim, correction=0)"),
        "surprise": src.ref(f + "features.py", "out.put(\"surprise_boundary_count\", 0.0)"),
        "summ<2": src.ref(f + "attn_res.py", "if ts.numel() < 2:"),
        "attnres_missing": src.ref(f + "attn_res.py", "v = torch.stack(vecs).mean(0) if vecs else zeros18"),
        "attnres_pairs": src.ref(f + "attn_res.py", "cos_means = [zero] * n_pairs"),
    }


OP_SERIES = re.compile(
    r"^(?P<prefix>.+?)_(?:w(?P<w>\d+)_(?P<stat>mean|std|slope)|"
    r"(?P<stft>spectral_centroid|bandwidth|low_band_energy|mid_band_energy|high_band_energy))$"
)


def operator_reqs(name: str, g: dict[str, str], offsets: dict[str, int], n_windows: int = 4) -> list[Req] | None:
    """Requirements for a temporal-operator name over a series of length steps - offset."""
    m = OP_SERIES.match(name)
    if not m or not any(m["prefix"].endswith(s) for s in offsets):
        return None
    series = next(s for s in offsets if m["prefix"].endswith(s))
    off = offsets[series]
    lead = f"series {series} has length steps-{off}" if off else f"series {series} has length steps"
    if m["stft"]:
        return [Req(8 + off, "literal", g["stft<8"], f"STFT needs n>=8; {lead}")]
    reqs = [Req(4 + off, "literal", g["win<4"], f"windowed stats need n>=4; {lead}")]
    w, stat = int(m["w"]), m["stat"]
    if stat in ("std", "slope"):
        # windows 0..n_windows-2 have width n//n_windows; the last takes the remainder
        need = 2 * n_windows if w < n_windows - 1 else 5  # last window: n - 3*(n//4) >= 2 from n=5
        if stat == "slope":
            reqs.append(Req(need + off, "literal", g["slope<2"], f"slope of a 1-point window (window w{w} needs n>={need}); {lead}"))
        else:
            reqs.append(Req(need + off, "degenerate", g["std"], f"population std of a 1-point window (w{w} needs n>={need}); {lead}"))
    return reqs


def predict(name: str, g: dict[str, str]) -> list[Req]:
    """Every length requirement the source imposes on ``name`` (empty: none below 1 step)."""
    R: list[Req] = []
    # -- attention flow (sampled layers)
    if name.startswith("attn_flow_L"):
        R.append(Req(2, "literal", g["flow<2"], "attention-flow block zero-filled for steps<2"))
        ops = operator_reqs(name, g, {"_prompt_mass": 0, "_recency_bias": 0})
        if ops:
            R += ops
        if re.search(r"_region_recent_(mean|std)$", name):
            R.append(Req(3, "degenerate", g["flow_third"],
                         "third=max(1,(len-c)//3) puts the recent region past every valid column at steps=2: an empty-region 0"))
        return R
    # -- value geometry
    if re.match(r"^value_L\d+_", name):
        R.append(Req(2, "literal", g["value<2"], "value-geometry layer block zero-filled for steps<2"))
        ops = operator_reqs(name, g, {"_novelty": 1, "_drift": 1})
        if ops:
            return R + ops
        if name.endswith("_novelty_std"):
            R.append(Req(3, "degenerate", g["std"], "std of the novelty series (length steps-1)"))
        if name.endswith("_key_drift_corr"):
            R.append(Req(4, "literal", g["corr<min"], "corr needs >=3 drift points (length steps-1)"))
        return R
    # -- q/k geometry
    if re.match(r"^qk_L\d+_", name):
        R.append(Req(2, "literal", g["query<2"], "qk-geometry layer block zero-filled for steps<2"))
        ops = operator_reqs(name, g, {"_q_novelty": 1, "_qk_self_align": 0})
        if ops:
            return R + ops
        if name.endswith("_q_novelty_std"):
            R.append(Req(3, "degenerate", g["std"], "std of the novelty series (length steps-1)"))
        return R
    # -- gate family (names absent, not zero, for steps<2)
    if re.match(r"^gate_L\d+_", name):
        R.append(Req(2, "absent", g["gate<2"], "gate family emits no names for steps<2"))
        ops = operator_reqs(name, g, {"_sparsity": 0, "_drift": 1})
        if ops:
            return R + ops
        if name.endswith("_drift_std"):
            R.append(Req(3, "degenerate", g["std"], "std of the drift series (length steps-1)"))
        return R
    if name.startswith("gate_cross_layer_"):
        return [Req(2, "absent", g["gate<2"], "present once >=2 gate layers have steps>=2")]
    # -- residual trajectory
    if name.startswith("res_traj_L"):
        R.append(Req(3, "literal", g["traj<3"], "residual-trajectory block zero-filled for steps<3"))
        ops = operator_reqs(name, g, {"_velocity_norm": 1, "_direction_change": 2})
        if ops:
            return R + ops
        if name.endswith(("_direction_change_std", "_acceleration_norm_std")):
            R.append(Req(4, "degenerate", g["std"], "std of a length steps-2 series"))
        return R
    # -- key geometry
    if re.match(r"^kv_key_(spread|eff_dim|drift|novelty_mean|novelty_std|novelty_traj\d)_L\d+$", name):
        R.append(Req(2, "literal", g["key<2"], "key-geometry layer block zero-filled for steps<2"))
        if "_novelty_std_" in name:
            R.append(Req(3, "degenerate", g["std"], "std of the novelty series (length steps-1)"))
        return R
    cka_two = Req(3, "degenerate", g["cka"], "linear CKA of two 2-row centred matrices is identically 1 at steps=2")
    if re.match(r"^kv_(key|value)_cka_(early_late|adjacent_mean|overall_mean)$", name):
        return [Req(2, "literal", g["cka"], "CKA summaries zero-filled when no layer has >=2 steps"), cka_two]
    if re.match(r"^kv_(key|value)_cka_L\d+_L\d+$", name):
        return [Req(2, "absent", g["cka"], "pairwise CKA names absent for steps<2"), cka_two]
    if name in ("value_crosslayer_spread_diversity", "q_crosslayer_spread_diversity"):
        return [Req(2, "literal", g["xlayer<2"], "needs >=2 layers with a spread (steps>=2)")]
    # -- per-head
    if name.startswith("ph_L"):
        return [Req(4, "literal", g["ph<4"], "per-head block zero-filled for steps<4")]
    if name.startswith("cache_attn_decay_rate_L"):
        return [Req(10, "literal", g["decay<10"], "attention decay fit zero for steps<10")]
    if name == "surprise_boundary_count":
        return [Req(20, "literal", g["surprise"], "needs steps >= surprise_window (20); a count, so 0 is also a legitimate measured value")]
    # -- epochs (cache-epoch detection over sampled key layers)
    if name.startswith("epoch_"):
        R.append(Req(75, "literal", g["epoch_empty"], "no epoch windows until steps >= epoch_window_size+epoch_stride (75); series is zeros(1)"))
        if name.startswith("epoch_regularity"):
            R.append(Req(100, "degenerate", g["std"], "std of one boundary until >=3 epoch centres (steps>=100)"))
        if name.startswith("epoch_n_transitions"):
            R.append(Req(150, "bound", g["epoch_threshold"],
                          "count of boundaries > mean+1.5*std: with b boundaries the max z-score is sqrt(b-1), so b<=3 "
                          "(steps<150) can never exceed 1.5; identically 0, not a guard. From 150 a discrete fraction "
                          "(_max can repeat one value, e.g. 1/6 at steps 200)"))
        return R
    # -- spectral (sampled layers): nodes m = steps if ceil(steps/10)<3 else ceil(steps/10)
    if name.startswith("spectral_"):
        if name.startswith("spectral_fiedler") or name.startswith("spectral_spectral_entropy"):
            key = "fiedler" if "fiedler" in name else "spec_entropy"
            R.append(Req(2, "literal", g[key], "len(ev)==1 (one node, steps=1) writes 0"))
        if name.startswith(("spectral_fiedler", "spectral_hfer")):
            R.append(Req(3, "degenerate", g["spec_nodes"],
                         "a 2-node graph (steps=2) fixes fiedler=1 and hfer at its 2-node value; 1 node gives 0"))
        elif name.startswith("spectral_smoothness"):
            R.append(Req(2, "degenerate", g["spec_nodes"], "a 1-node normalized Laplacian is [1]: smoothness = 1 at steps=1"))
        else:
            R.append(Req(2, "literal", g["spec_entropy"],
                         "at steps=2 the 2-node entropy is ~1e-9 (ridge eigenvalue only): computed, but near-degenerate"))
        return R
    # -- attention-residual routing family
    if re.match(r"^attnres_L\d+_.+_(std|drift)$", name):
        return [Req(2, "literal", g["summ<2"], "std/drift of a 1-position series written 0")]
    # -- one-step moments (population std of one value)
    if (re.match(r"^(activation_norm_std|delta_norm_std|head_agreement_std)_L\d+$", name)
            or name in ("logit_entropy_std", "top1_prob_std", "std_chosen_rank", "std_surprise")):
        return [Req(2, "degenerate", g["std"], "population std over a 1-step series")]
    return R


def binding(reqs: list[Req]) -> Req | None:
    return max(reqs, key=lambda r: r.min_steps) if reqs else None


# --------------------------------------------------------------------------- the sweep

def build_model_cfg(scale: str):
    from kotodama.model.llama import KotodamaModelConfig
    from kotodama.presets import SHAPES, TSUMUGI_LOOP

    shape = dict(SHAPES["1b-tsumugi"])
    if scale == "small":
        # same layer plan (72 effective; 19 prelude + 20x2 core + 13 coda) and boundaries, narrow widths
        shape.update(hidden_size=128, num_attention_heads=4, num_kv_heads=2, head_dim=32,
                     intermediate_size=512, vocab_size=1024)
    return KotodamaModelConfig(**shape, attn_impl="sdpa", attn_res=True,
                               attn_res_boundaries=list(TSUMUGI_LOOP), use_liger=False, rope_fp32=True)


def load_preset(registry: Path, name: str):
    from anamnesis.config.models import ModelPreset
    from anamnesis.extraction.replay_config import native_replay_configs

    raw = json.loads(registry.read_text())["presets"][name]
    fields = set(ModelPreset.model_fields)
    preset = ModelPreset.model_validate({k: v for k, v in raw.items() if k in fields})
    extraction, families = native_replay_configs(preset)
    return preset, extraction, families


@dataclass
class Sweep:
    names: dict[int, list[str]]                     # steps -> emitted names
    values: dict[int, list[dict[str, float]]]       # steps -> per (model,row) name->value


def _sweep_rows(models, lane_for, steps_list, prompt_lengths, vocab, sweep, meta, device) -> None:
    """Every (model, step count, row) through kotodama's replay_features_gpu."""
    import torch
    from kotodama.model.capture import replay_features_gpu

    for mi, (label, build) in enumerate(models):
        t0 = time.time()
        model = build()
        print(f"[model {mi}: {label}] built in {time.time() - t0:.1f}s", flush=True)
        for steps in steps_list:
            for ri, P in enumerate(prompt_lengths):
                ids = np.random.default_rng([mi, ri, steps]).integers(3, vocab, P + steps + 1)
                t1 = time.time()
                try:
                    res = replay_features_gpu(model, ids.tolist(), P, lane_for(steps))
                except Exception as exc:  # record refusals; the table reads them as absent
                    meta["refused"][f"m{mi}/s{steps}/r{ri}"] = f"{type(exc).__name__}: {exc}"
                    print(f"  steps={steps} row={ri} REFUSED {exc}", flush=True)
                    continue
                names = [str(n) for n in res.feature_names]
                if steps in sweep.names and sweep.names[steps] != names:
                    raise RuntimeError(f"name list differs between rows at steps={steps}")
                sweep.names[steps] = names
                sweep.values[steps].append(dict(zip(names, np.asarray(res.features, np.float64).tolist())))
                print(f"  model={mi} steps={steps} P={P} names={len(names)} "
                      f"zeros={int(np.sum(np.asarray(res.features) == 0))} {time.time() - t1:.1f}s", flush=True)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def run_sweep_node(args) -> tuple[Sweep, dict]:
    """The sweep on a GPU node with pleroma's deployed tree: pleroma's KotoGpuLane (the real
    tsumugi calibration and PCA), the real release weights, and a random full-width model."""
    import torch
    import anamnesis
    import kotodama
    import pleroma
    from kotodama.model.llama import KotodamaModel
    from pleroma.config.anamnesis_registry import ensure_registry
    from pleroma.harvest.koto_gpu_lane import KotoGpuLane
    from pleroma.model import koto

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {device} but no GPU is visible")
    torch.set_num_threads(args.threads)
    # bf16 on a GPU, as pleroma signs; fp32 on CPU, where bf16 matmuls are slow (placeholder
    # writes are length guards, not dtype effects)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    ensure_registry()
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    lanes = KotoGpuLane(args.preset, args.calib_dir, [], device)
    cfg = build_model_cfg("full")
    D = cfg.hidden_size

    def real():
        return koto.load_koto(args.checkpoint, size="1b-tsumugi", boundaries=koto.boundaries_for("tsumugi_loop"),
                              device=device, dtype=dtype, rope_fp32=True, eps_fp64_compat=True)

    def random_full(seed: int):
        def build():
            torch.manual_seed(seed)
            m = KotodamaModel(cfg)
            with torch.no_grad():
                for pname, p in m.named_parameters():
                    if pname.endswith("res_query"):
                        p.normal_(0.0, 1.0 / math.sqrt(D))
            return m.to(device=device, dtype=dtype).eval()
        return build

    models = [("tsumugi-release-s52450 (real weights)", real)]
    models += [(f"random full-width init seed {1000 + i}", random_full(1000 + i)) for i in range(args.models - 1)]
    steps_list = sorted(set(args.steps))
    prompt_lengths = [12, 16, 24, 9, 31][: args.rows]
    sweep = Sweep(names={}, values=defaultdict(list))
    meta = dict(backend="node", anamnesis_file=anamnesis.__file__, kotodama_file=kotodama.__file__,
                pleroma_file=pleroma.__file__, torch=torch.__version__, dtype=str(dtype),
                device=(torch.cuda.get_device_name(0) if device.startswith("cuda") else f"cpu x{args.threads} threads"),
                scale="full", model_cfg={k: getattr(cfg, k) for k in (
                    "hidden_size", "num_layers", "num_attention_heads", "num_kv_heads", "head_dim",
                    "intermediate_size", "vocab_size", "loop_start", "loop_len", "loop_times",
                    "attn_res_boundaries")},
                models=[label for label, _ in models], checkpoint=str(args.checkpoint),
                calibration=lanes.provenance,
                routing_queries="real weights; random model's randomised N(0,1/D)",
                sampled_layers=list(lanes.extraction.sampled_layers), pca_layers=list(lanes.extraction.pca_layers),
                pca_components=lanes.extraction.pca_components, prompt_lengths=prompt_lengths,
                n_models=len(models), steps=steps_list, refused={})
    _sweep_rows(models, lanes.lane, steps_list, prompt_lengths, cfg.vocab_size, sweep, meta, device)
    return sweep, meta


def run_sweep(args) -> tuple[Sweep, dict]:
    if args.backend == "node":
        return run_sweep_node(args)
    import torch
    import anamnesis
    import kotodama
    from anamnesis.extraction.fast.features import GpuFeatureLane
    from anamnesis.extraction.fast.schema import resolve_gpu_schema
    from kotodama.model.llama import KotodamaModel

    if not Path(anamnesis.__file__).resolve().is_relative_to(args.anamnesis_root.resolve()):
        raise SystemExit(f"anamnesis resolves to {anamnesis.__file__}, not {args.anamnesis_root}")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    preset, extraction, families = load_preset(args.registry, args.preset)
    cfg = build_model_cfg(args.scale)
    L, D = cfg.num_layers, cfg.hidden_size
    if L != preset.num_layers:
        raise SystemExit(f"model has {L} layers, preset {preset.num_layers}")
    steps_list = sorted(set(args.steps))
    prompt_lengths = [12, 16, 24, 9, 31][: args.rows]
    max_pos = max(prompt_lengths) + max(steps_list) + 4
    rng = np.random.default_rng(20261028)
    pm = (rng.standard_normal((L + 1, max_pos, D)) * 0.1).astype(np.float32)
    comps = {l: np.linalg.qr(rng.standard_normal((D, D)))[0][: extraction.pca_components].astype(np.float32)
             for l in extraction.pca_layers}
    means = {l: (rng.standard_normal(D) * 0.1).astype(np.float32) for l in extraction.pca_layers}
    lanes: dict[int, GpuFeatureLane] = {}

    def lane_for(steps: int) -> GpuFeatureLane:
        key = min(steps, 8)
        if key not in lanes:
            names = resolve_gpu_schema(L, steps, extraction, families, comps).feature_names
            lanes[key] = GpuFeatureLane(extraction, families, list(names), pm, comps, means, device="cpu",
                                        calibration_sha256="0" * 64, replay_path="full")
        return lanes[key]

    sweep = Sweep(names={}, values=defaultdict(list))
    meta = dict(anamnesis_file=anamnesis.__file__, kotodama_file=kotodama.__file__, torch=torch.__version__,
                scale=args.scale, model_cfg={k: getattr(cfg, k) for k in (
                    "hidden_size", "num_layers", "num_attention_heads", "num_kv_heads", "head_dim",
                    "intermediate_size", "vocab_size", "loop_start", "loop_len", "loop_times",
                    "attn_res_boundaries")},
                routing_queries=("zero init" if args.keep_zero_routing else "randomised N(0,1/D)"),
                sampled_layers=list(extraction.sampled_layers), pca_layers=list(extraction.pca_layers),
                pca_components=extraction.pca_components, prompt_lengths=prompt_lengths,
                n_models=args.models, steps=steps_list, refused={})
    def random_local(seed: int):
        def build():
            torch.manual_seed(seed)
            model = KotodamaModel(cfg).eval()
            if not args.keep_zero_routing:
                # AttnRes pseudo-queries initialise to zero, so a fresh model routes uniformly and
                # every routing statistic is a constant of the source count. A trained model's are
                # not; give them N(0, 1/D) entries so routing logits are O(1), as after training.
                with torch.no_grad():
                    for pname, p in model.named_parameters():
                        if pname.endswith("res_query"):
                            p.normal_(0.0, 1.0 / math.sqrt(D))
            return model
        return build

    models = [(f"random init seed {1000 + mi}", random_local(1000 + mi)) for mi in range(args.models)]
    meta["models"] = [label for label, _ in models]
    _sweep_rows(models, lane_for, steps_list, prompt_lengths, cfg.vocab_size, sweep, meta, "cpu")
    return sweep, meta


def save_raw(path: Path, sweep: Sweep, meta: dict) -> None:
    arrays = {"meta": np.array(json.dumps(meta, default=str))}
    for s, names in sweep.names.items():
        arrays[f"names_{s}"] = np.array(names)
        arrays[f"values_{s}"] = np.array([[r[n] for n in names] for r in sweep.values[s]], dtype=np.float64)
    np.savez_compressed(path, **arrays)


def load_raw(path: Path) -> tuple[Sweep, dict]:
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        sweep = Sweep(names={}, values={})
        for key in z.files:
            if key.startswith("names_"):
                s = int(key[6:])
                names = [str(n) for n in z[key]]
                sweep.names[s] = names
                sweep.values[s] = [dict(zip(names, row.tolist())) for row in z[f"values_{s}"]]
    return sweep, meta


# --------------------------------------------------------------------------- analysis

def family_of(name: str, slices: dict[str, tuple[int, int]], order: list[str]) -> str:
    if name.startswith("attnres_"):
        return "attn_res"
    i = order.index(name) if name in order else -1
    for fam, (a, b) in slices.items():
        if a <= i < b:
            return CORE_BLOCK_LABELS.get(fam, fam)
    return "unknown"


def analyse(sweep: Sweep, meta: dict, g: dict[str, str], slices, schema_order) -> tuple[list[dict], dict]:
    steps_list = sorted(sweep.names)
    all_names: list[str] = []
    for s in reversed(steps_list):  # longest-span order first, then any short-only names
        for n in sweep.names[s]:
            if n not in all_names:
                all_names.append(n)
    seen_set = set(all_names)
    status: dict[str, dict[int, tuple[str, float | None]]] = {n: {} for n in all_names}
    for s in steps_list:
        rows = sweep.values[s]
        present = set(sweep.names[s])
        for n in all_names:
            if n not in present:
                status[n][s] = ("absent", None)
                continue
            vals = {r[n] for r in rows}
            status[n][s] = ("const", vals.pop()) if len(vals) == 1 else ("var", None)

    fam = {n: family_of(n, slices, schema_order) for n in all_names}
    table, curve = [], {}
    for s in steps_list:
        c = Counter()
        for n in all_names:
            st, v = status[n][s]
            if st == "const":
                c["const"] += 1
                c["const_zero" if v == 0.0 else "const_nonzero"] += 1
            elif st == "absent":
                c["absent"] += 1
        banked = [n for n in sweep.names[s] if fam[n] not in NOT_BANKED]
        bz = [sum(1 for n in banked if r[n] == 0.0) for r in sweep.values[s]]
        c["banked_names"] = len(banked)
        c["banked_zeros_min"] = int(min(bz)) if bz else -1
        c["banked_zeros_max"] = int(max(bz)) if bz else -1
        c["banked_const_zero"] = sum(1 for n in banked if status[n][s] == ("const", 0.0))
        zeros = [sum(1 for n in sweep.names[s] if r[n] == 0.0) for r in sweep.values[s]]
        c["zeros_per_row_mean"] = float(np.mean(zeros)) if zeros else float("nan")
        c["zeros_per_row_min"] = int(min(zeros)) if zeros else -1
        c["zeros_per_row_max"] = int(max(zeros)) if zeros else -1
        c["n_names"] = len(sweep.names[s])
        curve[s] = dict(c)

    for n in all_names:
        st = status[n]
        # min steps such that the feature is present and varying at every tested length >= it
        threshold = None
        for s in reversed(steps_list):
            if st[s][0] == "var":
                threshold = s
            else:
                break
        const_vals = sorted({v for (k, v) in st.values() if k == "const"})
        const_lengths = [s for s in steps_list if st[s][0] == "const"]
        absent_lengths = [s for s in steps_list if st[s][0] == "absent"]
        reqs = predict(n, g)
        b = binding([r for r in reqs if r.kind != "absent"])
        absent_req = next((r for r in reqs if r.kind == "absent"), None)
        pred_min = b.min_steps if b else 1
        # empirical threshold expressed on the tested grid: the smallest tested length >= predicted
        pred_on_grid = next((s for s in steps_list if s >= pred_min), None)
        absent_on_grid = next((s for s in steps_list if s >= absent_req.min_steps), None) if absent_req else None
        if threshold is None:
            klass = "structural" if not b or pred_on_grid is None or b.kind != "bound" else "bound"
            if b and b.kind == "bound":
                klass = "bound"
        elif threshold == steps_list[0] or (const_lengths == [] and absent_lengths):
            klass = "computed"
        else:
            klass = b.kind if b else "UNGUARDED"
        agrees = (threshold == max(pred_on_grid or 0, absent_on_grid or 0) if threshold is not None
                  else (b is not None and b.kind == "bound" and pred_on_grid is None))
        if threshold is not None and threshold == steps_list[0] and pred_min <= steps_list[0]:
            agrees = True
        notes = []
        if const_lengths:
            notes.append(f"constant at steps {compress(const_lengths)}")
        if absent_lengths:
            notes.append(f"absent at steps {compress(absent_lengths)}")
        if b:
            notes.append(b.note)
        if not agrees:
            notes.append(f"MISMATCH: source predicts {pred_min}{' (absent<' + str(absent_req.min_steps) + ')' if absent_req else ''}")
        guard_refs = sorted({r.guard for r in reqs if (b and r.min_steps == b.min_steps) or r.kind == "absent"}) if reqs else []
        table.append(dict(
            feature_name=n,
            family=family_of(n, slices, schema_order),
            min_steps_computed=("never" if threshold is None else threshold),
            placeholder_value=("" if not const_vals else ",".join(f"{v:.9g}" for v in const_vals)),
            source_guard=";".join(guard_refs),
            notes="; ".join(notes),
            klass=klass,
            predicted=pred_min,
            agrees=agrees,
        ))
    return table, curve


def compress(xs: list[int]) -> str:
    return ",".join(str(x) for x in xs)


# --------------------------------------------------------------------------- output

def setup_lines(meta: dict) -> list[str]:
    models = "; ".join(meta.get("models", [])) or f"{meta['n_models']} random models"
    rows = f"{meta['n_models']} models x {len(meta['prompt_lengths'])} random token rows (prompt lengths {meta['prompt_lengths']}) per length"
    if meta.get("backend") == "node":
        calib = meta.get("calibration", {})
        return [
            f"- SWEEP on node2 with pleroma's deployed tree: anamnesis `{meta['anamnesis_file']}` (vendored 2c715e0), "
            f"kotodama `{meta['kotodama_file']}`, pleroma `{meta['pleroma_file']}`; torch {meta['torch']}, "
            f"{meta['device']}, {meta['dtype']}. Path: kotodama `replay_features_gpu` -> anamnesis `ExternalSpan`, "
            "lanes from pleroma `KotoGpuLane` (the lane pleroma harvests on).",
            f"- SOURCE AUDIT (guard file:line) against anamnesis main f7ad8af. The vendored 2c715e0 fast lane differs from "
            "f7ad8af only by byte-preserving refactors (FeatureCollector assembly, cached STFT/slope constants, "
            "spectral_hfer helper, vectorised gate top-k); every placeholder guard is the same.",
            f"- Models: {models}. {rows}. Config {json.dumps(meta['model_cfg'])}.",
            f"- Calibration: real, `release-s52450-ext4096` (sha {calib.get('calibration_sha256', '?')}, "
            f"{calib.get('positions_calibrated', '?')} positions); sampled layers {meta['sampled_layers']}, "
            f"PCA layers {meta['pca_layers']} x {meta['pca_components']} components.",
        ]
    return [
        f"- anamnesis: `{meta['anamnesis_file']}`; kotodama: `{meta['kotodama_file']}`; torch {meta['torch']}.",
        f"- Model scale: **{meta['scale']}**, config {json.dumps(meta['model_cfg'])}. Models: {models}. {rows}.",
        f"- Preset `tsumugi-1b-release-s52450` from pleroma's registry; sampled layers {meta['sampled_layers']}, "
        f"PCA layers {meta['pca_layers']} with {meta['pca_components']} random orthonormal components (no local calibration; "
        "positional means random). Path: kotodama `replay_features_gpu` -> anamnesis `ExternalSpan`, CPU.",
    ]


def write_outputs(table, curve, meta, out_tsv: Path, out_md: Path, out_json: Path, steps_list):
    cols = ["feature_name", "family", "min_steps_computed", "placeholder_value", "source_guard", "notes"]
    with out_tsv.open("w") as fh:
        fh.write("\t".join(c if c != "source_guard" else "source_guard (file:line)" for c in cols) + "\tclass\n")
        for r in table:
            fh.write("\t".join(str(r[c]).replace("\t", " ") for c in cols) + f"\t{r['klass']}\n")
    out_json.write_text(json.dumps(dict(meta=meta, curve={str(k): v for k, v in curve.items()},
                                        rows=table), indent=1, default=str))

    by_klass = Counter(r["klass"] for r in table)
    fam_thr: dict[str, Counter] = defaultdict(Counter)
    for r in table:
        fam_thr[r["family"]][str(r["min_steps_computed"])] += 1
    thr_counts = Counter(str(r["min_steps_computed"]) for r in table)
    lines = [
        "# Feature support table — fast lane on tsumugi geometry",
        "",
        f"Produced by `pipeline/anamnesis/scripts/feature_support_table.py`. Table: `{out_tsv.name}`; "
        f"raw per-length readings: `{out_json.name}`.",
        "",
        "## Setup",
        "",
        *setup_lines(meta),
        "- A feature is *constant* at a length when it takes one exact value across every row and model. "
        "`min_steps_computed` = smallest tested steps from which it is present and varying at every longer tested length.",
        f"- Lengths tested (steps): {steps_list}. Refusals: {len(meta['refused'])}.",
        "",
        "## Classes",
        "",
        "| class | features | meaning |",
        "|---|---:|---|",
    ]
    meaning = dict(computed="input-dependent at every tested length",
                   literal="constant below threshold: a guard writes a literal (0.0)",
                   degenerate="constant below threshold: statistic undefined (1-point std/slope, 2-node spectrum), no literal",
                   bound="identically 0 by arithmetic over the tested range (epoch transition count)",
                   structural="constant at every tested length on this architecture",
                   absent="never emitted",
                   UNGUARDED="constant at short lengths with NO source guard found")
    for k, v in sorted(by_klass.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} | {meaning.get(k, '')} |")
    lines += ["", "## Threshold counts (features by min_steps_computed)", "", "| min_steps | features |", "|---:|---:|"]
    for k in sorted(thr_counts, key=lambda x: (x == "never", int(x) if x != "never" else 0)):
        lines.append(f"| {k} | {thr_counts[k]} |")
    lines += ["", "## Per family", "", "| family | thresholds (min_steps: count) |", "|---|---|"]
    for fam in sorted(fam_thr):
        items = sorted(fam_thr[fam].items(), key=lambda kv: (kv[0] == "never", int(kv[0]) if kv[0] != "never" else 0))
        lines.append(f"| {fam} | " + ", ".join(f"{k}: {v}" for k, v in items) + " |")
    lines += ["", "## Placeholder curve", "",
              "Constant features at each length (over names emitted at that length), and exact zeros per row "
              "(the quantity pleroma counted, over all emitted names here, not the banked 3,885).", "",
              "| steps | tokens | names | constant | const=0 | const!=0 | absent | zeros/row (min-max) | banked-proxy names | banked const=0 | banked zeros/row | pleroma zeros |",
              "|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|"]
    for s in steps_list:
        c = curve[s]
        pz = PLEROMA_CURVE.get(s + 1, "")
        lines.append(f"| {s} | {s + 1} | {c['n_names']} | {c.get('const', 0)} | {c.get('const_zero', 0)} | "
                     f"{c.get('const_nonzero', 0)} | {c.get('absent', 0)} | {c['zeros_per_row_min']}-{c['zeros_per_row_max']} | "
                     f"{c['banked_names']} | {c['banked_const_zero']} | {c['banked_zeros_min']}-{c['banked_zeros_max']} | {pz} |")
    lines += ["", "### Against pleroma (lane 659aea48, 10 probe rows, banked 3,885 names; tokens = steps + 1)", "",
              "| tokens | steps | pleroma zeros | banked-proxy const=0 here | banked-proxy zeros/row here | match |",
              "|---:|---:|---:|---:|---|---|"]
    for tok, pz in PLEROMA_CURVE.items():
        s = tok - 1
        if s not in curve:
            lines.append(f"| {tok} | {s} | {pz} | not tested | | |")
            continue
        c = curve[s]
        ok = c["banked_zeros_min"] <= pz <= c["banked_zeros_max"] or c["banked_const_zero"] == pz
        lines.append(f"| {tok} | {s} | {pz} | {c['banked_const_zero']} | {c['banked_zeros_min']}-{c['banked_zeros_max']} | "
                     f"{'yes' if c['banked_const_zero'] == pz else ('within row range' if ok else 'NO')} |")
    mism = [r for r in table if not r["agrees"]]
    lines += ["", f"## Source/empirical mismatches ({len(mism)})", ""]
    for r in mism[:200]:
        lines.append(f"- `{r['feature_name']}` empirical {r['min_steps_computed']}, predicted {r['predicted']} — {r['notes']}")
    struct = [r for r in table if r["klass"] in ("structural", "bound")]
    lines += ["", f"## Bound or structural: constant with no short-span guard ({len(struct)})", ""]
    for r in struct:
        lines.append(f"- `{r['feature_name']}` = {r['placeholder_value']} ({r['klass']}) — {r['notes']}")
    out_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    home = Path.home() / "projects"
    ap.add_argument("--scale", choices=("full", "small"), default="full")
    ap.add_argument("--models", type=int, default=2)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--steps", type=int, nargs="*", default=DEFAULT_STEPS + EXTRA_STEPS)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--backend", choices=("local", "node"), default="local",
                    help="local: CPU, random weights and calibration; node: GPU with pleroma's lane, "
                         "real calibration, real tsumugi weights + random full-width models")
    ap.add_argument("--device", default="cuda:0", help="node backend: cuda:0 or cpu")
    ap.add_argument("--sweep-only", action="store_true",
                    help="save <tag>.values.npz and stop (analyse later with --reanalyse against the audited tree)")
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("/models/pleroma/koto-assets/tsumugi-release-s52450/model.pt"))
    ap.add_argument("--calib-dir", type=Path,
                    default=Path("/mnt/beegfs/luxia/pleroma/tsumugi/calibration/release-s52450-ext4096"))
    ap.add_argument("--reanalyse", action="store_true",
                    help="re-read the saved <tag>.values.npz instead of sweeping")
    ap.add_argument("--keep-zero-routing", action="store_true",
                    help="leave AttnRes pseudo-queries at their zero init (uniform routing)")
    ap.add_argument("--anamnesis-root", type=Path, default=home / "anamnesis-worktrees/main-ro")
    ap.add_argument("--registry", type=Path, default=home / "pleroma/profiles/anamnesis_models.json")
    ap.add_argument("--preset", default="tsumugi-1b-release-s52450")
    ap.add_argument("--out-dir", type=Path, default=home / "anamnesis_exps/outputs/analysis")
    ap.add_argument("--tag", default="feature_support_tsumugi_2026-10-28")
    args = ap.parse_args()

    g = {} if args.sweep_only else guards(SourceIndex(args.anamnesis_root))
    suffix = ""
    raw_path = args.out_dir / f"{args.tag}{suffix}.values.npz"
    if args.reanalyse:
        sweep, meta = load_raw(raw_path)
    else:
        sweep, meta = run_sweep(args)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        save_raw(raw_path, sweep, meta)
        if args.sweep_only:
            print(f"wrote {raw_path}")
            return
    from anamnesis.extraction.fast.schema import resolve_gpu_schema
    preset, extraction, families = load_preset(args.registry, args.preset)
    cfg = build_model_cfg(args.scale)
    comps = {l: np.zeros((extraction.pca_components, cfg.hidden_size), np.float32) for l in extraction.pca_layers}
    schema = resolve_gpu_schema(cfg.num_layers, 8, extraction, families, comps)
    table, curve = analyse(sweep, meta, g, schema.family_slices, list(schema.feature_names))
    write_outputs(table, curve, meta, args.out_dir / f"{args.tag}{suffix}.tsv",
                  args.out_dir / f"{args.tag}{suffix}.md", args.out_dir / f"{args.tag}{suffix}.raw.json",
                  sorted(sweep.names))
    print(f"wrote {args.out_dir / (args.tag + suffix)}.{{tsv,md,raw.json}}")


if __name__ == "__main__":
    main()
