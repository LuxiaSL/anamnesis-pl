"""ARM A4b — the FAITHFUL exp11 extension (ARM-A4b-faithful-substrate-2026-07-13).

The executed A4 ran a substrate the ratified block accidentally invented (short bare
Stage-0 gens, mid-block eviction). A4b runs exp11's ACTUAL substrate at battery grain:
  - Regime A (dialogue, 20 cells): 3B-native multi-turn conversations
    (kv-rotation native3b_convs + boost), turn-aligned OLDEST-FIRST eviction
    (protect system + last 2 turns + 4 sinks).
  - Regime B (docs, 4 cells): eval_docs (NOT model-generated — the provenance
    contrast inside A4b), contiguous-EARLY-block eviction.

Per cell:
  context  = rendered dialogue (system + alt user/asst, +generation prompt) OR doc text
  cont     = N=256 frozen GREEDY rollout from FULL (exp11 protocol; teacher-forced
             identically across all conditions)
  FULL     = unmodified-cache bridge replay (reference; same code path → deltas pure)
  NAIVE_f  = evict oldest turns / early block, NO re-rotation, survivors keep positions
  ROTATE_f = evict + exact re-rotation to compacted positions 0..S'-1
  REC_f    = fresh prefill of the shortened (kept) text

Fractions f ∈ {0.0625,0.125,0.25,0.5}; target_tokens = f·C. Where protections make a
fraction UNREACHABLE, the condition is reported unreachable (a5b_unreachable=true) and
its signature is NOT written — never silently clipped (ARM-A4b §Cells).

Signatures land in <out-run>/signatures_v3/<condition>/gen_XXX.json with the dissociation
columns (tf_nll, token-KL vs FULL, top-1 agreement). Instrument rider: first cell runs
FULL twice → asserts bitwise-equal features (12b faithfulness-floor-is-zero).

Cross-model A4b requires per-model native dialogue generation (exp11 "Regime A must be
3B-generated") — 3B-LOCKED here; Regime B docs are the only model-shareable cells.

Launcher mode (--launch) partitions CELLS across (gpus × workers-per-gpu) subprocesses.
Attention at ~5k tokens is memory-heavy per replay → 2 workers/GPU (ARM-A4b §Ops).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from anamnesis.config import MODEL_PRESETS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FRACTIONS = [0.0625, 0.125, 0.25, 0.5]
KINDS = ["naive", "rotate", "recompute"]
NUM_SINKS = 4
PROTECT_LAST_TURNS = 2
DOC_RECENT_PROTECT = 128       # docs have no turns; protect a recent tail (tokens)
N_ROLLOUT = 256
MIN_GEN_TOKENS = 24
ROLE_FOR_SPEAKER = {"deepseek": "user", "trinity": "assistant"}


def conditions(kinds: list[str] | None = None) -> list[str]:
    ks = kinds if kinds is not None else KINDS
    return ["full"] + [f"{k}_f{f}" for k in ks for f in FRACTIONS]  # 'full' = always the delta reference


def _log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    s = x - m
    return s - np.log(np.exp(s).sum(axis=-1, keepdims=True))


def _dissoc_columns(cond_logits: list[np.ndarray], full_lp: np.ndarray | None,
                    chosen: np.ndarray) -> tuple[dict, np.ndarray]:
    lg = np.stack(cond_logits).astype(np.float32)
    lp = _log_softmax(lg)
    idx = chosen.astype(np.int64)
    nll = -lp[np.arange(len(idx)), idx]
    out = {"tf_nll_mean": float(nll.mean()), "tf_nll_sum": float(nll.sum())}
    if full_lp is not None:
        kl = (np.exp(full_lp) * (full_lp - lp)).sum(axis=-1)  # KL(FULL || cond) per step
        out["token_kl_vs_full_mean"] = float(kl.mean())
        out["token_kl_vs_full_max"] = float(kl.max())
        out["top1_agree_vs_full"] = float((lp.argmax(-1) == full_lp.argmax(-1)).mean())
    return out, lp


# ── substrate loaders ────────────────────────────────────────────────────────────

def _messages_for_trinity(system_prompt: str, turns: list[dict]) -> list[dict]:
    """system + alternating user/assistant (deepseek→user, trinity→assistant), strict
    alternation from deepseek — the exp11 `messages_for_trinity` render (ported)."""
    msgs = [{"role": "system", "content": system_prompt}]
    for i, t in enumerate(turns):
        expected = "deepseek" if i % 2 == 0 else "trinity"
        if t.get("speaker") != expected:
            raise ValueError(f"turn {i} speaker {t.get('speaker')!r} != expected {expected!r}")
        msgs.append({"role": ROLE_FOR_SPEAKER[t["speaker"]], "content": t["text"].strip()})
    return msgs


def load_cells(dialogue_paths: list[Path], doc_path: Path | None,
               n_docs: int) -> list[dict]:
    """Unified cell list: dialogue cells (with messages) + doc cells (with text)."""
    cells: list[dict] = []
    for p in dialogue_paths:
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cells.append({"cell_id": rec["conv_id"], "regime": "dialogue",
                          "messages": _messages_for_trinity(rec["system_prompt"], rec["turns"]),
                          "topic": rec.get("topic", "")})
    if doc_path is not None and n_docs > 0:
        docs = [json.loads(l) for l in doc_path.read_text().splitlines() if l.strip()]
        for i, d in enumerate(docs[:n_docs]):
            cells.append({"cell_id": f"doc-{i:02d}", "regime": "doc", "text": d["text"],
                          "topic": d.get("topic", f"doc{i}")})
    return cells


def _early_region_keep(context_len: int, evict_frac: int, *, num_sinks: int,
                       recent_protect: int):
    """Docs: evict ONE contiguous EARLY block of round(frac·C) tokens starting right
    after the sinks; sinks + recent tail always kept. Returns (keep_tensor|None); None
    = fraction UNREACHABLE (evictable region too small)."""
    import torch
    n_evict = evict_frac
    lo, hi = num_sinks, context_len - recent_protect
    if n_evict <= 0:
        return torch.arange(context_len, dtype=torch.long)
    if hi - lo < n_evict:
        return None
    evicted = set(range(lo, lo + n_evict))
    keep = [i for i in range(context_len) if i not in evicted]
    return torch.tensor(keep, dtype=torch.long)


def run_worker(args) -> None:
    import torch

    from anamnesis.config import ExtractionConfig, FeaturePipelineConfig, ModelConfig
    from anamnesis.extraction.cache_surgery import (
        assert_rotation_homomorphism,
        evict,
        from_hf_cache,
        operative_inv_freq,
        oldest_turns_to_evict,
        reindex,
        to_hf_dynamic_cache,
        turn_keep_indices,
        turn_token_spans,
    )
    from anamnesis.extraction.feature_pipeline import compute_features_v2_from_data, save_features
    from anamnesis.extraction.model_loader import load_model
    from anamnesis.extraction.replay_cached import replay_extract_cached

    preset = MODEL_PRESETS[args.model]
    all_layers = list(range(preset.num_layers))
    extraction_config = ExtractionConfig(
        sampled_layers=preset.sampled_layers, pca_layers=preset.pca_layers,
        early_layer_cutoff=preset.early_layer_cutoff, late_layer_cutoff=preset.late_layer_cutoff,
        enable_tier3=True,
    )
    family_config = FeaturePipelineConfig(
        include_baseline_tiers=True, enable_residual_trajectory=True,
        enable_attention_flow=True, enable_gate_features=True,
        enable_temporal_dynamics=False, enable_per_head=True, enable_stft=True,
        enable_contrastive_projection=False, enable_value_geometry=True,
        enable_qk_geometry=True, enable_kv_cka=True,
        trajectory_layers=preset.trajectory_layers, contrastive_layers=preset.contrastive_layers,
    )
    model_config = ModelConfig(
        model_id=args.model_path, torch_dtype=preset.torch_dtype,
        num_layers=preset.num_layers, hidden_dim=preset.hidden_dim,
        num_attention_heads=preset.num_attention_heads, num_kv_heads=preset.num_kv_heads,
        head_dim=preset.head_dim,
    )
    loaded = load_model(
        model_config, sampled_layers=preset.sampled_layers, register_gate_hooks=True,
        key_layers=all_layers, value_layers=all_layers,
        query_layers=all_layers, attn_output_layers=all_layers,
    )
    device = next(loaded.model.parameters()).device
    tokenizer = loaded.tokenizer
    inv_freq = operative_inv_freq(loaded.model).to(device)  # 14e: LIVE-buffer value gate + operative
    assert_rotation_homomorphism(inv_freq)  # smoke 1: RoPE homomorphism, fails loud at load
    logger.info(f"[{args.label}] RoPE value+homomorphism gates PASS (14e; live buffer operative)")

    pm_path = args.calib_dir / "positional_means.npz"
    positional_means = None
    if pm_path.exists():
        positional_means = np.load(pm_path)["positional_means"].astype(np.float32)
    else:
        logger.warning(f"NO positional_means at {pm_path}")
    pca_components = pca_mean = None
    pca_path = args.calib_dir / "pca_model.pkl"
    if pca_path.exists():
        import pickle
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
        if isinstance(pca, dict):
            pca_components = np.asarray(pca["components"], dtype=np.float32)
            pca_mean = np.asarray(pca["mean"], dtype=np.float32)
        else:
            pca_components = np.asarray(pca.components_, dtype=np.float32)
            pca_mean = np.asarray(pca.mean_, dtype=np.float32)

    all_cells = load_cells(
        [Path(p) for p in args.dialogue], Path(args.docs) if args.docs else None, args.n_docs)
    cells = [all_cells[i] for i in args.cell_ids]

    conds = conditions([k for k in args.kinds.split(",") if k])
    sig_root = args.out_run_dir / "signatures_v3"
    for c in conds:
        (sig_root / c).mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def prefill(ids_list: list[int]):
        ids = torch.tensor([ids_list], dtype=torch.long, device=device)
        loaded.disable_hooks()
        out = loaded.model(ids, use_cache=True, return_dict=True)
        loaded.enable_hooks()
        loaded.clear_hook_state()
        return out.past_key_values

    @torch.no_grad()
    def greedy_rollout(context_ids: list[int]) -> list[int]:
        ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        loaded.disable_hooks()
        out = loaded.model.generate(
            ids, max_new_tokens=N_ROLLOUT, do_sample=False, num_beams=1,
            pad_token_id=tokenizer.eos_token_id, use_cache=True)
        loaded.enable_hooks()
        loaded.clear_hook_state()
        return out[0, len(context_ids):].tolist()

    rider_done = False
    n_done = n_fail = 0
    t0 = time.time()
    for ci, cell in enumerate(cells):
        cid = cell["cell_id"]
        gid = args.cell_ids[ci]
        try:
            # ── build FULL context + per-turn spans (dialogue) or plain ids (doc) ──
            if cell["regime"] == "dialogue":
                messages = cell["messages"]
                if args.truncate_context_tokens:
                    # F1-MID rung (factorial cell (i), 14q item 1): TURN-ALIGNED truncation of
                    # the banked dialogue to ~N context tokens ("truncated-dialogue", the
                    # spec's sanctioned mid-rung route). Keep a prefix of whole messages,
                    # ending on a USER turn so the generation prompt stays well-formed.
                    _, full_spans = turn_token_spans(tokenizer, messages,
                                                     add_generation_prompt=False)
                    cut = None
                    total = 0
                    for sp in full_spans:
                        total += sp.n_tokens
                        if total > args.truncate_context_tokens:
                            break
                        if messages[sp.index]["role"] == "user":
                            cut = sp.index
                    if cut is None or cut < 2:
                        logger.warning(f"[{args.label}] {cid} EXCLUDED: cannot truncate to "
                                       f"{args.truncate_context_tokens} tokens on a user turn")
                        continue
                    messages = messages[: cut + 1]
                ctx_t, spans = turn_token_spans(tokenizer, messages,
                                                add_generation_prompt=True)
                context_ids = ctx_t[0].tolist()
            else:
                context_ids = tokenizer(cell["text"], add_special_tokens=True)["input_ids"]
                if len(context_ids) > args.doc_max_tokens:
                    context_ids = context_ids[: args.doc_max_tokens]
                spans = None
            C = len(context_ids)

            cont_ids = greedy_rollout(context_ids)
            if len(cont_ids) < MIN_GEN_TOKENS + 1:
                logger.warning(f"[{args.label}] {cid} EXCLUDED: rollout only {len(cont_ids)} tokens")
                continue
            chosen = np.asarray(cont_ids[1:], dtype=np.int64)

            base_meta = {"generation_id": gid, "a4b_cell_id": cid,
                         "a4b_regime": cell["regime"], "a4b_topic": cell.get("topic", ""),
                         "a4b_context_len": C, "a4b_cont_len": len(cont_ids),
                         "a4b_num_sinks": NUM_SINKS, "a4b_n_turns": (len(spans) if spans else 0)}
            if args.truncate_context_tokens:
                base_meta["a4b_truncated_to"] = int(args.truncate_context_tokens)   # F1-mid rung
            if args.rope_fix_tag:
                base_meta["rope_fix"] = args.rope_fix_tag   # 14e provenance stamp

            base_cache = prefill(context_ids)
            snapshot = from_hf_cache(base_cache, positions=torch.arange(C, device=device))
            full_lp = None

            for cond in conds:
                sig_path = sig_root / cond / f"gen_{gid:03d}.json"
                if sig_path.exists() and cond != "full":
                    continue
                unreachable = False
                if cond == "full":
                    snap = snapshot
                    offset = C
                    surg_meta = {"kind": "full", "evict_frac": 0.0, "n_evicted": 0}
                else:
                    kind, ftag = cond.split("_f")
                    f = float(ftag)
                    target = int(round(f * C))
                    if cell["regime"] == "dialogue":
                        evict_turns = oldest_turns_to_evict(
                            spans, target_tokens=target, protect_roles=("system",),
                            protect_last=PROTECT_LAST_TURNS)
                        freed = sum(sp.n_tokens for sp in spans if sp.index in set(evict_turns))
                        if not evict_turns or freed < target * args.reach_tol:
                            unreachable = True
                        else:
                            keep = turn_keep_indices(spans, evict_turns, C,
                                                     num_sink_tokens=NUM_SINKS).to(device)
                    else:
                        keep_t = _early_region_keep(C, target, num_sinks=NUM_SINKS,
                                                    recent_protect=DOC_RECENT_PROTECT)
                        if keep_t is None:
                            unreachable = True
                        else:
                            keep = keep_t.to(device)
                    if unreachable:
                        meta = dict(base_meta)
                        meta.update({"a4b_condition": cond, "a4b_kind": kind,
                                     "a4b_evict_frac": f, "a4b_unreachable": True})
                        (sig_root / cond / f"gen_{gid:03d}.UNREACHABLE.json").write_text(
                            json.dumps(meta, indent=1))
                        logger.info(f"[{args.label}] {cid} {cond}: UNREACHABLE (target={target})")
                        continue
                    n_evicted = C - int(keep.shape[0])
                    if kind == "naive":
                        snap = evict(snapshot, keep)
                        offset = C
                    elif kind == "rotate":
                        snap = evict(snapshot, keep)
                        s_prime = int(keep.shape[0])
                        snap = reindex(snap, torch.arange(s_prime, device=device), inv_freq)
                        offset = s_prime
                    elif kind == "recompute":
                        keep_cpu = keep.cpu().tolist()
                        short_ids = [context_ids[j] for j in keep_cpu]
                        cache = prefill(short_ids)
                        snap = from_hf_cache(cache, positions=torch.arange(len(short_ids), device=device))
                        offset = len(short_ids)
                    else:
                        raise ValueError(kind)
                    surg_meta = {"kind": kind, "evict_frac": f, "n_evicted": n_evicted}

                raw = replay_extract_cached(
                    loaded, to_hf_dynamic_cache(snap), cont_ids, offset,
                    positional_means=positional_means)
                dis, lp = _dissoc_columns(raw.logits, full_lp, chosen)
                if cond == "full":
                    full_lp = lp
                    dis["token_kl_vs_full_mean"] = 0.0
                    dis["token_kl_vs_full_max"] = 0.0
                    dis["top1_agree_vs_full"] = 1.0

                result = compute_features_v2_from_data(
                    raw, extraction_config, family_config, pca_components, pca_mean)

                if cond == "full" and not rider_done:
                    raw2 = replay_extract_cached(
                        loaded, to_hf_dynamic_cache(snap), cont_ids, offset,
                        positional_means=positional_means)
                    r2 = compute_features_v2_from_data(
                        raw2, extraction_config, family_config, pca_components, pca_mean)
                    if not np.array_equal(np.asarray(result.features), np.asarray(r2.features)):
                        raise RuntimeError("A4b rider FAILED: FULL bridge replay not bitwise-deterministic")
                    logger.info(f"[{args.label}] rider OK: FULL bridge bitwise-deterministic")
                    rider_done = True

                metadata = dict(base_meta)
                metadata["a4b_condition"] = cond
                metadata.update({f"a4b_{k}": v for k, v in surg_meta.items()})
                metadata["a4b_position_offset"] = offset
                metadata["a4b_cache_len_after"] = snap.seq_len()
                metadata["dissociation"] = dis
                metadata["num_features"] = int(len(result.features))
                metadata["extraction_version"] = 3
                save_features(gid, result, metadata, sig_root / cond)

            n_done += 1
            el = time.time() - t0
            logger.info(f"[{args.label}] {ci+1}/{len(cells)} {cid} (C={C}, "
                        f"cont={len(cont_ids)}) done in {el:.0f}s")
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            logger.error(f"[{args.label}] {cid} FAILED: {exc}", exc_info=True)

    logger.info(f"[{args.label}] done: {n_done} ok, {n_fail} failed in {time.time()-t0:.0f}s")
    if n_fail:
        sys.exit(1)


def run_launcher(args) -> None:
    from anamnesis.scripts._gpu import resolve_physical_gpus

    gpu_ids = resolve_physical_gpus([g.strip() for g in args.gpus.split(",") if g.strip() != ""])
    all_cells = load_cells(
        [Path(p) for p in args.dialogue], Path(args.docs) if args.docs else None, args.n_docs)
    n_cells = len(all_cells)
    n_workers = min(len(gpu_ids) * args.workers_per_gpu, n_cells)
    logger.info(f"A4b: {n_cells} cells across {n_workers} workers ({len(gpu_ids)} GPUs)")

    worker_cells: list[list[int]] = [[] for _ in range(n_workers)]
    for i in range(n_cells):
        worker_cells[i % n_workers].append(i)

    log_dir = args.out_run_dir / "a4b_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for w, ids in enumerate(worker_cells):
        if not ids:
            continue
        gpu = gpu_ids[w % len(gpu_ids)]
        cmd = [sys.executable, "-m", "anamnesis.scripts.vmb_a4b_surgery_replay",
               "--model", args.model, "--model-path", args.model_path,
               "--calib-dir", str(args.calib_dir), "--out-run-dir", str(args.out_run_dir),
               "--dialogue", *args.dialogue, "--docs", str(args.docs),
               "--n-docs", str(args.n_docs), "--doc-max-tokens", str(args.doc_max_tokens),
               "--label", f"w{w}g{gpu}", "--kinds", args.kinds,
               "--rope-fix-tag", args.rope_fix_tag, "--cell-ids", *[str(g) for g in ids]]
        if args.truncate_context_tokens:
            cmd += ["--truncate-context-tokens", str(args.truncate_context_tokens)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu,
               "PYTHONPATH": os.environ.get("PYTHONPATH", "."),
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        lf = open(log_dir / f"worker_{w}.log", "w")
        procs.append((w, subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT), lf))
        logger.info(f"worker {w} (GPU {gpu}): cells {ids}")

    fails = 0
    for w, p, lf in procs:
        rc = p.wait()
        lf.close()
        if rc != 0:
            fails += 1
            logger.error(f"worker {w} exited rc={rc}")
    logger.info(f"A4b launcher done: {len(procs) - fails}/{len(procs)} workers clean")
    sys.exit(1 if fails else 0)


def main() -> None:
    ap = argparse.ArgumentParser(description="ARM A4b faithful cache-surgery replay")
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--out-run-dir", type=Path, required=True)
    ap.add_argument("--dialogue", nargs="+", required=True, help="native3b conv jsonl path(s)")
    ap.add_argument("--docs", default=None, help="eval_docs jsonl (Regime B)")
    ap.add_argument("--n-docs", type=int, default=4)
    ap.add_argument("--doc-max-tokens", type=int, default=4608)
    ap.add_argument("--truncate-context-tokens", type=int, default=None,
                    help="F1-MID rung (factorial cell (i)): turn-aligned truncation of each "
                         "dialogue to ~N context tokens (prefix of whole messages, ends on a "
                         "user turn). ~1200 = the spec's mid rung. Dialogue regime only.")
    ap.add_argument("--reach-tol", type=float, default=0.9,
                    help="dialogue fraction counts as reached if freed >= reach_tol*target")
    ap.add_argument("--cell-ids", type=int, nargs="+", help="worker: cell indices to process")
    ap.add_argument("--kinds", default="naive,rotate,recompute",
                    help="14e: comma-sep surgery kinds to run ('full' always included as delta reference)")
    ap.add_argument("--rope-fix-tag", default="",
                    help="14e: provenance stamp written into every gen's metadata (e.g. '14e')")
    ap.add_argument("--label", default="w")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--workers-per-gpu", type=int, default=2)
    args = ap.parse_args()

    if args.launch:
        run_launcher(args)
    else:
        if args.cell_ids is None:
            all_cells = load_cells([Path(p) for p in args.dialogue],
                                   Path(args.docs) if args.docs else None, args.n_docs)
            args.cell_ids = list(range(len(all_cells)))
        run_worker(args)


if __name__ == "__main__":
    main()
