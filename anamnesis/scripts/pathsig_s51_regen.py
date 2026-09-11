"""LEG 2 (GPU) — regenerate the §5.1 resolver cell's per-position residuals, run the PARITY
GATE, and take the three-way read (hand summary · path-signature · raw-linear).

Brief: ``research/planning/SPEC-path-signature-family-2026-09-11.md`` §3 E2 + §4b; data
inventory ``DIG-residual-tensors-2026-09-11.md`` §1. **Exploratory; nothing here is quotable
pre-first-read (C§8).**

WHY A REGENERATION AT ALL
-------------------------
The resolver cell's raw per-position residuals lived only in ``/dev/shm`` during the
2026-07-14 replay and were discarded once the hand features were computed. Everything needed
to re-derive them is banked: the exact token ids (``vmb_stage0_3b/replay_manifest.json``) and
the exact steering vector + dose (recovered verbatim from the cell's own replay logs —
``V2_L13``, layer 13, ``alpha=0.3276513576507568``, ``alpha_frac=0.03``). Teacher-forced
replay of 160 gens × 2 arms reproduces the states deterministically from the token ids.

THE PARITY GATE — A HARD STOP
-----------------------------
Before any new number is computed, the regenerated states must reproduce the two banked
numbers on the banked split (``arms/A5/s51_resolver_3b.json``):

    hand_means.logit.test  = 0.5781      (banked signatures — a SPLIT check: it moves only if
                                          the matched gid set / covariates changed)
    raw_linear.logit.test  = 0.7844      (regenerated states — the real data check)

If either misses, the script reports the delta and **stops without emitting the three-way
read**: a comparison against states that are not the same data would be void. ``--force``
exists only so the desk can rule a near-miss through; it is never the enactor's call.

WHAT IS REPRODUCED, EXACTLY
---------------------------
* the same ``load_model`` capture surface as ``run_replay_extraction`` (eager attention,
  all-layer k/v/q/o hooks, gate hooks on sampled layers) and the same ``replay_extract``;
* the same ``hidden_states`` float16 cast the discarded raw npz applied
  (``raw_saver.save_raw_tensors_v3``, ``hidden_dtype='float16'``), so the raw-linear surface
  is bit-for-bit the same function of the states;
* the same ``surface_vector(z, 'residual', sample_positions(T), T)`` = 5 sampled generated
  positions × 29 hidden_states rows × 3072 = 445,440 dims;
* the same eligibility filter the resolver applied (its windowed-no-means arm returned None,
  and therefore dropped the gen, when fewer than W=8 rows survived its ``[plen:T]`` slice);
* the same ``_cv_ladder`` (GroupKFold(5) × 3 seeds, per-fold length-residualisation on
  [prompt_len, gen_len], Gram-reduced, logit + deep) imported from the original script.

Raw npz files are NOT re-banked: at ~510 MB/gen that is ~163 GB for no benefit here. The
needed surfaces are computed in-process and only the 445k raw-linear rows (~143 MB/arm,
node-side, transient) and the tiny ``[T, k]`` projected paths are materialised.

Usage (node1, one GPU, via heimdall)::

    python -m anamnesis.scripts.pathsig_s51_regen \\
      --model 3b --model-path /models/llama-3.2-3b-instruct \\
      --calib-dir /models/anamnesis-extract/calibration/3b \\
      --stage0-run /models/anamnesis-extract/runs/vmb_stage0_3b \\
      --unsteered-run /models/anamnesis-extract/runs/vmb_a5_s51_3b/unsteered \\
      --steered-run /models/anamnesis-extract/runs/vmb_a5_s51_3b/V2_steered_a003 \\
      --inject-npz /models/anamnesis-extract/battery/a5_vectors_3b/a5_vectors.npz \\
      --inject-key V2_L13 --inject-layer 13 --inject-alpha 0.3276513576507568 \\
      --site-layer 14 --k 8 --out-dir ~/luxi-files/anamnesis-pathsig/out/s51
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

F32 = NDArray[np.float32]

# Banked numbers of record — arms/A5/s51_resolver_3b.json (desk-verified 2026-09-07).
BANKED_HAND_LOGIT = 0.5781
BANKED_RAWLIN_LOGIT = 0.7844
W_WINDOWS = 8          # the resolver's windowed arm; only its ELIGIBILITY rule matters here


class RegenConfig(BaseModel):
    """Frozen provenance for the regeneration — written verbatim into the result JSON."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    model_path: str
    calib_dir: str
    stage0_run: str
    unsteered_run: str
    steered_run: str
    inject_npz: str
    inject_key: str
    inject_layer: int
    inject_alpha: float
    inject_alpha_frac: float | None = 0.03
    site_layer: int = Field(description="Residual site for the path signature (3B: L14).")
    k: int = 8
    pca_file: str = "pca_model_corrected.pkl"
    gen_ids: tuple[int, ...] = ()


def _sample_positions(T: int, n: int = 5) -> NDArray[np.int_]:
    """Verbatim ``build_surface_caches.sample_positions`` (imported below; kept as a guard)."""
    if T <= 0:
        return np.zeros(n, dtype=int)
    return np.linspace(0, T - 1, n).round().astype(int)


def _per_layer_pca(calib_dir: Path, pca_file: str, layer: int, k: int) -> tuple[F32, F32]:
    import pickle
    p = calib_dir / pca_file
    with open(p, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"{p}: expected per-layer dict")
    vals = list(obj.values())
    if not (vals and isinstance(vals[0], dict) and "components" in vals[0]):
        raise TypeError(f"{p}: pooled PCA, not per-layer — refusing to substitute")
    keyed = {int(kk): v for kk, v in obj.items()}
    if layer not in keyed:
        raise KeyError(f"{p}: no basis for layer {layer} (have {sorted(keyed)})")
    comp = np.asarray(keyed[layer]["components"], dtype=np.float32)
    mean = np.asarray(keyed[layer]["mean"], dtype=np.float32)
    if k > comp.shape[0]:
        raise ValueError(f"k={k} > basis rank {comp.shape[0]}")
    return np.ascontiguousarray(comp[:k]), mean


def _collect_arm(loaded, entries: dict[str, Any], gen_ids: list[int], pos_means: F32,
                 write_handle, arm_label: str, site_layer: int,
                 pca_comp: F32, pca_mean: F32,
                 hand_run: Path) -> dict[str, Any]:
    """Replay one arm; return per-gen raw-linear vectors, site paths, hand vectors, T."""
    from anamnesis.analysis.v3_audit.build_surface_caches import sample_positions
    from anamnesis.extraction.replay_extract import replay_extract
    from anamnesis.scripts.vmb_s51_encoder_on_raw import _hand_vec

    rawlin: list[F32] = []
    paths: list[F32] = []
    hands: list[F32] = []
    gids: list[int] = []
    Ts: list[int] = []
    plens: list[int] = []
    skipped: list[dict[str, Any]] = []

    arr_idx = site_layer + 1          # hidden_states row (index 0 = embedding)
    max_pos = int(pos_means.shape[1])
    pm_layer = np.ascontiguousarray(pos_means[arr_idx])

    t0 = time.time()
    for i, gid in enumerate(gen_ids):
        key = str(gid)
        if key not in entries:
            skipped.append({"gen_id": gid, "reason": "absent from replay manifest"})
            continue
        try:
            e = entries[key]
            input_ids = e["input_ids"]
            plen = int(e["prompt_length"])
            if write_handle is not None:
                write_handle.spec.start_pos = plen
                write_handle.reset_stats()
            raw = replay_extract(loaded, input_ids, plen, positional_means=None)
            if write_handle is not None:
                st = write_handle.stats
                expected = len(input_ids) - plen
                got = int(st.get("positions", 0))
                if not st.get("saw_cache_position", False) or got != expected:
                    raise RuntimeError(
                        f"gen_{gid:03d}: replay injection gating broken "
                        f"(saw_cache_position={st.get('saw_cache_position')}, "
                        f"positions={got}, expected={expected})")

            # float16 cast EXACTLY as save_raw_tensors_v3 did before the surface was read
            hs16 = np.stack([h.astype(np.float16) for h in raw.hidden_states])  # [T, L+1, d]
            T = int(hs16.shape[0])
            if T <= 0:
                skipped.append({"gen_id": gid, "reason": "T=0"})
                del raw
                continue
            # the resolver's own eligibility rule (its windowed arm returned None below W)
            lo = min(max(plen, 0), T - 1)
            if (T - lo) < W_WINDOWS:
                skipped.append({"gen_id": gid,
                                "reason": f"resolver eligibility: T-plen={T-lo} < W={W_WINDOWS}"})
                del raw
                continue
            hv = _hand_vec(hand_run, gid)
            if hv is None:
                skipped.append({"gen_id": gid, "reason": f"no banked hand signature in {hand_run}"})
                del raw
                continue

            pos = sample_positions(T)
            assert np.array_equal(pos, _sample_positions(T)), "sample_positions drift"
            rl = hs16[pos].reshape(-1).astype(np.float16)

            X = hs16[:, arr_idx, :].astype(np.float32)
            abs_pos = np.minimum(plen + np.arange(T), max_pos - 1)
            Xc = X - pm_layer[abs_pos]
            proj = ((Xc - pca_mean) @ pca_comp.T).astype(np.float32)
            if not np.all(np.isfinite(proj)):
                raise RuntimeError("non-finite projection")

            rawlin.append(rl)
            paths.append(proj)
            hands.append(hv.astype(np.float32))
            gids.append(gid)
            Ts.append(T)
            plens.append(plen)
            del raw, hs16
        except Exception as exc:  # noqa: BLE001 — report, never silently drop a whole arm
            skipped.append({"gen_id": gid, "reason": f"{type(exc).__name__}: {exc}"})
            logger.error(f"[{arm_label}] gen_{gid:03d} FAILED: {exc}", exc_info=True)
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            logger.info(f"[{arm_label}] {i+1}/{len(gen_ids)} in {el:.0f}s "
                        f"(ETA {(len(gen_ids)-i-1)/max((i+1)/el,1e-9):.0f}s)")
            gc.collect()
    logger.info(f"[{arm_label}] collected {len(gids)} gens, {len(skipped)} skipped, "
                f"{time.time()-t0:.0f}s")
    return {"rawlin": rawlin, "paths": paths, "hand": hands, "gids": gids,
            "T": Ts, "plen": plens, "skipped": skipped}


def main() -> int:  # noqa: C901 — a linear pipeline, read top to bottom
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="3b")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--calib-dir", type=Path, required=True)
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--unsteered-run", type=Path, required=True)
    ap.add_argument("--steered-run", type=Path, required=True)
    ap.add_argument("--inject-npz", type=Path, required=True)
    ap.add_argument("--inject-key", default="V2_L13")
    ap.add_argument("--inject-layer", type=int, default=13)
    ap.add_argument("--inject-alpha", type=float, required=True)
    ap.add_argument("--inject-alpha-frac", type=float, default=0.03)
    ap.add_argument("--site-layer", type=int, default=14)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--pca-file", default="pca_model_corrected.pkl")
    ap.add_argument("--gen-stride", type=int, default=5,
                    help="the July job replayed every 5th stage-0 gen (0,5,...,795)")
    ap.add_argument("--n-gens", type=int, default=160)
    ap.add_argument("--gate-tol", type=float, default=0.02,
                    help="absolute tolerance on each banked number for the parity gate")
    ap.add_argument("--force", action="store_true",
                    help="DESK-ONLY: proceed past a failed parity gate (never the enactor's call)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from anamnesis.config import ExtractionConfig, MODEL_PRESETS, ModelConfig
    from anamnesis.extraction.model_loader import ResidualWriteSpec, load_model
    from anamnesis.scripts.pathsig_features import PathBank, signature_matrix
    from anamnesis.scripts.vmb_s51_encoder_on_raw import _cv_ladder
    from anamnesis.analysis.v3_audit._common import gen_metadata_by_id

    device = args.device if torch.cuda.is_available() else "cpu"
    gen_ids = [args.gen_stride * i for i in range(args.n_gens)]
    cfg = RegenConfig(
        model=args.model, model_path=args.model_path, calib_dir=str(args.calib_dir),
        stage0_run=str(args.stage0_run), unsteered_run=str(args.unsteered_run),
        steered_run=str(args.steered_run), inject_npz=str(args.inject_npz),
        inject_key=args.inject_key, inject_layer=args.inject_layer,
        inject_alpha=args.inject_alpha, inject_alpha_frac=args.inject_alpha_frac,
        site_layer=args.site_layer, k=args.k, pca_file=args.pca_file,
        gen_ids=tuple(gen_ids),
    )
    logger.info(f"config: {cfg.model_dump_json()}")

    # ── model: the run_replay_extraction v3 capture surface, verbatim ──
    preset = MODEL_PRESETS[args.model]
    all_layers = list(range(preset.num_layers))
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
    _ = ExtractionConfig  # capture-surface parity is via load_model; ec is feature-side only

    pm = np.load(args.calib_dir / "positional_means.npz")["positional_means"].astype(np.float32)
    logger.info(f"positional_means {pm.shape}")
    pca_comp, pca_mean = _per_layer_pca(args.calib_dir, args.pca_file, args.site_layer, args.k)
    logger.info(f"PCA L{args.site_layer}: components {pca_comp.shape}")

    manifest = json.loads((args.stage0_run / "replay_manifest.json").read_text())
    entries = manifest["entries"]
    s0 = gen_metadata_by_id(args.stage0_run / "metadata.json")
    logger.info(f"manifest {len(entries)} entries; stage0 metadata {len(s0)} gens")

    # ── arm 1: unsteered (no injection) ──
    U = _collect_arm(loaded, entries, gen_ids, pm, None, "unsteered", args.site_layer,
                     pca_comp, pca_mean, args.unsteered_run)

    # ── arm 2: V2_L13 @ alpha (the exact spec recovered from the July replay logs) ──
    vec_bank = np.load(args.inject_npz)
    if args.inject_key not in vec_bank:
        raise SystemExit(f"{args.inject_key!r} not in {args.inject_npz}")
    spec = ResidualWriteSpec(
        layer_idx=int(args.inject_layer),
        vector=torch.from_numpy(vec_bank[args.inject_key].astype(np.float32)),
        alpha=float(args.inject_alpha), start_pos=None, normalize=True,
    )
    handle = loaded.add_residual_write(spec)
    logger.info(f"injection active: key={args.inject_key} layer={args.inject_layer} "
                f"alpha={args.inject_alpha} (frac {args.inject_alpha_frac})")
    S = _collect_arm(loaded, entries, gen_ids, pm, handle, "steered_a003", args.site_layer,
                     pca_comp, pca_mean, args.steered_run)
    handle.remove()
    del loaded
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── matched-token pairing, verbatim from vmb_s51_resolver.main ──
    common = sorted(set(S["gids"]) & set(U["gids"]))
    logger.info(f"steered {len(S['gids'])} unsteered {len(U['gids'])} matched {len(common)}")
    if not common:
        raise SystemExit("no matched pairs — nothing to compare")

    def sel(arm: dict[str, Any], key: str) -> list[Any]:
        idx = {g: i for i, g in enumerate(arm["gids"])}
        return [arm[key][idx[g]] for g in common]

    def stack(key: str) -> F32:
        return np.stack(sel(S, key) + sel(U, key)).astype(np.float32)

    topic_c = [s0[g].get("topic_idx", s0[g].get("topic", g)) for g in common] * 2
    ut = {t: i for i, t in enumerate(sorted(set(map(str, topic_c))))}
    topic = np.array([ut[str(t)] for t in topic_c])
    C = np.asarray(
        [[float(s0[g].get("prompt_length", 0) or 0),
          float(s0[g].get("num_generated_tokens", s0[g].get("gen_length", 0)) or 0)]
         for g in common] * 2, dtype=np.float64)
    y = np.array([1] * len(common) + [0] * len(common))

    Xhand = stack("hand")
    Xraw = stack("rawlin")
    logger.info(f"X_hand {Xhand.shape}  X_rawlin {Xraw.shape}  topics={len(set(topic.tolist()))}")

    # ── THE PARITY GATE ──
    logger.info("=== PARITY GATE — regenerated states vs the banked resolver numbers ===")
    hand = _cv_ladder(Xhand, y, topic, C, device, "HAND (means-based sig)")
    raw = _cv_ladder(Xraw, y, topic, C, device, "RAW-LINEAR (5-pos snapshots)")
    d_hand = float(hand["logit"]["test"]) - BANKED_HAND_LOGIT
    d_raw = float(raw["logit"]["test"]) - BANKED_RAWLIN_LOGIT
    gate_pass = abs(d_hand) <= args.gate_tol and abs(d_raw) <= args.gate_tol
    gate = {
        "banked_hand_logit": BANKED_HAND_LOGIT,
        "regen_hand_logit": hand["logit"]["test"],
        "delta_hand": round(d_hand, 4),
        "banked_rawlin_logit": BANKED_RAWLIN_LOGIT,
        "regen_rawlin_logit": raw["logit"]["test"],
        "delta_rawlin": round(d_raw, 4),
        "tolerance": args.gate_tol,
        "n_matched_pairs": len(common),
        "banked_n_matched_pairs": 160,
        "P_hand": int(Xhand.shape[1]), "banked_P_hand": 3358,
        "P_rawlin": int(Xraw.shape[1]), "banked_P_rawlin": 445440,
        "RESULT": "PASS" if gate_pass else "FAIL",
    }
    logger.info(f"PARITY GATE: {json.dumps(gate)}")

    # paths + arm bookkeeping are banked regardless — they are the expensive part
    sig_paths = sel(S, "paths") + sel(U, "paths")
    bank = PathBank.from_list(
        sig_paths, gen_ids=[*common, *common],
        prompt_lengths=[*sel(S, "plen"), *sel(U, "plen")],
        source=f"s51 regen L{args.site_layer} k{args.k}", variant="pc",
    )
    np.savez_compressed(
        args.out_dir / f"s51_paths_L{args.site_layer}_k{args.k}.npz",
        paths_pc=bank.paths, paths_nopc=bank.paths,     # projection already positional-corrected
        offsets=bank.offsets, gen_ids=bank.gen_ids, lengths=bank.lengths,
        prompt_lengths=bank.prompt_lengths,
        y=y, topic=topic, C=C,
    )
    (args.out_dir / "s51_regen_arms.json").write_text(json.dumps({
        "config": cfg.model_dump(),
        "unsteered_skipped": U["skipped"], "steered_skipped": S["skipped"],
        "n_unsteered": len(U["gids"]), "n_steered": len(S["gids"]),
        "n_matched": len(common), "matched_gids": common,
        "T_stats": {"min": int(min(sel(S, "T") + sel(U, "T"))),
                    "max": int(max(sel(S, "T") + sel(U, "T"))),
                    "mean": round(float(np.mean(sel(S, "T") + sel(U, "T"))), 2)},
    }, indent=1))

    result: dict[str, Any] = {
        "cell": "E2 — §5.1 resolver, regenerated (path-signature spec 2026-09-11)",
        "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED, not quotable",
        "config": cfg.model_dump(),
        "parity_gate": gate,
        "ladder": {"hand_means": hand, "raw_linear": raw},
        "chance": 0.5,
    }
    if not gate_pass and not args.force:
        result["ABORTED"] = ("parity gate FAILED — the three-way read was NOT computed. "
                             "The regenerated states are not demonstrably the same data; a "
                             "comparison on them would be void. Desk rules on the delta.")
        (args.out_dir / f"s51_pathsig_e2_{args.model}.json").write_text(json.dumps(result, indent=1))
        logger.error("PARITY GATE FAILED — stopping before the three-way read (see JSON).")
        return 2

    # ── THE THREE-WAY READ (gate passed) ──
    logger.info("=== E2 three-way: hand summary · path-signature · raw-linear ===")
    sig_rows: dict[str, Any] = {}
    for k_ in sorted({4, args.k}):
        for level in (1, 2):
            Xs, names, kept = signature_matrix(bank, layer=args.site_layer, k=k_, level=level)
            if len(kept) != bank.n:
                logger.warning(f"path-signature k={k_} lvl{level}: {bank.n - len(kept)} paths "
                               f"dropped (short) — NOT imputed")
            label = f"PATH-SIG k={k_} lvl1{'+2' if level == 2 else ''} ({Xs.shape[1]}d, aug)"
            if len(kept) != bank.n:
                sig_rows[f"k{k_}_lvl{level}"] = {
                    "ERROR": f"{bank.n - len(kept)} paths dropped — split no longer identical to "
                             f"the hand/raw arms; refusing to report a non-comparable number"}
                continue
            res = _cv_ladder(Xs, y, topic, C, device, label)
            sig_rows[f"k{k_}_lvl{level}"] = {"P": int(Xs.shape[1]), "names_head": names[:3],
                                             **res}

    # ── the increment-permutation null on REAL paths, >=3 seeds ──
    logger.info("=== increment-permutation null on REAL paths (>=3 seeds) ===")
    nulls: dict[str, Any] = {}
    for k_ in sorted({4, args.k}):
        for level in (1, 2):
            per_seed = []
            for seed in (101, 202, 303):
                Xn, _n, kept = signature_matrix(bank, layer=args.site_layer, k=k_, level=level,
                                                permutation_seed=seed)
                if len(kept) != bank.n:
                    per_seed.append({"seed": seed, "ERROR": "path drop"})
                    continue
                r = _cv_ladder(Xn, y, topic, C, device,
                               f"NULL(seed {seed}) k={k_} lvl1{'+2' if level == 2 else ''}")
                per_seed.append({"seed": seed, **r})
            nulls[f"k{k_}_lvl{level}"] = per_seed

    result["path_signature"] = sig_rows
    result["increment_permutation_null"] = nulls
    result["projection"] = {"basis": f"{args.pca_file} per-layer L{args.site_layer}",
                            "positional_corrected": True, "augmentation": "t/(T-1) normalised",
                            "note": "pacing, not duration"}
    (args.out_dir / f"s51_pathsig_e2_{args.model}.json").write_text(json.dumps(result, indent=1))
    logger.info(f"E2 banked (first-read pending) -> "
                f"{args.out_dir / f's51_pathsig_e2_{args.model}.json'}")
    logger.info("PATHSIG_LEG2_COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
