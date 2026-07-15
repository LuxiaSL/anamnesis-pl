"""§5.1 — V2 encoder-on-raw + D1 extraction-ladder rung (WAVE2-A5 runbook item 5;
D1 verdict scope-add 2026-07-14).

THE RULING QUESTION (whole-vector ⊃ stake, WHAT-A-SIGNATURE-IS §4): the A5 verdict
found V2 (SAE steering) puts the STATE at the random floor under HAND-features while
behavior reached 2AFC .775 — the designated encoder-on-raw test case. If a learned
encoder on the RAW state distinguishes V2-steered from unsteered where hand-features
cannot, the whole-vector carries the stake the hand projection misses (hierarchy holds).
If it MISSES too, the V2 counterexample is real → stop-and-surface (framing-doc-changing).

MATCHED-TOKEN by construction: steered and unsteered are the SAME Stage-0 continuations
replayed with / without V2_L13 injection (identical input_ids; only the injection differs).
The classifier must read the injection's imprint on the state, never content.

THE LADDER (D1 rung, same GroupKFold-by-topic split, per-fold length-resid):
  hand-features (banked v3 signature)  vs  raw-linear (logit floor)  vs  raw-encoder (deep).
Reuses the June-15 all-surface machinery verbatim (`preprocess_fold_gpu` Gram-trick reduce +
`train_eval`), so the numbers are directly comparable to the mode-task encoder floors.

CPU/GPU-light (one small card or CPU) — the WINDOW cost was the raw replay; this is post-hoc.
Run (node1):
    python -m anamnesis.scripts.vmb_s51_encoder_on_raw --model 3b \
      --steered-raw /dev/shm/s51_raw_steered --unsteered-raw /dev/shm/s51_raw_unsteered \
      --steered-run <.../V2_steered> --unsteered-run <.../unsteered> \
      --stage0-run <vmb_stage0_3b> --out-dir <arms/A5> --device cuda:0
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from anamnesis.analysis.v3_audit._common import gen_metadata_by_id, train_eval
from anamnesis.analysis.v3_audit.build_surface_caches import sample_positions, surface_vector
from anamnesis.analysis.v3_audit.surface_encoder_floor import preprocess_fold_gpu

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_SURFACES = ["residual", "attention"]   # the two top sources (attention ≫ residual); --surfaces overrides
SEEDS = 3
K = 32
DEEP_EPOCHS = 800
LBFGS_L2 = 1e-3


def _raw_vec(npz_path: Path) -> np.ndarray | None:
    """residual+attention concat for one gen's v3 raw npz (build_surface_caches convention)."""
    try:
        z = np.load(npz_path, allow_pickle=True)
    except Exception:
        return None
    T = int(z["actual_lengths"].shape[0]) if "actual_lengths" in z.files else 0
    if T <= 0:
        return None
    pos = sample_positions(T)
    parts = []
    for s in RAW_SURFACES:
        v = surface_vector(z, s, pos, T)
        if v is None:
            return None
        parts.append(v.astype(np.float32))
    return np.concatenate(parts)


def _hand_vec(run_dir: Path, gid: int) -> np.ndarray | None:
    # the banked hand-feature signature vector lives in the sig NPZ (z["features"], 3358-dim),
    # NOT the JSON (which holds only metadata + tier slice indices).
    p = run_dir / "signatures_v3" / f"gen_{gid:03d}.npz"
    if not p.exists():
        return None
    try:
        z = np.load(p, allow_pickle=True)
    except Exception:
        return None
    return np.asarray(z["features"], dtype=np.float32) if "features" in z.files else None


def _load_arm(raw_dir: Path, run_dir: Path, stage0_meta: dict[int, dict], label: int):
    """Returns lists: (raw_vecs, hand_vecs, topics, C[plen,glen], y) for one arm."""
    raws, hands, topics, C, y, gids = [], [], [], [], [], []
    files = sorted(raw_dir.glob("gen_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    for f in files:
        gid = int(f.stem.split("_")[1])
        md = stage0_meta.get(gid)
        if md is None:
            continue
        rv = _raw_vec(f)
        hv = _hand_vec(run_dir, gid)
        if rv is None or hv is None:
            continue
        topic = md.get("topic_idx", md.get("topic", gid))
        plen = float(md.get("prompt_length", 0) or 0)
        glen = float(md.get("num_generated_tokens", md.get("gen_length", 0)) or 0)
        raws.append(rv); hands.append(hv); topics.append(topic)
        C.append([plen, glen]); y.append(label); gids.append(gid)
    return raws, hands, topics, C, y, gids


def _cv_ladder(X, y, topic, C, device, name):
    """Both archs (logit floor + deep) over GroupKFold(5) × SEEDS; resid=True. Returns dict."""
    accs = {"logit": [], "deep": []}
    for seed in range(SEEDS):
        for tr, te in GroupKFold(5).split(X, y, topic):
            Ztr, Zte = preprocess_fold_gpu(X[tr], X[te], C[tr], C[te], True, device)
            for arch in ("logit", "deep"):
                ta, _ = train_eval(Ztr, y[tr], Zte, y[te], arch, seed, device,
                                   deep_epochs=DEEP_EPOCHS, lbfgs_l2=LBFGS_L2, nclass=2, k=K)
                accs[arch].append(ta)
    out = {a: {"test": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
           for a, v in accs.items()}
    logger.info(f"  {name}: logit {out['logit']['test']:.1%}±{out['logit']['std']:.1%}  "
                f"deep {out['deep']['test']:.1%}±{out['deep']['std']:.1%}  (chance 50%)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steered-raw", type=Path, required=True)
    ap.add_argument("--unsteered-raw", type=Path, required=True)
    ap.add_argument("--steered-run", type=Path, required=True)
    ap.add_argument("--unsteered-run", type=Path, required=True)
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--surfaces", default="residual,attention",
                    help="raw surfaces to concat (residual is fast; attention is slow to resample)")
    ap.add_argument("--dose-frac", type=float, default=0.3,
                    help="the V2_L13 injection frac this run's steered arm used (for the record)")
    args = ap.parse_args()
    global RAW_SURFACES
    RAW_SURFACES = [s.strip() for s in args.surfaces.split(",") if s.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    s0_meta = gen_metadata_by_id(args.stage0_run / "metadata.json")
    logger.info(f"stage0 metadata: {len(s0_meta)} gens")

    sr, sh, st, sc, sy, sg = _load_arm(args.steered_raw, args.steered_run, s0_meta, 1)
    ur, uh, ut, uc, uy, ug = _load_arm(args.unsteered_raw, args.unsteered_run, s0_meta, 0)
    logger.info(f"steered {len(sy)} gens, unsteered {len(uy)} gens")
    # keep only gids present in BOTH arms (matched-token pairing)
    common = sorted(set(sg) & set(ug))
    logger.info(f"matched gids in both arms: {len(common)}")

    def _sel(vecs, gids):
        idx = {g: i for i, g in enumerate(gids)}
        return [vecs[idx[g]] for g in common]

    Xraw = np.stack(_sel(sr, sg) + _sel(ur, ug)).astype(np.float32)
    Xhand = np.stack(_sel(sh, sg) + _sel(uh, ug)).astype(np.float32)
    topic_c = _sel(st, sg) + _sel(ut, ug)
    # map topics to ints for GroupKFold
    utop = {t: i for i, t in enumerate(sorted(set(map(str, topic_c))))}
    topic = np.array([utop[str(t)] for t in topic_c])
    C = np.asarray(_sel(sc, sg) + _sel(uc, ug), dtype=np.float64)
    y = np.array([1] * len(common) + [0] * len(common))
    logger.info(f"X_raw {Xraw.shape}  X_hand {Xhand.shape}  topics={len(set(topic.tolist()))}")

    logger.info("=== §5.1 ladder (V2 steered-vs-unsteered, matched-token, GroupKFold-by-topic) ===")
    raw_label = f"RAW ({'+'.join(RAW_SURFACES)})"
    hand = _cv_ladder(Xhand, y, topic, C, device, "HAND-features (banked v3 sig)")
    raw = _cv_ladder(Xraw, y, topic, C, device, raw_label)

    verdict = ("ENCODER-CATCHES-V2: raw-encoder ≫ hand ≈ chance → whole-vector⊃stake HOLDS"
               if raw["deep"]["test"] - max(hand["deep"]["test"], hand["logit"]["test"]) > 0.08
               and max(hand["logit"]["test"], hand["deep"]["test"]) < 0.62
               else "SEE-NUMBERS: not the clean catch pattern — outer loop rules"
               if raw["deep"]["test"] > 0.62
               else "ENCODER-MISSES-V2: raw at floor too → COUNTEREXAMPLE REAL (stop-and-surface)")

    out = {
        "model": args.model, "cell": "S5.1 V2 encoder-on-raw + D1 ladder rung",
        "STATUS": "FIRST_READ_PENDING (C§8) — ⛔RULING feeder (whole-vector⊃stake)",
        "n_matched_pairs": len(common), "n_topics": len(set(topic.tolist())),
        "P_raw": int(Xraw.shape[1]), "P_hand": int(Xhand.shape[1]),
        "vector": "V2_L13", "dose_frac": args.dose_frac, "surfaces_raw": RAW_SURFACES,
        "ladder": {
            "hand_features": hand,     # the projection that put V2 at floor
            "raw_linear_and_encoder": raw,
        },
        "chance": 0.5,
        "verdict_heuristic": verdict,
        "notes": ["matched-token: same Stage-0 continuations, ±V2_L13 injection at frac 0.3;",
                  "raw = residual+attention Gram-reduced per fold (June-15 machinery, verbatim);",
                  "D1 rung: hand vs raw-linear vs raw-encoder on the identical split."],
    }
    p = args.out_dir / f"s51_encoder_on_raw_{args.model}.json"
    p.write_text(json.dumps(out, indent=1))
    logger.info(f"VERDICT: {verdict}")
    logger.info(f"banked (first-read pending) -> {p}")


if __name__ == "__main__":
    main()
