"""§5.1 RESOLVER — coverage vs aggregation-granularity (session-5 item 7; WAIS §3 decider).

§5.1 found, on the α=.03 V2-steered-vs-unsteered residual surface (matched-token, identical
GroupKFold-by-topic split): HAND-features (means-based engineered signature) 57.8% linear vs
RAW-LINEAR (5 sampled position snapshots of the residual) 78.4%. A ~20pp gap on the SAME surface,
both linear. This resolver asks WHY:

  - If a WINDOWED NO-MEANS aggregate of the residual (finer than hand's means, still an aggregate
    not raw snapshots) reaches raw-linear → the gap is AGGREGATION-GRANULARITY (hand's means were
    too coarse; the residual surface carries the stake, you just need no-means windowed reads).
    ⇒ rewrites the WAIS §3 sentence — STOP-AND-SURFACE.
  - If the windowed aggregate stays at hand (<< raw-linear) → the gap is SURFACE-COVERAGE (no
    aggregate recovers it; raw needs the full per-position surface). WAIS §3 stands.

Windowed-no-means feature: gen positions [plen:T] split into W windows; per window/layer/dim
the STD (dispersion) and SLOPE (linear drift) — NO plain mean (the discipline law). Same residual
raw as raw-linear, same 160-pair split, same ladder (logit + deep, GroupKFold-by-topic, len-resid).
Reuses /dev/shm/s51_raw_{steered_a003,unsteered}. First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from anamnesis.analysis.v3_audit._common import gen_metadata_by_id
from anamnesis.analysis.v3_audit.build_surface_caches import sample_positions, surface_vector
from anamnesis.scripts.vmb_s51_encoder_on_raw import _cv_ladder, _hand_vec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

W_WINDOWS = 8


def _windowed_nomeans(z, T: int) -> np.ndarray | None:
    """Residual (T, L+1, hidden) → per-window STD + SLOPE over GEN positions, no means.
    Feature = concat over {std, slope} × W windows × (L+1) × hidden."""
    if "hidden_states" not in z.files:
        return None
    hs = z["hidden_states"]                         # (T, L+1, hidden) f16
    if hs.size == 0 or hs.shape[0] < T:
        return None
    plen = int(z["prompt_length"]) if "prompt_length" in z.files else 0
    lo = min(max(plen, 0), T - 1)
    gen = hs[lo:T].astype(np.float32)               # (G, L+1, hidden) generated span
    G = gen.shape[0]
    if G < W_WINDOWS:
        return None
    edges = np.linspace(0, G, W_WINDOWS + 1).round().astype(int)
    stds, slopes = [], []
    for w in range(W_WINDOWS):
        a, b = edges[w], max(edges[w + 1], edges[w] + 1)
        seg = gen[a:b]                              # (g, L+1, hidden)
        stds.append(seg.std(axis=0))               # no-means dispersion
        g = seg.shape[0]
        if g >= 2:
            t = np.arange(g, dtype=np.float32)
            t -= t.mean()
            denom = float((t * t).sum())
            slope = np.tensordot(t, seg, axes=([0], [0])) / denom   # (L+1, hidden) OLS slope
        else:
            slope = np.zeros_like(seg[0])
        slopes.append(slope)
    return np.concatenate([np.stack(stds).reshape(-1),
                           np.stack(slopes).reshape(-1)]).astype(np.float16)


def _load_arm(raw_dir: Path, run_dir: Path, s0_meta, label: int):
    rawlin, wnm, hand, topics, C, gids = [], [], [], [], [], []
    for f in sorted(raw_dir.glob("gen_*.npz"), key=lambda p: int(p.stem.split("_")[1])):
        gid = int(f.stem.split("_")[1])
        md = s0_meta.get(gid)
        if md is None:
            continue
        try:
            z = np.load(f, allow_pickle=True)
        except Exception:
            continue
        T = int(z["actual_lengths"].shape[0]) if "actual_lengths" in z.files else 0
        if T <= 0:
            continue
        rl = surface_vector(z, "residual", sample_positions(T), T)
        wv = _windowed_nomeans(z, T)
        hv = _hand_vec(run_dir, gid)
        if rl is None or wv is None or hv is None:
            continue
        rawlin.append(rl); wnm.append(wv); hand.append(hv)
        topics.append(md.get("topic_idx", md.get("topic", gid)))
        C.append([float(md.get("prompt_length", 0) or 0),
                  float(md.get("num_generated_tokens", md.get("gen_length", 0)) or 0)])
        gids.append(gid)
    return dict(rawlin=rawlin, wnm=wnm, hand=hand, topics=topics, C=C, gids=gids, y=[label] * len(gids))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b")
    ap.add_argument("--steered-raw", type=Path, required=True)
    ap.add_argument("--unsteered-raw", type=Path, required=True)
    ap.add_argument("--steered-run", type=Path, required=True)
    ap.add_argument("--unsteered-run", type=Path, required=True)
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import torch
    device = args.device if torch.cuda.is_available() else "cpu"

    s0 = gen_metadata_by_id(args.stage0_run / "metadata.json")
    S = _load_arm(args.steered_raw, args.steered_run, s0, 1)
    U = _load_arm(args.unsteered_raw, args.unsteered_run, s0, 0)
    common = sorted(set(S["gids"]) & set(U["gids"]))
    logger.info(f"steered {len(S['gids'])} unsteered {len(U['gids'])} matched {len(common)}")

    def sel(arm, key):
        idx = {g: i for i, g in enumerate(arm["gids"])}
        return [arm[key][idx[g]] for g in common]

    def stack(key):
        return np.stack(sel(S, key) + sel(U, key)).astype(np.float32)

    topic_c = sel(S, "topics") + sel(U, "topics")
    ut = {t: i for i, t in enumerate(sorted(set(map(str, topic_c))))}
    topic = np.array([ut[str(t)] for t in topic_c])
    C = np.asarray(sel(S, "C") + sel(U, "C"), dtype=np.float64)
    y = np.array([1] * len(common) + [0] * len(common))

    Xhand, Xraw, Xwnm = stack("hand"), stack("rawlin"), stack("wnm")
    logger.info(f"X_hand {Xhand.shape}  X_rawlin {Xraw.shape}  X_wnm {Xwnm.shape}")

    logger.info("=== §5.1 resolver ladder (identical a003 split, residual surface) ===")
    hand = _cv_ladder(Xhand, y, topic, C, device, "HAND (means-based sig)")
    wnm = _cv_ladder(Xwnm, y, topic, C, device, f"WINDOWED-NO-MEANS (W={W_WINDOWS} std+slope)")
    raw = _cv_ladder(Xraw, y, topic, C, device, "RAW-LINEAR (5-pos snapshots)")

    h = max(hand["logit"]["test"], hand["deep"]["test"])
    w = max(wnm["logit"]["test"], wnm["deep"]["test"])
    r = max(raw["logit"]["test"], raw["deep"]["test"])
    gap = r - h
    closed = (w - h) / gap if gap > 1e-6 else 0.0
    verdict = ("GRANULARITY: windowed-no-means closes the hand→raw gap "
               f"({closed:.0%}) → the residual surface carries the stake, hand's MEANS were too "
               "coarse → WAIS §3 REWRITE (stop-and-surface)"
               if closed >= 0.60 else
               "COVERAGE: windowed-no-means stays near hand "
               f"({closed:.0%} of gap) → no aggregate recovers raw; the whole-vector's advantage is "
               "surface-coverage → WAIS §3 sentence STANDS"
               if closed <= 0.35 else
               f"INTERMEDIATE ({closed:.0%} of gap closed) → partial granularity; outer loop rules")

    out = {"model": args.model, "cell": "S5.1 resolver — coverage vs granularity (item 7)",
           "STATUS": "FIRST_READ_PENDING (C§8) — WAIS §3 decider",
           "n_matched_pairs": len(common), "n_topics": len(set(topic.tolist())),
           "P_hand": int(Xhand.shape[1]), "P_rawlin": int(Xraw.shape[1]), "P_wnm": int(Xwnm.shape[1]),
           "split": "α=.03 V2_L13-steered vs unsteered, matched-token, residual surface",
           "ladder": {"hand_means": hand, "windowed_no_means": wnm, "raw_linear": raw},
           "gap_hand_to_raw": round(gap, 4), "fraction_of_gap_closed_by_wnm": round(closed, 3),
           "chance": 0.5, "verdict_heuristic": verdict}
    p = args.out_dir / f"s51_resolver_{args.model}.json"
    p.write_text(json.dumps(out, indent=1))
    logger.info(f"gap hand→raw = {gap:.3f}; wnm closes {closed:.0%}")
    logger.info(f"VERDICT: {verdict}")
    logger.info(f"banked -> {p}")


if __name__ == "__main__":
    main()
