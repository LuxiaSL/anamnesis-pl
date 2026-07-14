"""P1 — §1.5 token-KL + §4.3 entropy reducer over matched-token steered vs unsteered replays.

§1.5 (the isolated P3 dissociation column): steering the state at FIXED (teacher-forced) tokens
changes the next-token distribution by some KL, while the SIGNATURE separates strongly — the
matched-token analog of the A1/A2 "token-visible vs internal" dissociation. This reducer emits
the token-KL channel: per-position KL(unsteered ‖ steered) over the MT continuations, to sit
beside the banked signature deformation (the state channel).

§4.3 (re-prioritized into the graded-Goodhart panel): the per-position next-token ENTROPY
under steering — the "does steering flatten/sharpen the distribution" trajectory. Both readouts
come from ONE logit-retaining replay pass (run_replay_extraction --logits-top-k <big>): the
driver retains enough per-position logit mass, this reducer reads it. No GPU.

Inputs = two raw dirs of `gen_*.npz` (v3 raw with logits_values/logits_indices/logits_entropy)
over the SAME gen ids and matched tokens: the α=0 (unsteered) replay and one steered
(vector,dose) replay. The window loops this over vectors/doses.

    PYTHONPATH=pipeline python -m anamnesis.scripts.vmb_a5_kl_entropy \
        --unsteered-raw <run>/raw_unsteered --steered-raw <run>/raw_V3_L14_a0.3 \
        --label V3_L14_a0.3 --out-dir outputs/battery/arms/A5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Missing-support floor for the truncated KL: a logit this far below a distribution's
# smallest retained logit is treated as the mass a missing index carries. With a LARGE
# --logits-top-k (near full vocab) this never binds and the KL is ~exact; documented so the
# outer loop reads any KL computed at small top-k as an upper-bounded approximation.
MISSING_LOGIT_MARGIN = 10.0


def _pos_dist(values: NDArray, indices: NDArray, support: NDArray) -> NDArray:
    """Softmax prob over `support` from a position's top-k (value, index); missing → floored."""
    lut = {int(ix): float(v) for ix, v in zip(indices, values)}
    floor = float(values.min()) - MISSING_LOGIT_MARGIN
    logits = np.array([lut.get(int(s), floor) for s in support], dtype=np.float64)
    logits -= logits.max()
    p = np.exp(logits)
    return p / p.sum()


def _gen_kl(un: dict, st: dict) -> tuple[NDArray, NDArray, NDArray]:
    """Per-position KL(unsteered‖steered) + entropies over a gen's matched continuation."""
    T = min(un["logits_values"].shape[0], st["logits_values"].shape[0])
    kl = np.empty(T, dtype=np.float64)
    for t in range(T):
        support = np.union1d(un["logits_indices"][t], st["logits_indices"][t])
        P = _pos_dist(un["logits_values"][t], un["logits_indices"][t], support)
        Q = _pos_dist(st["logits_values"][t], st["logits_indices"][t], support)
        kl[t] = float(np.sum(P * (np.log(P + 1e-12) - np.log(Q + 1e-12))))
    return kl, un["logits_entropy"][:T].astype(np.float64), st["logits_entropy"][:T].astype(np.float64)


def _load(raw_dir: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for p in sorted(raw_dir.glob("gen_*.npz")):
        gid = int(p.stem.split("_")[1])
        z = np.load(p, allow_pickle=True)
        if "logits_values" not in z:
            continue
        out[gid] = {k: z[k] for k in ("logits_values", "logits_indices", "logits_entropy")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unsteered-raw", type=Path, required=True)
    ap.add_argument("--steered-raw", type=Path, required=True)
    ap.add_argument("--label", required=True, help="cell label, e.g. V3_L14_a0.3")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--traj-len", type=int, default=64, help="positions in the emitted trajectory")
    args = ap.parse_args()

    un, st = _load(args.unsteered_raw), _load(args.steered_raw)
    gids = sorted(set(un) & set(st))
    if not gids:
        raise SystemExit("no shared gen ids between the two raw dirs (matched-token required)")

    per_gen_meankl: list[float] = []
    kl_by_pos: list[list[float]] = [[] for _ in range(args.traj_len)]
    ent_un_by_pos: list[list[float]] = [[] for _ in range(args.traj_len)]
    ent_st_by_pos: list[list[float]] = [[] for _ in range(args.traj_len)]
    top_k = int(un[gids[0]]["logits_values"].shape[1])
    for g in gids:
        kl, eu, es = _gen_kl(un[g], st[g])
        per_gen_meankl.append(float(kl.mean()))
        for t in range(min(args.traj_len, len(kl))):
            kl_by_pos[t].append(float(kl[t]))
            ent_un_by_pos[t].append(float(eu[t])); ent_st_by_pos[t].append(float(es[t]))

    def _mean(xs: list[list[float]]) -> list[float]:
        return [round(float(np.mean(x)), 5) if x else None for x in xs]

    out = {
        "arm": "A5", "cell": args.label,
        "readout": "§1.5 token-KL(unsteered‖steered) [the isolated matched-token dissociation "
                   "column — sits beside the banked signature deformation: KL small + signature "
                   "large = the P3-class state-vs-token dissociation] + §4.3 per-position entropy.",
        "n_gens": len(gids), "logits_top_k": top_k,
        "kl_approx_note": ("EXACT if the replay retained ~full-vocab logits (--logits-top-k "
                           "large); at small top_k the union-support KL is an upper-bounded "
                           f"approximation (missing-index floor = min_logit − {MISSING_LOGIT_MARGIN})."),
        "mean_kl": round(float(np.mean(per_gen_meankl)), 5),
        "mean_kl_sem": round(float(np.std(per_gen_meankl) / max(len(per_gen_meankl) ** 0.5, 1)), 5),
        "kl_trajectory": _mean(kl_by_pos),
        "entropy_unsteered_trajectory": _mean(ent_un_by_pos),
        "entropy_steered_trajectory": _mean(ent_st_by_pos),
        "mean_entropy_unsteered": round(float(np.nanmean([np.mean(x) for x in ent_un_by_pos if x])), 5),
        "mean_entropy_steered": round(float(np.nanmean([np.mean(x) for x in ent_st_by_pos if x])), 5),
        "law": {"n": len(gids), "M": "from raw dirs",
                "law": "per-position KL(unsteered‖steered) over union top-k support + banked "
                       "exact entropy; matched-token (teacher-forced) continuations",
                "floor_type": "n/a (dissociation column; pairs with signature deformation)"},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    p = args.out_dir / f"a5_kl_entropy_{args.label}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"[{args.label}] n={len(gids)} top_k={top_k}  mean_KL={out['mean_kl']}  "
          f"entropy {out['mean_entropy_unsteered']}→{out['mean_entropy_steered']}")
    print(f"  → {p}")


if __name__ == "__main__":
    main()
