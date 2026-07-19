"""14w swap-continuity DETECTOR — the analysis half (CPU, reads the per-token npz from
vmb_a7_swap_continuity). Separated from the GPU extraction so the readout is fast + re-runnable.

Reads {rung}__{col} arrays (rung ∈ baseline/shared_ablate/topk2; cols pos/token_id/resid_velocity/
attn_regime/top1_switch/set_churn). Computes the frozen 14w readouts (ADDENDUM 2026-07-18w):

  Leg 1 (P .65) — swap events discontinuity-locked ABOVE the matched-token null. Partial correlation
    swap<->discontinuity after removing token_id×position-bin means from BOTH (the position-tracking +
    token-type confound rider), with an asymptotic p (n~1e5, permutation is overkill and was the perf
    trap in the inline version). Raw correlation reported beside.
  Confound rider (causal) — baseline vs topk2 at MATCHED (gen,pos): corr(Δswap, Δdisc). Position is
    identical by construction, so a positive link is swap-locked, not position-locked.
  Leg 2 (P .55) — shared_ablate amplifies swap-locked discontinuity vs baseline (SAME swaps — ablate
    leaves routing unchanged — backbone removed => bigger seams): coupling slope + at-hi-swap disc gap.

Vectorized demean (bincount over strata codes) + asymptotic Spearman p. Nan-robust (ablation can zero
states -> cos NaN). First-read -> outer loop; UNSTAMPED.

    python -m anamnesis.scripts.vmb_a7_swap_analyze --npz .../swap14w/pertoken.npz --out-json .../swap_continuity.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

RUNGS = ("baseline", "shared_ablate", "topk2")
DISC = ("resid_velocity", "attn_regime")
SWAP = ("top1_switch", "set_churn")


def _rank(x):
    return np.argsort(np.argsort(x)).astype(np.float64)


def _spearman_p(x, y):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 10:
        return float("nan"), float("nan"), n
    rx, ry = _rank(x), _rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = math.sqrt((rx * rx).sum() * (ry * ry).sum())
    r = float((rx * ry).sum() / d) if d > 0 else float("nan")
    if not math.isfinite(r) or abs(r) >= 1.0:
        return r, float("nan"), n
    t = r * math.sqrt((n - 2) / (1 - r * r))          # ~ t_{n-2}; normal approx at large n
    p = math.erfc(abs(t) / math.sqrt(2))              # two-sided
    return round(r, 4), float(f"{p:.2e}"), n


def _demean_by_strata(v, strata_codes, n_codes):
    """subtract per-stratum mean (vectorized). NaN-safe: NaNs excluded from group means."""
    v = np.asarray(v, np.float64)
    finite = np.isfinite(v)
    sums = np.bincount(strata_codes[finite], weights=v[finite], minlength=n_codes)
    cnts = np.bincount(strata_codes[finite], minlength=n_codes)
    means = np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0)
    return v - means[strata_codes]


def run_analysis(npz_path: Path, out_json: Path, posbin: int = 16) -> dict:
    """Compute + write the 14w Leg1/confound/Leg2 readouts from a per-token npz. Fast (vectorized
    demean + asymptotic p); shared by the standalone detector and the extractor (vmb_a7_swap_continuity)."""
    args = argparse.Namespace(npz=Path(npz_path), out_json=Path(out_json), posbin=posbin)
    d = np.load(args.npz)
    data = {r: {c: d[f"{r}__{c}"] for c in
                ("gen", "pos", "token_id", "resid_velocity", "attn_regime", "top1_switch", "set_churn")}
            for r in RUNGS if f"{r}__pos" in d}
    base = data["baseline"]

    # strata = token_id × position-bin  (the confound rider's matched null)
    strata = base["token_id"].astype(np.int64) * 100000 + (base["pos"] // args.posbin).astype(np.int64)
    codes, inv = np.unique(strata, return_inverse=True)
    n_codes = len(codes)

    res = {"arm": "14w continuity-through-swaps — DETECTOR", "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED",
           "swap_contrast_rung": "topk2", "n_baseline_tokens": int(len(base["pos"])),
           "n_strata_token_x_posbin": int(n_codes), "posbin": args.posbin}

    # Leg 1 — matched-null partial correlation (+ raw)
    res["leg1_baseline"] = {}
    for disc in DISC:
        for swap in SWAP:
            r_raw, p_raw, n = _spearman_p(base[swap], base[disc])
            sw_d = _demean_by_strata(base[swap], inv, n_codes)
            di_d = _demean_by_strata(base[disc], inv, n_codes)
            r_par, p_par, _ = _spearman_p(sw_d, di_d)
            res["leg1_baseline"][f"{swap}->{disc}"] = {
                "raw_spearman": r_raw, "partial_spearman_matchednull": r_par,
                "partial_p_asymptotic": p_par, "n": n}

    # Confound rider — baseline vs topk2, matched (gen,pos)
    def align(a, b):
        ka = {(int(g), int(p)): i for i, (g, p) in enumerate(zip(a["gen"], a["pos"]))}
        kb = {(int(g), int(p)): i for i, (g, p) in enumerate(zip(b["gen"], b["pos"]))}
        common = [k for k in ka if k in kb]
        return np.array([ka[k] for k in common]), np.array([kb[k] for k in common])

    ia, ib = align(base, data["topk2"])
    res["confound_causal_baseline_vs_topk2"] = {"n_matched": int(len(ia))}
    for disc in DISC:
        dswap = data["topk2"]["top1_switch"][ib] - base["top1_switch"][ia]
        ddisc = data["topk2"][disc][ib] - base[disc][ia]
        r, p, n = _spearman_p(dswap, ddisc)
        res["confound_causal_baseline_vs_topk2"][f"dtop1_switch->d{disc}"] = {"spearman": r, "p": p, "n": n}

    # Leg 2 — shared_ablate amplifies swap-locked discontinuity
    ia, ib = align(base, data["shared_ablate"])
    res["leg2_shared_ablate"] = {"n_matched": int(len(ia))}
    for disc in DISC:
        cb, _, _ = _spearman_p(base["top1_switch"][ia], base[disc][ia])
        ca, _, _ = _spearman_p(data["shared_ablate"]["top1_switch"][ib], data["shared_ablate"][disc][ib])
        bsw = base["top1_switch"][ia]
        hi = bsw > np.quantile(bsw, 0.75)
        abl_hi = np.asarray(data["shared_ablate"][disc][ib][hi], np.float64)
        base_hi = np.asarray(base[disc][ia][hi], np.float64)
        amp = (float(np.nanmean(abl_hi) - np.nanmean(base_hi))
               if np.isfinite(abl_hi).any() and np.isfinite(base_hi).any() else float("nan"))
        res["leg2_shared_ablate"][disc] = {
            "coupling_baseline": cb, "coupling_ablate": ca,
            "coupling_amplified": bool(math.isfinite(cb) and math.isfinite(ca) and ca > cb),
            "hi_swap_disc_ablate_minus_baseline": round(amp, 5) if math.isfinite(amp) else None}

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(res, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--posbin", type=int, default=16, help="position-bin width for the matched null")
    args = ap.parse_args()
    res = run_analysis(args.npz, args.out_json, args.posbin)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
