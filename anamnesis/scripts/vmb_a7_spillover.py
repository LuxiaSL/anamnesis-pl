"""A7 spillover + coherence readout (vmb arm A7, M6) — the co-primary headline reads.

Teacher-forced: each perturbation rung replays the SAME 80 floor gens as the unperturbed baseline
(topk6 == noise0), so per-gen deltas are ⊥-content by construction. Per the desk MORNING RULINGS:

  HEADLINE (spillover): for each NON-routing source s (attention/residual/gate/keys/output) compute the
  paired per-gen shift Δ_s = RMS over s's features of mean_g(z_perturbed − z_baseline) in floor-z units.
  Headline FIRES iff ≥1 non-routing source clears the 0.1× floor-z visibility BAR dose-monotonically
  within an arm; the sign-flip permutation is the GATE (replay is bitwise-deterministic → p≈0 for any
  real shift, so the BAR is the claim, not significance). xrt source Δ reported as the manipulation check
  (trivially-expected, excluded from the headline). §2c PAIR reads: shared_ablate vs routed_ablate
  (branch specialness); drop_top2 vs drop_rand2 (targeting).
  SECONDARY (coherence): within-cell tightness (mean pairwise cosine of the rung's whole-vector sigs)
  vs dose — graceful committee degradation vs cliff.

Run (node1, CPU): python -m anamnesis.scripts.vmb_a7_spillover \
  --a7-tf-root /models/anamnesis-extract/battery/arms/A7_dsv2/tf \
  --floor-dir /models/anamnesis-extract/runs/vmb_stage0_dsv2_lite/signatures_v3_x2 --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anamnesis.analysis.battery.deltas import load_floor_scale
from anamnesis.analysis.battery.floors import load_signature_matrix
from anamnesis.analysis.feature_map import FeatureMap, Source

NONROUTING = [Source.attention, Source.residual, Source.gate, Source.keys, Source.output]
BAR = 0.1                     # 0.1× floor-z visibility bar (12b convention)
TOPK_ARM = ["topk4", "topk2", "topk1"]        # increasing perturbation (fewer experts)
NOISE_ARM = ["noise0.25", "noise0.5", "noise1.0"]
PAIRS = [("shared_ablate", "routed_ablate"), ("drop_top2", "drop_rand2")]


def _load_z(d: Path, med, scale, subdir="signatures_v3_x2"):
    X, names, ids = load_signature_matrix(d / subdir)
    return (X - med) / scale, names, list(ids)


def _paired(zp, ids_p, zb, id_base):
    common = [g for g in ids_p if g in id_base]
    pi = np.array([zp[ids_p.index(g)] for g in common])
    bi = np.array([zb[id_base[g]] for g in common])
    return pi - bi, len(common)


def _delta_source(delta, idx, rng):
    d = delta[:, idx]                              # [n, |s|]
    Delta = float(np.sqrt((d.mean(0) ** 2).mean()))     # RMS per-feature mean shift (floor-z units)
    perm = np.empty(3000)
    for k in range(3000):
        sgn = rng.choice([-1.0, 1.0], size=d.shape[0])[:, None]
        perm[k] = np.sqrt(((d * sgn).mean(0) ** 2).mean())
    return Delta, float((perm >= Delta).mean())


def _coherence(z):
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    G = zn @ zn.T
    n = G.shape[0]
    return float((G.sum() - np.trace(G)) / (n * (n - 1)))   # mean off-diagonal cosine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a7-tf-root", type=Path, required=True)
    ap.add_argument("--floor-dir", type=Path, required=True)
    ap.add_argument("--baseline", default="topk6")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    med, scale = load_floor_scale(args.floor_dir)
    zb, names, ids_b = _load_z(args.a7_tf_root / args.baseline, med, scale)
    id_base = {g: i for i, g in enumerate(ids_b)}
    fmap = FeatureMap(names, 27)
    src_idx = {s: [i for i, t in enumerate(fmap.tags) if t.source == s]
               for s in NONROUTING + [Source.expert_routing]}
    rng = np.random.default_rng(0)

    rungs = TOPK_ARM + NOISE_ARM + [c for p in PAIRS for c in p]
    spill: dict = {}
    coh: dict = {args.baseline: round(_coherence(zb), 4)}
    for rung in rungs:
        cell = args.a7_tf_root / rung
        if not (cell / "signatures_v3_x2").exists():
            continue
        zp, _nm, ids_p = _load_z(cell, med, scale)
        delta, n = _paired(zp, ids_p, zb, id_base)
        coh[rung] = round(_coherence(zp), 4)
        row = {"n": n}
        for s, idx in src_idx.items():
            if idx:
                D, p = _delta_source(delta, idx, rng)
                row[s.value] = {"Delta_floorz": round(D, 4), "p_perm": round(p, 4),
                                "clears_bar": bool(D >= BAR)}
        spill[rung] = row

    # ── headline decision: ≥1 non-routing source ≥ bar, dose-monotone, in ≥1 arm ──
    def arm_monotone(arm, s):
        vals = [spill[r][s.value]["Delta_floorz"] for r in arm if r in spill and s.value in spill[r]]
        return len(vals) == len(arm) and all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)) \
            and vals[-1] >= BAR
    headline = {}
    for s in NONROUTING:
        headline[s.value] = {"topk_arm_monotone_at_bar": arm_monotone(TOPK_ARM, s),
                             "noise_arm_monotone_at_bar": arm_monotone(NOISE_ARM, s)}
    fires = any(v["topk_arm_monotone_at_bar"] or v["noise_arm_monotone_at_bar"]
                for v in headline.values())

    # §2c pair reads (which branch / targeting moves the non-routing sources more)
    pairs = {}
    for a, b in PAIRS:
        if a in spill and b in spill:
            pairs[f"{a}_vs_{b}"] = {
                s.value: {a: spill[a][s.value]["Delta_floorz"], b: spill[b][s.value]["Delta_floorz"]}
                for s in NONROUTING if s.value in spill[a]}

    out = {
        "arm": "A7 spillover (headline) + coherence (secondary) — teacher-forced",
        "bar_floorz": BAR, "baseline": args.baseline,
        "HEADLINE_FIRES": fires,
        "headline_by_source": headline,
        "spillover_per_rung": spill,
        "pair_reads_s2c": pairs,
        "coherence_by_rung": coh,
        "note": "replay bitwise-deterministic → p_perm≈0 for any real shift; the 0.1× BAR is the claim. "
                "xrt source Δ = manipulation check (trivially-expected), excluded from HEADLINE_FIRES.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({"HEADLINE_FIRES": fires, "headline_by_source": headline,
                      "coherence_by_rung": coh}, indent=1))
    print("\nspillover Δ_floorz (non-routing sources) + xrt manip-check:")
    for rung in rungs:
        if rung in spill:
            cells = " ".join(f"{s.value[:4]}={spill[rung][s.value]['Delta_floorz']}"
                             for s in NONROUTING + [Source.expert_routing] if s.value in spill[rung])
            print(f"  {rung:16s} {cells}")


if __name__ == "__main__":
    main()
