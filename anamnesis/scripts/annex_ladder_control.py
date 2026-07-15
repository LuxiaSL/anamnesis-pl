"""ANNEX A6″ — THE DEPTH-STRATIFIED CONTROL: rebuilding the ladder's positive control correctly.

⚠ ANNEX LANE: nothing here is quotable until it graduates through a frozen prereg cell.
Ledger: outputs/battery/annex/ANNEX-LEDGER.md

WHY THIS EXISTS — the session-3 baton's #1 item rests on a FALSE PREMISE, and this rung both
proves that and replaces it.

  THE BATON SAID: "`pca_*` features are EXACT linear functionals of the residual. They cannot
  saturate ⇒ they MUST read b=1 at ANY dose. Use them as the positive control."

  ⛔ BOTH HALVES ARE WRONG, and each is wrong for an instructive reason.

  1. **"Linear functional" ⇏ "linear response to α".** `pca_L{l}` reads `hidden_states[t][l+1]`
     (state_extractor.py:1081, `layer_offset = l_idx + 1`) — the OUTPUT of block l. The injection
     is a forward PRE-hook adding αv to the INPUT of block 14 (model_loader.py:279). So

         Δh_out_14 = αv + [f(h_in + αv) − f(h_in)]        f = attn+mlp of block 14

     A linear functional OF A NONLINEAR RESPONSE is nonlinear. The premise reasoned from a true
     property to a false conclusion by skipping the step in between.

  2. **The fix was already inside the failure.** `pca_*` is **1,250 of the trivial block's
     2,076 features — 60%**. The block that read b=1.19–1.63 IS mostly pca_*. Switching to pca_*
     re-runs the failure and returns the same number.

  ⛔ AND THE STATED CAUSE IS FALSE TOO. Session 2 wrote: "the block is dominated by BOUNDED logit
  statistics (entropy, top1_prob, surprise) — linear under small perturbations, SATURATING under
  large ones." Measured: those are **~22 of 2,076 features ≈ 1%**. And the sign is backwards —
  **saturation makes a response grow SLOWER than α (b < 1); the block reads b = 1.19–1.63, i.e.
  SUPERlinear.** An explanation that predicts the wrong sign of the effect it explains is not an
  explanation. Recorded so nobody re-raises it.

★ WHAT IS ACTUALLY WRONG: **C§1's "trivial" classifies by FEATURE FAMILY, not by DEPTH relative
to the injection site.** That lumps together three animals with three different responses:

    UPSTREAM   (layers < site)  Δ ≡ 0 EXACTLY. Matched tokens ⇒ everything below the injection
                                depends only on tokens ≤ t, which are identical. Bitwise.
    AT-SITE    (L14)            Δ = αv·comp + block-14's own nonlinear correction. The ONLY
                                place "the linear image of αv" is even approximately the truth.
    DOWNSTREAM (L18/21/24)      fully routed through 4–10 more blocks. Nothing about this is
                                linear, and calling it "trivial" was always about WHAT IT READS,
                                never about HOW IT RESPONDS.

So a positive control must be DEPTH-SCOPED. This rung builds the two that exist:

  · **THE ZERO CONTROL (exact, and nobody has ever run it).** Every upstream feature must read
    Δ = 0.0 to the bit, at every dose, for every vector. It validates the twin pairing, the
    determinism claim, the delta machinery, and the absence of leakage — the things a failed
    positive control cast doubt on. It cannot saturate, cannot rotate, and has no free parameter.
  · **THE AT-SITE CONTROL (approximate, and honest about it).** pca_L14 is αv·comp plus one
    block's correction. b→1 as α→0 for any smooth map; how far it stays near 1 up the ladder is
    a MEASUREMENT of block 14's nonlinearity, not an assumption.

    PYTHONPATH=pipeline python -m anamnesis.scripts.annex_ladder_control
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from anamnesis.analysis.battery.deltas import load_floor_scale
from anamnesis.scripts.annex_corpus import REPO, VENUE_DIR, load_venue
from anamnesis.scripts.annex_linearity import MT_PAT, OUT, load_dose_cells

logger = logging.getLogger(__name__)
F32 = NDArray[np.float32]

# Injection sites in the bank: L14 for every vector except V2 (L13).
LAYER_RE = re.compile(r"L(\d+)")


def layers_of(name: str) -> set[int]:
    return {int(x) for x in LAYER_RE.findall(name)}


def effective_max_layer(name: str) -> int | None:
    """The DEEPEST block a feature actually reads — which is NOT always its deepest NAMED layer.

    ⚠ `delta_*_L{l}` reads `hs_array[:, l+1]` AND `hs_array[:, l+2]` (state_extractor.py:479-481)
    = the outputs of blocks l and l+1 — so it reads **one block deeper than its name**. Verified
    as the ONLY such family: `l + 2` appears nowhere else in the extractor (only lines 481/486,
    both in the delta block); every other family reads at most its named block.

    This off-by-one is exactly what the zero control caught on its FIRST run, and the catch is the
    reason to trust the rung: V2 (the lone L13 vector) moved `delta_norm_mean_L12` by up to
    **9.09 z**, while all six L14 vectors left all 883 columns at **exactly 0.0**. Both facts are
    the same fact — delta_*_L12 reads block 13, which is upstream of L14 and IS the injected
    surface at L13. The control was right and the naive name-based cut was wrong.
    """
    ls = layers_of(name)
    if not ls:
        return None
    return max(ls) + (1 if name.startswith("delta_") else 0)


def upstream_cols(names: list[str], site: int) -> NDArray[np.int64]:
    """Columns reading strictly ABOVE the injection site ⇒ Δ must be exactly 0.

    Site-dependent by necessity: V2 injects at L13, everything else at L14, and a feature that is
    upstream of L14 may be AT-SITE for L13 (delta_*_L12 is exactly that feature). A single
    conservative cut would either miss the V2 boundary or needlessly weaken the L14 control.
    """
    return np.array([i for i, n in enumerate(names)
                     if (e := effective_max_layer(n)) is not None and e < site], dtype=np.int64)


def depth_spaces(names: list[str]) -> dict[str, NDArray[np.int64]]:
    """Depth-stratified column sets — the decomposition C§1's family-based split cannot make."""
    idx: dict[str, list[int]] = {"pca_L7_zero": [], "at_site_pca_L14": [],
                                 "downstream_pca": [], "pca_all": []}
    for i, n in enumerate(names):
        if n.startswith("pca_"):
            ls = layers_of(n)
            idx["pca_all"].append(i)
            l = min(ls) if ls else -1
            if l == 7:
                idx["pca_L7_zero"].append(i)
            elif l == 14:
                idx["at_site_pca_L14"].append(i)
            elif l > 14:
                idx["downstream_pca"].append(i)
    return {k: np.array(v, dtype=np.int64) for k, v in idx.items()}


def loglog_slope(alphas: list[float], mags: list[float]) -> float:
    """b in ‖Δs‖ ∝ α^b. Linear map ⇒ b = 1 exactly."""
    if len(alphas) < 2 or min(mags) <= 0:
        return float("nan")
    return float(np.polyfit(np.log(alphas), np.log(mags), 1)[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT / "annex_ladder_control.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    med, scale = load_floor_scale(VENUE_DIR / "signatures_v3")
    c = load_venue(capped_only=True)
    names = c.feature_names

    print("=== MATCHED-TOKEN LADDER (per-gen Δ vs stage0 twins) ===")
    cells = load_dose_cells(med, scale, names)

    by_vec: dict[str, dict[float, object]] = {}
    for name, cell in cells.items():
        m = MT_PAT.match(name)
        key = m.group("vec") if not m.group("site") else f"{m.group('vec')}_L{m.group('site')}"
        by_vec.setdefault(key, {})[cell.alpha] = cell

    sp = depth_spaces(names)
    print("\n=== DEPTH-STRATIFIED SPACES (the decomposition C§1's family split cannot make) ===")
    for k, v in sp.items():
        print(f"  {k:20s} d={len(v):5d}")

    # ── ★ RUNG 1: THE ZERO CONTROL — exact, and it has no free parameter ────────────────────
    print(f"\n{'='*78}\n=== ★ ZERO CONTROL: upstream of the injection ⇒ Δ MUST be 0.0 to the bit\n{'='*78}")
    zero_rows: dict[str, dict] = {}
    worst = 0.0
    for vec in sorted(by_vec):
        site = int(vec.rsplit("_L", 1)[1]) if "_L" in vec else 14   # randoms are site-independent
        up = upstream_cols(names, site)
        print(f"  {vec:10s} site L{site}  upstream d={len(up):4d}", end="")
        v_worst = 0.0
        for a in sorted(by_vec[vec]):
            cell = by_vec[vec][a]
            D = np.stack([cell.deltas[g] for g in sorted(cell.deltas)])
            for space, cols in (("upstream_zero", up), ("pca_L7_zero", sp["pca_L7_zero"])):
                mx = float(np.abs(D[:, cols]).max())
                nz = int((np.abs(D[:, cols]) > 0).sum())
                zero_rows[f"{vec}_a{a}_{space}"] = {"max_abs_delta": mx, "n_nonzero_cells": nz,
                                                    "site": site, "d": int(len(cols))}
                worst, v_worst = max(worst, mx), max(v_worst, mx)
        print(f"   worst |Δz| over {len(by_vec[vec])} doses = {v_worst:.3e}"
              f"{'  ✓' if v_worst == 0.0 else '  ✗'}")
    n_bad = sum(1 for r in zero_rows.values() if r["max_abs_delta"] > 0)
    print(f"\n  cells x spaces checked : {len(zero_rows)}")
    print(f"  worst |Δz| anywhere    : {worst:.3e}")
    print(f"  rows with ANY nonzero  : {n_bad}")
    zero_verdict = ("PASS (exact) — machinery sound: twins paired, replay bitwise-deterministic, "
                    "no leakage below the site, delta algebra correct"
                    if worst == 0.0 else
                    f"FAIL — upstream moved by {worst:.3e}; the MT bank is NOT what it claims")
    print(f"  ⇒ {zero_verdict}")

    # ── ★ RUNG 2: the at-site vs downstream slope ladder ────────────────────────────────────
    results: dict[str, dict] = {"zero_control": {"rows": zero_rows, "worst_abs_delta": worst,
                                                 "verdict": zero_verdict}}
    for space in ("at_site_pca_L14", "downstream_pca", "pca_all"):
        cols = sp[space]
        print(f"\n{'='*78}\n=== SPACE: {space} (d={len(cols)}) — linear ⇒ b = 1\n{'='*78}")
        print(f"  {'vector':10s} {'b_global':>9s} {'b@.1':>8s} {'b@.2':>8s} {'b@.3':>8s}  "
              f"{'‖Δ‖@.03':>9s} {'‖Δ‖@.5':>9s}")
        rows: dict[str, dict] = {}
        for vec in sorted(by_vec):
            doses = sorted(by_vec[vec])
            mags = []
            for a in doses:
                cell = by_vec[vec][a]
                D = np.stack([cell.deltas[g][cols] for g in sorted(cell.deltas)])
                mags.append(float(np.linalg.norm(D, axis=1).mean()))
            b_global = loglog_slope(doses, mags)
            b_local: dict[float, float] = {}
            for i in range(1, len(doses) - 1):
                b_local[doses[i]] = float(
                    (np.log(mags[i + 1]) - np.log(mags[i - 1]))
                    / (np.log(doses[i + 1]) - np.log(doses[i - 1])))
            rows[vec] = {"alphas": doses, "response_magnitude": [round(m, 5) for m in mags],
                         "loglog_slope_global": round(b_global, 4),
                         "loglog_slope_local": {str(k): round(v, 4) for k, v in b_local.items()}}
            g = lambda a: f"{b_local[a]:8.3f}" if a in b_local else "       —"
            print(f"  {vec:10s} {b_global:9.3f} {g(0.1)} {g(0.2)} {g(0.3)}  "
                  f"{mags[0]:9.4f} {mags[-1]:9.4f}")
        results[space] = {"vectors": rows}

    # ── ★ RUNG 3: WHERE the trivial block's superlinearity actually lives ───────────────────
    # pca_* is 60% of trivial_2076 BY COUNT and reads b≈1.0, while the whole block reads
    # 1.19–1.63. ‖Δ‖ is dominated by MAGNITUDE, not by count — so a minority of features must be
    # carrying it. This names them instead of asserting a cause (session 2 asserted "bounded logit
    # statistics saturate"; that is ~1% of the block and predicts the wrong sign).
    z2 = np.load(REPO / "outputs/battery/arms/C2/c2_orphaned_axis_3b.npz")
    nt = {str(x) for x in z2["feature_names"]}
    triv = np.array([i for i, n in enumerate(names) if n not in nt], dtype=np.int64)
    fam_of = {}
    for i in triv:
        n = names[i]
        f = ("pca" if n.startswith("pca_") else
             "activation_norm" if n.startswith("activation_norm") else
             "delta" if n.startswith("delta_") else
             "res_traj" if n.startswith("res_traj") else
             "value" if n.startswith("value_") else
             "kv_value_cka" if n.startswith("kv_value_cka") else
             "logit_bounded" if any(s in n for s in ("surprise", "logit_entropy", "top1_prob",
                                                     "top5_mass", "chosen_rank")) else "other")
        fam_of.setdefault(f, []).append(i)

    print(f"\n{'='*78}\n=== ★ TRIVIAL BLOCK, DECOMPOSED BY FAMILY — where does b>1 come from?\n{'='*78}")
    print(f"  {'family':16s} {'d':>5s} {'%count':>7s} {'%‖Δ‖² @.5':>10s} "
          f"{'b(R2)':>7s} {'b(V3)':>7s} {'b(V1)':>7s}")
    ref = np.stack([by_vec["V3_L14"][0.5].deltas[g] for g in sorted(by_vec["V3_L14"][0.5].deltas)])
    tot = float((ref[:, triv] ** 2).sum())
    fam_rows = {}
    for f, cols_l in sorted(fam_of.items(), key=lambda kv: -len(kv[1])):
        cols = np.array(cols_l, dtype=np.int64)
        bs = {}
        for vec in ("R2", "V3_L14", "V1_L14"):
            doses = sorted(by_vec[vec])
            mags = [float(np.linalg.norm(
                np.stack([by_vec[vec][a].deltas[g][cols] for g in sorted(by_vec[vec][a].deltas)]),
                axis=1).mean()) for a in doses]
            bs[vec] = loglog_slope(doses, mags)
        shr = float((ref[:, cols] ** 2).sum()) / max(tot, 1e-30)
        fam_rows[f] = {"d": int(len(cols)), "pct_count": round(100 * len(cols) / len(triv), 1),
                       "pct_delta_sq_at_alpha0.5": round(100 * shr, 2),
                       "b_global": {k: round(v, 3) for k, v in bs.items()}}
        print(f"  {f:16s} {len(cols):5d} {100*len(cols)/len(triv):6.1f}% {100*shr:9.2f}% "
              f"{bs['R2']:7.3f} {bs['V3_L14']:7.3f} {bs['V1_L14']:7.3f}")
    results["trivial_family_decomposition"] = fam_rows

    # ── ★ RUNG 4: THE DECISIVE TEST — is the trivial block's b>1 the MAP, or the FORCING? ────
    # `chosen_rank[i] = (x > x[cid]).sum()` (state_extractor.py:191) = the rank of the FORCED
    # token under the STEERED logits. In matched-token replay the forced tokens ARE the banked
    # stage0 tokens, so mean/std_chosen_rank measure HOW OFF-POLICY THE FORCING IS — the very
    # quantity the on-policy gate thresholds (spearman(agreement, mean_chosen_rank) = −0.83,
    # p=1e-10), not a property of the injection→computation map.
    #
    # If that diagnosis is right, deleting TWO features out of 2,076 must collapse the trivial
    # block from b = 1.19–1.63 to b ≈ 1. If it does not, the diagnosis is wrong.
    forcing = {i for i, n in enumerate(names) if "chosen_rank" in n}
    print(f"\n{'='*78}\n=== ★ DECISIVE: drop the {len(forcing)} chosen_rank features "
          f"(they measure the FORCING, not the map)\n{'='*78}")
    all_c = np.arange(len(names), dtype=np.int64)
    tests = {
        "trivial_2076_AS_IS": triv,
        "trivial_2074_minus_forcing": np.array([i for i in triv if i not in forcing], np.int64),
        "full_3358_AS_IS": all_c,
        "full_3356_minus_forcing": np.array([i for i in all_c if i not in forcing], np.int64),
    }
    print(f"  {'space':30s} {'d':>5s} " + " ".join(f"{v:>7s}" for v in sorted(by_vec)))
    dec: dict[str, dict] = {}
    for sname, cols in tests.items():
        bs = {}
        for vec in sorted(by_vec):
            doses = sorted(by_vec[vec])
            mags = [float(np.linalg.norm(
                np.stack([by_vec[vec][a].deltas[g][cols] for g in sorted(by_vec[vec][a].deltas)]),
                axis=1).mean()) for a in doses]
            bs[vec] = round(loglog_slope(doses, mags), 3)
        dec[sname] = {"d": int(len(cols)), "b_global": bs}
        print(f"  {sname:30s} {len(cols):5d} " + " ".join(f"{bs[v]:7.3f}" for v in sorted(by_vec)))
    results["decisive_forcing_test"] = dec

    OUT.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "ANNEX_RULE": "NOT QUOTABLE until graduated via a frozen prereg cell",
        "rung": "A6″ — the depth-stratified control (replaces the baton's false-premise #1)",
        "why_the_baton_was_wrong": {
            "premise": "'pca_* are exact linear functionals ⇒ cannot saturate ⇒ must read b=1'",
            "refutation_1": "pca_L{l} reads hidden_states[t][l+1] = the OUTPUT of block l "
                            "(state_extractor.py:1081); the injection is a PRE-hook on block 14's "
                            "INPUT (model_loader.py:279). Δh_out = αv + [f(h+αv) − f(h)]. A linear "
                            "functional OF A NONLINEAR RESPONSE is nonlinear.",
            "refutation_2": "pca_* is 1,250 of the trivial block's 2,076 features (60%). The "
                            "proposed control IS most of the thing that failed; it would have "
                            "reproduced the same number.",
            "refutation_3": "the stated CAUSE ('dominated by BOUNDED logit statistics that "
                            "saturate') is false twice: those are ~22/2076 ≈ 1% of the block, and "
                            "saturation predicts b < 1 while the block reads b = 1.19–1.63 "
                            "(SUPERlinear). Wrong composition, wrong sign.",
        },
        "the_real_defect": "C§1's 'trivial' classifies by FEATURE FAMILY, not by DEPTH relative to "
                           "the injection site — so it lumps upstream (Δ≡0), at-site (αv + one "
                           "block's correction), and downstream (fully routed) into one block and "
                           "calls the mixture 'the linear image of αv'. 'Trivial' was always about "
                           "WHAT A FEATURE READS, never about HOW IT RESPONDS TO α.",
        "zero_control": "upstream features must read Δ = 0.0 to the bit (matched tokens ⇒ "
                        "everything below the site depends only on identical tokens). Exact, no "
                        "free parameter, never run before. It validates twin pairing, replay "
                        "determinism, the delta machinery, and the absence of leakage.",
        "at_site_control": "pca_L14 = αv·comp + block-14's correction. b→1 as α→0 for any smooth "
                           "map; how far it stays near 1 MEASURES block 14's nonlinearity rather "
                           "than assuming it away.",
        "spaces": {k: int(len(v)) for k, v in sp.items()},
        "results": results,
    }, indent=1))
    print(f"\n  → {args.out}")


if __name__ == "__main__":
    main()
