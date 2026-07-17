"""D3 — prompt-vs-lever behavioral equivalence at MATCHED RATE (14q item 2, Pd3=.70).

Extends R9's state-side answer (profiles distinct at full strength) to the
behavior-conditioned case: at MATCHED BEHAVIORAL RATE, do prompt-induced and lever-induced
generations remain distinguishable in PROFILE space (readout-vector × dose; NEVER per-facet)?

⛔ COMMON-INSTRUMENT RULE (session-11 UPDATE block, binding): the matched behavioral rate is
computed on instruments valid for BOTH arms — (a) frac_analogical from the A2-pures LDA
applied to NO-INJECT signatures (prompt arm: its plain sigs — no injection exists; lever arm:
the banked `signatures_v3_noinject` expression column), and (b) the text-level analogy-marker
rate. Injected-sig frac is NEVER used for matching (α≤.1 injected-frac = upper bound, not
certificate — the apples-to-contaminated failure this rule exists to prevent).

Design:
  rate table   — per graded prompt cell (g0 none … g4 = RUN4 analogical verbatim) and per
                 banked V3 dose (α .03/.1/.3): frac-noinject + markers/1k.
  matching     — per V3 dose, nearest grade under each instrument (both reported; frac is
                 primary, markers corroborate; disagreement is data, not noise — 14r tension).
  profiles     — Δ_prompt(g)      = centroid(g.Z) − centroid(g0.Z)          [plain sigs]
                 Δ_inject_state(α)= centroid(V3@α injected) − pooled rider  [state column]
                 Δ_inject_expr(α) = centroid(V3@α noinject) − pooled rider  [expression column]
                 Δ_random(α)      = per R1-3@α − pooled rider               [generic null]
  metrics      — cos(Δ_prompt, Δ_inject_*) vs the Δ_random null band; dir0 projection of each
                 Δ (the shared mode component); off-dir0 residual cos (distinctness after
                 removing the shared axis). Two-column readout standing: state AND expression
                 rows, both, always.
Both outcomes pre-worded (14q): distinct-profiles ⇒ "not an expensive prompt" quotable;
indistinguishable ⇒ the lever's uniqueness claim scoped to state-side (R9) only.

CPU-only over banked + D3-generated corpora. First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from anamnesis.analysis.battery.deltas import ConditionCorpus, load_floor_scale
from anamnesis.analysis.battery.manifest import MODEL_META

DIR0_PAIR = ("analogical", "contrastive")
# analogy markers — vmb_a5_postverdict_gates lexicon verbatim (the banked marker instrument)
MARKERS = [r'\blike a\b', r'\blike an\b', r'\bas if\b', r'\bimagine\b', r'\bthink of\b',
           r'\bsimilar to\b', r'\bjust as\b', r'\bakin to\b', r'\bmetaphor\b', r'\banalogy\b',
           r'\banalogous\b', r'\bresembles\b', r'\bcompare it to\b', r'\bmuch like\b']


def marker_rate(meta_path: Path) -> float:
    md = json.loads(meta_path.read_text())
    gens = md["generations"] if "generations" in md else md
    tot_m = tot_w = 0
    for g in gens:
        t = (g.get("generated_text") or "").lower()
        w = len(t.split())
        tot_m += sum(len(re.findall(m, t)) for m in MARKERS)
        tot_w += max(w, 1)
    return tot_m / tot_w * 1000.0


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--promptarm-run", type=Path, required=True,
                    help="vmb_d3_3b_promptarm (cells g0_none..g4_full, replayed plain sigs)")
    ap.add_argument("--model", default="3b")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    br, mm = args.battery_root, MODEL_META[args.model]
    med, scale = load_floor_scale(br / mm.stage0_dir / "signatures_v3")

    def corpus(d: Path, label: str, sig_subdir: str = "signatures_v3") -> ConditionCorpus:
        return ConditionCorpus(d / sig_subdir, d / "metadata.json", med, scale, label)

    # dir0 + frac classifier — identical construction to the gg/F-rung (one source of truth)
    pures = {m: corpus(br / f"vmb_a2_{args.model}_pure_{m}", m) for m in DIR0_PAIR}
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
        np.vstack([pures[DIR0_PAIR[0]].Z, pures[DIR0_PAIR[1]].Z]),
        np.r_[np.ones(len(pures[DIR0_PAIR[0]].Z)), np.zeros(len(pures[DIR0_PAIR[1]].Z))])
    dir0 = clf.coef_[0].astype(np.float64)
    dir0 /= np.linalg.norm(dir0)

    a5 = br / f"vmb_a5_{args.model}"
    rider_Z = np.vstack([corpus(d, d.name).Z for d in sorted(a5.iterdir())
                         if d.name.endswith("_a0.0") and (d / "signatures_v3").exists()])
    ref = rider_Z.mean(0)

    # ── rate table ──────────────────────────────────────────────────────────────
    grades: dict[str, dict] = {}
    for d in sorted(args.promptarm_run.iterdir()):
        if not (d / "signatures_v3").exists():
            continue
        C = corpus(d, d.name)
        grades[d.name] = {"Z": C.Z, "n": len(C.Z),
                          "frac": float(clf.predict(C.Z).mean()),
                          "markers_per_1k": round(marker_rate(d / "metadata.json"), 3)}
    if "g0_none" not in grades:
        raise SystemExit("prompt arm missing its g0_none baseline cell")

    lever: dict[str, dict] = {}
    for a in ("0.03", "0.1", "0.3"):
        d = a5 / f"V3_L14_L14_a{a}"
        if not (d / "signatures_v3_noinject").exists():
            raise SystemExit(f"{d.name}: no banked signatures_v3_noinject (expression column "
                             "required by the common-instrument rule)")
        state = corpus(d, f"V3@{a}-state")
        expr = corpus(d, f"V3@{a}-expr", sig_subdir="signatures_v3_noinject")
        lever[a] = {"state_Z": state.Z, "expr_Z": expr.Z, "n": len(state.Z),
                    "frac_noinject": float(clf.predict(expr.Z).mean()),
                    "frac_injected_STATE_SIDE_ONLY": float(clf.predict(state.Z).mean()),
                    "markers_per_1k": round(marker_rate(d / "metadata.json"), 3)}

    # ── matching (per V3 dose, nearest grade; both instruments) ─────────────────
    gnames = [g for g in grades if g != "g0_none"]
    matches = {}
    for a, row in lever.items():
        by_frac = min(gnames, key=lambda g: abs(grades[g]["frac"] - row["frac_noinject"]))
        by_mark = min(gnames, key=lambda g: abs(grades[g]["markers_per_1k"] - row["markers_per_1k"]))
        matches[a] = {"by_frac": by_frac, "by_markers": by_mark,
                      "instruments_agree": by_frac == by_mark}

    # ── profiles at the matched pairs ────────────────────────────────────────────
    g0c = grades["g0_none"]["Z"].mean(0)

    def profile_row(gname: str, a: str) -> dict:
        dp = grades[gname]["Z"].mean(0) - g0c
        ds = lever[a]["state_Z"].mean(0) - ref
        de = lever[a]["expr_Z"].mean(0) - ref
        rands = []
        for r in ("R1", "R2", "R3"):
            rd = a5 / f"{r}_L14_a{a}"
            if (rd / "signatures_v3").exists():
                rands.append(corpus(rd, f"{r}@{a}").Z.mean(0) - ref)
        null_cos = [round(cos(dp, dr), 4) for dr in rands]

        def deco(v: np.ndarray) -> dict:
            proj = float(v @ dir0)
            resid = v - proj * dir0
            return {"dir0_projection": round(proj, 4), "offdir0_norm": round(float(np.linalg.norm(resid)), 4)}

        dp_r = dp - float(dp @ dir0) * dir0
        ds_r = ds - float(ds @ dir0) * dir0
        de_r = de - float(de @ dir0) * dir0
        return {
            "grade": gname, "alpha": a,
            "cos_prompt_vs_state": round(cos(dp, ds), 4),
            "cos_prompt_vs_expr": round(cos(dp, de), 4),
            "cos_prompt_vs_random_null": null_cos,
            "offdir0_residual_cos_state": round(cos(dp_r, ds_r), 4),
            "offdir0_residual_cos_expr": round(cos(dp_r, de_r), 4),
            "prompt": deco(dp), "lever_state": deco(ds), "lever_expr": deco(de),
        }

    profile_rows = []
    for a, m in matches.items():
        profile_rows.append(profile_row(m["by_frac"], a))
        if m["by_markers"] != m["by_frac"]:
            profile_rows.append(profile_row(m["by_markers"], a) | {"matched_by": "markers"})

    out = {
        "arm": "D3 — prompt-vs-lever behavioral equivalence at matched rate (14q item 2)",
        "STATUS": "FIRST_READ_PENDING (C§8)",
        "model": args.model,
        "filed_P": {"Pd3_profiles_remain_distinguishable": 0.70},
        "law": ("common-instrument matching (frac on no-inject sigs + text markers; injected "
                "frac NEVER used for matching, reported state-side only); profiles = centroid "
                "deltas in floor-z, readout-vector×dose frame (never per-facet); two-column "
                "readout (state + expression rows, both); dir0 = A2-pures LDA (gg-identical)"),
        "pre_worded_outcomes": {
            "distinct": "prompt and lever remain profile-distinguishable at matched rate ⇒ "
                        "'not an expensive prompt' quotable (outer loop words it)",
            "indistinguishable": "lever uniqueness claim scoped to state-side (R9) only",
        },
        "rate_table": {
            "grades": {g: {k: v for k, v in row.items() if k != "Z"} for g, row in grades.items()},
            "lever": {a: {k: v for k, v in row.items() if not k.endswith("_Z")}
                      for a, row in lever.items()},
        },
        "matches": matches,
        "profiles_at_matched_rate": profile_rows,
    }
    args.out_json.write_text(json.dumps(out, indent=1))
    print(json.dumps({"matches": matches}, indent=1))
    for r in profile_rows:
        print(f"g={r['grade']} α={r['alpha']}: cos(prompt,state)={r['cos_prompt_vs_state']} "
              f"cos(prompt,expr)={r['cos_prompt_vs_expr']} null={r['cos_prompt_vs_random_null']}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
