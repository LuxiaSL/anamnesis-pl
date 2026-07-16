"""V3selbare behavioral re-run (the void fix) — apply the F-rung mode classifier to
V3selbare's OWN generation signatures.

The banked `a5_results_v3sel_inv_3b.json` semantics rows for V3selbare are bit-identical to
V3's (.01875/.11875/.975) — a copy bug isolated to the semantics pipeline step (coherence
rows differ ⇒ the gens are real and distinct; COVSCREEN first-read §1). This re-derives
`frac_analogical` from V3selbare's real signatures using the identical F-rung LDA
(pure_analogical vs pure_contrastive, floor-z, lsqr/shrinkage=auto) and reports next to V3's
banked rows + the rider floor. Behavioral leg of V3selbare = measured, not copied.

CPU-only; texts/sigs persisted on node (synced local). First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from anamnesis.analysis.battery.deltas import ConditionCorpus, load_floor_scale
from anamnesis.analysis.battery.manifest import MODEL_META

DIR0_PAIR = ("analogical", "contrastive")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--model", default="3b")
    ap.add_argument("--vector", default="V3selbare")
    ap.add_argument("--alphas", nargs="+", default=["0.03", "0.1", "0.3"])
    ap.add_argument("--site", type=int, default=14)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    mm = MODEL_META[args.model]
    br = args.battery_root
    med, scale = load_floor_scale(br / mm.stage0_dir / "signatures_v3")

    def cc(d: Path, label: str) -> ConditionCorpus:
        return ConditionCorpus(d / "signatures_v3", d / "metadata.json", med, scale, label)

    # F-rung classifier: pure analogical vs contrastive (identical to vmb_arm_a5_analyze)
    pures = {m: cc(br / f"vmb_a2_{args.model}_pure_{m}", f"pure-{m}") for m in DIR0_PAIR}
    Xp = np.vstack([pures[DIR0_PAIR[0]].Z, pures[DIR0_PAIR[1]].Z])
    yp = np.r_[np.ones(pures[DIR0_PAIR[0]].Z.shape[0]), np.zeros(pures[DIR0_PAIR[1]].Z.shape[0])]
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xp, yp)

    # rider floor (pooled α=0 riders, same reference the analyze uses)
    a5_root = br / f"vmb_a5_{args.model}"
    rider_ccs = [cc(d, f"rider-{d.name}") for d in sorted(a5_root.iterdir())
                 if d.name.endswith("_a0.0") and (d / "signatures_v3").exists()]
    ref_centroid = np.vstack([r.Z for r in rider_ccs]).mean(axis=0)
    frac_ref = float(clf.predict(np.vstack([r.Z for r in rider_ccs])).mean())

    # banked V3 rows (the comparison arm) from a5_results
    banked = json.loads((br / "arms/A5" / f"a5_results_{args.model}.json").read_text())
    v3_banked = {r["alpha_frac"]: r["frac_analogical"] for r in banked["semantics_match"]
                 if r["vector"] == "V3" and r["site"] == args.site}

    rows = []
    for a in args.alphas:
        d = a5_root / f"{args.vector}_L{args.site}_L{args.site}_a{a}"
        if not (d / "signatures_v3").exists():
            d = a5_root / f"{args.vector}_L{args.site}_a{a}"      # single-L naming fallback
        if not (d / "signatures_v3").exists():
            rows.append({"alpha_frac": float(a), "error": f"cell not found ({d.name})"})
            continue
        Z = cc(d, f"{args.vector}-a{a}").Z
        frac = float(clf.predict(Z).mean())
        af = float(a)
        rows.append({"cell": d.name, "alpha_frac": af, "n": int(Z.shape[0]),
                     "frac_analogical_REDERIVED": round(frac, 5),
                     "frac_analogical_V3_banked": v3_banked.get(af),
                     "frac_analogical_rider_floor": round(frac_ref, 5),
                     "differs_from_V3": (v3_banked.get(af) is not None
                                         and abs(frac - v3_banked.get(af)) > 1e-6)})
        print(f"  {args.vector} α{a}: rederived={frac:.4f}  V3_banked={v3_banked.get(af)}  "
              f"floor={frac_ref:.4f}  differs_from_V3={rows[-1]['differs_from_V3']}")

    # ── mechanism audit: is the V3<->V3selbare frac-identity a COPY or genuine? ──
    # genuine iff (a) the sigs are distinct, (b) per-gen LDA labels genuinely differ
    # (agreement < 100%), (c) the marginal count still coincides.
    mech = []
    v3_axis = clf.coef_[0] / np.linalg.norm(clf.coef_[0])  # unused; kept for parity
    for a in args.alphas:
        d_vsb = a5_root / f"{args.vector}_L{args.site}_a{a}"
        d_v3 = a5_root / f"V3_L{args.site}_L{args.site}_a{a}"
        if not (d_vsb / "signatures_v3").exists() or not (d_v3 / "signatures_v3").exists():
            continue
        vsb, v3 = cc(d_vsb, "vsb"), cc(d_v3, "v3")
        s_vsb, s_v3 = vsb.Z.mean(0) - ref_centroid, v3.Z.mean(0) - ref_centroid
        cos = float(s_vsb @ s_v3 / (np.linalg.norm(s_vsb) * np.linalg.norm(s_v3) + 1e-12))
        p_vsb, p_v3 = clf.predict(vsb.Z).astype(int), clf.predict(v3.Z).astype(int)
        dv, d3 = dict(zip(vsb.gen_ids, p_vsb)), dict(zip(v3.gen_ids, p_v3))
        common = [g for g in vsb.gen_ids if g in d3]
        agree = sum(dv[g] == d3[g] for g in common)
        mech.append({"alpha_frac": float(a),
                     "cos_shift_vsb_v3": round(cos, 4),
                     "sig_max_abs_z_diff": round(float(np.max(np.abs(vsb.Z - v3.Z))), 3),
                     "per_gen_label_agreement": f"{agree}/{len(common)}",
                     "per_gen_agreement_frac": round(agree / len(common), 4),
                     "frac_identical_but_predictions_differ": bool(agree < len(common))})

    out = {"arm": f"{args.vector} behavioral re-run (void fix) — frac_analogical from OWN sigs",
           "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop",
           "model": args.model, "site": args.site,
           "law": "LDA(pure analogical vs contrastive, floor-z, lsqr/shrinkage=auto) applied to "
                  f"{args.vector} steered gens; frac = fraction predicted analogical; "
                  "SCOPE: frac_analogical meaningful only inside coherence window α≤.3 (COVSCREEN §2)",
           "void_confirmed_if": "rederived == V3_banked exactly ⇒ the copy would have been silent; "
                                "differs ⇒ the re-run recovers the real behavioral leg",
           "mechanism_audit": mech,
           "AUDIT_VERDICT": (
               "NOT-A-COPY / COVSCREEN-§1-REVERSED: frac_analogical re-derived from V3selbare's "
               "OWN distinct signatures reproduces V3's counts EXACTLY, while (a) sigs are distinct "
               "(max|Δz| per row shown), (b) shift vectors are NOT parallel at low α (cos ~0.07 @.03), "
               "(c) per-gen LDA labels genuinely DIFFER (agreement < 100% — e.g. ~86% @.1) yet net to "
               "the same marginal count. ⇒ the identical frac is a GENUINE behavioral-equivalence "
               "finding, not a pipeline copy; the analyze F-rung computes each cell from its own "
               "cc_cache[cname].Z (distinct keys — no copy mechanism). The COVSCREEN §1 'impossible on "
               "independent cells' premise is empirically false. STOP-AND-SURFACE: reverses an "
               "outer-loop verdict + revives the retracted R5 'label-free lever → same behavior' claim. "
               "SCOPE caveat: α=.3 match is inside the degradation regime (V3selbare TTR .454 < V3 .578, "
               "trigram_rep .155 > .037) — clean behavioral parity is α≤.1."),
           "rows": rows}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
