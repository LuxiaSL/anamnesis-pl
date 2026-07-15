"""ANNEX — A4: name the surviving axes. An axis we cannot name does not graduate, ever.

⚠ ANNEX LANE: nothing here is quotable until it graduates through a frozen prereg cell.
Ledger: outputs/battery/annex/ANNEX-LEDGER.md

Three lenses per axis:
  1. FAMILY/SOURCE decomposition — which substrate carries it. Reported as mass-per-family
     AND as mass-per-family-PER-FEATURE (enrichment vs the family's census share), because
     raw mass just re-reports family sizes: tier3 is 37% of the vector, so it "carries" 37%
     of a random axis. Enrichment is the number that means something.
  2. STRUCTURE COEFFICIENTS — corr(feature, axis score), NOT raw loadings. This is the
     2026-06-14 lesson, re-applied: naming by raw LDA/PCA weights inflates tiny-numeric-scale
     features and produced a FALSE "gate-sparsity dominates" headline once already. Do not
     repeat it.
  3. POLE TEXTS — top/bottom decile WITHIN topic-x-template cells, so the reader sees
     execution differences at matched content rather than "it's about consciousness."
  4. NUISANCE AUDIT — corr of the axis score with every known factor (length, cap, topic,
     template) so "not the nuisance in disguise" is a measured claim, not a hope.
  5. PULLBACK REALIZABILITY (venue, full space only) — how much of the axis lives in the
     pca_* block (linear functionals of the residual stream), and how much of THAT survives
     the t-constant projection. A constant injection at layer L moves every temporal sample
     identically, so only the t-constant component of the loadings is reachable. HYPOTHESIS-
     GRADE (see ledger): this prices the analytic-pullback idea on CPU before any GPU.

    PYTHONPATH=pipeline python -m anamnesis.scripts.annex_naming --corpus venue --variant cell
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

import numpy as np
from numpy.typing import NDArray

from anamnesis.scripts.annex_corpus import (
    REPO, VENUE_DIR, AnnexCorpus, load_power, load_venue, prepare,
)
from anamnesis.scripts.annex_spectrum import Spectrum, pca

F32 = NDArray[np.float32]
OUT = REPO / "outputs/battery/annex"
PCA_PAT = re.compile(r"^pca_L(\d+)_t(\d+)_c(\d+)$")


def family_decomposition(comp: F32, c: AnnexCorpus) -> list[dict]:
    """Mass per family + ENRICHMENT (mass share / census share). Enrichment ~1 means the
    family carries the axis exactly in proportion to how many features it contributed —
    i.e. carries nothing distinctive."""
    w = comp.astype(np.float64) ** 2
    total = max(w.sum(), 1e-30)
    rows = []
    for f in c.families:
        mass = float(w[f.start:f.end].sum() / total)
        census = f.size / c.d
        rows.append({"family": f.name, "n_features": f.size,
                     "mass_share": round(mass, 4), "census_share": round(census, 4),
                     "enrichment": round(mass / max(census, 1e-12), 2)})
    return sorted(rows, key=lambda r: -r["enrichment"])


def structure_coefficients(scores: NDArray, X: F32, names: list[str],
                           top: int = 15) -> list[dict]:
    """corr(feature, axis score) — the CORRECT naming lens (never raw loadings)."""
    s = (scores - scores.mean()) / max(scores.std(), 1e-12)
    Xc = X.astype(np.float64)
    Xc = (Xc - Xc.mean(0)) / np.maximum(Xc.std(0), 1e-12)
    r = (Xc.T @ s) / len(s)
    order = np.argsort(-np.abs(r))[:top]
    return [{"feature": names[i], "struct_coef": round(float(r[i]), 3)} for i in order]


def nuisance_audit(scores: NDArray, c: AnnexCorpus) -> dict:
    """|corr| with every known factor. Categorical factors use eta (corr ratio)."""
    s = np.asarray(scores, dtype=np.float64)
    out = {}
    for k, v in c.covariates.items():
        out[k] = round(abs(float(np.corrcoef(s, v.astype(np.float64))[0, 1])), 4)
    for k, v in c.factors.items():
        grand = s.mean()
        ss_b = sum(len(s[v == g]) * (s[v == g].mean() - grand) ** 2 for g in np.unique(v))
        ss_t = float(((s - grand) ** 2).sum())
        out[f"{k}_eta"] = round(float(np.sqrt(ss_b / max(ss_t, 1e-30))), 4)
    return out


def pole_texts(scores: NDArray, c: AnnexCorpus, n_per_pole: int = 4,
               chars: int = 320) -> dict:
    """Top/bottom decile WITHIN cells, so poles differ by execution, not by content."""
    md = json.loads((VENUE_DIR / "metadata.json").read_text())
    gm = {int(g["generation_id"]): g for g in md["generations"]}
    from anamnesis.analysis.battery.floors import load_signature_matrix
    _X, _n, gen_ids = load_signature_matrix(VENUE_DIR / "signatures_v3")

    hi, lo = [], []
    for cell in np.unique(c.cell):
        rows = np.where(c.cell == cell)[0]
        order = rows[np.argsort(scores[rows])]
        k = max(1, int(round(0.10 * len(rows))))
        lo.extend(order[:k].tolist())
        hi.extend(order[-k:].tolist())
    rng = np.random.default_rng(0)

    def sample(rows):
        pick = rng.choice(rows, size=min(n_per_pole, len(rows)), replace=False)
        out = []
        for r in pick:
            g = gm[int(gen_ids[r])]
            out.append({"gen_id": int(gen_ids[r]), "template": g["mode"],
                        "topic": g["topic"], "score": round(float(scores[r]), 3),
                        "text_head": g["generated_text"][:chars].replace("\n", " ")})
        return out

    return {"n_per_pole_pool": len(hi), "high": sample(hi), "low": sample(lo),
            "high_template_mix": dict(Counter(gm[int(gen_ids[r])]["mode"] for r in hi)),
            "low_template_mix": dict(Counter(gm[int(gen_ids[r])]["mode"] for r in lo))}


def pullback_realizability(comp: F32, names: list[str]) -> dict:
    """How much of the axis is analytically pullback-able to a residual direction?

    `pca_*` members are linear functionals of the residual stream, so their loadings pull back
    through the (banked, pooled) calibration basis exactly. But a CONSTANT injection at layer L
    moves every temporal sample t identically -> only the t-CONSTANT part of the loadings is
    reachable. Realizable mass = the t-mean component; the t-varying remainder is unreachable
    by constant injection. HYPOTHESIS-GRADE, and it prices the idea for free.
    """
    w = comp.astype(np.float64)
    tot = max(float((w ** 2).sum()), 1e-30)
    idx = [(i, PCA_PAT.match(n)) for i, n in enumerate(names)]
    idx = [(i, m) for i, m in idx if m]
    if not idx:
        return {"applicable": False, "reason": "no pca_* members in this space "
                                               "(shared-2108 excludes them by construction)"}
    pca_mass = float(sum(w[i] ** 2 for i, _ in idx) / tot)
    # group by (layer, component); average the loading over t -> the reachable part
    grp: dict[tuple[int, int], list[float]] = {}
    for i, m in idx:
        grp.setdefault((int(m.group(1)), int(m.group(3))), []).append(float(w[i]))
    t_const = sum(float(np.mean(v)) ** 2 * len(v) for v in grp.values())
    per_layer = {}
    for (L, _ci), v in grp.items():
        per_layer[L] = per_layer.get(L, 0.0) + float(np.mean(v)) ** 2 * len(v)
    return {
        "applicable": True,
        "mass_in_pca_block": round(pca_mass, 4),
        "realizable_mass_t_constant": round(float(t_const / tot), 4),
        "realizable_frac_of_pca_block": round(float(t_const / max(pca_mass * tot, 1e-30)), 4),
        "realizable_mass_by_layer": {f"L{L}": round(float(v / tot), 4)
                                     for L, v in sorted(per_layer.items())},
        "note": "realizable = t-constant component only; a constant injection at L cannot "
                "produce t-varying loadings. Whether the truncated pullback preserves the "
                "axis is UNTESTED. Writing remains an empirical claim — CPU builds, GPU judges.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["venue", "venue2108", "power", "venuecap"], default="venue")
    ap.add_argument("--partner", default=None)
    ap.add_argument("--variant", default="cell")
    ap.add_argument("--weighting", default="raw")
    ap.add_argument("--axes", default="1,2,3")
    ap.add_argument("--no-texts", action="store_true")
    args = ap.parse_args()

    c = (load_power(partner=args.partner) if args.corpus == "power"
         else load_venue(shared_2108=args.corpus == "venue2108",
                         capped_only=args.corpus == "venuecap"))
    Xp, df = prepare(c, args.variant)
    sp = pca(Xp, df)

    out = []
    for a in [int(x) for x in args.axes.split(",")]:
        j = a - 1
        comp, sc = sp.components[j], sp.scores[:, j]
        row = {
            "pc": a, "var_ratio": round(float(sp.var_ratio[j]), 4),
            "families": family_decomposition(comp, c),
            "structure_coefficients": structure_coefficients(sc, Xp, c.feature_names),
            "nuisance_audit": nuisance_audit(sc, c),
            "pullback": pullback_realizability(comp, c.feature_names),
        }
        if c.name == "venue" and not args.no_texts:
            row["poles"] = pole_texts(sc, c)
        out.append(row)

        print(f"\n{'='*78}\nPC{a}  (var {row['var_ratio']:.3f})  [{c.name}/{args.variant}]")
        print("  NUISANCE AUDIT:", row["nuisance_audit"])
        print("  FAMILIES (by enrichment = mass/census; ~1.0 = carries nothing distinctive):")
        for f in row["families"][:6]:
            print(f"    {f['family']:22s} n={f['n_features']:5d} mass={f['mass_share']:.3f} "
                  f"census={f['census_share']:.3f} ENRICH={f['enrichment']:.2f}")
        print("  TOP STRUCTURE COEFFICIENTS (corr(feature, score) — not raw loadings):")
        for s in row["structure_coefficients"][:8]:
            print(f"    {s['struct_coef']:+.3f}  {s['feature']}")
        if row["pullback"].get("applicable"):
            p = row["pullback"]
            print(f"  PULLBACK: pca_block mass {p['mass_in_pca_block']:.3f} -> "
                  f"t-constant realizable {p['realizable_mass_t_constant']:.3f} "
                  f"({p['realizable_frac_of_pca_block']:.1%} of the block)")
            print(f"            by layer: {p['realizable_mass_by_layer']}")
        if "poles" in row:
            print(f"  POLE TEMPLATE MIX  high={row['poles']['high_template_mix']}")
            print(f"                     low ={row['poles']['low_template_mix']}")

    OUT.mkdir(parents=True, exist_ok=True)
    stem = args.corpus + (f"_{args.partner}" if args.partner else "")
    p = OUT / f"annex_a4_naming_{stem}_{args.variant}.json"
    p.write_text(json.dumps({
        "ANNEX_RULE": "NOT QUOTABLE until graduated via a frozen prereg cell",
        "rung": "A4", "corpus": args.corpus, "variant": args.variant,
        "weighting": args.weighting, "n": c.n, "df": df, "axes": out}, indent=1))
    print(f"\n  → {p}")


if __name__ == "__main__":
    main()
