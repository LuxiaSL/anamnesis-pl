"""A8 Leg-4 / L4-a — expression-coupling regression (CPU; frozen bar P8-L4a .60).

The computational-mechanics frame's entry test: g is fit on paired OBSERVATIONS, so it
can only align the observation-coupled subspace; crypticity (state structure that the
text does not reveal) is the obstruction to transport.  Operationally, per banked
axis-instance:

    coupling  := |corr| over the pair's own fit-corpus texts between
                 (a) the per-text SOURCE-side state projection onto the axis
                     (native arm, source anchor site, banked mean-state bank), and
                 (b) that axis's TEXT-OBSERVABLE functional (defined in FUNCTIONALS
                     below, stamped BEFORE any coupling is computed).
    fidelity  := the axis's banked transport read (primary native proc_k512 cos).

Then Spearman(coupling, fidelity) over the pooled instance table (add-2: the bar binds
the single POOLED regression; no per-pair split scoring).

Discipline (baton §L4-a):
  * phase order is enforced by --phase: `functionals` writes the definition stamp;
    `couplings` computes couplings (and refuses to run without that stamp);
    `join` merges the banked fidelity table in LAST.  No functional definition is ever
    revised after a coupling or a fidelity value has been seen (the definitions are
    fixed by each axis's CONSTRUCTION semantics, not tuned).
  * an axis with no honest text observable is EXCLUDED with a named reason in the
    stamp (pre-ruled fork); n must stay >= 10 or the leg parks.

Everything UNSTAMPED (C§8).  No P is scored here — the desk scores.

Usage:
  python -m anamnesis.scripts.a8_leg4_coupling --phase functionals
  python -m anamnesis.scripts.a8_leg4_coupling --phase couplings
  python -m anamnesis.scripts.a8_leg4_coupling --phase join
  python -m anamnesis.scripts.a8_leg4_coupling --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anamnesis.scripts.a8_rosetta import load_axes, _gs, _unit  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARM = Path("outputs/battery/arms/A8_conjugation")
OUT = ARM / "leg4" / "readouts_cpu"
WORD_RE = re.compile(r"[A-Za-z']+")

# --------------------------------------------------------------------- lexicons
HEDGE = ("perhaps", "maybe", "might", "may", "could", "possibly", "arguably",
         "seems", "seem", "appears", "appear", "suggests", "suggest", "likely",
         "unlikely", "presumably", "roughly", "somewhat", "often", "sometimes",
         "generally", "tend", "tends", "unclear", "uncertain", "probably")
ANALOGICAL = ("like", "as", "similar", "similarly", "analogous", "analogy",
              "resembles", "resemble", "imagine", "picture", "metaphor",
              "akin", "just", "mirrors", "mirror", "echoes")
CONTRASTIVE = ("however", "whereas", "unlike", "contrast", "conversely",
               "although", "though", "but", "yet", "instead", "rather",
               "differs", "differ", "difference", "opposed", "versus")


# ------------------------------------------------------------------ functionals
def _words(text: str) -> list[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def f_lex_entropy(words: list[str], ctx: dict) -> float:
    """Normalized Shannon entropy of the text's own word distribution (bits / max)."""
    if len(words) < 2:
        return float("nan")
    c = Counter(words)
    p = np.array(list(c.values()), dtype=float) / len(words)
    h = float(-(p * np.log2(p)).sum())
    return h / np.log2(len(words))


def f_rep_rate(words: list[str], ctx: dict) -> float:
    """1 - distinct-4gram ratio (repetition mass, text-observable)."""
    n = 4
    if len(words) < n + 1:
        return float("nan")
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def f_hedge_rate(words: list[str], ctx: dict) -> float:
    """Hedge/uncertainty markers per 100 words (margin/confidence proxy)."""
    if not words:
        return float("nan")
    return 100.0 * sum(w in HEDGE for w in words) / len(words)


def f_rare_rate(words: list[str], ctx: dict) -> float:
    """Fraction of word tokens outside the corpus-wide top-1000 words (temp proxy)."""
    if not words:
        return float("nan")
    top = ctx["top1000"]
    return sum(w not in top for w in words) / len(words)


def f_mode_marker_diff(words: list[str], ctx: dict) -> float:
    """(analogical - contrastive) marker rate per 100 words (dir0 = analogical-contrastive)."""
    if not words:
        return float("nan")
    a = sum(w in ANALOGICAL for w in words)
    c = sum(w in CONTRASTIVE for w in words)
    return 100.0 * (a - c) / len(words)


def f_completion_len(words: list[str], ctx: dict) -> float:
    """Text length in words — the termination proxy (censored on the capped corpus)."""
    return float(len(words))


FUNCTIONALS = {
    "lex_entropy": {
        "fn": f_lex_entropy, "axes": ["V7"],
        "definition": "normalized Shannon entropy of the text's empirical word "
                      "distribution: H(words)/log2(n_words); regex words [A-Za-z']+, lowercased",
        "rationale": "V7 = entropy-gradient dial; its text-observable shadow is lexical "
                     "unpredictability of the emitted stream",
    },
    "rep_rate": {
        "fn": f_rep_rate, "axes": ["Vrep_perp"],
        "definition": "1 - (distinct 4-grams / total 4-grams) over word tokens",
        "rationale": "Vrep = repetition-mass dial; repetition is directly text-observable",
    },
    "hedge_rate": {
        "fn": f_hedge_rate, "axes": ["Vconf"],
        "definition": f"markers per 100 words from the stamped hedge lexicon ({len(HEDGE)} terms)",
        "rationale": "Vconf = margin/confidence dial; the honest text shadow of low margin "
                     "is hedged/uncertain phrasing. PROXY-GRADE: weakest of the set — a "
                     "logit margin is not a surface feature (flagged, not excluded)",
    },
    "rare_rate": {
        "fn": f_rare_rate, "axes": ["Vtemp"],
        "definition": "fraction of word tokens outside the corpus-wide top-1000 word list "
                      "(list built per corpus, from that corpus's own texts)",
        "rationale": "V_temp = temperature dial; hot sampling reaches further into the tail",
    },
    "mode_marker_diff": {
        "fn": f_mode_marker_diff, "axes": ["dir0"],
        "definition": "(analogical markers - contrastive markers) per 100 words from the "
                      f"stamped lexicons ({len(ANALOGICAL)}/{len(CONTRASTIVE)} terms)",
        "rationale": "dir0 = analogical - contrastive (V3 route-1 recipe order); the mode "
                     "statistic is the marker-rate difference in the same contrast direction",
    },
    "completion_len": {
        "fn": f_completion_len, "axes": ["Veos_perp"],
        "definition": "n word tokens of the text",
        "rationale": "eos-perp = stopping/termination dial; the text observable of a "
                     "termination decision is where the text stops. MEASURED ON THE CAPPED "
                     "CORPUS by design (desk interim note 2: 620/780 texts are cap-censored) "
                     "— its predicted place is the bottom of the coupling axis",
    },
}

EXCLUSIONS = {
    "oblique": "no single text-observable functional exists for a constructed oblique "
               "(unit(V7 + Vrep_raw)); any composite would be an improvised definition "
               "authored tonight — excluded per the pre-ruled fork rather than invented",
}

# ------------------------------------------------------------------- instances
# (pair, source model, corpus path, states path, source site, fits-family label)
PAIRS = {
    "3b->8b": {"src": "3b", "corpus": ARM / "corpus/corpus_manifest.json",
               "states": ARM / "states/states_3b_native.npz", "site": "L14",
               "rung": "same-family + tokenizer"},
    "8b->qwen": {"src": "8b", "corpus": ARM / "leg1/corpus/corpus_manifest.json",
                 "states": ARM / "leg1/states/states_8b_native.npz", "site": "L16",
                 "rung": "cross-family dense"},
    "8b->dsv2": {"src": "8b", "corpus": ARM / "leg2/corpus/corpus_manifest.json",
                 "states": ARM / "leg2/states/states_8b_native.npz", "site": "L16",
                 "rung": "dense -> MoE"},
}

# axis -> functional key
AXIS_FUNCTIONAL = {"V7": "lex_entropy", "Vrep_perp": "rep_rate", "Vconf": "hedge_rate",
                   "Vtemp": "rare_rate", "dir0": "mode_marker_diff",
                   "Veos_perp": "completion_len"}

# mode-free instances: same axis vector, coupling restricted to the mode-free strata
# (S1+S2 — the rows the mode-free g was fit on), fidelity = the mode-free banked read.
MODEFREE_STRATA = ("S1", "S2")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------- phase 1: stamp
def phase_functionals() -> dict:
    stamp = {
        "STATUS": "UNSTAMPED (C§8)",
        "leg": "A8 Leg-4 / L4-a",
        "prereg": "A8-add-2 P8-L4a (bar binds the pooled Spearman)",
        "authored": "phase 1 of 3 — written BEFORE any coupling or fidelity join "
                    "(baton no-peeking clause). The banked fidelity numbers are the arm's "
                    "public record (Leg-0/1/3 first-reads); what this stamp fixes is that "
                    "the functional DEFINITIONS were chosen from each axis's construction "
                    "semantics and are not revised after any coupling is seen.",
        "word_tokenizer": "regex [A-Za-z']+ lowercased (model-independent, so the same "
                          "definition applies to every pair's corpus); digits are dropped "
                          "by construction",
        "known_confound": "every functional except completion_len is a length-normalized "
                          "rate (lex_entropy is H/log2(n_words)); completion_len IS length "
                          "by definition. Length structure therefore enters the coupling "
                          "column differently for eos-perp than for the others — filed at "
                          "definition time, not discovered after the fact",
        "functionals": {k: {"definition": v["definition"], "rationale": v["rationale"],
                            "axes": v["axes"]} for k, v in FUNCTIONALS.items()},
        "exclusions": EXCLUSIONS,
        "coupling_metric": "primary |Pearson r| between per-text projection (source-side, "
                           "native arm, source anchor site, banked mean-state) and the "
                           "functional, over ALL rows of that pair's fit corpus; "
                           "|Spearman rho| filed beside (skewed functionals)",
        "beside_columns": {
            "eos_generated_only": "completion_len coupling restricted to model-generated "
                                  "strata (S1+S3) — S2 lengths are CHUNKER cuts, not "
                                  "termination decisions (filed beside, primary stays "
                                  "the uniform all-rows read)",
        },
        "lexicons": {"hedge": list(HEDGE), "analogical": list(ANALOGICAL),
                     "contrastive": list(CONTRASTIVE)},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "l4a_functional_definitions.json"
    p.write_text(json.dumps(stamp, indent=1))
    logger.info("functional definitions stamped -> %s", p)
    return stamp


# ----------------------------------------------------------- phase 2: couplings
def _corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(ok.sum())
    if n < 10 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), float("nan"), n
    pear = float(np.corrcoef(x, y)[0, 1])
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    spear = float(np.corrcoef(rx, ry)[0, 1])
    return pear, spear, n


def _source_axes(model: str) -> dict[str, np.ndarray]:
    reads, extras, _ = load_axes(model)
    ax = {k: v.vec for k, v in reads.items()}
    if "Veos_perp" in extras:
        ax["Veos_perp"] = extras["Veos_perp"].vec
    else:  # 3B has no banked Veos_perp — construct as the anatomy read did
        ax["Veos_perp"] = _gs(extras["Veos_raw"].vec, reads["V7"].vec)
    return ax


def phase_couplings(corpus_override: dict | None = None) -> dict:
    defs_path = OUT / "l4a_functional_definitions.json"
    if not defs_path.exists():
        raise SystemExit("phase order violated: run --phase functionals first")
    rows = []
    pairs = corpus_override or PAIRS
    for pair, cfg in pairs.items():
        entries = json.loads(Path(cfg["corpus"]).read_text())["entries"]
        z = np.load(cfg["states"])
        ids = list(z["text_ids"])
        states = z[cfg["site"]].astype(np.float64)
        by_id = {t: i for i, t in enumerate(ids)}
        # corpus rows aligned to the state bank order
        idx = [by_id[e["text_id"]] for e in entries]
        S = states[idx]
        words = [_words(e["text"]) for e in entries]
        strata = np.array([e["stratum"] for e in entries])
        top1000 = {w for w, _ in Counter(w for ws in words for w in ws).most_common(1000)}
        ctx = {"top1000": top1000}
        fvals = {k: np.array([spec["fn"](ws, ctx) for ws in words])
                 for k, spec in FUNCTIONALS.items()}
        axes = _source_axes(cfg["src"])
        for axis, fkey in AXIS_FUNCTIONAL.items():
            if axis not in axes:
                continue
            proj = S @ axes[axis]
            pear, spear, n = _corr(proj, fvals[fkey])
            rows.append({"pair": pair, "rung": cfg["rung"], "axis": axis,
                         "variant": "primary", "functional": fkey,
                         "coupling_pearson_abs": abs(pear), "coupling_spearman_abs": abs(spear),
                         "corr_signed_pearson": pear, "n_texts": n,
                         "source_model": cfg["src"], "site": cfg["site"]})
            # mode-free variant for the needle (the add-2 clause names this pair)
            if axis == "dir0":
                m = np.isin(strata, MODEFREE_STRATA)
                pe, sp, nn = _corr(proj[m], fvals[fkey][m])
                rows.append({"pair": pair, "rung": cfg["rung"], "axis": "dir0_modefree",
                             "variant": "mode-free (S1+S2 rows only — the mode-free g's "
                                        "own fit universe)", "functional": fkey,
                             "coupling_pearson_abs": abs(pe), "coupling_spearman_abs": abs(sp),
                             "corr_signed_pearson": pe, "n_texts": nn,
                             "source_model": cfg["src"], "site": cfg["site"]})
            if axis == "Veos_perp":
                m = np.isin(strata, ("S1", "S3"))
                pe, sp, nn = _corr(proj[m], fvals[fkey][m])
                rows.append({"pair": pair, "rung": cfg["rung"], "axis": "Veos_perp",
                             "variant": "BESIDE: model-generated strata only (S1+S3)",
                             "functional": fkey, "coupling_pearson_abs": abs(pe),
                             "coupling_spearman_abs": abs(sp), "corr_signed_pearson": pe,
                             "n_texts": nn, "source_model": cfg["src"], "site": cfg["site"]})
    out = {"STATUS": "UNSTAMPED (C§8)", "phase": "2/3 couplings",
           "definitions": str(defs_path), "rows": rows}
    p = OUT / "l4a_couplings.json"
    p.write_text(json.dumps(out, indent=1))
    logger.info("couplings -> %s (%d rows)", p, len(rows))
    return out


# ---------------------------------------------------------------- phase 3: join
def _banked_fidelity() -> dict:
    """Banked primary transport reads (native proc_k512, anchor pair), from the
    readout jsons — read ONLY in this phase."""
    fid: dict[tuple[str, str], dict] = {}

    def _from_rosetta(path: Path, pair: str, anchor: str, variant: str) -> None:
        d = json.loads(path.read_text())
        for row in d.get("axis", []):
            if (row.get("site_pair") == anchor and row.get("arm") == "native"
                    and row.get("family") == "proc_k512"):
                fid[(pair, row["axis"] + variant)] = {
                    "fidelity": float(row["cos_fwd"]), "source": f"{path}::axis[{anchor}]",
                    "envelope_q95": row.get("null_q95_fwd"),
                    "top_pc": row.get("top_pc_max_fwd")}

    _from_rosetta(ARM / "readouts/rosetta_readout.json", "3b->8b", "3bL14->8bL16", "")
    _from_rosetta(ARM / "readouts_modefree/rosetta_readout.json", "3b->8b",
                  "3bL14->8bL16", "_modefree")
    _from_rosetta(ARM / "leg1/readouts/rosetta_readout.json", "8b->qwen",
                  "8bL16->qwen-7bL21", "")
    _from_rosetta(ARM / "leg1/readouts_modefree/rosetta_readout.json", "8b->qwen",
                  "8bL16->qwen-7bL21", "_modefree")
    # Leg-2 (DSV2) has no full rosetta pass — its reads are the prep/curve-row artifacts
    cur = ARM / "leg3/readouts_cpu/dsv2_curverows.json"
    if cur.exists():
        d = json.loads(cur.read_text())
        f, n = d["fields_series"], d["needle_series"]
        for axis, val, note in (
                ("V7", f["a_hat_V7_L22_primary"], "fields_series::a_hat_V7_L22_primary"),
                ("Vtemp", f["Vtemp_L18_primary"],
                 "fields_series::Vtemp_L18_primary (SITE NOTE: L18, not the L22 anchor "
                 "— the banked dsv2 Vtemp target lives at L18)"),
                ("dir0", n["dir0_primary"], "needle_series::dir0_primary (target V3_L18)"),
                ("dir0_modefree", n["dir0_modefree"], "needle_series::dir0_modefree")):
            fid[("8b->dsv2", axis)] = {"fidelity": float(val),
                                       "source": f"{cur}::{note}"}
    # eos-perp fidelity = the anatomy-read commutation cos (same object shape as an
    # axis read: cos(g.Veos_perp_src, Veos_perp_tgt))
    ana = ARM / "leg3/readouts_cpu/veos_anatomy.json"
    if ana.exists():
        d = json.loads(ana.read_text())
        for anat_pair, pair in (("3b->8b", "3b->8b"), ("8b->qwen-7b", "8b->qwen")):
            row = d["commutation_rederived"].get(anat_pair, {}).get("Veos_perp")
            if row:
                fid[(pair, "Veos_perp")] = {
                    "fidelity": float(row["cos_gv_vs_target_own"]),
                    "source": f"{ana}::commutation_rederived[{anat_pair}][Veos_perp] "
                              "(same object shape as an axis read: cos(g·v_src, v_tgt))",
                    "envelope_q95": row.get("null_q95_abs")}
    return fid


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def phase_join(fidelity_manual: dict | None = None) -> dict:
    cj = json.loads((OUT / "l4a_couplings.json").read_text())
    fid = _banked_fidelity()
    if fidelity_manual:
        fid.update({tuple(k.split("|")): v for k, v in fidelity_manual.items()})
    joined, unmatched = [], []
    for r in cj["rows"]:
        if r["variant"].startswith("BESIDE"):
            continue
        key = (r["pair"], r["axis"])
        if key not in fid:
            unmatched.append(f"{r['pair']}::{r['axis']}")
            continue
        joined.append({**r, **fid[key]})
    x = np.array([r["coupling_pearson_abs"] for r in joined])
    y = np.array([r["fidelity"] for r in joined])
    xs = np.array([r["coupling_spearman_abs"] for r in joined])
    res = {
        "STATUS": "UNSTAMPED (C§8) — mechanics only, no P scored (desk scores P8-L4a)",
        "phase": "3/3 join",
        "n_instances": len(joined),
        "pooled_spearman_coupling_vs_fidelity": round(_spearman(x, y), 4) if len(joined) >= 3 else None,
        "pooled_spearman_using_spearman_coupling": round(_spearman(xs, y), 4) if len(joined) >= 3 else None,
        "pooled_pearson": round(float(np.corrcoef(x, y)[0, 1]), 4) if len(joined) >= 3 else None,
        "bar_reads_on": "pooled Spearman (add-2: single pooled regression, no per-pair split)",
        "eos_rank": None, "unmatched_instances": unmatched,
        "rows": sorted(joined, key=lambda r: -r["coupling_pearson_abs"]),
        "beside_rows": [r for r in cj["rows"] if r["variant"].startswith("BESIDE")],
    }
    order = sorted(joined, key=lambda r: r["coupling_pearson_abs"])
    res["eos_rank"] = {r["pair"] + "::" + r["axis"]: i + 1 for i, r in enumerate(order)
                       if r["axis"] == "Veos_perp"}
    res["dir0_pair_ordering"] = [
        {"pair": p,
         "mode_aware": next((r for r in joined if r["pair"] == p and r["axis"] == "dir0"), None),
         "mode_free": next((r for r in joined if r["pair"] == p and r["axis"] == "dir0_modefree"), None)}
        for p in sorted({r["pair"] for r in joined})]
    # ---- pre-declared beside variants (NON-SCORING; the bar binds the primary above)
    besides = {}
    # (a) eos-perp measured on model-generated strata only (the stamped beside column):
    #     S2 lengths are chunker cuts, so the primary all-rows completion_len coupling
    #     carries wikitext chunk-length variance that is not termination behaviour.
    swap = {(r["pair"], r["axis"]): r for r in cj["rows"] if r["variant"].startswith("BESIDE")}
    rows_b = [dict(r) for r in joined]
    for r in rows_b:
        k = (r["pair"], r["axis"])
        if k in swap:
            r["coupling_pearson_abs"] = swap[k]["coupling_pearson_abs"]
            r["coupling_spearman_abs"] = swap[k]["coupling_spearman_abs"]
            r["variant"] = swap[k]["variant"]
    xb = np.array([r["coupling_pearson_abs"] for r in rows_b])
    yb = np.array([r["fidelity"] for r in rows_b])
    besides["eos_generated_strata_only"] = {
        "pooled_spearman": round(_spearman(xb, yb), 4),
        "what_changed": "the two/three eos-perp rows use the S1+S3 coupling; all other "
                        "rows identical. Pre-declared at definition time as a beside.",
    }
    # (b) within-pair Spearman (diagnostic only — add-2 forbids per-pair split SCORING,
    #     this is filed because pooling mixes pairs with different attenuation constants)
    besides["within_pair_spearman_diagnostic"] = {}
    for pair in sorted({r["pair"] for r in joined}):
        rs = [r for r in joined if r["pair"] == pair]
        if len(rs) >= 4:
            besides["within_pair_spearman_diagnostic"][pair] = {
                "n": len(rs),
                "spearman": round(_spearman(
                    np.array([r["coupling_pearson_abs"] for r in rs]),
                    np.array([r["fidelity"] for r in rs])), 4)}
    besides["within_pair_note"] = ("rake-14: fidelity scales differ per pair (â .514 / "
                                   ".332 / .266) and the needle class is not on the "
                                   "fields' attenuation scale — pooling across pairs mixes "
                                   "those. NO ÷â normalisation is applied anywhere here "
                                   "(rake-14 forbids it for needle rows).")
    res["beside_variants"] = besides

    # ---- the add-2 clause facts, stated mechanically (the desk scores)
    order = sorted(joined, key=lambda r: r["coupling_pearson_abs"])
    n = len(order)
    res["add2_clause_facts"] = {
        "clause_1_pooled_spearman_ge_0.7": {
            "value": res["pooled_spearman_coupling_vs_fidelity"],
            "primary_definition": "|Pearson| coupling, all corpus rows"},
        "clause_2_eos_in_bottom_two_couplings": {
            "eos_ranks_1_is_lowest": {f"{r['pair']}::{r['axis']}": i + 1
                                      for i, r in enumerate(order)
                                      if r["axis"] == "Veos_perp"},
            "n_instances": n,
            "under_eos_beside_column": {
                f"{r['pair']}::{r['axis']}": i + 1
                for i, r in enumerate(sorted(rows_b,
                                             key=lambda r: r["coupling_pearson_abs"]))
                if r["axis"] == "Veos_perp"}},
        "clause_3_dir0_modefree_pair_consistent": [
            {"pair": p,
             "coupling_mode_aware": next(r["coupling_pearson_abs"] for r in joined
                                         if r["pair"] == p and r["axis"] == "dir0"),
             "coupling_mode_free": next(r["coupling_pearson_abs"] for r in joined
                                        if r["pair"] == p and r["axis"] == "dir0_modefree"),
             "fidelity_mode_aware": next(r["fidelity"] for r in joined
                                         if r["pair"] == p and r["axis"] == "dir0"),
             "fidelity_mode_free": next(r["fidelity"] for r in joined
                                        if r["pair"] == p and r["axis"] == "dir0_modefree"),
             "same_order": (next(r["coupling_pearson_abs"] for r in joined
                                 if r["pair"] == p and r["axis"] == "dir0")
                            > next(r["coupling_pearson_abs"] for r in joined
                                   if r["pair"] == p and r["axis"] == "dir0_modefree"))
                           == (next(r["fidelity"] for r in joined
                                    if r["pair"] == p and r["axis"] == "dir0")
                               > next(r["fidelity"] for r in joined
                                      if r["pair"] == p and r["axis"] == "dir0_modefree"))}
            for p in sorted({r["pair"] for r in joined}
                            & {r["pair"] for r in joined if r["axis"] == "dir0_modefree"})],
    }

    p = OUT / "l4a_coupling_regression.json"
    p.write_text(json.dumps(res, indent=1))
    logger.info("join -> %s (n=%d, pooled spearman %s)", p, len(joined),
                res["pooled_spearman_coupling_vs_fidelity"])
    _write_md(res)
    _write_scatter(res)
    return res


def _write_md(res: dict) -> None:
    lines = ["# L4-a — expression-coupling vs transport fidelity (UNSTAMPED, C§8)", "",
             f"n instances = {res['n_instances']}; pooled Spearman = "
             f"**{res['pooled_spearman_coupling_vs_fidelity']}** "
             f"(Spearman-coupling variant {res['pooled_spearman_using_spearman_coupling']}; "
             f"Pearson {res['pooled_pearson']})", "",
             "| pair | rung | axis | functional | coupling \\|r\\| | coupling \\|rho\\| | fidelity | n |",
             "|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        lines.append(f"| {r['pair']} | {r['rung']} | {r['axis']} | {r['functional']} | "
                     f"{r['coupling_pearson_abs']:.3f} | {r['coupling_spearman_abs']:.3f} | "
                     f"{r['fidelity']:+.3f} | {r['n_texts']} |")
    lines += ["", "## Beside rows (not in the regression)", "",
              "| pair | axis | variant | coupling \\|r\\| | n |", "|---|---|---|---|---|"]
    for r in res["beside_rows"]:
        lines.append(f"| {r['pair']} | {r['axis']} | {r['variant']} | "
                     f"{r['coupling_pearson_abs']:.3f} | {r['n_texts']} |")
    (OUT / "l4a_coupling_regression.md").write_text("\n".join(lines) + "\n")


def _write_scatter(res: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("no scatter (%s)", exc)
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    marks = {"3b->8b": "o", "8b->qwen": "s", "8b->dsv2": "^"}
    for pair, m in marks.items():
        rs = [r for r in res["rows"] if r["pair"] == pair]
        if not rs:
            continue
        ax.scatter([r["coupling_pearson_abs"] for r in rs], [r["fidelity"] for r in rs],
                   marker=m, label=pair, s=60)
        for r in rs:
            ax.annotate(r["axis"], (r["coupling_pearson_abs"], r["fidelity"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="grey", lw=.5)
    ax.set_xlabel("expression coupling  |corr(state projection, text functional)|")
    ax.set_ylabel("transport fidelity  cos(g·v_src, v_tgt)")
    ax.set_title(f"A8 L4-a — coupling vs fidelity (n={res['n_instances']}, "
                 f"pooled Spearman {res['pooled_spearman_coupling_vs_fidelity']})\n"
                 "UNSTAMPED (C§8)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "l4a_coupling_scatter.png", dpi=140)
    logger.info("scatter -> %s", OUT / "l4a_coupling_scatter.png")


# ------------------------------------------------------------------- selftest
def selftest() -> int:
    ok = True
    ws = _words("The cat sat on the mat. The cat sat on the mat.")
    r = f_rep_rate(ws, {})
    ok &= r > 0.3
    print(f"[{'OK' if r > 0.3 else 'BAD'}] rep_rate on a doubled sentence = {r:.3f} (>0.3)")
    r2 = f_rep_rate(_words("alpha beta gamma delta epsilon zeta eta theta iota"), {})
    ok &= r2 == 0.0
    print(f"[{'OK' if r2 == 0 else 'BAD'}] rep_rate on distinct words = {r2}")
    import string
    vocab = [a + b for a in string.ascii_lowercase for b in string.ascii_lowercase][:50]
    h1 = f_lex_entropy(_words(" ".join(["word"] * 50)), {})
    h2 = f_lex_entropy(_words(" ".join(vocab)), {})
    ok &= h1 < h2
    print(f"[{'OK' if h1 < h2 else 'BAD'}] lex_entropy monotone: repeated {h1:.3f} < varied {h2:.3f}")
    hd = f_hedge_rate(_words("perhaps this might possibly be true"), {})
    ok &= hd > 0
    print(f"[{'OK' if hd > 0 else 'BAD'}] hedge_rate fires = {hd:.1f}/100w")
    md = f_mode_marker_diff(_words("similar analogous like imagine"), {})
    md2 = f_mode_marker_diff(_words("however whereas unlike instead"), {})
    ok &= md > 0 > md2
    print(f"[{'OK' if md > 0 > md2 else 'BAD'}] mode_marker_diff signs: {md:.1f} / {md2:.1f}")
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    p, s, n = _corr(x, 2 * x + 1e-9 * rng.normal(size=200))
    ok &= abs(p - 1) < 1e-3 and n == 200
    print(f"[{'OK' if abs(p - 1) < 1e-3 else 'BAD'}] corr positive control r={p:.4f} n={n}")
    p0, _, _ = _corr(x, rng.normal(size=200))
    ok &= abs(p0) < 0.25
    print(f"[{'OK' if abs(p0) < 0.25 else 'BAD'}] corr null control r={p0:.4f}")
    sp = _spearman(np.array([1., 2, 3, 4]), np.array([2., 4, 6, 9]))
    ok &= abs(sp - 1) < 1e-9
    print(f"[{'OK' if abs(sp - 1) < 1e-9 else 'BAD'}] spearman monotone = {sp}")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["functionals", "couplings", "join", "all"])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.phase in ("functionals", "all"):
        phase_functionals()
    if args.phase in ("couplings", "all"):
        phase_couplings()
    if args.phase in ("join", "all"):
        phase_join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
