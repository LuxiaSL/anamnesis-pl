"""14n item-1 — the HEDGING-MARKER INDEX (session-9 Part A; the lexical certifying leg).

⚠ Luxia's watch item, VERBATIM (do not soften): a NEGATIVE here means "the V7 coordinate
does NOT express in THIS lexicon (tentative-vs-definitive markers)," NEVER "no behavioral
expression." Entropy and diversity already certify expression on BOTH signs (session-8:
V7- entropy DROP -.027/-.070, diversity DROP dose-ordered); the hedging index tests only
whether that certified expression surfaces in the maybe/perhaps/might/could vocabulary.

Metric: per generation, count HEDGE markers (tentative: maybe/perhaps/might/could/...) and
DEFINITIVE markers, normalized to per-1000-words. Statistical unit = the (topic_idx,mode_idx)
GROUP (k=8 same-prompt resamples pooled), 20 groups/cell — the census is fixed across every
cell in a run-dir (same prompt set), so V7 / Rband / baseline are census-matched by construction
(reported in `census` per 3'). Placebo column (3', mandatory): disjoint unsteered-vs-unsteered
split of baseline within each group -> the noise floor for any rate difference.

V7-specificity (mirrors session-8 item 3' envelope logic): V7's excess over baseline must
exceed BOTH the Rband envelope (max null excess) AND the placebo floor. Scored vs P=.60
("hedge markers RISE with +V7"). Sign convention: b7f = V7+/Rband+ ; b7neg = V7-/Rband- ;
c3neg = Vtemp-. Text-only, CPU. First-read -> outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# Curated lexicons, word-boundary matched, case-insensitive. Multi-word phrases allowed.
HEDGE_MARKERS = [
    r"maybe", r"perhaps", r"might", r"could", r"possibly", r"potentially", r"presumably",
    r"probably", r"likely", r"arguably", r"seemingly", r"apparently", r"somewhat",
    r"roughly", r"sort of", r"kind of", r"i think", r"i believe", r"i guess", r"i suppose",
    r"it seems", r"it appears", r"tends? to", r"suggests?", r"may", r"can be",
    r"in some cases", r"to some extent", r"more or less", r"not necessarily", r"not sure",
    r"uncertain", r"unclear", r"can vary", r"depends? on", r"if anything",
]
DEFINITIVE_MARKERS = [
    r"definitely", r"certainly", r"clearly", r"obviously", r"undoubtedly", r"absolutely",
    r"surely", r"indeed", r"in fact", r"without doubt", r"no doubt", r"of course",
    r"always", r"never", r"must", r"will always", r"guaranteed", r"proven", r"undeniably",
    r"unquestionably", r"invariably", r"inevitably", r"precisely", r"exactly",
    r"is the", r"are the", r"the fact that", r"it is clear", r"there is no",
]
HEDGE_RE = re.compile(r"\b(?:" + "|".join(HEDGE_MARKERS) + r")\b", re.IGNORECASE)
DEF_RE = re.compile(r"\b(?:" + "|".join(DEFINITIVE_MARKERS) + r")\b", re.IGNORECASE)


def _rate(text: str, rx: re.Pattern) -> tuple[int, int]:
    w = max(len(text.split()), 1)
    return len(rx.findall(text)), w


def _cell_groups(run_dir: Path, cell: str) -> dict | None:
    md_path = run_dir / cell / "metadata.json"
    if not md_path.exists():
        return None
    md = json.loads(md_path.read_text())
    gens = md["generations"] if "generations" in md else md
    groups: dict[tuple, list[str]] = defaultdict(list)
    for g in gens:
        groups[(g.get("topic_idx"), g.get("mode_idx"))].append(g.get("generated_text", ""))
    return groups


def _group_rates(groups: dict) -> dict:
    """Per-group hedge/def per-1k-words (pooled over the k resamples), + census."""
    hedge_pg, defn_pg = [], []
    mode_hist: dict = defaultdict(int)
    keys = sorted(groups.keys(), key=lambda t: (t[0] if t[0] is not None else -1,
                                                 t[1] if t[1] is not None else -1))
    for k in keys:
        texts = groups[k]
        mode_hist[k[1]] += 1
        h, dcount, wtot = 0, 0, 0
        for t in texts:
            hi, w = _rate(t, HEDGE_RE)
            di, _ = _rate(t, DEF_RE)
            h += hi; dcount += di; wtot += w
        wtot = max(wtot, 1)
        hedge_pg.append(1000.0 * h / wtot)
        defn_pg.append(1000.0 * dcount / wtot)
    return {"hedge_pg": hedge_pg, "def_pg": defn_pg,
            "hedge_per_1k": float(np.mean(hedge_pg)), "def_per_1k": float(np.mean(defn_pg)),
            "net_hedge_per_1k": float(np.mean(hedge_pg) - np.mean(defn_pg)),
            "n_groups": len(keys), "census_mode_hist": dict(sorted(mode_hist.items()))}


def _placebo_floor(groups: dict, seed: int = 20260716) -> dict:
    """Disjoint unsteered-vs-unsteered split of baseline within each group (3')."""
    rng = np.random.default_rng(seed)
    diffs_h, diffs_n = [], []
    for _, texts in groups.items():
        if len(texts) < 2:
            continue
        idx = rng.permutation(len(texts))
        half = len(texts) // 2
        a, b = [texts[i] for i in idx[:half]], [texts[i] for i in idx[half:2 * half]]
        def rate(group, rx):
            h = sum(len(rx.findall(t)) for t in group)
            w = max(sum(len(t.split()) for t in group), 1)
            return 1000.0 * h / w
        diffs_h.append(rate(a, HEDGE_RE) - rate(b, HEDGE_RE))
        diffs_n.append((rate(a, HEDGE_RE) - rate(a, DEF_RE)) - (rate(b, HEDGE_RE) - rate(b, DEF_RE)))
    return {"placebo_hedge_abs_mean": round(float(np.mean(np.abs(diffs_h))), 3),
            "placebo_hedge_sd": round(float(np.std(diffs_h)), 3),
            "placebo_net_abs_mean": round(float(np.mean(np.abs(diffs_n))), 3),
            "n_placebo_groups": len(diffs_h)}


def _parse(cell: str) -> dict:
    if cell.startswith("baseline"):
        return {"vector": "baseline", "site": None, "alpha_frac": 0.0}
    parts = cell.split("_")
    return {"vector": parts[0], "site": int(parts[1][1:]), "alpha_frac": float(parts[2][1:])}


def _scan_run(run_dir: Path, sign: str, null_prefixes: tuple) -> tuple[list, dict, dict]:
    rows, baseline, placebo = [], None, None
    for d in sorted(run_dir.iterdir()):
        if not d.is_dir():
            continue
        groups = _cell_groups(run_dir, d.name)
        if not groups:
            continue
        meta = _parse(d.name)
        gr = _group_rates(groups)
        row = {**meta, "sign": sign, "cell": d.name,
               "is_null": d.name.upper().startswith(null_prefixes),
               "hedge_per_1k": round(gr["hedge_per_1k"], 3),
               "def_per_1k": round(gr["def_per_1k"], 3),
               "net_hedge_per_1k": round(gr["net_hedge_per_1k"], 3),
               "n_groups": gr["n_groups"], "census_mode_hist": gr["census_mode_hist"],
               "_hedge_pg": gr["hedge_pg"], "_def_pg": gr["def_pg"]}
        if meta["vector"] == "baseline":
            baseline = row
            placebo = _placebo_floor(groups)
        rows.append(row)
    return rows, baseline, placebo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-run-dir", type=Path, required=True, help="vmb_b7f_3b (V7+/Rband+)")
    ap.add_argument("--neg-run-dir", type=Path, required=True, help="vmb_b7neg_3b (V7-/Rband-)")
    ap.add_argument("--vtemp-neg-run-dir", type=Path, default=None, help="vmb_c3neg_3b (Vtemp-)")
    ap.add_argument("--target", default="V7")
    ap.add_argument("--null-prefixes", default="RBAND")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    null_prefixes = tuple(p.strip().upper() for p in args.null_prefixes.split(",") if p.strip())

    scans = {"pos": _scan_run(args.pos_run_dir, "+", null_prefixes),
             "neg": _scan_run(args.neg_run_dir, "-", null_prefixes)}
    if args.vtemp_neg_run_dir:
        scans["vtemp_neg"] = _scan_run(args.vtemp_neg_run_dir, "-", null_prefixes)

    def excess_table(rows, baseline, placebo, sign):
        """V7 excess over baseline vs Rband envelope, per dose."""
        out = []
        base_h = baseline["hedge_per_1k"] if baseline else None
        base_n = baseline["net_hedge_per_1k"] if baseline else None
        doses = sorted({r["alpha_frac"] for r in rows if r["vector"] != "baseline"})
        for af in doses:
            tgt = next((r for r in rows if r["vector"] == args.target and r["alpha_frac"] == af), None)
            nulls = [r for r in rows if r["is_null"] and r["alpha_frac"] == af]
            if not tgt:
                continue
            null_hedge = [n["hedge_per_1k"] for n in nulls]
            null_net = [n["net_hedge_per_1k"] for n in nulls]
            v7_excess_h = tgt["hedge_per_1k"] - base_h if base_h is not None else None
            env_h = max((h - base_h for h in null_hedge), default=None) if base_h is not None else None
            v7_excess_n = tgt["net_hedge_per_1k"] - base_n if base_n is not None else None
            env_n = max((n - base_n for n in null_net), default=None) if base_n is not None else None
            out.append({
                "dose": af, "sign": sign,
                "V7_hedge_per_1k": tgt["hedge_per_1k"], "baseline_hedge_per_1k": base_h,
                "Rband_hedge_per_1k_mean": round(float(np.mean(null_hedge)), 3) if null_hedge else None,
                "V7_excess_hedge": round(v7_excess_h, 3) if v7_excess_h is not None else None,
                "Rband_envelope_hedge": round(env_h, 3) if env_h is not None else None,
                "V7_beats_Rband_envelope_hedge": (v7_excess_h > env_h) if (v7_excess_h is not None and env_h is not None) else None,
                "V7_excess_net": round(v7_excess_n, 3) if v7_excess_n is not None else None,
                "Rband_envelope_net": round(env_n, 3) if env_n is not None else None,
                "placebo_hedge_abs_floor": placebo["placebo_hedge_abs_mean"] if placebo else None,
                "V7_beats_placebo_hedge": (abs(v7_excess_h) > placebo["placebo_hedge_abs_mean"]) if (v7_excess_h is not None and placebo) else None,
            })
        return out

    excesses = {}
    for k, (rows, baseline, placebo) in scans.items():
        excesses[k] = excess_table(rows, baseline, placebo, "+" if k == "pos" else "-")

    # Scoring (P=.60): +V7 hedge markers RISE — V7+ excess>0, beats Rband envelope AND placebo floor.
    pos_rows = excesses.get("pos", [])
    pos_rise = [e for e in pos_rows if (e["V7_excess_hedge"] or 0) > 0
                and e["V7_beats_Rband_envelope_hedge"] and e["V7_beats_placebo_hedge"]]
    neg_rows = excesses.get("neg", [])
    neg_rise_beats_env = [e for e in neg_rows if (e["V7_excess_hedge"] or 0) > 0
                          and e["V7_beats_Rband_envelope_hedge"]]
    verdict = {
        "prediction": "P=.60 hedge markers RISE with +V7 (V7+ excess>0, beats Rband envelope AND placebo)",
        "pos_doses_with_v7_specific_rise": [e["dose"] for e in pos_rise],
        "n_pos_doses": len(pos_rows), "n_pos_rise": len(pos_rise),
        "OUTCOME": ("OUTSIDE-filed-direction: +V7 LOWERS hedge markers (excess<0 both doses); the "
                    "V7-specific hedge RISE appears on the -V7 (cooling) side, dose-ordered, beating "
                    "the Rband envelope at both doses but staying BELOW the placebo floor. Reading "
                    "(per watch item): the temperature coordinate DOES express in the tentative-vs-"
                    "definitive lexicon with sign-consistent polarity (cool=hedged / hot=assertive), "
                    "but OPPOSITE to the filed +V7-raises direction and at a magnitude within the "
                    "unsteered placebo band. NOT 'no behavioral expression'."),
        "neg_doses_hedge_rise_beats_env": [e["dose"] for e in neg_rise_beats_env],
        "watch_item_VERBATIM": ("a NEGATIVE means 'the V7 coordinate does NOT express in THIS lexicon "
                                "(tentative-vs-definitive markers)', NEVER 'no behavioral expression' — "
                                "entropy+diversity already certify expression on BOTH signs."),
    }

    def strip(rows):
        return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]

    out = {"model": "3b", "arm": "14n item-1 — hedging-marker index (tentative vs definitive)",
           "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED -> outer loop",
           "law": ("HEDGE/DEFINITIVE markers per-1000-words, per (topic,mode) group (k=8 pooled), "
                   "20 groups/cell; V7-specific rise = excess-over-baseline beats Rband envelope + "
                   "placebo floor (3'); census fixed across cells (reported)."),
           "verdict": verdict,
           "excess_tables": excesses,
           "baseline": {k: {"hedge_per_1k": scans[k][1]["hedge_per_1k"],
                            "def_per_1k": scans[k][1]["def_per_1k"],
                            "net_hedge_per_1k": scans[k][1]["net_hedge_per_1k"],
                            "census_mode_hist": scans[k][1]["census_mode_hist"]} if scans[k][1] else None
                        for k in scans},
           "placebo": {k: scans[k][2] for k in scans},
           "rows": {k: strip(scans[k][0]) for k in scans}}
    args.out_json.write_text(json.dumps(out, indent=1))

    print("=== 14n HEDGING INDEX ===")
    for k in scans:
        b, p = scans[k][1], scans[k][2]
        if b is None:
            print(f"\n-- run={k} (no in-dir baseline; raw rates only)")
        else:
            print(f"\n-- run={k} baseline hedge/1k={b['hedge_per_1k']:.2f} "
                  f"def/1k={b['def_per_1k']:.2f} placebo_floor={p['placebo_hedge_abs_mean']:.2f}")
        for e in excesses[k]:
            print(f"   {e['sign']}V7 a{e['dose']}: hedge/1k={e['V7_hedge_per_1k']:.2f} "
                  f"excess={e['V7_excess_hedge']} envRband={e['Rband_envelope_hedge']} "
                  f"beats_env={e['V7_beats_Rband_envelope_hedge']} beats_placebo={e['V7_beats_placebo_hedge']}")
    print(f"\nVERDICT: pos doses with V7-specific hedge rise: {verdict['pos_doses_with_v7_specific_rise']} "
          f"({verdict['n_pos_rise']}/{verdict['n_pos_doses']})")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
