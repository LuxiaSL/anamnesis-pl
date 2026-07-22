"""A8 Leg-4F / L4-d — termination-stratum rescue readout (CPU; frozen bar P8-L4d .55).

The desk's interim-note-2 diagnosis: the Leg-0/1 fit corpora are TERMINATION-CENSORED
(~90% of S1/S3 sit at the 512 cap), so eos-perp-relevant covariance was censored out of the
data g was fit on — which would explain why Veos-perp is the one axis whose commutation sits
AT NULL cross-family (.0558 vs q95 .0638) while its field siblings hold.

S5 (uncapped, natural terminations, 160/voice, 100% natural-EOS) was added to the fit.  This
reads:

  BAR (add-2 P8-L4d): Veos-perp commutation 8b->qwen rises from at-null (.056) to
                      > 2x its envelope q95 under the S5-augmented fit.
  FREE COLUMN (no bar): the full axis panel under all three fits — do the OTHER axes move
                      when S5 enters?
  L4-a RIDER (reported, unscored): eos-perp expression-coupling recomputed on the
                      S5-AUGMENTED corpus — the frame's cleanest internal check, since the
                      capped-corpus coupling was measured on censored data.

Three maps compared, all native proc_k512 at the 8bL16->qwen-7bL21 anchor:
  baseline   = Leg-1 fits/            (S1+S2+S3, 780 texts — the fit the .056 came from)
  s5aug      = leg4/fits_s5aug/       (S1+S2+S3+S5, 1100)
  s5aug_mf   = leg4/fits_s5aug_modefree/ (S1+S2+S5 — mode-free variant, rides free)

UNSTAMPED (C§8).  Mechanics only — the desk scores P8-L4d.

Run (repo root): PYTHONPATH=pipeline python -m anamnesis.scripts.a8_leg4_termination
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from anamnesis.scripts.a8_fit_g import load_transport_map
from anamnesis.scripts.a8_leg4_coupling import FUNCTIONALS, _corr, _words
from anamnesis.scripts.a8_rosetta import _unit, cos, load_axes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_leg4_termination")

ARM = Path("outputs/battery/arms/A8_conjugation")
LEG4 = ARM / "leg4"
OUT = LEG4 / "readouts_cpu"
ANCHOR = "fit_8bL16__qwen-7bL21_native_proc_k512.npz"
MAPS = {
    "baseline_leg1 (S1+S2+S3, 780)": ARM / "leg1/fits" / ANCHOR,
    "s5aug (S1+S2+S3+S5, 1100)": LEG4 / "fits_s5aug" / ANCHOR,
    "s5aug_modefree (S1+S2+S5)": LEG4 / "fits_s5aug_modefree" / ANCHOR,
}
PANEL = ("V7", "Vrep_perp", "Vconf", "Vtemp", "dir0")
EXTRAS = ("Veos_raw", "Veos_perp")
N_RANDOM = 100
SEED = 80
BAR_BASELINE = 0.0558


def axis_table() -> dict:
    src_reads, src_extras, src_pool = load_axes("8b")
    tgt_reads, tgt_extras, _ = load_axes("qwen-7b")
    src = {**{k: src_reads[k].vec for k in PANEL},
           **{k: src_extras[k].vec for k in EXTRAS}}
    tgt = {**{k: tgt_reads[k].vec for k in PANEL},
           **{k: tgt_extras[k].vec for k in EXTRAS}}
    rng = np.random.default_rng(SEED)
    randoms = rng.standard_normal((N_RANDOM, src["V7"].shape[0]))
    randoms /= np.linalg.norm(randoms, axis=1, keepdims=True)

    out = {}
    for label, path in MAPS.items():
        if not path.exists():
            out[label] = {"unavailable": str(path)}
            continue
        tm = load_transport_map(path)
        nulls = [_unit(tm.transport(r)) for r in randoms] + \
                [_unit(tm.transport(a.vec)) for a in src_pool]
        rows = {}
        for name, v in src.items():
            u = _unit(tm.transport(v))
            null_cos = np.array([abs(cos(n, tgt[name])) for n in nulls])
            q95 = float(np.quantile(null_cos, 0.95))
            c = cos(u, tgt[name])
            rows[name] = {"cos": round(c, 4), "null_q95_abs": round(q95, 4),
                          "ratio_to_q95": round(abs(c) / q95, 3) if q95 else None,
                          "exceeds_envelope": bool(abs(c) > q95)}
        out[label] = {"fit": str(path), "rows": rows}
    return out


def eos_coupling_on_s5() -> dict:
    """L4-a's eos-perp coupling recomputed on the S5-augmented corpus (rider, unscored)."""
    entries = json.loads((LEG4 / "corpus/corpus_manifest.json").read_text())["entries"]
    z = np.load(LEG4 / "states/states_8b_native.npz")
    ids = list(z["text_ids"])
    S = z["L16"].astype(np.float64)
    pos = {t: i for i, t in enumerate(ids)}
    idx = [pos[e["text_id"]] for e in entries]
    S = S[idx]
    words = [_words(e["text"]) for e in entries]
    strata = np.array([e["stratum"] for e in entries])
    top1000 = {w for w, _ in Counter(w for ws in words for w in ws).most_common(1000)}
    ctx = {"top1000": top1000}
    fn = FUNCTIONALS["completion_len"]["fn"]
    fvals = np.array([fn(ws, ctx) for ws in words])
    axes, extras, _ = load_axes("8b")
    proj = S @ extras["Veos_perp"].vec
    subsets = {
        "ALL rows (S1+S2+S3+S5, n=1100)": np.ones(len(entries), bool),
        "S5 only (uncensored, n=320)": strata == "S5",
        "model-generated incl. S5 (S1+S3+S5)": np.isin(strata, ("S1", "S3", "S5")),
        "capped only (S1+S3) — the Leg-0/1 condition": np.isin(strata, ("S1", "S3")),
    }
    out = {}
    for name, m in subsets.items():
        p, s, n = _corr(proj[m], fvals[m])
        out[name] = {"coupling_pearson_abs": round(abs(p), 4),
                     "coupling_spearman_abs": round(abs(s), 4), "n": n,
                     "length_sd_words": round(float(np.std(fvals[m])), 1)}
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tab = axis_table()
    res = {
        "STATUS": "UNSTAMPED (C§8) — mechanics only, desk scores P8-L4d",
        "leg": "A8 Leg-4F / L4-d", "prereg": "A8-add-2 P8-L4d",
        "gate_achieved": json.loads((LEG4 / "corpus/corpus_stamp.json").read_text())
        ["gate_add2"],
        "bar": {
            "form": "Veos_perp commutation 8b->qwen rises from at-null (.0558) to "
                    "> 2x its envelope q95 under the S5-augmented fit",
            "baseline_of_record": BAR_BASELINE,
        },
        "axis_table_all_fits": tab,
        "eos_coupling_rider": eos_coupling_on_s5(),
    }
    # the bar's own numbers, pulled out
    bar_rows = {}
    for label, blk in tab.items():
        if "rows" in blk:
            r = blk["rows"]["Veos_perp"]
            bar_rows[label] = {**r, "meets_2x_q95": bool(abs(r["cos"]) > 2 * r["null_q95_abs"])}
    res["bar_reads"] = bar_rows
    (OUT / "l4d_termination_rescue.json").write_text(json.dumps(res, indent=1))

    lines = ["# L4-d — termination-stratum rescue (UNSTAMPED, C§8)", "",
             "S5 gate achieved: 100% natural-EOS both voices, 160/voice, 0% at cap.", "",
             "## Veos-perp commutation (the bar)", "",
             "| fit | cos | null q95 | ratio | > 2x q95 |", "|---|---|---|---|---|"]
    for label, r in bar_rows.items():
        lines.append(f"| {label} | {r['cos']:+.4f} | {r['null_q95_abs']:.4f} | "
                     f"{r['ratio_to_q95']} | {'YES' if r['meets_2x_q95'] else 'no'} |")
    lines += ["", "## Full axis panel (free column — do the others move?)", "",
              "| axis | " + " | ".join(MAPS) + " |",
              "|---" * (len(MAPS) + 1) + "|"]
    for name in list(PANEL) + list(EXTRAS):
        cells = []
        for label in MAPS:
            blk = tab[label]
            cells.append(f"{blk['rows'][name]['cos']:+.4f}" if "rows" in blk else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## eos-perp expression coupling (L4-a rider, unscored)", "",
              "| corpus subset | coupling \\|r\\| | \\|rho\\| | n | length sd (words) |",
              "|---|---|---|---|---|"]
    for k, v in res["eos_coupling_rider"].items():
        lines.append(f"| {k} | {v['coupling_pearson_abs']:.3f} | "
                     f"{v['coupling_spearman_abs']:.3f} | {v['n']} | {v['length_sd_words']} |")
    (OUT / "l4d_termination_rescue.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
