"""A7 FREE-GEN coherence-vs-dose readout (completes the .60 coherence-secondary row, already MIXED on TF).

The teacher-forced A7 ladder banked the spillover headline; this reads coherence in the model's OWN
sampled text under each MoE-perturbation rung. Per gen (word-based, tokenizer-free), degeneracy metrics:
  ttr          = unique / total words              (LOW  = degenerate)
  trigram_rep  = 1 - unique / total trigrams       (HIGH = repetitive collapse, e.g. "balena balena…")
  selfrep      = frac of words seen earlier in gen  (HIGH = looping)
  gen_len      = realized generated tokens
Aggregated per rung + length-residualized (regress each metric on gen_len pooled across rungs; report
per-rung mean residual so a rung looking degenerate purely by running longer/shorter is controlled).
Rungs are dose-ordered within family (topk 6->1, noise 0->1, ablate, drop) for the graceful-vs-cliff read.

M6 text is byte-BPE -> maybe_decode. Reads {root}/{rung}/*.json gen records. First-read -> outer loop; UNSTAMPED.

    python -m anamnesis.scripts.vmb_a7_freegen_coherence --root $RUNS/vmb_a7_dsv2_freegen \
        --out-json $BANK/arms/A7_dsv2/freegen/coherence_vs_dose.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from anamnesis.analysis.battery.text_decode import maybe_decode

# dose ordering within family (for the graceful-vs-cliff curve)
RUNG_ORDER = {
    "baseline": ("baseline", 0), "topk6": ("topk", 0), "topk4": ("topk", 1),
    "topk2": ("topk", 2), "topk1": ("topk", 3),
    "noise0.0": ("noise", 0), "noise0.25": ("noise", 1), "noise0.5": ("noise", 2), "noise1.0": ("noise", 3),
    "shared_ablate": ("ablate", 1), "routed_ablate": ("ablate", 2),
    "drop_topm2": ("drop", 1), "drop_randm2": ("drop", 1),
}


def gen_metrics(text: str, ntok: int) -> dict | None:
    words = text.split()
    if len(words) < 3:
        return None
    tri = [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]
    seen: set[str] = set()
    selfrep = 0
    for w in words:
        if w in seen:
            selfrep += 1
        seen.add(w)
    return {"ttr": len(set(words)) / len(words),
            "trigram_rep": 1.0 - len(set(tri)) / max(len(tri), 1),
            "selfrep": selfrep / len(words),
            "gen_len": float(ntok)}


def load_rung(rung_dir: Path) -> list[dict]:
    rows = []
    # gen_multicell writes to {rung}/gen_records/*.json; single-process run_gen_tokens to {rung}/*.json
    files = (glob.glob(str(rung_dir / "gen_records" / "*.json"))
             or glob.glob(str(rung_dir / "*.json")))
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(d, dict) or "generated_text" not in d:
            continue
        m = gen_metrics(maybe_decode(d["generated_text"]),
                        d.get("num_generated_tokens", len(d["generated_text"].split())))
        if m:
            rows.append(m)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="ladder run root ({root}/{rung}/*.json)")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    metrics = ("ttr", "trigram_rep", "selfrep", "gen_len")
    per_rung: dict[str, list[dict]] = {}
    for d in sorted(args.root.iterdir()):
        if d.is_dir() and (d.name in RUNG_ORDER):
            rows = load_rung(d)
            if rows:
                per_rung[d.name] = rows

    if not per_rung:
        raise SystemExit(f"no rung gens under {args.root}")

    # length-residualization: pooled OLS of each metric on gen_len (control length)
    all_len = np.concatenate([[r["gen_len"] for r in rows] for rows in per_rung.values()])
    resid_coef = {}
    for mk in ("ttr", "trigram_rep", "selfrep"):
        y = np.concatenate([[r[mk] for r in rows] for rows in per_rung.values()])
        A = np.vstack([all_len, np.ones_like(all_len)]).T
        b, m0 = np.linalg.lstsq(A, y, rcond=None)[0]
        resid_coef[mk] = (float(b), float(m0))

    summary = {}
    for rung, rows in per_rung.items():
        s = {"n": len(rows), "family": RUNG_ORDER[rung][0], "dose": RUNG_ORDER[rung][1]}
        for mk in metrics:
            v = np.array([r[mk] for r in rows], np.float64)
            s[mk] = round(float(v.mean()), 4)
        for mk in ("ttr", "trigram_rep", "selfrep"):
            b, m0 = resid_coef[mk]
            resid = np.array([r[mk] - (b * r["gen_len"] + m0) for r in rows])
            s[f"{mk}_lenresid"] = round(float(resid.mean()), 4)
        s["frac_degenerate"] = round(float(np.mean([r["trigram_rep"] > 0.5 for r in rows])), 3)
        summary[rung] = s

    base = summary.get("baseline") or summary.get("topk6") or summary.get("noise0.0")
    out = {
        "arm": "A7 free-gen coherence-vs-dose (secondary; completes the .60 MIXED row)",
        "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED -> outer loop",
        "law": "per-rung degeneracy metrics (ttr low / trigram_rep high / selfrep high = collapse), "
               "length-residualized (pooled OLS on gen_len); dose-ordered within family for graceful-vs-cliff.",
        "baseline_ref": base,
        "by_rung": summary,
        "families": {fam: sorted([r for r in summary if summary[r]["family"] == fam],
                                 key=lambda r: summary[r]["dose"])
                     for fam in ("topk", "noise", "ablate", "drop")},
    }
    args.out_json.write_text(json.dumps(out, indent=1))
    # dose-ordered print
    for fam in ("topk", "noise", "ablate", "drop"):
        rungs = out["families"][fam]
        print(f"[{fam}] " + " | ".join(
            f"{r}: ttr {summary[r]['ttr']:.2f} trirep {summary[r]['trigram_rep']:.2f} "
            f"deg {summary[r]['frac_degenerate']:.2f}" for r in rungs))
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
