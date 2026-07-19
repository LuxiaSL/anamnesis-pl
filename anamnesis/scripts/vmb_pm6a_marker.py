"""PM6-a marker readout (vmb A5 / M6) — the EXPRESSION side of the state-lever.

The companion to vmb_pm6a_statelever (which scores the on-axis STATE shift). This scores
whether steering EXPRESSES in the generated text, via the canonical socratic markers
(question rate + SOCRATIC lexicon per 1k words, reused from vmb_14k_soclin_lever). Per-gen
permutation, both signs, byte-BPE-decoded (M6 banks generated_text encoded).

The expression bar (THE STANDING RULE, D3 graded-prompt parity ladder): a steered cell's
marker rate is scored against the appropriate PROMPT GRADE (faint/medium/full socratic
suggestion) AND above the dose-matched R band — NEVER against the full pole. Pass
--graded-dirs grade=dir,... to bring the ladder in; without it, only pole-ceiling +
baseline/R contrasts are reported (item A1: the baseline/R q-rates the original block omitted).

Run (node1, CPU): python -m anamnesis.scripts.vmb_pm6a_marker \
  --run-dir /models/anamnesis-extract/runs/vmb_a5_dsv2_lite_pm6a \
  --pole-socratic-dir .../vmb_a2_dsv2-lite_pure_socratic \
  --pole-linear-dir .../vmb_a2_dsv2-lite_pure_linear \
  --site 9 --out-json <json>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from anamnesis.analysis.battery.text_decode import maybe_decode
from anamnesis.scripts.vmb_14k_soclin_lever import SOCRATIC  # canonical marker lexicon

# sign+dose suffix, same grammar as the state-lever: m=neg (socratic-ward), a/p=pos.
CELL = re.compile(r"^(?P<vec>V3|V1|R1|R2|R3)_L(?P<site>\d+)_(?P<sd>[apm][\d.]+)$")


def _per_gen_rates(texts: list[str]) -> dict[str, np.ndarray]:
    """Per-gen question_per_1k and socratic_marker_per_1k (decoded, ≥1-word gens)."""
    q, s = [], []
    for raw in texts:
        t = maybe_decode(raw)
        w = max(len(t.split()), 1)
        q.append(1000.0 * t.count("?") / w)
        s.append(1000.0 * len(SOCRATIC.findall(t)) / w)
    return {"q": np.array(q), "s": np.array(s)}


def _texts(cell_dir: Path) -> list[str]:
    md = json.loads((cell_dir / "metadata.json").read_text())
    gens = md["generations"] if "generations" in md else md
    return [g.get("generated_text", "") for g in gens]


def _perm(a: np.ndarray, b: np.ndarray, nperm: int = 20000, seed: int = 0) -> tuple[float, float]:
    """One-sided permutation: P(mean(a) - mean(b) >= observed) under label shuffle."""
    if len(a) == 0 or len(b) == 0:
        return 0.0, 1.0
    rng = np.random.default_rng(seed)
    obs = float(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    na = len(a)
    c = 0
    for _ in range(nperm):
        rng.shuffle(pool)
        c += (pool[:na].mean() - pool[na:].mean()) >= obs
    return obs, float((c + 1) / (nperm + 1))


def _summ(r: dict[str, np.ndarray]) -> dict:
    return {"n": int(len(r["q"])),
            "question_per_1k": round(float(r["q"].mean()), 3),
            "socratic_marker_per_1k": round(float(r["s"].mean()), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True, help="PM6-a cells root")
    ap.add_argument("--pole-socratic-dir", type=Path, required=True)
    ap.add_argument("--pole-linear-dir", type=Path, required=True)
    ap.add_argument("--baseline-cell", default="baseline")
    ap.add_argument("--site", type=int, default=9)
    ap.add_argument("--graded-dirs", default=None,
                    help="parity ladder, e.g. faint=/path,medium=/path,full=/path")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    # pole ceiling (socratic vs linear) — the marker's dynamic range
    pole_soc = _per_gen_rates(_texts(args.pole_socratic_dir))
    pole_lin = _per_gen_rates(_texts(args.pole_linear_dir))
    q_obs, q_p = _perm(pole_soc["q"], pole_lin["q"])
    s_obs, s_p = _perm(pole_soc["s"], pole_lin["s"])
    poles_pass = bool(q_p < 0.05 and q_obs > 0)

    base = _per_gen_rates(_texts(args.run_dir / args.baseline_cell))

    # collect steered + R cells at the site
    cells: dict[str, dict[str, np.ndarray]] = {}
    for d in sorted(args.run_dir.iterdir()):
        m = CELL.match(d.name)
        if not m or int(m.group("site")) != args.site or not (d / "metadata.json").exists():
            continue
        cells[d.name] = _per_gen_rates(_texts(d))

    # graded parity ladder (optional) — the expression-bar reference
    graded: dict[str, dict[str, np.ndarray]] = {}
    if args.graded_dirs:
        for kv in args.graded_dirs.split(","):
            grade, path = kv.split("=", 1)
            graded[grade.strip()] = _per_gen_rates(_texts(Path(path.strip())))

    # per-dose contrasts: V3 vs baseline, V3 vs dose-matched R band
    lever = {}
    doses = sorted({CELL.match(name).group("sd") for name in cells})  # all keys matched CELL
    for dose in doses:
        v3 = cells.get(f"V3_L{args.site}_{dose}")
        rcells = [cells[f"R{i}_L{args.site}_{dose}"] for i in (1, 2, 3)
                  if f"R{i}_L{args.site}_{dose}" in cells]
        if v3 is None or not rcells:
            continue
        Rq = np.concatenate([r["q"] for r in rcells])
        Rs = np.concatenate([r["s"] for r in rcells])
        db_obs, db_p = _perm(v3["q"], base["q"])          # V3 vs baseline (question rate)
        dR_obs, dR_p = _perm(v3["q"], Rq)                  # V3 vs matched-R band
        sR_obs, sR_p = _perm(v3["s"], Rs)                  # socratic-lexicon vs R
        lever[f"L{args.site}_{dose}"] = {
            "V3": _summ(v3),
            "baseline_q_per_1k": round(float(base["q"].mean()), 3),
            "R_band_q_per_1k": [round(float(r["q"].mean()), 3) for r in rcells],
            "R_band_q_mean": round(float(Rq.mean()), 3),
            "V3_vs_baseline": {"delta_q": round(db_obs, 3), "p": round(db_p, 4)},
            "V3_vs_Rband": {"delta_q": round(dR_obs, 3), "p": round(dR_p, 4)},
            "V3_socratic_vs_Rband": {"delta": round(sR_obs, 3), "p": round(sR_p, 4)},
            # fires iff expressed above BOTH baseline and the matched-R band (question rate)
            "expressed": bool(db_p < 0.05 and db_obs > 0 and dR_p < 0.05 and dR_obs > 0),
        }

    out = {
        "arm": "PM6-a marker readout (expression side; socratic-ward −V3)",
        "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop",
        "site": args.site,
        "pole_ceiling": {"socratic": _summ(pole_soc), "linear": _summ(pole_lin),
                         "q_delta": round(q_obs, 3), "q_p": round(q_p, 4),
                         "socratic_delta": round(s_obs, 3), "socratic_p": round(s_p, 4),
                         "poles_separate": poles_pass},
        "baseline": _summ(base),
        "graded_ladder": {g: _summ(r) for g, r in graded.items()} or "not-provided (item A2 needs graded corpora)",
        "lever_by_dose": lever,
        "law": "per-gen question_per_1k + SOCRATIC lexicon/1k (canonical); expression bar = "
               "steered rate matches the appropriate PROMPT GRADE AND exceeds the dose-matched "
               "R band, NEVER the full pole (D3 standing rule). Per-gen permutation, 20k.",
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=1))
    print(f"pole ceiling: socratic q/1k={out['pole_ceiling']['socratic']['question_per_1k']} "
          f"vs linear {out['pole_ceiling']['linear']['question_per_1k']} "
          f"(Δ{q_obs:+.2f}, p={q_p:.4f}) → {'SEPARATE' if poles_pass else 'MARKER-WEAK'}")
    print(f"baseline q/1k={out['baseline']['question_per_1k']}")
    for k, v in lever.items():
        print(f"  {k}: V3 q/1k={v['V3']['question_per_1k']} vs base {v['baseline_q_per_1k']} "
              f"(Δ{v['V3_vs_baseline']['delta_q']:+}, p={v['V3_vs_baseline']['p']}) | "
              f"vs Rband {v['R_band_q_mean']} (Δ{v['V3_vs_Rband']['delta_q']:+}, "
              f"p={v['V3_vs_Rband']['p']}) → expressed={v['expressed']}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
