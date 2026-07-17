"""Register-stiffening cross-arm watch item (13b; session-11 Part A.5, the Part-F survivor).

13b logged: random residual perturbations produce a GENERIC judged-formality bump at trace
doses (R1 2AFC .750@.03 → .325 across the ladder) — "perturbation-induced register
stiffening", cross-arm watch item, never tested since. This scores it CPU-side across every
banked perturbation family with free-gen TEXT:

  A5  R1/R2/R3  @L14 α{.03,.1,.3,1.0}, n=160/cell   (vmb_a5_3b)
  b7  Rband1-3  @L14 α{.03,.1,.3},     n=40/cell    (vmb_b7_3b)
  c3  Rc1-3     @L14+L21 α{.03,.1,.3}, n=160/cell   (vmb_c3_3b)
  context: V1 ladder (formality lever, positive control) · V3 ladder (13b re-based
  anti-formal) — direction anchors, not members of the watch class.
  A4 surgery EXCLUDED by construction: surgery replays the same tokens — no new text exists.
  Annex Rband symlink farms excluded (same underlying cells; annex-attributed lane).

INSTRUMENT (declared here; no formality text lexicon existed before this):
  two SEPARATE per-1k-token columns, never composited —
  formal_connectives/1k  (furthermore, moreover, consequently, thus, hence, notably,
                          accordingly, therefore, in addition, it is important/essential to)
  contractions/1k        (n't, 're, 'll, 've, 'm, 's-as-clitic approximated by the
                          apostrophe-form list — direction: DOWN = stiffer)
  "Stiffening" = formal connectives UP and/or contractions DOWN vs the pooled α=0 riders.
⚠ SCOPING LAW (14n watch-item precedent, Luxia's wording): a negative here is scoped as
"the bump does not express in THIS lexicon" — NEVER "no register stiffening" (the original
detection instrument was a 2AFC judge; this is its cheap text proxy). No filed P — watch
item, descriptive; the judge form is the escalation path if the proxy fires.

Statistic per cell: rates + z vs the rider distribution (rider spread = across the pooled
rider cells' per-gen bootstrap, B=2000, seeded). Cross-family read = the watch question:
does the SAME direction show at trace dose (α=.03) in ≥2 independent R families?

CPU-only, banked texts only. First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

FORMAL_RE = re.compile(
    r"\b(?:furthermore|moreover|consequently|thus|hence|notably|accordingly|therefore|"
    r"in addition|it is (?:important|essential) to)\b", re.IGNORECASE)
CONTRACTION_RE = re.compile(
    r"\b\w+(?:n't|'re|'ll|'ve|'m|'d)\b", re.IGNORECASE)

SEED = 20260716
N_BOOT = 2000


def cell_rates(run: Path) -> dict | None:
    md_path = run / "metadata.json"
    if not md_path.exists():
        return None
    md = json.loads(md_path.read_text())
    gens = md["generations"] if "generations" in md else md
    per_gen = []
    for g in gens:
        t = g.get("generated_text") or ""
        ntok = g.get("num_generated_tokens") or max(1, len(t.split()))
        per_gen.append((len(FORMAL_RE.findall(t)) * 1000.0 / ntok,
                        len(CONTRACTION_RE.findall(t)) * 1000.0 / ntok))
    if not per_gen:
        return None
    arr = np.array(per_gen)
    return {"n": len(per_gen),
            "formal_per_1k": round(float(arr[:, 0].mean()), 4),
            "contractions_per_1k": round(float(arr[:, 1].mean()), 4),
            "_per_gen": arr}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery-root", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    root = args.battery_root

    # pooled α=0 riders (the shared reference convention)
    rider_cells = sorted((root / "vmb_a5_3b").glob("*_a0.0"))
    rider_arrs = []
    for d in rider_cells:
        c = cell_rates(d)
        if c:
            rider_arrs.append(c["_per_gen"])
    if not rider_arrs:
        raise SystemExit("no α=0 rider cells found under vmb_a5_3b")
    riders = np.vstack(rider_arrs)
    rng = np.random.default_rng(SEED)
    boots = np.array([riders[rng.integers(0, len(riders), len(riders))].mean(axis=0)
                      for _ in range(N_BOOT)])
    rider_mean = riders.mean(axis=0)
    rider_sd = boots.std(axis=0)

    families = {
        "A5_R": [("vmb_a5_3b", f"{v}_L14_a{a}") for v in ("R1", "R2", "R3")
                 for a in ("0.03", "0.1", "0.3", "1.0")],
        "b7_Rband": [("vmb_b7_3b", f"{v}_L14_a{a}") for v in ("Rband1", "Rband2", "Rband3")
                     for a in ("0.03", "0.1", "0.3")],
        "c3_Rc": [("vmb_c3_3b", f"{v}_L{s}_a{a}") for v in ("Rc1", "Rc2", "Rc3")
                  for s in (14, 21) for a in ("0.03", "0.1", "0.3")],
        "context_V": ([("vmb_a5_3b", f"V1_L14_a{a}") for a in ("0.03", "0.1", "0.3")]
                      + [("vmb_a5_3b", f"V3_L14_L14_a{a}") for a in ("0.03", "0.1", "0.3")]),
    }

    rows = []
    for fam, cells in families.items():
        for run_name, cell in cells:
            c = cell_rates(root / run_name / cell)
            if c is None:
                rows.append({"family": fam, "cell": cell, "MISSING": True})
                continue
            arr = c.pop("_per_gen")
            z_f = (arr[:, 0].mean() - rider_mean[0]) / rider_sd[0] if rider_sd[0] > 0 else None
            z_c = (arr[:, 1].mean() - rider_mean[1]) / rider_sd[1] if rider_sd[1] > 0 else None
            rows.append({"family": fam, "cell": cell, **c,
                         "z_formal_vs_riders": round(float(z_f), 2) if z_f is not None else None,
                         "z_contractions_vs_riders": round(float(z_c), 2) if z_c is not None else None,
                         "stiffening_direction": bool((z_f or 0) > 0 and (z_c or 0) < 0)})

    # the watch question: same direction at trace dose in >=2 independent R families?
    trace = [r for r in rows if not r.get("MISSING") and r["family"] != "context_V"
             and r["cell"].endswith("a0.03")]
    fam_fire = {}
    for fam in ("A5_R", "b7_Rband", "c3_Rc"):
        fr = [r for r in trace if r["family"] == fam]
        fam_fire[fam] = {
            "n_cells": len(fr),
            "n_stiffening_direction": sum(1 for r in fr if r["stiffening_direction"]),
            "mean_z_formal": round(float(np.mean([r["z_formal_vs_riders"] for r in fr])), 2) if fr else None,
            "mean_z_contractions": round(float(np.mean([r["z_contractions_vs_riders"] for r in fr])), 2) if fr else None,
        }

    out = {
        "arm": "register-stiffening cross-arm watch item (13b; s11 Part A.5)",
        "STATUS": "FIRST_READ_PENDING (C§8)",
        "model": "3b",
        "law": ("formal-connective + contraction rates per 1k tokens on banked free-gen texts; "
                f"z vs pooled α=0 riders (bootstrap B={N_BOOT}, seed={SEED}); watch question = "
                "stiffening direction (formal↑ and contractions↓) at trace dose α=.03 in ≥2 "
                "independent R families; NO filed P — descriptive watch item"),
        "scoping_law": "negative = 'does not express in THIS lexicon' (14n precedent), never "
                       "'no stiffening' — the original instrument was a 2AFC judge; judge form "
                       "is the escalation path",
        "instrument_provenance": "lexicon DEFINED IN THIS SCRIPT (no prior formality text lexicon "
                                 "existed); two columns never composited",
        "census_note": "all three R families + riders share one census (same prompt_set, "
                       "expository, 20 topics, 512-tok gens — verified on metadata before this "
                       "read); the cross-run rider reference is composition-clean (3′ checked)",
        "rider_reference": {"n_rider_cells": len(rider_cells), "n_rider_gens": int(len(riders)),
                            "formal_per_1k": round(float(rider_mean[0]), 4),
                            "contractions_per_1k": round(float(rider_mean[1]), 4),
                            "boot_sd": [round(float(s), 5) for s in rider_sd]},
        "trace_dose_family_read": fam_fire,
        "rows": rows,
    }
    args.out_json.write_text(json.dumps(out, indent=1))
    print(json.dumps(fam_fire, indent=1))
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
