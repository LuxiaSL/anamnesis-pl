"""A8 Leg-5 (L4-f) — owl transport readout: the de-dicto ladder vs its four controls.

Reads the two `vmb_a6_2b_probe` outputs written by this session:
  owl_transported_8b.json  — Vdiverge/Valign carried Qwen -> 8B through the REVERSE
                             Procrustes (the leg's positive arm) + AR1-3 = Qwen's banked
                             R-band members through the SAME reverse map
  owl_rawnull_8b.json      — the same ladder for the zero-padded RAW Qwen vector (the
                             "coordinates transfer" null; see the bank stamp)

and tabulates, per add-3's execution clauses:
  primary  = de-dicto owl lexicon rate (the owl battery's own ruler, judge-free)
  controls = transported-R band (AR1-3) · RAW unconjugated · placebo floor · alpha=0 AR floor
  beside   = de-se lexicon rate (no bar), coherence gate, animal-pick modal

No P is scored here (desk scores P8-5). UNSTAMPED (C§8).

Run: PYTHONPATH=pipeline python -m anamnesis.scripts.a8_leg4f_owl_readout
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_leg4f_owl_readout")

LEG4 = Path("outputs/battery/arms/A8_conjugation/leg4")
GPU = LEG4 / "readouts_gpu"
OUT = LEG4 / "readouts_cpu"
ARMS = {"transported": GPU / "owl_transported_8b.json",
        "rawnull": GPU / "owl_rawnull_8b.json"}
DOSES = ("0.0", "0.03", "0.1", "0.3", "-0.03", "-0.1", "-0.3")


def _rate(blk: dict, key: str) -> float | None:
    v = blk.get(key)
    return round(float(v), 4) if isinstance(v, (int, float)) else None


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"STATUS": "UNSTAMPED (C§8) — mechanics only, desk scores P8-5",
           "leg": "A8 Leg-5 (L4-f) — the transported owl",
           "prereg": "A8-add-3 P8-5 execution clauses",
           "primary_metric": "de-dicto owl lexicon rate (judge-free, the owl battery's ruler)",
           "arms": {}}
    for name, path in ARMS.items():
        if not path.exists():
            res["arms"][name] = {"unavailable": str(path)}
            continue
        d = json.loads(path.read_text())
        vecs = d.get("results", d)
        arm = {"placebo": d.get("placebo"), "census": d.get("census"),
               "ladders": {}}
        for vk, ladder in vecs.items():
            rows = {}
            for a in DOSES:
                blk = ladder.get(a)
                if not isinstance(blk, dict):
                    continue
                rows[a] = {"de_dicto": _rate(blk, "de_dicto_rate"),
                           "de_se": _rate(blk, "de_se_floor_rate"),
                           "coherence": _rate(blk, "coherence"),
                           "coherence_gate_pass": blk.get("coherence_gate_pass"),
                           "animal_pick_modal": (blk.get("animal_pick_modal")
                                                 or blk.get("modal_pick"))}
            arm["ladders"][vk] = rows
        res["arms"][name] = arm

    # the comparison the letter asks for, pulled out
    def ladder(arm: str, vk: str) -> dict:
        return res["arms"].get(arm, {}).get("ladders", {}).get(vk, {})

    tr, rn = ladder("transported", "Vdiverge"), ladder("rawnull", "Vdiverge")
    ars = {k: ladder("transported", k) for k in ("AR1", "AR2", "AR3")}
    pos = [a for a in ("0.03", "0.1", "0.3") if a in tr]
    dd = [tr[a]["de_dicto"] for a in pos if tr[a]["de_dicto"] is not None]
    base = tr.get("0.0", {}).get("de_dicto")
    ar_max = max([ars[k][a]["de_dicto"] for k in ars for a in pos
                  if a in ars[k] and ars[k][a]["de_dicto"] is not None] or [None])
    res["headline"] = {
        "alpha0_baseline_de_dicto": base,
        "transported_de_dicto_by_dose": {a: tr[a]["de_dicto"] for a in pos},
        "rawnull_de_dicto_by_dose": {a: rn.get(a, {}).get("de_dicto") for a in pos},
        "AR_band_max_over_positive_doses": ar_max,
        "placebo_de_dicto_abs_floor": res["arms"].get("transported", {})
        .get("placebo", {}).get("de_dicto_abs_floor"),
        "dose_ordered_positive_arm": (all(x <= y for x, y in zip(dd, dd[1:]))
                                      if len(dd) >= 2 else None),
        "coherence_gate_all_pass": all(
            tr[a].get("coherence_gate_pass") is not False for a in tr if a != "0.0"),
    }
    (OUT / "l4f_owl_transport_readout.json").write_text(json.dumps(res, indent=1))

    lines = ["# L4-f / Leg-5 — transported owl, de-dicto ladder (UNSTAMPED, C§8)", "",
             "| arm / vector | " + " | ".join(DOSES) + " |",
             "|---" * (len(DOSES) + 1) + "|"]
    for arm in res["arms"]:
        for vk, rows in res["arms"][arm].get("ladders", {}).items():
            cells = [f"{rows[a]['de_dicto']:.3f}" if a in rows
                     and rows[a]["de_dicto"] is not None else "—" for a in DOSES]
            lines.append(f"| {arm}/{vk} | " + " | ".join(cells) + " |")
    lines += ["", "## Headline", "", "```",
              json.dumps(res["headline"], indent=1), "```"]
    (OUT / "l4f_owl_transport_readout.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
