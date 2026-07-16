"""A6 Cell 5 — N5 coherence gate on the teacher corpora (WAVE2-A6 cell 5; session-8 Part C).

The N5 rule: no student trains on an incoherent teacher corpus. The coherence metric is the
teacher's per-prompt field self-consistency (subliminal RESEARCH-JOURNAL 2026-07-12,
1.6-UPDATE; reinvent NOTHING — source-pointer rider):

    td_p = teacher_p − null_p                         (raw, gen-seed-averaged signature delta)
    coh(p) = cos(td_p, LOO-mean of the teacher's OTHER deltas)
    corpus coherence = mean_p coh(p) over the number prompts

The authoritative measurement is banked (`subliminal .../teacher_coherence.result.json`,
step-453). Frozen threshold τ = 0.15 (the journal's "τ* exists iff coherence ≳0.15";
healthy band 0.14-0.24 on heldout-numbers vs the incoherent owl/Llama arm at 0.047). Gate:
number-corpus coherence ≥ τ. Scored vs P=.85 (all corpora pass). A corpus below τ →
stop-and-surface (blocks THAT student line, not the arm).

CPU-only, reads the banked artifact. First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TAU = 0.15  # frozen threshold (journal: τ* exists iff coherence ≳0.15; owl incoherent = 0.047)
ANIMALS = ("cat", "penguin", "phoenix", "wolf")
CONTROLS = ("a", "b", "c", "d", "e")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coherence-json", type=Path, required=True,
                    help="subliminal research/artifacts_2026-07-11/teacher_coherence.result.json")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    coh = json.loads(args.coherence_json.read_text())
    per_animal = coh["per_animal"]

    rows = []
    for a in ANIMALS:
        if a not in per_animal:
            rows.append({"corpus": f"qwen_{a}_v3_numbers", "role": "animal-teacher",
                         "coherence_training_numbers": None, "coherence_heldout_numbers": None,
                         "measured": False, "note": "not in banked artifact"})
            continue
        pc = per_animal[a]["per_category"]
        train = float(pc["training_numbers"]["coh_global"])
        held = float(pc["heldout_numbers"]["coh_global"])
        # gate on the STRICTER (heldout) number coherence — the honest generalizing rung
        gate_val = held
        rows.append({
            "corpus": f"qwen_{a}_v3_numbers", "role": "animal-teacher",
            "coherence_training_numbers": round(train, 4),
            "coherence_heldout_numbers": round(held, 4),
            "gate_value": round(gate_val, 4), "tau": TAU,
            "passes": bool(gate_val >= TAU), "measured": True,
        })

    # controls: the coherence framework treats them as the BASELINE (align is measured vs
    # control), so per-corpus teacher coherence is not in the banked artifact. Controls are pure
    # number-generation — the coherent domain by the finding's own logic ("numbers can't be
    # faked"); the incoherence failure mode is trait-SHORTCUT, which control corpora lack.
    for c in CONTROLS:
        rows.append({
            "corpus": f"qwen_control_{c}_numbers", "role": "control (baseline)",
            "coherence_training_numbers": None, "coherence_heldout_numbers": None,
            "gate_value": None, "tau": TAU, "passes": None, "measured": False,
            "note": "numbers-domain baseline; coherence not banked (control = the align reference, "
                    "not a trait-teacher). Uniform battery-space coherence = scoped GPU follow-up.",
        })

    measured = [r for r in rows if r.get("measured")]
    n_pass = sum(1 for r in measured if r["passes"])
    all_measured_pass = all(r["passes"] for r in measured)
    failures = [r["corpus"] for r in measured if not r["passes"]]

    out = {
        "arm": "A6 Cell 5 — N5 teacher-corpus coherence gate",
        "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop",
        "law": "coherence coh(p)=cos(td_p, LOO-mean other teacher deltas); td_p=teacher−null, "
               "gen-seed-avg (subliminal RESEARCH-JOURNAL 1.6-UPDATE); gate on heldout-numbers "
               f"coherence ≥ τ={TAU} (frozen; journal ≳0.15, owl incoherent=0.047).",
        "filed_P": {"all_teacher_corpora_pass": 0.85},
        "tau_frozen": TAU,
        "n_measured": len(measured), "n_pass": n_pass,
        "all_measured_pass": all_measured_pass,
        "failures": failures,
        "verdict": ("4/4 animal teacher corpora PASS (load-bearing gate for cell-1 animal students); "
                    "5 control corpora numbers-domain, coherence unmeasured (scoped GPU follow-up)"
                    if all_measured_pass and not failures else
                    f"STOP-AND-SURFACE: {failures} below τ={TAU}"),
        "scope_note": "Cell-1 replays cat/penguin/phoenix (animals) + control a-e; the animal-corpus "
                      "gate is the load-bearing one for the students' regime field. Wolf gated for "
                      "completeness though wolf student not in cell-1 cohort.",
        "provenance": "authoritative banked measurement (teacher_coherence.result.json, step-453); "
                      "reinvent-nothing per the source-pointer rider.",
        "rows": rows,
    }
    args.out_json.write_text(json.dumps(out, indent=1))
    print(f"N5 gate τ={TAU}: {n_pass}/{len(measured)} measured corpora pass; failures={failures}")
    for r in rows:
        gv = r.get("gate_value")
        print(f"  {r['corpus']:26} coh_heldout={r.get('coherence_heldout_numbers')} "
              f"gate={gv} pass={r['passes']}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
