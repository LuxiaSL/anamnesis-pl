"""A6 Cell 2 — THE A5↔A6 bridge readout (first-of-kind; C§8 absolute; session-8 Part C).

Tests "subliminal learning = steering-vector distillation" (2606.00995): does steering the
TEACHER with the animal vector deform its signature the SAME WAY that distillation deforms
the STUDENT? Compared in PROFILE space (the A5-twist standing rule — readout-vector × dose
correlation vs the matched-R cosine band; NEVER per-facet):

  teacher_profile(α) = mean(Z_steered@α) − mean(Z_unsteered)     (Qwen base + animal-vector)
  student_profile    = mean(Z_cat_student_final) − mean(Z_base)   (from the cohort)
  match = cos(teacher_profile(α), student_profile)  vs  the R-null band
          cos(teacher_profile_R(α), student_profile)  over R1..R3

Both profiles live in stage-0-floor-z signature space (feature_map cells reported as the
sub-decomposition, but the MATCH is the whole-vector cosine, never per-facet). Convergence
(cos beats the R band) = mechanism evidence; divergence = the vector-distillation account
is incomplete. BOTH pre-worded — the CALL is the outer loop's. Scored vs P=.60.

REPORT, DO NOT INTERPRET. CPU (reads banked steered + cohort + base sigs). First-read → outer loop.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from anamnesis.analysis.battery.deltas import load_floor_scale
from anamnesis.analysis.battery.floors import load_signature_matrix, build_cells
from anamnesis.analysis.battery.manifest import MODEL_META

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _mean_z(sig_dir: Path, med, scale) -> NDArray | None:
    if not sig_dir.exists():
        return None
    X, names, _ = load_signature_matrix(sig_dir)
    return ((X - med) / scale).mean(0), names


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steer-run-dir", type=Path, required=True, help="Qwen teacher steer cells")
    ap.add_argument("--base-sig", type=Path, required=True, help="unsteered Qwen base signatures_v3 (α=0)")
    ap.add_argument("--student-final-sig", type=Path, required=True,
                    help="cohort cat_dense/step-0075/signatures_v3")
    ap.add_argument("--student-base-sig", type=Path, required=True,
                    help="base-model signatures for the SAME probes as the student (floor stage-0)")
    ap.add_argument("--floor-dir", type=Path, required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    ap.add_argument("--animal", default="cat")
    ap.add_argument("--null-prefix", default="AR")
    ap.add_argument("--model", default="qwen-7b")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    mm = MODEL_META[args.model]
    med, scale = load_floor_scale(args.floor_dir)

    # student post-distill profile (cat_dense final − base, on shared probes → base cancels
    # in the mean-difference; we subtract the base mean for the same probe set)
    sf = _mean_z(args.student_final_sig, med, scale)
    sb = _mean_z(args.student_base_sig, med, scale)
    if sf is None or sb is None:
        raise SystemExit("missing student final/base signatures")
    student_profile = sf[0] - sb[0]
    names = sf[1]

    base = _mean_z(args.base_sig, med, scale)
    if base is None:
        raise SystemExit("missing unsteered base signatures")
    base_mean = base[0]

    # per-source decomposition (report only; match is whole-vector)
    cells = {c: m for c, m in build_cells(names, mm.n_layers).items()
             if c in ("whole_vector", "source:attention", "source:residual", "source:gate",
                      "source:keys", "source:output")}

    import re
    def parse(name):
        m = re.match(r"^(?P<vec>[A-Za-z][A-Za-z0-9]*?)_L(?P<s>\d+)_a(?P<a>[\d.]+)$", name)
        return (m.group("vec"), int(m.group("s")), float(m.group("a"))) if m else (None, None, None)

    rows = []
    for d in sorted(args.steer_run_dir.iterdir()) if args.steer_run_dir.exists() else []:
        vec, site, af = parse(d.name)
        if vec is None or af == 0.0:
            continue
        z = _mean_z(d / "signatures_v3", med, scale)
        if z is None:
            continue
        teacher_profile = z[0] - base_mean
        is_null = vec.upper().startswith(args.null_prefix.upper())
        row = {"cell": d.name, "vector": vec, "site": site, "alpha_frac": af, "is_null": is_null,
               "cos_teacher_vs_student_wholevector": round(cos(teacher_profile, student_profile), 4),
               "teacher_deformation_norm": round(float(np.linalg.norm(teacher_profile)), 4)}
        for c, mask in cells.items():
            row[f"cos_{c}"] = round(cos(teacher_profile[mask], student_profile[mask]), 4)
        rows.append(row)

    # match vs R-null band at each α
    def null_band(af):
        vals = [r["cos_teacher_vs_student_wholevector"] for r in rows
                if r["is_null"] and abs(r["alpha_frac"] - af) < 1e-9]
        return (round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4), len(vals)) if vals else (None, None, 0)

    summary = []
    for af in args.alphas:
        tgt = next((r for r in rows if not r["is_null"] and abs(r["alpha_frac"] - af) < 1e-9), None)
        nb_mean, nb_std, nb_n = null_band(af)
        beats = (tgt is not None and nb_mean is not None
                 and tgt["cos_teacher_vs_student_wholevector"] > nb_mean + 2 * (nb_std or 0))
        summary.append({"alpha_frac": af,
                        "target_cos": tgt["cos_teacher_vs_student_wholevector"] if tgt else None,
                        "null_band_mean": nb_mean, "null_band_std": nb_std, "null_n": nb_n,
                        "beats_null_band_2sd": bool(beats)})

    out = {"arm": "A6 Cell 2 — A5↔A6 bridge (steered-teacher vs distilled-student profile)",
           "STATUS": "FIRST_READ_PENDING (C§8 ABSOLUTE — first-of-kind; REPORT, DO NOT INTERPRET)",
           "model": args.model, "animal": args.animal,
           "law": "profile-space match: cos(mean(steered)−mean(base), mean(student_final)−mean(base)) "
                  "whole-vector; vs matched-R cosine band (mean±2sd over AR1-3). Per-source cos = "
                  "sub-decomposition only, NEVER the match criterion.",
           "filed_P": {"profile_match_beats_R_band": 0.60},
           "readings": "cos beats R band = convergence (steering≈distillation mechanism); within/below "
                       "band = divergence (vector-distillation account incomplete). CALL = outer loop.",
           "student_profile_norm": round(float(np.linalg.norm(student_profile)), 4),
           "summary_by_alpha": summary, "rows": rows}
    args.out_json.write_text(json.dumps(out, indent=1))
    for s in summary:
        logger.info(f"α={s['alpha_frac']}: target_cos={s['target_cos']} null_band={s['null_band_mean']}"
                    f"±{s['null_band_std']} beats={s['beats_null_band_2sd']}")
    logger.info(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
