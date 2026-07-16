"""A6 Cell 1 — retroactive transplant validation (outer-loop order #3, 2026-07-15/16).

The 14o transplant-validation rider was never executed. Redefined (ruling): run the
transplanted L1-backbone metric on echo-sandbox's OWN cached dense-cohort data and
reproduce its banked backbone mag_g / cos_to_final_g per step. If our transplant of
`node_stats_at_step` is faithful, the numbers match echo's `trajectory_dense.result.json`.

Recipe (echo field.py, self-contained here — no cross-repo import): gen_seed-average X over
sorted (logical, pid); standardize with the STEP-453 chart (mu/sd of the 453 aggregate);
backbone g(t) = mean(Z[student]) − mean(Z[control]); mag_g = ‖g‖; cos_to_final_g =
cos(g(t), g(453)). This is exactly the battery cohort analyzer's backbone computation, run
on echo's feature space instead of ours. CPU. First-read → outer loop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def gen_seed_avg(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"].astype(np.float64)
    cols = {k: d[k] for k in d.files if k != "X"}
    keys = np.array([f"{lg}|{p}" for lg, p in zip(cols["logical"], cols["pid"])])
    uniq = sorted(set(keys.tolist()))
    Xa, role = [], []
    for u in uniq:
        m = keys == u
        Xa.append(X[m].mean(0))
        role.append(str(cols["role"][m][0]))
    return np.asarray(Xa), np.asarray(role)


def backbone(Z, role):
    return Z[role == "student"].mean(0) - Z[role == "control"].mean(0)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True, help="echo-sandbox experiments/cache")
    ap.add_argument("--ref-npz", type=Path, required=True,
                    help="the STEP-453 aggregate cache (fixed chart), e.g. cache/sub_step453.npz")
    ap.add_argument("--banked-json", type=Path, required=True, help="trajectory_dense.result.json")
    ap.add_argument("--dense-prefix", default="sub_dense_step")
    ap.add_argument("--steps", nargs="+", default=["1", "2", "3", "5", "8", "13", "21", "34", "55", "75"])
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    # step-453 chart: mu/sd + g_final
    Xa453, role453 = gen_seed_avg(args.ref_npz)
    mu, sd = Xa453.mean(0), Xa453.std(0)
    sd = sd.copy(); sd[sd < 1e-9] = 1.0
    g_final = backbone((Xa453 - mu) / sd, role453)

    banked = json.loads(args.banked_json.read_text())["tilings"]["rec_L0"]

    rows = []
    max_mag_err = max_cos_err = 0.0
    for s in args.steps:
        p = args.cache_dir / f"{args.dense_prefix}{s}.npz"
        if not p.exists():
            rows.append({"step": s, "present": False}); continue
        Xa, role = gen_seed_avg(p)
        Z = (Xa - mu) / sd
        g = backbone(Z, role)
        mag_g = float(np.linalg.norm(g))
        cos_f = cos(g, g_final)
        bk = banked.get(s) or banked.get(str(int(s)))
        bmag = float(bk["backbone"]["mag_g"]) if bk else None
        bcos = float(bk["backbone"]["cos_to_final_g"]) if bk else None
        merr = abs(mag_g - bmag) if bmag is not None else None
        cerr = abs(cos_f - bcos) if bcos is not None else None
        if merr is not None:
            max_mag_err = max(max_mag_err, merr); max_cos_err = max(max_cos_err, cerr)
        rows.append({"step": s, "present": True,
                     "repro_mag_g": round(mag_g, 5), "banked_mag_g": round(bmag, 5) if bmag else None,
                     "mag_abs_err": round(merr, 6) if merr is not None else None,
                     "repro_cos_to_final": round(cos_f, 5), "banked_cos_to_final": round(bcos, 5) if bcos else None,
                     "cos_abs_err": round(cerr, 6) if cerr is not None else None})
        print(f"  step-{s}: mag_g repro={mag_g:.4f} banked={bmag} (err {merr}); "
              f"cos repro={cos_f:.4f} banked={bcos}")

    faithful = max_mag_err < 1e-3 and max_cos_err < 1e-3
    out = {"arm": "A6 Cell 1 — transplant validation vs echo-sandbox dense backbone",
           "STATUS": "FIRST_READ_PENDING (C§8) — UNSTAMPED → outer loop",
           "law": "reproduce echo's L1 backbone mag_g/cos_to_final_g from its OWN cached dense data "
                  "using the battery analyzer's backbone recipe (gen-seed-avg, 453-chart standardize, "
                  "student_mean − control_mean).",
           "max_mag_abs_err": round(max_mag_err, 6), "max_cos_abs_err": round(max_cos_err, 6),
           "transplant_faithful": bool(faithful),
           "verdict": ("TRANSPLANT FAITHFUL — reproduces echo backbone to <1e-3" if faithful
                       else f"MISMATCH — max mag err {max_mag_err:.4f}, cos err {max_cos_err:.4f}"),
           "rows": rows}
    args.out_json.write_text(json.dumps(out, indent=1))
    print(f"max_mag_err={max_mag_err:.2e} max_cos_err={max_cos_err:.2e} faithful={faithful}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
