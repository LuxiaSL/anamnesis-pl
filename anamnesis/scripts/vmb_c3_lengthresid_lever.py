"""C3 (d) lever — LENGTH-RESIDUALIZATION robustness column (14f completeness item).

The lever (d) projects the per-cell mean signature shift onto the C2 orphaned axis. This
robustness column asks: is the V_temp targeting an artifact of LENGTH differences between
steered and rider populations (V_temp samples hotter → possibly longer/shorter)? Residualize
`num_generated_tokens` out of every per-gen signature feature (C2 convention: per-feature OLS,
β fit on the pooled steered+rider set, applied to all), THEN recompute the mean-shift targeting
and the Vtemp ÷ mean(Rc) ratio at matched (site,α).

14f prediction: V_temp targeting stays ≥2× matched null at every cell after residualization
(P=0.90; the C2 axis itself was length-insensitive, .678 both ways). Reports raw vs len-resid
side by side. CPU-only. First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from anamnesis.analysis.battery.deltas import ConditionCorpus
from anamnesis.scripts.vmb_arm_a5_analyze import parse_cell_dir, rekey_topic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _lengths(run: Path, gids) -> np.ndarray:
    md = json.loads((run / "metadata.json").read_text())
    gens = md["generations"] if "generations" in md else md
    by = {int(g.get("generation_id", g.get("gen_id", -1))): g.get("num_generated_tokens", 0) for g in gens}
    return np.array([by.get(int(g), 0) for g in gids], dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3-run-dir", type=Path, required=True)
    ap.add_argument("--main-run-dir", type=Path, required=True, help="vmb_a5_3b (α=0 riders)")
    ap.add_argument("--axis-npz", type=Path, required=True)
    ap.add_argument("--model", default="3b")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    A = np.load(args.axis_npz)
    axis = A["axis"].astype(np.float64)
    idx = A["feature_indices"].astype(int)
    med, scale = A["med"].astype(np.float32), A["scale"].astype(np.float32)

    def corpus(d: Path, label: str) -> ConditionCorpus:
        return rekey_topic(ConditionCorpus(d / "signatures_v3", d / "metadata.json", med, scale, label))

    # collect per-gen Z (on the axis idx columns) + lengths, tagged by cell
    cells = {}   # cell -> (Z_idx [n,1282], length [n], is_null, site, alpha)
    riders_Z, riders_len = [], []
    for d in sorted(args.main_run_dir.iterdir()):
        if d.name.endswith("_a0.0") and (d / "signatures_v3").exists():
            cc = corpus(d, f"rider-{d.name}")
            riders_Z.append(cc.Z[:, idx]); riders_len.append(_lengths(d, cc.gen_ids))
    ref_Zidx = np.vstack(riders_Z); ref_len = np.concatenate(riders_len)
    logger.info(f"reference: {ref_Zidx.shape[0]} rider gens")

    for d in sorted(args.c3_run_dir.iterdir()):
        info = parse_cell_dir(d.name)
        if info is None or not (d / "signatures_v3").exists() or info["alpha_frac"] == 0.0:
            continue
        cc = corpus(d, d.name)
        cells[d.name] = {"Z": cc.Z[:, idx], "len": _lengths(d, cc.gen_ids),
                         "is_null": info["vector"].upper().startswith("RC"),
                         "site": info["site"], "alpha_frac": info["alpha_frac"],
                         "vector": info["vector"]}

    # pooled length-resid: β fit on {all steered + riders}, per-feature OLS, applied to all
    pool_Z = np.vstack([ref_Zidx] + [c["Z"] for c in cells.values()])
    pool_L = np.concatenate([ref_len] + [c["len"] for c in cells.values()])
    L = np.c_[np.ones(len(pool_L)), pool_L]
    beta, *_ = np.linalg.lstsq(L, pool_Z.astype(np.float64), rcond=None)

    def resid(Z, ln):
        return (Z.astype(np.float64) - np.c_[np.ones(len(ln)), ln] @ beta)

    ref_c_raw = ref_Zidx.mean(axis=0)
    ref_c_res = resid(ref_Zidx, ref_len).mean(axis=0)

    rows = []
    for name, c in cells.items():
        tgt_raw = float(abs((c["Z"].mean(axis=0) - ref_c_raw) @ axis))
        tgt_res = float(abs((resid(c["Z"], c["len"]).mean(axis=0) - ref_c_res) @ axis))
        rows.append({"cell": name, "vector": c["vector"], "site": c["site"],
                     "alpha_frac": c["alpha_frac"], "n": int(c["Z"].shape[0]),
                     "is_null": c["is_null"], "targeting_raw": tgt_raw, "targeting_lenresid": tgt_res})

    def rc_mean(site, af, key):
        v = [r[key] for r in rows if r["is_null"] and r["site"] == site and r["alpha_frac"] == af]
        return float(np.mean(v)) if v else None

    for r in rows:
        if r["is_null"]:
            continue
        for key in ("targeting_raw", "targeting_lenresid"):
            base = rc_mean(r["site"], r["alpha_frac"], key)
            r[f"{key}_over_Rc"] = round(float(r[key] / max(base, 1e-12)), 3) if base else None
        r["lever_2x_lenresid"] = bool((r.get("targeting_lenresid_over_Rc") or 0) >= 2.0)

    out = {"model": args.model, "arm": "C3 (d) lever — length-resid robustness (14f item)",
           "STATUS": "FIRST_READ_PENDING (C§8)",
           "law": "per-gen signature length-residualized (num_generated_tokens, pooled-OLS, C2 convention) "
                  "before mean-shift·C2-axis targeting; (d) = Vtemp ÷ mean(Rc) matched (site,α); "
                  "14f P=0.90: stays ≥2× after residualization",
           "rows": sorted(rows, key=lambda r: (r["is_null"], r["site"], r["alpha_frac"]))}
    args.out_json.write_text(json.dumps(out, indent=1))
    for r in out["rows"]:
        if r["is_null"]:
            continue
        logger.info(f"  {r['cell']:16} tgt/Rc raw={r.get('targeting_raw_over_Rc')} "
                    f"lenresid={r.get('targeting_lenresid_over_Rc')} 2x={r['lever_2x_lenresid']}")
    logger.info(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
