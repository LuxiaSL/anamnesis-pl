"""14h item 1 — MT-bank upstream-zero STANDING check (per bank, per vector).

At matched tokens under bitwise-deterministic replay, every signature feature reading
strictly UPSTREAM of a vector's injection site must read Delta = 0.0 EXACTLY vs the
banked UNSTEERED sig of the same gen (identical tokens => layers below the site are
byte-identical). One check certifies twin pairing + replay determinism + no leakage +
delta algebra, no free parameter. Any nonzero => QUARANTINE the bank, stop-and-surface.

Layer keying (state_extractor: feature `_L{n}` reads hidden_states[n+1] = output of
model-layer n; the `delta_*` family reads blocks n+1 AND n+2 = model-layers n,n+1):
  plain  `X_L{n}`        upstream (must be 0) iff  n <  L_inj
  delta  `delta_*_L{n}`  upstream (must be 0) iff  n <= L_inj - 2   (one block shallower)
Features with no parseable single `_L{n}` (cross-layer / output-source) are not
layer-attributable and are excluded from the upstream set (reported as skipped).

Usage:
  python -m anamnesis.scripts.vmb_a5_upstream_zero --mt-root <out_root> \
      --stage0-run <floor_run> --out <json> [--sample 12]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_L_RE = re.compile(r"_L(\d+)")   # NO trailing boundary: catches BOTH layers of
                                 # cross-layer names like kv_value_cka_L0_L28.


def _layers_of(name: str) -> list[int]:
    """ALL layer indices in the name (cross-layer features name >1)."""
    return [int(m.group(1)) for m in _L_RE.finditer(name)]


def _layer_of(name: str) -> int | None:
    """Deepest layer the feature reads (max over all _L tokens), or None if none.

    Cross-layer features (e.g. kv_*_cka_L{a}_L{b}) read the DEEPER layer too, so a
    feature is upstream-safe only if ALL its layers are upstream => key on the max."""
    ls = _layers_of(name)
    return max(ls) if ls else None


def _is_upstream(name: str, layer: int, site: int) -> bool:
    # `layer` is the deepest read layer (max over the name's _L tokens).
    if name.startswith("delta") or "_delta" in name:
        return layer <= site - 2          # delta reads model-layers n, n+1
    return layer < site


def _load(sig_npz: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(sig_npz, allow_pickle=True)
    return d["feature_names"].astype(str), d["features"].astype(np.float64)


def check_cell(cell_dir: Path, floor_sig: Path, site: int, sample: int) -> dict:
    inj_dir = cell_dir / "signatures_v3"
    gen_files = sorted(inj_dir.glob("gen_*.npz"))[:sample]
    if not gen_files:
        return {"error": f"no gen sigs under {inj_dir}"}
    names0, _ = _load(gen_files[0])
    up_mask = np.array([
        (_layer_of(n) is not None) and _is_upstream(n, _layer_of(n), site)
        for n in names0])
    n_up = int(up_mask.sum())
    worst = 0.0
    worst_feat = None
    n_nonzero = 0
    offenders: list[str] = []
    for gf in gen_files:
        gid = gf.stem.replace("gen_", "")
        fnames, finj = _load(gf)
        fbase_names, fbase = _load(floor_sig / f"gen_{gid}.npz")
        if not np.array_equal(fnames, fbase_names):
            return {"error": f"feature-name mismatch for gen {gid}"}
        diff = np.abs(finj - fbase)[up_mask]
        nz = np.nonzero(diff)[0]
        if nz.size:
            n_nonzero += int(nz.size)
            up_names = fnames[up_mask]
            for i in nz[:5]:
                offenders.append(f"{up_names[i]}={diff[i]:.3e}(gen{gid})")
            j = int(np.argmax(diff))
            if diff[j] > worst:
                worst, worst_feat = float(diff[j]), str(up_names[j])
    return {"site": site, "n_upstream_features": n_up, "n_gens": len(gen_files),
            "n_nonzero_upstream": n_nonzero, "max_abs_upstream_delta": worst,
            "worst_feature": worst_feat, "offenders_sample": offenders[:10],
            "PASS": n_nonzero == 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mt-root", type=Path, required=True)
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=12, help="gens checked per cell")
    ap.add_argument("--map-site", type=int, required=True,
                    help="injection site for site-independent keys (randoms R*)")
    args = ap.parse_args()

    floor_sig = args.stage0_run / "signatures_v3"
    cells = sorted(d for d in args.mt_root.iterdir()
                   if d.is_dir() and (d / "signatures_v3").is_dir())
    report: dict = {"mt_root": str(args.mt_root), "sample": args.sample, "cells": {}}
    all_pass = True
    for cd in cells:
        key = cd.name.rsplit("_a", 1)[0]
        lay = _layer_of(key)
        if lay is None:            # site-independent randoms (R1-R3) inject at the map site
            lay = args.map_site
        res = check_cell(cd, floor_sig, lay, args.sample)
        report["cells"][cd.name] = res
        status = "PASS" if res.get("PASS") else f"FAIL {res.get('worst_feature')}"
        logger.info(f"{cd.name} (site L{lay}): {res.get('n_upstream_features')} upstream, "
                    f"max|Δ|={res.get('max_abs_upstream_delta')} → {status}")
        all_pass &= bool(res.get("PASS"))
    report["ALL_PASS"] = all_pass
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    logger.info(f"upstream-zero → {args.out}  ALL_PASS={all_pass}")
    if not all_pass:
        raise SystemExit("UPSTREAM-ZERO FAIL — quarantine bank, stop-and-surface (14h)")


if __name__ == "__main__":
    main()
