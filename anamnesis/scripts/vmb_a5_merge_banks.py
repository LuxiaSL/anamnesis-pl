"""Merge selected vector keys from several a5-vector banks into ONE gen-ready bank.

The V7-stack build produces several separate banks (V7+Rband from stage-2, RA from annex_14r,
V3/V4 from the base bank). `vmb_a5_gen_multicell` wants a SINGLE `--inject-npz` holding every
inject_key plus an `--inject-norms-json` carrying `median_resid_norms`. This is the numpy merge
step of the recipe (diary 2026-07-19 §"V7-stack recipe" steps 5/final) promoted to a tracked,
reusable primitive (also the M7 gen-bank assembler).

Each `--source` is `PATH:key1,key2,...` (keys copied verbatim from that npz). `--norms-from` is a
stamps json whose `median_resid_norms` (or top-level `L{n}` map) is carried into the output stamps
AND written to `<out-dir>/norms.json` for `--inject-norms-json`. Vector orthogonality/sign are NOT
touched here — sign-anchoring happens at build time, upstream. Duplicate keys across sources ->
hard error (never silently pick one). Output: `<out-dir>/a5_vectors.npz` + `a5_vectors_stamps.json`
+ `norms.json`.

    python -m anamnesis.scripts.vmb_a5_merge_banks --out-dir $BANK/a5_vectors_qwen_7b_v7stack \
        --source $V7S2/a5_vectors.npz:V7_L21,Rband1_L21,Rband2_L21,Rband3_L21 \
        --source $RA/a5_vectors.npz:RA_L21 \
        --source $BANK/a5_vectors_qwen_7b/a5_vectors.npz:V3_L21 \
        --norms-from $BANK/a5_vectors_qwen_7b/a5_vectors_stamps.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_norms(stamps_path: Path) -> dict[str, float]:
    """Pull median_resid_norms from a stamps json (nested or top-level L{n} form)."""
    s = json.loads(stamps_path.read_text())
    if isinstance(s.get("median_resid_norms"), dict):
        return {k: float(v) for k, v in s["median_resid_norms"].items()}
    # fall back: any top-level "L<int>": number entries
    out = {k: float(v) for k, v in s.items()
           if isinstance(k, str) and k.startswith("L") and k[1:].isdigit()
           and isinstance(v, (int, float))}
    if not out:
        raise SystemExit(f"no median_resid_norms found in {stamps_path}")
    return out


def _parse_source(spec: str) -> tuple[Path, list[str]]:
    if ":" not in spec:
        raise SystemExit(f"--source must be PATH:key1,key2 (got {spec!r})")
    path_str, keys_str = spec.rsplit(":", 1)
    keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    if not keys:
        raise SystemExit(f"--source {spec!r} lists no keys")
    return Path(path_str), keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True,
                    help="PATH:key1,key2,... (repeatable)")
    ap.add_argument("--norms-from", type=Path, required=True,
                    help="stamps json carrying median_resid_norms")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default=None, help="stamp label only")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    merged: dict[str, np.ndarray] = {}
    stamps: dict[str, dict] = {}
    for spec in args.source:
        path, keys = _parse_source(spec)
        if not path.exists():
            raise SystemExit(f"source npz missing: {path}")
        d = np.load(path)
        avail = set(d.keys())
        for k in keys:
            if k not in avail:
                raise SystemExit(f"key {k!r} not in {path} (have: {sorted(avail)[:12]}…)")
            if k in merged:
                raise SystemExit(f"duplicate key {k!r} across sources — refusing to guess")
            merged[k] = np.asarray(d[k], dtype=np.float32)
            stamps[k] = {"source": str(path), "raw_norm": float(np.linalg.norm(d[k]))}

    norms = _load_norms(args.norms_from)
    out_stamps = {"vectors": stamps, "median_resid_norms": norms,
                  "provenance": f"vmb_a5_merge_banks; sources={args.source}"}
    if args.model:
        out_stamps["model"] = args.model

    np.savez(args.out_dir / "a5_vectors.npz", **merged)
    (args.out_dir / "a5_vectors_stamps.json").write_text(json.dumps(out_stamps, indent=1))
    (args.out_dir / "norms.json").write_text(json.dumps({"median_resid_norms": norms}, indent=1))
    print(f"merged {len(merged)} keys -> {args.out_dir}/a5_vectors.npz")
    for k in merged:
        print(f"  {k}: |v|={stamps[k]['raw_norm']:.4f}")
    print(f"norms sites: {sorted(norms)} -> {args.out_dir}/norms.json")


if __name__ == "__main__":
    main()
