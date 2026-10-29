"""Gate 2 for Column C: does the depth family see the channel-8026 switch in Llama-3.1-70B-Instruct?

CPU only, over retained captures (no 70B forward). Inputs on node1, read-only:

* vLLM lane captures: ``lane-70b-001/l2/captures/population-capture-70b-native-001/full-b1-order0/
  row-<gid>.pt`` (``hidden`` [80, T, 8192]: the output of each layer at each span position);
* HF raw captures of the same rows: ``crossing-diagnostic-001/hf-substrate{,-controls}/row-<gid>.pt``
  (the three mode-swap rows 10011, 10018, 10022 and the three rule-chosen controls).

For every span position of every capture it computes the Column C depth family
(``column_c_features.depth_features`` on the 80-point path; the 70B geometry has no routing and
these captures carry no attention rows for HF, so the FLOOR here is depth-only) and records the
labelling quantities: channel 8026's magnitude per layer, and the per-layer largest channel.

Output (``--out``): an npz with the depth features [N, F], names, digest, row id, engine,
position, chosen token id, |h[l, 8026]| for every layer [N, 80], and the per-layer max-|h|
channel at L40. The analysis (paired and observational tests) runs on that file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from column_c_features import (Geometry, depth_features, floor_digest,  # noqa: E402
                               floor_names)

LANE = Path("/models/anamnesis-vllm-integration/lane-70b-001")
VLLM = LANE / "l2/captures/population-capture-70b-native-001/full-b1-order0"
HF_DIRS = (LANE / "crossing-diagnostic-001/hf-substrate",
           LANE / "crossing-diagnostic-001/hf-substrate-controls")
SWAP = (10011, 10018, 10022)
CONTROLS = (10012, 10003, 10007)
CHANNEL = 8026


def capture_rows(path: Path):
    import torch
    d = torch.load(path, map_location="cpu", weights_only=True)
    h = d["hidden"]
    if h.ndim != 3 or h.shape[0] != 80:
        raise ValueError(f"{path}: hidden {tuple(h.shape)}, expected [80, T, d]")
    return h, d["chosen"].numpy()


def reduce_capture(h, chosen, gid: int, engine: str, geo: Geometry):
    import torch
    T = h.shape[1]
    feats = np.empty((T, len(floor_names(geo))), dtype=np.float32)
    ch = np.empty((T, 80), dtype=np.float32)
    maxdim40 = np.empty(T, dtype=np.int64)
    for p in range(T):
        path = h[:, p].float().numpy()
        feats[p] = depth_features(path)
        ch[p] = np.abs(path[:, CHANNEL])
        maxdim40[p] = int(np.abs(path[40]).argmax())
    del torch
    return dict(feats=feats, ch8026=ch, maxdim40=maxdim40, gid=np.full(T, gid), engine=np.array([engine] * T),
                pos=np.arange(T), chosen=chosen[:T])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-population", type=int, default=34,
                    help="extra vLLM rows, drawn with seed 0 from the population capture")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)
    import torch
    torch.set_num_threads(args.threads)

    geo = Geometry(model="llama-3.1-70b-instruct", n_points=80, n_read_layers=0)
    names = floor_names(geo)
    all_ids = sorted(int(p.stem.split("-")[1]) for p in VLLM.glob("row-*.pt"))   # retained captures only
    fixed = set(SWAP) | set(CONTROLS)
    rng = np.random.default_rng(0)
    pool = [g for g in all_ids if g not in fixed]
    population = sorted(rng.choice(pool, size=min(args.n_population, len(pool)), replace=False).tolist())
    jobs = [(g, "hf", next(d for d in HF_DIRS if (d / f"row-{g:05d}.pt").exists()) / f"row-{g:05d}.pt")
            for g in (*SWAP, *CONTROLS)]
    # the controls' vLLM captures were not retained; they enter on the HF side only
    jobs += [(g, "vllm", VLLM / f"row-{g:05d}.pt") for g in (*SWAP, *population)]
    parts, t0 = [], time.time()
    for k, (gid, engine, path) in enumerate(jobs, 1):
        h, chosen = capture_rows(path)
        parts.append(reduce_capture(h, chosen, gid, engine, geo))
        print(f"[8026] {k}/{len(jobs)} row {gid} {engine}: T={h.shape[1]} ({time.time() - t0:.0f}s)", flush=True)
        del h
    out = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, names=np.array(names), digest=np.array(floor_digest(geo)),
             population=np.array(population), swap=np.array(SWAP), controls=np.array(CONTROLS), **out)
    (args.out.with_suffix(".json")).write_text(json.dumps(dict(
        geometry=geo.to_json(), digest=floor_digest(geo), n_names=len(names), jobs=[(g, e, str(p)) for g, e, p in jobs],
        n_positions=int(len(out["pos"])), seconds=round(time.time() - t0, 1)), indent=1))
    print(f"[8026] wrote {args.out}: {len(out['pos'])} positions x {len(names)} names", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
