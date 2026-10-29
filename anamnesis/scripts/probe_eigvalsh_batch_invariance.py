"""Is ``torch.linalg.eigvalsh`` batchmate-invariant? Each matrix's eigenvalues, solved alone and
inside batches of other matrices, compared byte for byte.

The matrices are the attention reduction's spectral Laplacians in form: ``diag(d) - G + 1e-10 I``
with ``G`` a nonnegative symmetric Gram matrix of unit rows, zero diagonal, ``k x k`` for the
spectral step counts a span can have. Per ``k`` and device, a matrix is solved as ``[k, k]``,
as ``[1, k, k]``, and at position 0 and at a random position inside batches of ``--sizes``
random mates; any differing byte is reported. CUDA dispatches small batched matrices to a
different cuSOLVER routine than single ones, so the single-matrix forms are reported apart.

Usage: python probe_eigvalsh_batch_invariance.py --out result.json [--ks 3,5,13,26,32,33,52]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def laplacians(torch, count: int, k: int, width: int, seed: int, device):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(count, k, width, generator=g, dtype=torch.float64)
    x = x * (torch.rand(count, k, width, generator=g, dtype=torch.float64) < 0.3)
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    graph = (x @ x.transpose(1, 2)).clamp_min(0)
    graph = (graph + graph.transpose(1, 2)) * 0.5
    graph.diagonal(dim1=1, dim2=2).zero_()
    lap = torch.diag_embed(graph.sum(-1)) - graph + torch.eye(k, dtype=torch.float64) * 1e-10
    return lap.to(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="2,3,5,13,26,32,33,52")
    ap.add_argument("--sizes", default="2,8,10,80,128,1280")
    ap.add_argument("--targets", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import torch

    torch.use_deterministic_algorithms(True)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    sizes = [int(s) for s in args.sizes.split(",")]
    report: dict = dict(torch=torch.__version__,
                        cuda=torch.version.cuda, results={})
    for device in devices:
        for k in (int(v) for v in args.ks.split(",")):
            pool = laplacians(torch, max(sizes) + args.targets, k, 400, k, device)
            targets = pool[: args.targets]
            mates = pool[args.targets:]
            ref_2d = torch.stack([torch.linalg.eigvalsh(t) for t in targets])
            ref_1 = torch.stack([torch.linalg.eigvalsh(t[None])[0] for t in targets])
            pair = torch.stack([torch.linalg.eigvalsh(torch.stack([t, mates[0]]))[0]
                                for t in targets])
            entry = dict(single_2d_vs_batch1=int((ref_2d != ref_1).sum()),
                         batch1_vs_batch2=int((ref_1 != pair).sum()), batches={})
            gen = torch.Generator().manual_seed(1)
            for size in sizes:
                moved_first = moved_random = 0
                for i, t in enumerate(targets):
                    batch = torch.cat([t[None], mates[: size - 1]])
                    moved_first += int((torch.linalg.eigvalsh(batch)[0] != pair[i]).sum())
                    place = int(torch.randint(0, size, (1,), generator=gen))
                    batch = torch.cat([mates[:place], t[None], mates[place: size - 1]])
                    moved_random += int((torch.linalg.eigvalsh(batch)[place] != pair[i]).sum())
                entry["batches"][str(size)] = dict(moved_at_0=moved_first,
                                                   moved_at_random=moved_random)
            entry["vs_cpu_max_abs"] = float((ref_2d.cpu() - torch.linalg.eigvalsh(
                targets.cpu())).abs().max()) if device == "cuda" else 0.0
            report["results"][f"{device}-k{k}"] = entry
            print(device, k, json.dumps(entry), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
