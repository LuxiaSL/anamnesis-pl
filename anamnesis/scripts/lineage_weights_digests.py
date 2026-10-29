"""The weights-v1 digest of each checkpoint, as a kotodama signer computes it for the rows it signs.

Each checkpoint is loaded with kotodama's serving loader on the drain's device (``kotodama.serve.server.load_model``), and
``anamnesis.extraction.weights_digest.weights_digest`` is taken of the loaded model. That is the digest a drain's
rows carry, so it's the ``expected_weights`` the drain's EngineReducer binds. It runs inside a kotodama tree
(``tools/run_py.sh``) with anamnesis on PYTHONPATH.

Usage: python lineage_weights_digests.py --checkpoint release=<path> --checkpoint sft=<path> ... --out <json>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, action="append", help="label=<path>")
    ap.add_argument("--model-size", default="1b-tsumugi")
    ap.add_argument("--device", default="cuda", help="load as the drain does (it loads on cuda)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    import torch
    from anamnesis.extraction.weights_digest import weights_digest
    from kotodama.serve import server

    out: dict[str, dict[str, object]] = {}
    for spec in args.checkpoint:
        label, _, path = spec.partition("=")
        if not label or not path:
            raise SystemExit(f"--checkpoint wants label=<path>, got {spec!r}")
        t0 = time.time()
        try:
            model, _ = server.load_model(path, args.device, mode="base", model_size=args.model_size,
                                         engine="reference")
            digest = weights_digest(model)
        except Exception as exc:  # report every checkpoint, even when one fails
            out[label] = dict(path=path, error=f"{type(exc).__name__}: {exc}")
            print(json.dumps({label: out[label]}), flush=True)
            continue
        out[label] = dict(path=path, weights_sha256=digest, seconds=round(time.time() - t0, 1))
        print(json.dumps({label: out[label]}), flush=True)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
