"""14h Qwen-gate contingency rider (outer-loop ruling 2026-07-15) — confirm-or-refute the
fp16 attribution of Qwen's depressed alpha=0 self-agreement (check C = .914 at alpha=0,
no injection).

Method: greedy-generate with the SMOKE's own incremental path (fidelity to check C),
then teacher-force the SAME realized tokens through a full forward at fp16 vs fp32
(same weights). Self-agreement = argmax-vs-realized top-1.
  fp32 self-agreement -> ~>=.99  => fp16-path attribution CONFIRMED (record says so).
  fp32 stays ~.91                => NOT dtype-explained; a structural incremental-vs-full
                                    path asymmetry that could reach beyond the gate ->
                                    STOP-AND-SURFACE again.
Diagnosis-verification ONLY; the Qwen MT/gate rung stays DROPPED regardless (Option 1).
n~=20. First-read -> outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from anamnesis.config import MODEL_PRESETS
from anamnesis.scripts._a5_common import teacher_forced_agreement
from anamnesis.scripts.vmb_a5_smoke_incremental import _generate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _agrees(model, realized: list[tuple[list[int], int]]) -> list[float]:
    return [teacher_forced_agreement(model, ids, P) for ids, P in realized]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_PRESETS.keys()), required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--stage0-run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--compute-dtype", choices=["float16", "float32"], default="float16",
                    help="dtype for BOTH generation and the primary teacher-forced pass "
                         "(self-consistent). float32 = the airtight no-path-asymmetry run.")
    args = ap.parse_args()

    preset = MODEL_PRESETS[args.model]
    eos = list(preset.eos_token_ids)
    dt = {"float16": torch.float16, "float32": torch.float32}[args.compute_dtype]
    other_dt = torch.float32 if dt == torch.float16 else torch.float16

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dt, attn_implementation="eager").to("cuda").eval()

    entries = json.loads((args.stage0_run / "replay_manifest.json").read_text())["entries"]
    gen_kw = dict(max_new_tokens=args.max_new_tokens, eos_ids=eos, pad_id=pad,
                  temperature=float(preset.temperature), top_p=0.9)

    realized: list[tuple[list[int], int]] = []
    for k in range(args.n):
        e = entries.get(str(k * 40))
        if not e:
            continue
        P = int(e["prompt_length"])
        ids = torch.tensor([e["input_ids"][:P]], dtype=torch.long, device="cuda")
        r = _generate(model, ids, seed=0, greedy=True, **gen_kw)
        if len(r) - P >= 8:
            realized.append((r, P))
    logger.info(f"{len(realized)} greedy self-generations")

    # SELF: generation and teacher-forcing BOTH at compute_dtype (the meaningful number —
    # the regime the cells actually run in). CROSS: TF the same tokens at the OTHER dtype
    # (diagnostic only; a cross-dtype argmax-divergence artifact, NOT a path asymmetry).
    self_ag = _agrees(model, realized)
    logger.info(f"{args.compute_dtype} SELF-agreement mean {np.mean(self_ag):.4f} "
                f"min {np.min(self_ag):.4f}")

    model = model.to(other_dt)
    cross = _agrees(model, realized)
    logger.info(f"cross-dtype TF ({other_dt}) mean {np.mean(cross):.4f} min {np.min(cross):.4f}")

    # for the fp32 confirmatory: SELF >= .98 => no structural path asymmetry (14f P=.85).
    no_path_asymmetry = bool(np.mean(self_ag) >= 0.98)
    res = {"model": args.model, "n": len(realized), "compute_dtype": args.compute_dtype,
           "self_agreement_mean": float(np.mean(self_ag)),
           "self_agreement_min": float(np.min(self_ag)),
           "cross_dtype_TF_mean": float(np.mean(cross)),
           "cross_dtype_TF_min": float(np.min(cross)),
           "self_agreement_ge_0.98": no_path_asymmetry,
           "note": ("SELF = gen+TF both at compute_dtype (the operative regime); "
                    "CROSS = TF at the other dtype on these tokens (cross-dtype argmax "
                    "divergence, expected, not a path asymmetry).")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    logger.info(f"dtype rider -> {args.out}\n{json.dumps(res, indent=2)}")


if __name__ == "__main__":
    main()
