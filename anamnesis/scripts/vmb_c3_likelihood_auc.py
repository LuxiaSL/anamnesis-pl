"""C3 certifying (c) — LIKELIHOOD rung as an AUC on A1's exact ruler (14f completeness item).

A1's likelihood detector = base-model NLL (surprisal of the text under the UNSTEERED base model).
A1 separated temperature conditions by it (t09 hotter → more surprising). Here: can that same
ruler separate V_temp-steered text from matched-norm Rc-steered text? Forward each gen's tokens
under the UNSTEERED model (no injection), mean NLL over the generated span (= nll_u, the exact
quantity the entropy replay's likelihood rung reports), then AUC(Vtemp vs pooled Rc) with
GroupKFold-by-topic (leak-proof, A1's convention). 1-D score → decision_function is the NLL itself.

14f prediction: visible (≥.60) at α=.1 AND weak (<.65) at α=.03 — the co-variation structure
reproduces on the standard ruler (P=0.85). Pairs with the content rung (blind at α.03, leaks at α.1).
First-read → outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from anamnesis.config import MODEL_PRESETS
from anamnesis.scripts.vmb_c3_entropy_replay import _ent_nll_over_gen


def _per_gen_nll(model, run_dir: Path, cell: str, dev):
    d = run_dir / cell
    meta = json.loads((d / "metadata.json").read_text())
    gens = meta["generations"] if "generations" in meta else meta
    entries = json.loads((d / "replay_manifest.json").read_text())["entries"]
    out = []
    for g in gens:
        e = entries.get(str(g["generation_id"]))
        if e is None:
            continue
        ids = torch.tensor([e["input_ids"]], dtype=torch.long, device=dev)
        P = int(e["prompt_length"])
        if ids.shape[1] - P < 2:
            continue
        _, nll = _ent_nll_over_gen(model, ids, P, None)   # unsteered forward, base-model surprisal
        out.append((float(np.mean(nll)), str(g.get("topic_idx", g.get("topic", "")))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--c3-run-dir", type=Path, required=True)
    ap.add_argument("--sites", nargs="+", type=int, default=[14, 21])
    ap.add_argument("--alphas", nargs="+", default=["0.03", "0.1"])
    ap.add_argument("--nulls", nargs="+", default=["Rc1", "Rc2", "Rc3"])
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM
    preset = MODEL_PRESETS[args.model]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=preset.torch_dtype, attn_implementation="eager").to("cuda").eval()
    dev = next(model.parameters()).device

    rows = []
    for s in args.sites:
        for a in args.alphas:
            vt = _per_gen_nll(model, args.c3_run_dir, f"Vtemp_L{s}_a{a}", dev)
            rc = []
            for nl in args.nulls:
                rc += _per_gen_nll(model, args.c3_run_dir, f"{nl}_L{s}_a{a}", dev)
            if not vt or not rc:
                continue
            nll = np.array([x[0] for x in vt] + [x[0] for x in rc])
            y = np.r_[np.ones(len(vt)), np.zeros(len(rc))]
            grp = np.array([x[1] for x in vt] + [x[1] for x in rc])
            scores = np.zeros(len(y))
            for tr, te in GroupKFold(n_splits=5).split(nll.reshape(-1, 1), y, grp):
                # 1-D ruler: the NLL itself is the decision score (orient by train means)
                sign = 1.0 if nll[tr][y[tr] == 1].mean() >= nll[tr][y[tr] == 0].mean() else -1.0
                scores[te] = sign * nll[te]
            auc = float(roc_auc_score(y, scores))
            rows.append({"cell": f"Vtemp_L{s}_a{a}", "site": s, "alpha_frac": float(a),
                         "n_vtemp": len(vt), "n_rc": len(rc),
                         "mean_nll_vtemp": round(float(nll[y == 1].mean()), 4),
                         "mean_nll_rc": round(float(nll[y == 0].mean()), 4),
                         "likelihood_auc_vs_Rc": round(auc, 4)})
            print(f"  Vtemp_L{s}_a{a}: likelihood AUC vs Rc = {auc:.3f} "
                  f"(nll {nll[y==1].mean():.3f} vs {nll[y==0].mean():.3f})")

    out = {"model": args.model, "arm": "C3 certifying (c) — likelihood rung AUC on A1's ruler (14f item)",
           "STATUS": "FIRST_READ_PENDING (C§8)",
           "law": "base-model NLL (unsteered forward, gen span) → AUC(Vtemp vs pooled Rc), "
                  "GroupKFold-by-topic, 1-D ruler; 14f P=0.85: ≥.60 at α.1 AND <.65 at α.03",
           "rows": rows}
    args.out_json.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
