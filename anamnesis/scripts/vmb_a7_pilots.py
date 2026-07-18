"""A7 pilots (vmb arm A7, M6) — measure the two ladder parameters on real floor gens.

Per SPEC-A7-BLOCK §2b/§2c + §8: before the ladders fire, RECORD
  (1) sigma_logit[L]  — per-MoE-layer std of the router logits (the noise ε-ladder scale;
                        ε then = fraction of that layer's own commitment scale, desk ruling 2).
  (2) mass profile    — mean shared_mass + the sorted top-k routed-weight profile per layer, and the
                        mass-matched m: the smallest m whose dropped top-m routed weight-fraction ≥
                        the shared branch's mass share, so shared-ablate and top-m-drop remove
                        comparable mass (the fair comparison; desk control triangle).

Teacher-forced over N floor continuations (their banked tokens). GPU, 1 device. Writes a JSON.

Run (node1): python -m anamnesis.scripts.vmb_a7_pilots --model-path <MP> \
  --floor-run-dir /models/anamnesis-extract/runs/vmb_stage0_dsv2_lite --n 20 --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from anamnesis.extraction.model_loader import _moe_modules


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--floor-run-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()

    manifest = json.loads((args.floor_run_dir / "replay_manifest.json").read_text())
    entries = manifest["entries"]
    gids = sorted(int(k) for k in entries)[: args.n]

    # accumulators per MoE layer
    mods = _moe_modules(model)
    layer_ids = [l for l, _ in mods]
    logit_std: dict[int, list] = {l: [] for l in layer_ids}
    shared_mass: dict[int, list] = {l: [] for l in layer_ids}
    weight_profile: dict[int, list] = {l: [] for l in layer_ids}   # sorted top-k weight fractions

    restores = []
    for l_idx, mlp in mods:
        orig = mlp.route_tokens_to_experts

        def _cap(router_logits, _orig=orig, _l=l_idx):
            rl = router_logits.detach().float()
            logit_std[_l].append(float(rl.std(dim=-1).mean()))          # mean over tokens of per-token σ
            idx, w = _orig(router_logits)
            wf = w.detach().float()
            wsum = wf.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            prof = (wf / wsum).sort(dim=-1, descending=True).values.mean(dim=0)  # [k] mean sorted frac
            weight_profile[_l].append(prof.cpu().numpy())
            return idx, w

        mlp.route_tokens_to_experts = _cap
        restores.append(lambda m=mlp, o=orig: setattr(m, "route_tokens_to_experts", o))

        def _sh(mod, a, o, _l=l_idx):
            shared_mass[_l].append(o.detach().float().reshape(-1, o.shape[-1]).norm(dim=-1).mean().item())

        h = mlp.shared_experts.register_forward_hook(_sh)
        restores.append(h.remove)

    # also capture routed-branch norm to form shared_mass = shared/(shared+routed)
    routed_norm: dict[int, list] = {l: [] for l in layer_ids}
    for l_idx, mlp in mods:
        def _ro(mod, a, o, _l=l_idx):
            routed_norm[_l].append(o.detach().float().reshape(-1, o.shape[-1]).norm(dim=-1).mean().item())
        restores.append(mlp.experts.register_forward_hook(_ro).remove)

    with torch.no_grad():
        for gid in gids:
            ids = torch.tensor([entries[str(gid)]["input_ids"]], device="cuda")
            model(input_ids=ids, use_cache=False)

    for r in restores:
        r()

    # ── reduce ──
    out = {"model": "dsv2-lite", "n_gens": len(gids), "layers": layer_ids, "per_layer": {}}
    sigma_logit = {}
    mass_m = {}
    for l in layer_ids:
        sig = float(np.mean(logit_std[l]))
        sh = float(np.mean(shared_mass[l]))
        ro = float(np.mean(routed_norm[l]))
        smass = sh / (sh + ro) if (sh + ro) > 0 else 0.0
        prof = np.mean(np.stack(weight_profile[l]), axis=0)          # [k] mean sorted weight fractions
        # mass-matched m: smallest m whose cumulative top-m routed weight-fraction ≥ shared mass share.
        # shared adds ~‖shared‖; routed adds ~‖routed‖; matching removed-routed-mass to shared ⇒
        # top-m fraction ≥ smass/(1-smass).
        target = smass / (1.0 - smass) if smass < 1.0 else 1.0
        cum = np.cumsum(prof)
        m = int(np.searchsorted(cum, target) + 1)
        m = max(1, min(m, len(prof)))
        sigma_logit[l] = sig
        mass_m[l] = m
        out["per_layer"][str(l)] = {
            "sigma_logit": round(sig, 4), "shared_mass": round(smass, 5),
            "weight_profile_sorted": [round(float(x), 4) for x in prof],
            "target_dropfrac": round(float(target), 5), "mass_matched_m": m,
        }
    # program-level recommendations (median across layers)
    out["sigma_logit_per_layer"] = {str(l): round(sigma_logit[l], 4) for l in layer_ids}
    out["mass_matched_m_per_layer"] = {str(l): mass_m[l] for l in layer_ids}
    out["mass_matched_m_median"] = int(np.median(list(mass_m.values())))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")
    print("sigma_logit per layer:", out["sigma_logit_per_layer"])
    print("shared_mass per layer:", {str(l): out['per_layer'][str(l)]['shared_mass'] for l in layer_ids})
    print("mass_matched_m per layer:", out["mass_matched_m_per_layer"], "| median m =", out["mass_matched_m_median"])


if __name__ == "__main__":
    main()
