"""A7 MoE-perturbation smoke (vmb arm A7, M6) — validates attach_moe_perturbation on the REAL model.

Loads DeepSeek-V2-Lite once, runs a fixed teacher-forced forward, and for each perturbation mode
asserts the effective routing changed as intended + is seed-reproducible + remove() restores baseline
EXACTLY (the §7 rider: "the offline test asserts the effective selection count per rung"). GPU, 1 device.

Run (node1, Heimdall custom or direct):
  python -m anamnesis.scripts.vmb_a7_perturb_smoke --model-path /models/anamnesis-extract/dsv2-lite-chat
"""
from __future__ import annotations

import argparse

import torch

from anamnesis.extraction.model_loader import (
    MoEPerturbSpec,
    _moe_modules,
    attach_moe_perturbation,
)


def _router_selection(model, input_ids):
    """Run a forward; return per-MoE-layer (topk_idx, topk_weight, shared_norm) by monkey-reading
    route_tokens_to_experts + shared branch. Uses a temporary capture wrapper (removed after)."""
    caps: dict[int, dict] = {}
    restores = []
    for l_idx, mlp in _moe_modules(model):
        orig = mlp.route_tokens_to_experts

        def _cap(router_logits, _orig=orig, _l=l_idx):
            idx, w = _orig(router_logits)
            caps[_l] = {"idx": idx.detach().cpu(), "w": w.detach().cpu()}
            return idx, w

        # Preserve any installed perturbation (also an instance attr on the same key): restore IT
        # after capture, not the class method — else this harness clobbers the perturbation under test.
        prev = mlp.__dict__.get("route_tokens_to_experts", None)
        mlp.route_tokens_to_experts = _cap
        if prev is None:
            restores.append(lambda m=mlp: m.__dict__.pop("route_tokens_to_experts", None))
        else:
            restores.append(lambda m=mlp, p=prev: setattr(m, "route_tokens_to_experts", p))
        sh = {}

        def _shcap(mod, a, o, _l=l_idx, _sh=sh):
            _sh["n"] = float(o.reshape(-1, o.shape[-1]).norm(dim=-1).mean())
            caps.setdefault(_l, {})["shared_norm"] = _sh["n"]

        h = mlp.shared_experts.register_forward_hook(_shcap)
        restores.append(h.remove)
    with torch.no_grad():
        model(input_ids=input_ids, use_cache=False)
    for r in restores:
        r()
    return caps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    ids = tok("The committee deliberated over the routing of scarce resources.",
              return_tensors="pt").input_ids.cuda()

    base = _router_selection(model, ids)
    l0 = sorted(base)[len(base) // 2]                 # a mid MoE layer
    k_base = base[l0]["idx"].shape[-1]
    print(f"baseline: {len(base)} MoE layers, native k={k_base}, mid layer L{l0}")
    fails = []

    # ── top-k ladder: effective selection count == the set k ──
    for k in (4, 2, 1):
        h = attach_moe_perturbation(model, MoEPerturbSpec(mode="topk", top_k=k))
        c = _router_selection(model, ids)
        got = c[l0]["idx"].shape[-1]
        ok = got == k
        print(f"  topk={k}: effective selection count L{l0} = {got} {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"topk {k}: got {got}")
        h.remove()

    # remove restored baseline EXACTLY (selection identical to pre-perturbation)
    c = _router_selection(model, ids)
    restored = torch.equal(c[l0]["idx"], base[l0]["idx"]) and torch.equal(c[l0]["w"], base[l0]["w"])
    print(f"  remove() restores baseline: {'OK' if restored else 'FAIL'}")
    if not restored:
        fails.append("remove did not restore baseline")

    # ── router-noise: changes selection + seed-reproducible ──
    sigma = {l: 1.0 for l in base}
    h = attach_moe_perturbation(model, MoEPerturbSpec(mode="noise", eps=1.0, sigma_logit=sigma, seed=7))
    n1 = _router_selection(model, ids)
    n2 = _router_selection(model, ids)
    changed = not torch.equal(n1[l0]["idx"].sort(-1).values, base[l0]["idx"].sort(-1).values)
    repro = torch.equal(n1[l0]["idx"], n2[l0]["idx"]) and torch.equal(n1[l0]["w"], n2[l0]["w"])
    print(f"  noise eps=1.0: selection changed={changed} seed-reproducible={repro} "
          f"{'OK' if changed and repro else 'FAIL'}")
    if not (changed and repro):
        fails.append(f"noise changed={changed} repro={repro}")
    h.remove()

    # ── shared-expert ablation: shared branch norm → 0 ──
    h = attach_moe_perturbation(model, MoEPerturbSpec(mode="shared_ablate"))
    c = _router_selection(model, ids)
    sn = c[l0].get("shared_norm", None)
    ok = sn is not None and sn == 0.0
    print(f"  shared_ablate: shared branch mean-norm L{l0} = {sn} {'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"shared_ablate norm={sn}")
    h.remove()

    # ── drop_topm / drop_randm: m weights zeroed per token ──
    for mode in ("drop_topm", "drop_randm"):
        h = attach_moe_perturbation(model, MoEPerturbSpec(mode=mode, m=2, seed=3))
        c = _router_selection(model, ids)
        zeros = int((c[l0]["w"] == 0).sum(-1).float().mean().round())
        ok = zeros == 2
        print(f"  {mode} m=2: mean zeroed weights/token L{l0} = {zeros} {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(f"{mode}: zeroed {zeros}")
        h.remove()

    print("\n" + ("A7 PERTURBATION SMOKE: PASS" if not fails else f"FAIL: {fails}"))
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
