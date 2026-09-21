"""Computer #5 install-lever PoC — S6: the propagation gate. THE DECISIVE PRE-BEHAVIORAL CHECK.

SPEC §6 G4. Behaviour-free. Teacher-force fixed sequences through base twice — once clean, once
with the residual write at the site — and measure:
  * per-layer ||Delta residual|| and the ON-AXIS component  -> does the write survive the depth?
  * KL(clean || injected) over the completion positions     -> does it reach the LOGITS?

WHY THIS GATE EXISTS. Koto's single-site write died at a block boundary: |Delta| collapsed ~100x,
downstream on-axis went to ~0, and the final-logit KL was **0.0021 nats**. Two full marker cells
came back NULL and the cause was the architecture eating the write, not the vector. Banked
reference points: real lever ~0.21 nats, junk ~0.54, dead write 0.0021. Llama-70B is dense with no
AttnRes-style commit-and-reset, so a single-site write SHOULD survive — this measures it instead
of hoping, for half an hour.

Also the STANDING WARNING (koto §5.7, D1X): the sensor-optimal site is not the actuation site.
Koto's L18 read best and never actuated at any tolerable dose; L14 actuated. If our alignment
diagnostic and this gate disagree, THIS WINS.

LOGITS NOTE: vLLM computes logits only at positions it needs to sample, so there is no all-position
logits tensor to read. We hook `model.model` (LlamaModel), whose forward returns the final
post-norm hidden states for every position, and apply lm_head ourselves. That is the same
computation vLLM would do, just at every position.

CONVENTION (pinned, spec §10c): residual stream at layer s = pre-hook arg1 + arg2. Injection adds
to arg1; since the layer immediately computes norm(arg1+arg2), adding to either is identical.

    c5_propagation_check.py --model-path /models/Llama-3.1-70B --vectors install_c5.npz \
        --floors floors_base.jsonl --site 38 --alphas 0.05 0.1 0.3 --n 12
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Banked reference KLs (nats) — the interpretive scale for the number this script produces.
REF_REAL_LEVER = 0.21
REF_JUNK = 0.54
REF_KOTO_DEAD = 0.0021


def find_model(llm) -> torch.nn.Module:
    engine = getattr(llm, "llm_engine", llm)
    for path in ("engine_core.engine_core.model_executor.driver_worker.model_runner.model",
                 "engine_core.model_executor.driver_worker.model_runner.model",
                 "model_executor.driver_worker.model_runner.model"):
        obj = engine
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.Module):
            return obj
    sys.exit("FATAL: model not reachable — VLLM_ENABLE_V1_MULTIPROCESSING=0 and tp=1 required")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--vectors", type=Path, required=True)
    ap.add_argument("--floors", type=Path, required=True)
    ap.add_argument("--site", type=int, required=True)
    ap.add_argument("--vector-keys", nargs="+", default=None,
                    help="default: Install_L<site>, InstallW_L<site>, Rband_L<site>")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.3])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--min-completion", type=int, default=24)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    args = ap.parse_args()

    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        sys.exit("FATAL: VLLM_ENABLE_V1_MULTIPROCESSING=0 required")

    vz = dict(np.load(args.vectors, allow_pickle=True))
    keys = args.vector_keys or [f"Install_L{args.site}", f"InstallW_L{args.site}",
                                f"Rband_L{args.site}"]
    for k in keys:
        if k not in vz:
            sys.exit(f"FATAL: {k} not in {args.vectors}. have: {sorted(vz)}")
    med_key = f"median_norm_L{args.site}"
    if med_key not in vz:
        sys.exit(f"FATAL: {med_key} missing — cannot resolve the dose currency")
    median_norm = float(vz[med_key])

    rows = [json.loads(l) for l in args.floors.read_text(encoding="utf-8").splitlines() if l.strip()]
    seqs = []
    for r in rows:
        pids, cids = r.get("prompt_ids") or [], r.get("completion_ids") or []
        if len(cids) >= args.min_completion:
            seqs.append((list(pids) + list(cids), len(pids), len(pids) + len(cids)))
        if len(seqs) >= args.n:
            break
    if len(seqs) < 4:
        sys.exit(f"FATAL: only {len(seqs)} usable sequences (need completions >= "
                 f"{args.min_completion})")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model=args.model_path, tensor_parallel_size=1, dtype="bfloat16",
              max_model_len=1024, gpu_memory_utilization=args.gpu_mem_util,
              enforce_eager=True, enable_prefix_caching=False, seed=20260729)
    model = find_model(llm)
    layers = model.model.layers
    n_layers = len(layers)
    dev = next(model.parameters()).device

    state: dict[str, object] = {"inject": None, "w0": 0, "w1": 0}
    resid: dict[int, torch.Tensor] = {}
    final_hidden: dict[str, torch.Tensor] = {}

    def mk_capture(s: int):
        def hook(mod, a, kw):
            if len(a) >= 2 and isinstance(a[1], torch.Tensor):
                h = a[1]
                r = a[2] if len(a) >= 3 and isinstance(a[2], torch.Tensor) else None
                resid[s] = (h + r).detach() if r is not None else h.detach()
            return None
        return hook

    def inject_hook(mod, a, kw):
        """Add alpha*v to the residual over COMPLETION positions only. Adding to arg1 is
        equivalent to adding to arg2: the layer's first act is norm(arg1+arg2)."""
        vec = state["inject"]
        if vec is None or not a or not isinstance(a[1], torch.Tensor):
            return None
        h = a[1]
        w0, w1 = int(state["w0"]), int(state["w1"])
        if h.shape[0] < w1:      # decode step, not the prefill we are measuring
            return None
        h2 = h.clone()
        h2[w0:w1] = h2[w0:w1] + vec.to(h.dtype)
        return (a[0], h2) + tuple(a[2:]), kw

    def final_layer_hook(mod, a, out):
        """Capture the LAST decoder layer's OUTPUT. vLLM returns (hidden, residual) still split,
        so the pre-norm final residual is their sum — same convention as everywhere else here.
        (Hooking `model.model` was tried first and yielded nothing usable; a post-hook on the
        final layer uses exactly the machinery the capture pass already proved.)"""
        outs = out if isinstance(out, tuple) else (out,)
        ts = [t for t in outs if isinstance(t, torch.Tensor)]
        if not ts:
            return None
        h = ts[0]
        r = ts[1] if len(ts) > 1 else None
        final_hidden["h"] = ((h + r) if r is not None else h).detach()
        return None

    # ⚠ HOOK ORDER IS LOAD-BEARING. PyTorch runs pre-hooks in registration order and passes each
    # one the previous hook's modified args. Registering the capture hooks first made the capture
    # at the injection site read the PRE-write residual, so |Delta|@site measured exactly 0.00
    # while downstream layers plainly moved. Injection must be registered FIRST so the site's
    # capture sees the written value.
    inj = layers[args.site].register_forward_pre_hook(inject_hook, with_kwargs=True)
    caps = [layers[s].register_forward_pre_hook(mk_capture(s), with_kwargs=True)
            for s in range(n_layers)]
    fin = layers[n_layers - 1].register_forward_hook(final_layer_hook)
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    def run(ids, w0, w1, vec):
        state.update({"inject": vec, "w0": w0, "w1": w1})
        resid.clear(); final_hidden.clear()
        llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
        snap = {s: resid[s][:w1].float().clone() for s in range(n_layers)
                if s in resid and resid[s].shape[0] >= w1}
        h = final_hidden.get("h")
        if h is None:
            logger.warning("drop: no final-layer capture (hook never fired)")
            return snap, None
        if h.shape[0] < w1:
            logger.warning("drop: final capture has %d rows < window end %d (decode-step "
                           "overwrite?)", h.shape[0], w1)
            return snap, None
        if len(snap) < n_layers:
            logger.warning("drop: only %d/%d layers captured at full length", len(snap), n_layers)
        return snap, h[:w1].float().clone()

    lm_head_w = model.lm_head.weight
    results: dict[str, object] = {}

    for key in keys:
        v_np = vz[key].astype(np.float32)
        results[key] = {}
        for alpha in args.alphas:
            scale = alpha * median_norm
            v = torch.tensor(v_np, device=dev, dtype=torch.float32) * scale
            v_unit = torch.tensor(v_np, device=dev, dtype=torch.float32)
            per_layer_dnorm = np.zeros(n_layers)
            per_layer_onaxis = np.zeros(n_layers)
            kls, n_ok = [], 0
            for ids, w0, w1 in seqs:
                clean_r, clean_h = run(ids, w0, w1, None)
                inj_r, inj_h = run(ids, w0, w1, v)
                if clean_h is None or inj_h is None or len(clean_r) < n_layers:
                    continue
                for s in range(n_layers):
                    d = (inj_r[s][w0:w1] - clean_r[s][w0:w1])
                    per_layer_dnorm[s] += float(d.norm(dim=-1).mean())
                    per_layer_onaxis[s] += float((d @ v_unit).mean())
                # logits = lm_head( final_norm(h) ). Skipping the norm would compare two
                # unnormalised streams and the KL would be meaningless.
                with torch.no_grad():
                    n0 = model.model.norm(clean_h[w0:w1].to(lm_head_w.dtype))
                    n1 = model.model.norm(inj_h[w0:w1].to(lm_head_w.dtype))
                    n0 = n0[0] if isinstance(n0, tuple) else n0
                    n1 = n1[0] if isinstance(n1, tuple) else n1
                lg0 = (n0.float() @ lm_head_w.T.float())
                lg1 = (n1.float() @ lm_head_w.T.float())
                p = torch.log_softmax(lg0, dim=-1)
                q = torch.log_softmax(lg1, dim=-1)
                kls.append(float(F.kl_div(q, p, log_target=True, reduction="batchmean")))
                n_ok += 1
                del lg0, lg1, p, q
            if not n_ok:
                logger.warning("%s alpha=%.3f: no usable sequences", key, alpha)
                continue
            per_layer_dnorm /= n_ok
            per_layer_onaxis /= n_ok
            at_site = float(per_layer_dnorm[args.site])
            at_final = float(per_layer_dnorm[-1])
            kl = float(np.mean(kls))
            verdict = ("DEAD — write never reaches the logits (koto's failure mode)"
                       if kl < 0.01 else
                       "LIVE — in the real-lever band" if kl >= 0.05 else
                       "WEAK — above dead, below the real-lever reference; raise dose or resite")
            results[key][str(alpha)] = {
                "alpha": alpha, "scale": round(scale, 4), "n_seqs": n_ok,
                "delta_norm_at_site": round(at_site, 3),
                "expected_scale": round(scale, 3),
                "delta_norm_final_layer": round(at_final, 3),
                "survival_ratio_final_over_site": round(at_final / max(at_site, 1e-9), 4),
                "onaxis_at_site": round(float(per_layer_onaxis[args.site]), 3),
                "onaxis_final_layer": round(float(per_layer_onaxis[-1]), 3),
                "logit_KL_nats": round(kl, 5),
                "reference": {"real_lever": REF_REAL_LEVER, "junk": REF_JUNK,
                              "koto_dead": REF_KOTO_DEAD},
                "VERDICT": verdict,
                "per_layer_delta_norm": [round(x, 3) for x in per_layer_dnorm.tolist()],
            }
            logger.info("%s a=%.3f: |D|@site=%.2f (dose %.2f) final=%.2f ratio=%.3f "
                        "KL=%.5f nats -> %s", key, alpha, at_site, scale, at_final,
                        at_final / max(at_site, 1e-9), kl, verdict.split(" —")[0])

    for h in caps + [inj, fin]:
        h.remove()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"site": args.site, "median_norm": median_norm,
                                    "n_layers": n_layers, "n_seqs": len(seqs),
                                    "results": results}, indent=1))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
