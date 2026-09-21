"""Computer #5 install-lever PoC — S7: steered generation, the dose ladder.

SPEC §3 (S7), §5. Steers PRISTINE base-70B along the install vector during generation and reads
the frozen marker battery. Matched-norm band nulls at the same doses; both signs; alpha=0 baseline.

INJECTION WINDOW — generated positions only. vLLM's decoder layer receives `positions` as arg0
(absolute position per row), so the mask is simply `positions >= prompt_len`. That one rule is
correct during prefill AND during every decode step, with no per-step toggling and no reliance on
batch shapes. It does require every sequence in flight to share a prompt length, which is why
prompts are submitted ONE AT A TIME (with n samples as the batch). Cheap at vLLM speed.

enforce_eager=True is mandatory: CUDA graphs would capture the decode path and the hook would stop
firing after capture — silently, producing a beautiful table of unsteered numbers.

Verdict rule (pre-registered, inherited): the lever is real iff the markers move
DOSE-RESPONSIVELY, SIGN-CONSISTENTLY, OUTSIDE the R-band envelope, with COHERENCE in window.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c5_freeze_markers import coherence_scores, texture_scores  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TEMPERATURE = 1.0
TOP_P = 0.98
STOP_SEQUENCE = "\n\n**User:**"


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


def summarize(texts: list[str]) -> dict:
    tex = [texture_scores(t) for t in texts]
    coh = [coherence_scores(t) for t in texts]
    out: dict[str, float] = {}
    for k in tex[0]:
        out[k] = round(statistics.fmean(x[k] for x in tex), 4)
    for k in ("ttr", "distinct2", "max_rep6"):
        vals = [x[k] for x in coh if not np.isnan(x[k])]
        out[k] = round(statistics.fmean(vals), 4) if vals else float("nan")
    out["n"] = len(texts)
    out["n_empty"] = sum(1 for t in texts if not t.strip())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--vectors", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--site", required=True,
                    help="layer, or a comma list for MULTI-SITE injection. Each site gets its own\n dose currency (its own median residual norm), so alpha means the same fraction\n everywhere and the sites stay comparable.")
    ap.add_argument("--cells", required=True,
                    help="semicolon list of KEY:alpha,alpha,... e.g. "
                         "'Install_L38:0.15,0.3,0.45,0.6,0.8,-0.3,-0.6;Rband_L38:0.3,0.6'")
    ap.add_argument("--arm", default="substantive")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--coherence-floor", type=float, default=0.45)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--raws-dir", type=Path, default=None)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=20260729)
    args = ap.parse_args()

    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        sys.exit("FATAL: VLLM_ENABLE_V1_MULTIPROCESSING=0 required")

    vz = dict(np.load(args.vectors, allow_pickle=True))
    sites = [int(x) for x in str(args.site).split(",")]
    med_norms = {}
    for st in sites:
        mk = f"median_norm_L{st}"
        if mk not in vz:
            sys.exit(f"FATAL: {mk} missing from {args.vectors}")
        med_norms[st] = float(vz[mk])
    median_norm = med_norms[sites[0]]

    ps = json.loads(args.prompts.read_text())
    prompts = [p for p in ps["prompts"] if p["arm"] == args.arm]
    if not prompts:
        sys.exit(f"FATAL: no prompts with arm={args.arm!r}")

    cells: list[tuple[str, list[float]]] = []
    for chunk in args.cells.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        stem, alphas = chunk.split(":", 1)
        # An alpha token may be a single value (same alpha at every site) OR a "/"-separated
        # per-site list in --site order. Per-site alphas are what make a MIXING-RATIO sweep at
        # fixed total displacement expressible: equal alpha is NOT equal displacement, because
        # each site is scaled by its own median residual norm.
        # multi-site: the cell names a STEM (e.g. "Install"), and each site uses its own
        # <stem>_L<site> vector. Single-site keeps the full key for backwards compatibility.
        for st in sites:
            k = f"{stem}_L{st}" if len(sites) > 1 else stem
            if k not in vz:
                sys.exit(f"FATAL: vector {k!r} not in {args.vectors}. have {sorted(vz)}")
        parsed: list[float | tuple[float, ...]] = []
        for tokn in alphas.split(","):
            if "/" in tokn:
                per = tuple(float(x) for x in tokn.split("/"))
                if len(per) != len(sites):
                    sys.exit(f"FATAL: alpha spec {tokn!r} has {len(per)} values but "
                             f"{len(sites)} sites were given")
                parsed.append(per)
            else:
                parsed.append(float(tokn))
        cells.append((stem, parsed))

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model_path, tensor_parallel_size=1, dtype="bfloat16",
              max_model_len=768, gpu_memory_utilization=args.gpu_mem_util,
              enforce_eager=True, enable_prefix_caching=False, seed=args.seed)
    model = find_model(llm)
    layers = model.model.layers
    dev = next(model.parameters()).device

    state: dict[str, object] = {"vecs": {}, "prompt_len": 0}

    def mk_inject(site_idx: int):
      def inject_hook(mod, a, kw):
        vec = state["vecs"].get(site_idx)
        if vec is None or len(a) < 2 or not isinstance(a[1], torch.Tensor):
            return None
        positions, h = a[0], a[1]
        if not isinstance(positions, torch.Tensor):
            return None
        # generated positions only — correct in prefill and in every decode step alike
        mask = (positions >= int(state["prompt_len"]))
        if not bool(mask.any()):
            return None
        h2 = h.clone()
        h2[mask] = h2[mask] + vec.to(h.dtype)
        return (a[0], h2) + tuple(a[2:]), kw
      return inject_hook

    handles = [layers[st].register_forward_pre_hook(mk_inject(st), with_kwargs=True)
               for st in sites]
    sp = SamplingParams(n=args.n_samples, temperature=TEMPERATURE, top_p=TOP_P,
                        max_tokens=args.max_new_tokens, stop=[STOP_SEQUENCE], seed=args.seed)
    tok = llm.get_tokenizer()

    def run_cell(vecs: dict) -> list[str]:
        texts: list[str] = []
        for p in prompts:
            ids = tok(p["prompt_text"], add_special_tokens=True)["input_ids"]
            state.update({"vecs": vecs, "prompt_len": len(ids)})
            from vllm.inputs import TokensPrompt
            outs = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
            texts.extend(o.text for o in outs[0].outputs)
        return texts

    results: dict[str, dict] = {}
    raws = args.raws_dir or args.out.parent / (args.out.stem + "_raws")
    raws.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # α=0 baseline once, shared by every cell (constraint: always record it)
    base_texts = run_cell({})
    baseline = summarize(base_texts)
    (raws / "baseline_a0.json").write_text(json.dumps(base_texts, indent=0))
    logger.info("BASELINE a=0: %s", json.dumps({k: baseline[k] for k in
                ("scarequote", "parenthetical", "note_marker", "modal_hedge", "n_words", "ttr")}))

    for key, alphas in cells:
        results[key] = {"0.0": baseline}
        for a in alphas:
            per_site = dict(zip(sites, a)) if isinstance(a, tuple) else {s_: a for s_ in sites}
            vecs = {}
            for st in sites:
                k = f"{key}_L{st}" if len(sites) > 1 else key
                # each site scaled by ITS OWN median norm — alpha is a fraction everywhere
                vecs[st] = (torch.tensor(vz[k].astype(np.float32), device=dev,
                                         dtype=torch.float32) * (per_site[st] * med_norms[st]))
            texts = run_cell(vecs)
            r = summarize(texts)
            r["coherence_gate_pass"] = bool(r["ttr"] >= args.coherence_floor)
            r["alpha"] = list(a) if isinstance(a, tuple) else a
            r["per_site_alpha"] = {str(k_): v for k_, v in per_site.items()}
            r["total_displacement"] = round(sum(per_site[s_] * med_norms[s_] for s_ in sites), 3)
            r["displacement_share"] = {str(s_): round(per_site[s_] * med_norms[s_]
                                       / max(sum(per_site[x] * med_norms[x] for x in sites), 1e-9), 3)
                                       for s_ in sites}
            results[key][str(a)] = r
            (raws / f"{key}_a{a}.json").write_text(json.dumps(texts, indent=0))
            logger.info("%s a=%s tot=%.1f: scarequote=%.2f paren=%.2f note=%.3f hedge=%.3f "
                        "nwords=%.1f ttr=%.3f gate=%s", key,
                        "/".join(f"{v:g}" for v in per_site.values()),
                        r["total_displacement"], r["scarequote"],
                        r["parenthetical"], r["note_marker"], r["modal_hedge"],
                        r["n_words"], r["ttr"], r["coherence_gate_pass"])

    for h in handles:
        h.remove()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "sites": sites, "median_norms": med_norms, "n_prompts": len(prompts),
        "n_samples": args.n_samples, "elapsed_s": round(time.time() - t0, 1),
        "coherence_floor": args.coherence_floor,
        "verdict_rule": ("real iff dose-responsive AND sign-consistent AND outside the R-band "
                         "envelope AND coherence in window"),
        "baseline_alpha0": baseline, "results": results}, indent=1))
    logger.info("wrote %s (raws in %s)", args.out, raws)


if __name__ == "__main__":
    main()
