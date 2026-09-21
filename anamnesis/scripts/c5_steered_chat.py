"""Computer #5 — the steering rack, served. Port of kotodama's `koto_steered_chat.py`.

Deliberately a 1:1 port of koto's wire contract so kotodama's `steer_webui.py` and `kotalk.py`
work against this backend UNMODIFIED (that UI builds its vector rack generically from whatever
`GET /info` returns, so matching the contract is the whole integration):

    GET  /v1/models     -> {"data": [{"id": <alias>}, ...]}
    GET  /info|/health  -> {"model", "site", "aliases": {alias: {key, alpha, site, desc}}, ...}
    POST /generate      -> {"model"|"vectors", "alpha", "messages"|"prompt", "stream", ...}
                           stream: SSE `data: {"token": ...}` then `data: {"done": true, ...}`
                           else:   {"text", "prompt_tokens", "completion_tokens",
                                    "tokens_per_second"}
    POST /              -> legacy batch: {"candidates": [...]}

WHAT DIFFERS FROM KOTO, AND WHY

1. Engine is vLLM, not koto's hand-written forward. Same in-process requirements as the rest of
   this arm: tensor_parallel_size=1, VLLM_ENABLE_V1_MULTIPROCESSING=0, enforce_eager=True.
   Without all three the injection hook silently never fires and you serve unsteered text.
2. NO CHAT TEMPLATE. Computer is a base-model-style document mind; koto's `render_chatml` is
   replaced by the corpus scaffold. Getting this wrong doesn't error, it just stops being Computer.
3. Injection is plain additive at generated positions across an arbitrary set of sites — no
   block-persistent re-assertion. Llama-70B is dense and the write grows ~6x on its own
   (spec §10e); koto needed persistence for AttnRes boundaries that do not exist here.
4. ⚠ SSE is POST-HOC CHUNKING, not true token streaming. vLLM's offline `LLM` has no incremental
   callback, so the text is generated in full and then emitted in pieces. The wire shape is
   identical and the UI animates, but time-to-first-token is really time-to-full-generation.
   Stated in /info so nobody reads a latency number off it.

LINEAGE. Every generation is stamped at birth to `--lineage-log` — vector(s), dose, site(s), run
id. Golden-path Phase 4: retroactively impossible, and the one rule that keeps the books clean if
steered text ever breeds into a checkpoint.
"""
from __future__ import annotations

import argparse
import http.server
import threading
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HEADER = "As follows is a conversation between another user and Model C{topic}."
SCAFFOLD_OPEN = "\n\nFull conversation with Model C:\n"
# Sampling + stops lifted VERBATIM from Celeste's own chat UIs (modelC_train/chat_c.py,
# chat_c_latest.py, chat_c_computer6.py) so the rack behaves like the interfaces she already
# uses. temp/top_p match the model card; the STOP LIST does not — the card names only the first.
# The other four matter for a base-model document mind, which will otherwise happily write the
# NEXT turn, start a fresh document, or run past a horizontal rule. (Measured on the banked
# corpora: boundary overrun was 1.6% of base rows and 0.0% of c5's under the single stop — tiny,
# and asymmetric AGAINST our own contrast — but at 2200 max_tokens in chat it is not tiny.)
STOPS = ["\n\n**User:**", "\n\n**User**", "\n\n**Model C:**",
         "\nAs follows is", "\n\n---"]
TEMPERATURE, TOP_P, MAX_NEW = 1.0, 0.98, 2200

# The rack. One alias = one (vector, site) pair with a sane default dose, exactly as koto's
# ALIASES worked — multi-site cells are composed CLIENT-side via the `vectors` stack (that is
# what /steer a:0.45 + b:0.45 does), which is why the champion is a preset, not an alias.
RACK: dict[str, tuple[str, int, float, str]] = {
    "computer-mid":     ("Install_L38", 38, 0.45, "Computer at 47% depth. Gentle alone; the "
                                                  "mid half of the champion pair."),
    "computer-deep":    ("Install_L48", 48, 0.45, "Computer at 60% depth. The strongest single "
                                                  "site that keeps coherence high."),
    "computer-deepest": ("Install_L76", 76, 0.18, "Computer at 95% depth. Heaviest per unit "
                                                  "dose — keep alpha low."),
    "rband-mid":        ("Rband_L38",   38, 0.45, "Matched-norm random control at L38. Should "
                                                  "do nothing Computer-ish."),
    "rband-deep":       ("Rband_L48",   48, 0.45, "Matched-norm random control at L48."),
    "whitened-deep":    ("InstallW_L48", 48, 0.45, "Whitened variant. Measurably weaker than raw "
                                                   "here; kept as a beside-column."),
}
# Champion = MULTI L38+L48 at 0.45 (spec §10g): 6.11 scarequote, and the cell that reaches
# chance against real Computer at haiku grade (§10h).
PRESETS = {
    "computer (champion)": [("computer-mid", 0.45), ("computer-deep", 0.45)],
    "computer (light)":    [("computer-mid", 0.30), ("computer-deep", 0.30)],
    "computer (deep mix)": [("computer-deep", 0.21), ("computer-deepest", 0.21)],
    "control":             [],
}


def render_scaffold(messages: list[dict] | None, prompt: str | None, topic: str | None) -> str:
    """Computer's corpus scaffold. NOT a chat template — this model has none, and the same
    scaffold on vanilla base is the persona-off control by construction."""
    if messages is None:
        return prompt or ""
    head = HEADER.format(topic=f" about {topic}" if topic else "") + SCAFFOLD_OPEN
    parts = [head]
    for m in messages:
        who = "User" if m.get("role") == "user" else "Model C"
        parts.append(f"\n**{who}:** {m.get('content', '')}\n")
    parts.append("\n**Model C:**")
    return "".join(parts)


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
    sys.exit("FATAL: model not reachable — need VLLM_ENABLE_V1_MULTIPROCESSING=0 and tp=1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/models/Llama-3.1-70B")
    ap.add_argument("--vectors", type=Path, required=True)
    ap.add_argument("--port", type=int, default=2555)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--lineage-log", type=Path,
                    default=Path("/mnt/beegfs/luxia/computer5/chat/induced_lineage.jsonl"))
    args = ap.parse_args()

    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        sys.exit("FATAL: VLLM_ENABLE_V1_MULTIPROCESSING=0 required (else hooks never fire and "
                 "you will serve UNSTEERED text with no error)")

    vz = dict(np.load(args.vectors, allow_pickle=True))
    rack = {a: v for a, v in RACK.items() if v[0] in vz}
    dropped = sorted(set(RACK) - set(rack))
    if dropped:
        logger.warning("aliases dropped (vector absent from bank): %s", dropped)
    if not rack:
        sys.exit(f"FATAL: no rack aliases resolve against {args.vectors}")
    med = {s: float(vz[f"median_norm_L{s}"]) for _, s, _, _ in rack.values()
           if f"median_norm_L{s}" in vz}

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    logger.info("loading %s", args.model_path)
    llm = LLM(model=args.model_path, tensor_parallel_size=1, dtype="bfloat16",
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem_util,
              enforce_eager=True, enable_prefix_caching=False, seed=20260729)
    model = find_model(llm)
    layers = model.model.layers
    dev = next(model.parameters()).device
    tok = llm.get_tokenizer()

    # vLLM's offline `LLM` is NOT re-entrant, and the injection `state` below is shared mutable
    # state read inside the forward hook. Under ThreadingHTTPServer two overlapping requests
    # corrupt vLLM's forward context (AssertionError: "Forward context is not set") AND clobber
    # each other's writes — i.e. one request silently steers with another's vector. A/B mode
    # fires steered + control together, so this is hit within a minute of real use, not an edge
    # case. One GPU, one generation at a time: serialize and let callers queue.
    GEN_LOCK = threading.Lock()
    state: dict = {"writes": {}, "prompt_len": 0}

    def mk_hook(site: int):
        def hook(mod, a, kw):
            v = state["writes"].get(site)
            if v is None or len(a) < 2 or not isinstance(a[1], torch.Tensor):
                return None
            positions, h = a[0], a[1]
            if not isinstance(positions, torch.Tensor):
                return None
            mask = positions >= int(state["prompt_len"])
            if not bool(mask.any()):
                return None
            h2 = h.clone()
            h2[mask] = h2[mask] + v.to(h.dtype)
            return (a[0], h2) + tuple(a[2:]), kw
        return hook

    for s in sorted({s for _, s, _, _ in rack.values()}):
        layers[s].register_forward_pre_hook(mk_hook(s), with_kwargs=True)
    args.lineage_log.parent.mkdir(parents=True, exist_ok=True)

    def resolve(body) -> tuple[dict[int, torch.Tensor], list[dict]]:
        """-> ({site: summed write}, recipe). Accepts either a `vectors` stack (alias or raw
        key + alpha, summed per site — koto's composition semantics) or a single `model` alias."""
        stack = body.get("vectors")
        if not stack:
            alias = body.get("model") or "control"
            if alias in ("control", "none"):
                return {}, []
            if alias not in rack:
                raise KeyError(f"unknown model {alias}; have {sorted(rack)} + control")
            stack = [{"name": alias, "alpha": body.get("alpha", rack[alias][2])}]
        per_site: dict[int, torch.Tensor] = {}
        recipe = []
        for item in stack:
            nm = item.get("name")
            if nm in ("control", "none"):
                continue
            if nm in rack:
                key, site, dflt, _ = rack[nm]
            elif nm in vz and isinstance(item.get("site"), int):
                key, site, dflt = nm, int(item["site"]), 0.45
            else:
                raise KeyError(f"unknown vector {nm}; aliases {sorted(rack)}")
            a = float(item.get("alpha", dflt))
            piece = torch.tensor(vz[key].astype(np.float32), device=dev) * (a * med.get(site, 1.0))
            per_site[site] = per_site.get(site, torch.zeros_like(piece)) + piece
            recipe.append({"alias": nm, "key": key, "site": site, "alpha": a})
        return per_site, recipe

    def generate(body) -> tuple[str, int, int, list[dict]]:
        with GEN_LOCK:
            return _generate_locked(body)

    def _generate_locked(body) -> tuple[str, int, int, list[dict]]:
        writes, recipe = resolve(body)
        text = render_scaffold(body.get("messages"), body.get("prompt"), body.get("topic"))
        ids = tok(text, add_special_tokens=True)["input_ids"]
        sp = SamplingParams(
            n=1, temperature=float(body.get("temperature", TEMPERATURE)),
            top_p=float(body.get("top_p", TOP_P)),
            max_tokens=int(body.get("max_new_tokens", MAX_NEW)), stop=STOPS)
        state.update({"writes": writes, "prompt_len": len(ids)})
        try:
            outs = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
        finally:
            state["writes"] = {}          # never leak a write into the next request
        comp = outs[0].outputs[0]
        run_id = uuid.uuid4().hex[:12]
        # induced lineage, stamped AT GENERATION (golden path Phase 4)
        with args.lineage_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "run_id": run_id, "induced": bool(recipe), "recipe": recipe,
                "model_path": args.model_path, "vectors_npz": str(args.vectors),
                "n_prompt": len(ids), "n_out": len(comp.token_ids),
                "text": comp.text, "ts": time.time()}) + "\n")
        return comp.text, len(ids), len(comp.token_ids), recipe

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            p = self.path.rstrip("/")
            if p == "/v1/models":
                self._json(200, {"data": [{"id": a} for a in sorted(rack)] +
                                         [{"id": "control"}]})
            elif p in ("/info", "/health", ""):
                self._json(200, {
                    "model": "computer5-steered (base Llama-3.1-70B + install vector)",
                    "site": sorted({s for _, s, _, _ in rack.values()}),
                    "aliases": {a: {"key": k, "alpha": d, "site": s, "desc": desc}
                                for a, (k, s, d, desc) in rack.items()},
                    "presets": PRESETS, "mode": "scaffold",
                    "sampling": {"temperature": TEMPERATURE, "top_p": TOP_P,
                                 "max_new_tokens": MAX_NEW, "stop": STOPS,
                                 "source": "Celeste's chat_c*.py defaults, verbatim"},
                    "note": ("no chat template — Computer's corpus scaffold. SSE is post-hoc "
                             "chunking, NOT true token streaming (vLLM offline has no "
                             "incremental callback), so first-token timing is not meaningful. "
                             "Every generation is lineage-stamped at birth.")})
            else:
                self._json(404, {"error": f"no route {self.path}"})

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            except Exception as exc:  # noqa: BLE001
                return self._json(400, {"error": f"bad body: {exc}"})
            p = self.path.rstrip("/")
            try:
                if p == "/generate":
                    return self._generate(body)
                return self._legacy(body)
            except KeyError as exc:
                return self._json(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001 — surface anything to the client
                logger.exception("request failed")
                try:
                    return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
                except Exception:
                    return None

        def _generate(self, body):
            t0 = time.time()
            text, n_in, n_out, recipe = generate(body)
            dt = max(time.time() - t0, 1e-6)
            if not body.get("stream"):
                return self._json(200, {"text": text.strip(), "prompt_tokens": n_in,
                                        "completion_tokens": n_out,
                                        "tokens_per_second": n_out / dt, "recipe": recipe})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            # post-hoc chunking on word boundaries so the UI animates; see /info note
            buf, chunk = text, ""
            for i, ch in enumerate(buf):
                chunk += ch
                if ch == " " or i == len(buf) - 1:
                    self.wfile.write(f"data: {json.dumps({'token': chunk})}\n\n".encode())
                    self.wfile.flush()
                    chunk = ""
            self.wfile.write(("data: " + json.dumps({
                "done": True, "prompt_tokens": n_in, "completion_tokens": n_out,
                "tokens_per_second": n_out / dt, "recipe": recipe}) + "\n\n").encode())
            self.wfile.flush()

        def _legacy(self, body):
            n = int(body.get("n", 1))
            if not body.get("steer", True):
                body = {**body, "model": "control", "vectors": None}
            cands = [generate(body)[0].strip() for _ in range(n)]
            return self._json(200, {"candidates": cands})

    logger.info("rack: %s", sorted(rack))
    logger.info("serving on 0.0.0.0:%d  (lineage -> %s)", args.port, args.lineage_log)
    http.server.ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
