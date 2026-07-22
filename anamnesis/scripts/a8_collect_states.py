"""A8 Leg-0 — T4+T5: Phase A paired collection (GPU) + bitwise spot-replay (CP-1 gate).

Forced replay of every corpus text through ONE model, both template arms (never mixed),
capturing the per-text MEAN residual state over completion positions at the site grid.
Banks raw fp32 means + per-site median norms via a8_fit_g.save_state_bank (the one
contract). Pre-normalizes NOTHING (normalization happens at fit time, in the fit stamp).

Conventions mirrored from the a5 machinery (the banked vectors' native space):
  - site L = forward_pre_hook on decoder_layers(model)[L] — the residual ENTERING
    layer L; batch-1; use_cache=False; .float() cast from the bf16 forward.
  - model load: AutoModelForCausalLM(dtype=bfloat16, attn_implementation="eager").
    bf16 forward + fp32 banked means per Luxia's dtype ruling 2026-07-22 — CERTIFIED
    BY the CP-1 bitwise spot-replay gate, not by assumption. Pin ONE card per model
    for the whole session (launcher sets CUDA_VISIBLE_DEVICES; determinism is
    within-model/within-card).
  - native arm: chat template from the manifest's system/user prompts,
    add_generation_prompt=True, date_string=VMB_CANONICAL_DATE (Llama strftime_now
    pin); completion ids = tokenize(text, no specials) appended after the prompt.
  - raw arm: tokenize(text, with specials); completion positions = everything after
    the specials prefix (computed, not assumed).

Re-tokenization caveat (GP-Q9, rake-report item): the corpus banks TEXT, not token
ids — fresh tokenization is the defined object for cross-replay, and Leg-0's shared
tokenizer guarantees both models see IDENTICAL token sequences per (text, arm).
Cross-tokenizer legs (Leg 1+) revisit this explicitly.

Modes (one model per invocation; run once per model on the assigned card):
  --collect              full pass over the corpus, both arms (or --arms native)
  --spot-replay K        FRESH-PROCESS re-run of K stratified texts per arm, byte-
                         compared against the banked means (the CP-1 gate; K>=3)
  --cp1-summary          the compact CP-1 table from banked stamps + spot results
                         (run locally after rsyncing states/ back; needs no GPU)

Launch template (node-side, after venv activation, from the deploy root):
  OMP_NUM_THREADS=1 python -m anamnesis.scripts.a8_collect_states --collect \
    --model 3b --model-path /models/llama-3.2-3b-instruct --arm-root <ARM_ROOT>
  python -m anamnesis.scripts.a8_collect_states --spot-replay 3 --model 3b ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from anamnesis.scripts.a8_fit_g import (
    ARMS, DEFAULT_ARM_ROOT, SITES, StateBank, load_state_bank, save_state_bank)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_collect_states")

STRATA = ("S1", "S2", "S3", "S5")  # S5 = Leg-4 uncapped/natural-termination stratum


# ---------------------------------------------------------------- corpus + ids
def load_corpus(arm_root: Path) -> tuple[list[dict], str]:
    path = arm_root / "corpus" / "corpus_manifest.json"
    with open(path) as f:
        entries = json.load(f)["entries"]
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    if not entries:
        raise RuntimeError(f"{path}: empty corpus")
    return entries, h


def build_ids(tok, entry: dict, arm: str, date_string: str) -> tuple[list[int], int]:
    """Returns (ids, P) — full token ids and the completion start position."""
    text = entry["text"]
    comp = tok.encode(text, add_special_tokens=False)
    if not comp:
        raise RuntimeError(f"{entry['text_id']}: text tokenizes to nothing")
    if arm == "native":
        msgs = []
        if entry.get("system_prompt"):
            msgs.append({"role": "system", "content": entry["system_prompt"]})
        msgs.append({"role": "user", "content": entry["user_prompt"]})
        try:
            res = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          date_string=date_string)
        except TypeError:      # date-free templates (e.g. Qwen) reject the kwarg
            res = tok.apply_chat_template(msgs, add_generation_prompt=True)
        prompt_ids = list(res["input_ids"] if hasattr(res, "keys") else res)
        return prompt_ids + comp, len(prompt_ids)
    if arm == "raw":
        ids = tok.encode(text, add_special_tokens=True)
        p = len(ids) - len(comp)
        if p < 0 or ids[p:] != comp:
            # specials interleaved unexpectedly — fall back to explicit BOS prefix
            bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
            ids, p = bos + comp, len(bos)
        return ids, p
    raise ValueError(f"unknown arm {arm!r}")


# ---------------------------------------------------------------- state capture
class SiteCapture:
    """forward_pre_hooks on the site layers; grabs input hidden_states per forward."""

    def __init__(self, model, sites: tuple[int, ...]):
        from anamnesis.extraction.model_loader import decoder_layers
        layers = decoder_layers(model)
        self.grab: dict[int, "object"] = {}
        self.handles = []
        for s in sites:
            def hook(module, hook_args, hook_kwargs, _s=s):
                hs = hook_args[0] if hook_args else hook_kwargs.get("hidden_states")
                self.grab[_s] = hs.detach()
                return None
            self.handles.append(
                layers[s].register_forward_pre_hook(hook, with_kwargs=True))

    def close(self):
        for h in self.handles:
            h.remove()


def compute_means(model, tok, entries: list[dict], sites: tuple[int, ...], arm: str,
                  date_string: str, device: str, log_every: int = 50
                  ) -> tuple[dict[int, np.ndarray], dict[int, list[float]], list[int]]:
    """One forward per text; per-text fp32 mean over completion positions per site.
    Also returns per-position residual norms (a5 dose-currency comparability) and
    sequence lengths."""
    import torch

    cap = SiteCapture(model, sites)
    means: dict[int, list[np.ndarray]] = {s: [] for s in sites}
    tok_norms: dict[int, list[float]] = {s: [] for s in sites}
    seq_lens: list[int] = []
    t0 = time.time()
    try:
        for i, e in enumerate(entries):
            ids, p = build_ids(tok, e, arm, date_string)
            t = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                model(t, use_cache=False, return_dict=True)
            for s in sites:
                h = cap.grab[s][0, p:, :].float()          # [n_comp, d] fp32
                if h.shape[0] == 0:
                    raise RuntimeError(f"{e['text_id']}: no completion positions")
                means[s].append(h.mean(dim=0).cpu().numpy().astype(np.float32))
                tok_norms[s].append(float(h.norm(dim=-1).median().cpu()))
            seq_lens.append(len(ids))
            if (i + 1) % log_every == 0:
                logger.info("[%s/%s] %d/%d texts (%.2fs/text)", arm, sites, i + 1,
                            len(entries), (time.time() - t0) / (i + 1))
    finally:
        cap.close()
    return ({s: np.stack(v) for s, v in means.items()},
            {s: v for s, v in tok_norms.items()}, seq_lens)


def load_model_and_tok(model_path: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="eager",
    ).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_path)
    return model, tok


def trunk_stamp(model_path: str, tok, device: str) -> dict:
    import torch
    import transformers
    cfg = Path(model_path) / "config.json"
    tpl = tok.chat_template or ""
    dev_name = (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "cpu")
    return {"model_path": str(model_path),
            "config_sha256": hashlib.sha256(cfg.read_bytes()).hexdigest()
            if cfg.exists() else None,
            "chat_template_sha256": hashlib.sha256(tpl.encode()).hexdigest(),
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "dtype_forward": "bfloat16", "dtype_banked": "float32",
            "attn_implementation": "eager", "device": device,
            "cuda_device_name": dev_name,
            "cuda_visible_devices":
                __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "(unset)")}


# ---------------------------------------------------------------- collect mode
def collect(arm_root: Path, model: str, model_path: str, arms: list[str],
            device: str, sites_override: tuple[int, ...] | None = None) -> None:
    from anamnesis.scripts.vmb_stage0_generate import VMB_CANONICAL_DATE

    entries, manifest_sha = load_corpus(arm_root)
    sites = sites_override or SITES[model]
    states_dir = arm_root / "states"
    logs_dir = arm_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    model_obj, tok = load_model_and_tok(model_path, device)
    trunk = trunk_stamp(model_path, tok, device)
    logger.info("trunk: %s", json.dumps(trunk)[:200])

    for arm in arms:
        t0 = time.time()
        means, tok_norms, seq_lens = compute_means(
            model_obj, tok, entries, sites, arm, VMB_CANONICAL_DATE, device)
        text_ids = [e["text_id"] for e in entries]
        median_norms = {s: float(np.median(np.linalg.norm(means[s], axis=1)))
                        for s in sites}
        bank = StateBank(model=model, arm=arm, text_ids=text_ids,
                         states=means, median_norms=median_norms)
        npz_path, norms_path = save_state_bank(states_dir, bank)
        counts: dict[str, int] = {}
        for e in entries:
            counts[e["stratum"]] = counts.get(e["stratum"], 0) + 1
        stamp = {
            "arm_dir": "A8_conjugation", "leg": arm_root.name, "builder": "a8_collect_states.py",
            "prereg_tag": "prereg-arm8-v1", "model": model, "template_arm": arm,
            "sites": list(sites), "n_texts": len(entries),
            "counts_per_stratum": counts,
            "corpus_manifest_sha256": manifest_sha,
            "date_string": VMB_CANONICAL_DATE,
            "median_mean_state_norms": {f"L{s}": v for s, v in median_norms.items()},
            "median_token_resid_norms": {
                f"L{s}": float(np.median(tok_norms[s])) for s in sites},
            "seq_len": {"min": int(min(seq_lens)), "max": int(max(seq_lens)),
                        "mean": round(float(np.mean(seq_lens)), 1)},
            "wall_seconds": round(time.time() - t0, 1),
            "trunk": trunk,
            "capture_convention": "forward_pre_hook on decoder_layers[L] "
                                  "(residual ENTERING layer L), completion "
                                  "positions >= P, fp32 mean; batch-1, no cache",
        }
        stamp_path = states_dir / f"collection_stamp_{model}_{arm}.json"
        with open(stamp_path, "w") as f:
            json.dump(stamp, f, indent=1)
        logger.info("banked %s (%s) -> %s + %s + %s", model, arm, npz_path.name,
                    norms_path.name, stamp_path.name)


# ---------------------------------------------------------------- spot replay (T5)
def pick_spot_ids(entries: list[dict], k: int) -> list[str]:
    """Deterministic stratified pick: k texts spread across strata by id-hash order."""
    by_stratum: dict[str, list[str]] = {s: [] for s in STRATA}
    for e in entries:
        by_stratum[e["stratum"]].append(e["text_id"])
    for s in STRATA:
        by_stratum[s].sort(key=lambda t: hashlib.sha256(t.encode()).hexdigest())
    picked, i = [], 0
    while len(picked) < k:
        s = STRATA[i % len(STRATA)]
        if by_stratum[s]:
            picked.append(by_stratum[s].pop(0))
        i += 1
    return picked


def spot_replay(arm_root: Path, model: str, model_path: str, arms: list[str],
                device: str, k: int) -> int:
    from anamnesis.scripts.vmb_stage0_generate import VMB_CANONICAL_DATE

    entries, _ = load_corpus(arm_root)
    by_id = {e["text_id"]: e for e in entries}
    sites = SITES[model]
    states_dir = arm_root / "states"
    model_obj, tok = load_model_and_tok(model_path, device)
    spot_ids = pick_spot_ids(entries, k)
    ok_all = True
    report: dict = {"model": model, "k": k, "spot_ids": spot_ids,
                    "trunk": trunk_stamp(model_path, tok, device), "results": {}}
    for arm in arms:
        bank = load_state_bank(states_dir, model, arm)
        idx = {t: i for i, t in enumerate(bank.text_ids)}
        sub = [by_id[t] for t in spot_ids]
        means, _, _ = compute_means(model_obj, tok, sub, sites, arm,
                                    VMB_CANONICAL_DATE, device, log_every=1000)
        res = {}
        for j, t in enumerate(spot_ids):
            per_site = {}
            for s in sites:
                banked = bank.states[s][idx[t]]
                fresh = means[s][j]
                exact = bool(np.array_equal(banked, fresh))
                per_site[f"L{s}"] = {
                    "bitwise_exact": exact,
                    "max_abs_diff": float(np.max(np.abs(
                        banked.astype(np.float64) - fresh.astype(np.float64))))}
                ok_all &= exact
            res[t] = per_site
        report["results"][arm] = res
        n_exact = sum(d["bitwise_exact"] for r in res.values() for d in r.values())
        n_tot = sum(len(r) for r in res.values())
        logger.info("spot-replay %s/%s: %d/%d site-cells bitwise exact",
                    model, arm, n_exact, n_tot)
    out = states_dir / f"spot_replay_{model}.json"
    with open(out, "w") as f:
        json.dump({**report, "all_bitwise_exact": ok_all}, f, indent=1)
    logger.info("spot report: %s  ALL_EXACT=%s", out, ok_all)
    return 0 if ok_all else 1


# ---------------------------------------------------------------- CP-1 summary
def cp1_summary(arm_root: Path, models: list[str]) -> int:
    states_dir = arm_root / "states"
    lines = ["CP-1 COLLECTION QC (UNSTAMPED, C§8)"]
    norms: dict[tuple[str, str], dict] = {}
    for model in models:
        for arm in ARMS:
            sp = states_dir / f"collection_stamp_{model}_{arm}.json"
            if not sp.exists():
                lines.append(f"  {model}/{arm}: MISSING")
                continue
            st = json.loads(sp.read_text())
            norms[(model, arm)] = st["median_mean_state_norms"]
            lines.append(
                f"  {model}/{arm}: n={st['n_texts']} strata={st['counts_per_stratum']}"
                f" norms={ {k: round(v, 2) for k, v in norms[(model, arm)].items()} }"
                f" seq_len={st['seq_len']['mean']} wall={st['wall_seconds']}s")
        spr = states_dir / f"spot_replay_{model}.json"
        if spr.exists():
            r = json.loads(spr.read_text())
            lines.append(f"  {model} spot-replay k={r['k']}: "
                         f"ALL_BITWISE_EXACT={r['all_bitwise_exact']}")
        else:
            lines.append(f"  {model} spot-replay: NOT RUN")
    if len(models) == 2 and all((m, a) in norms for m in models for a in ARMS):
        m0, m1 = models
        for arm in ARMS:
            r = {f"{k0}->{k1}": round(norms[(m1, arm)][k1] / norms[(m0, arm)][k0], 3)
                 for k0 in norms[(m0, arm)] for k1 in norms[(m1, arm)]}
            lines.append(f"  norm ratio {m1}/{m0} ({arm}): {r}")
    print("\n".join(lines))
    return 0


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm-root", type=Path, default=DEFAULT_ARM_ROOT)
    ap.add_argument("--model", choices=sorted(SITES), help="one model per invocation")
    ap.add_argument("--model-path", help="local weights dir (node-side)")
    ap.add_argument("--arms", default="native,raw",
                    help="comma list; strata/arms never mixed in banks")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sites", default=None,
                    help="comma-separated site override (default = the model's SITES grid); "
                         "used by Leg-4F to bank Qwen L18, where Vdiverge lives")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--spot-replay", type=int, metavar="K", default=None)
    ap.add_argument("--cp1-summary", nargs="*", metavar="MODEL", default=None,
                    help="e.g. --cp1-summary 3b 8b (local, no GPU)")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"unknown arms {bad}; valid: {ARMS}")

    if args.cp1_summary is not None:
        return cp1_summary(args.arm_root, args.cp1_summary or ["3b", "8b"])
    if not args.model or not args.model_path:
        raise SystemExit("--model and --model-path required for GPU modes")
    if args.collect:
        collect(args.arm_root, args.model, args.model_path, arms, args.device,
                sites_override=(tuple(int(x) for x in args.sites.split(','))
                                if args.sites else None))
        return 0
    if args.spot_replay is not None:
        if args.spot_replay < 3:
            raise SystemExit("CP-1 gate requires K >= 3")
        return spot_replay(args.arm_root, args.model, args.model_path, arms,
                           args.device, args.spot_replay)
    raise SystemExit("pick a mode: --collect / --spot-replay K / --cp1-summary")


if __name__ == "__main__":
    sys.exit(main())
