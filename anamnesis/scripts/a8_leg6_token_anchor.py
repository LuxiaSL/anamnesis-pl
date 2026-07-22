"""A8 Leg-6 / Item 6 — the special-token anchor hypothesis (P8-TOK1 .55 / P8-TOK2 .40).

Luxia's mechanism for the arm's one unexplained failure. Veos-perp is the axis that never
transported: commutation .131 same-tokenizer vs .056 cross-tokenizer, the S5 termination
rescue raised its COUPLING but left its TRANSPORT flat, and Veos_raw carries while
Veos-perp does not. The hypothesis: Veos-perp is anchored to per-model SPECIAL-TOKEN
unembedding geometry — the mechanical EOS-trigger component — which shared-text pairing
structurally cannot align, because special tokens have no cross-model semantic referent
in the paired observations.

TOK1 — LOGIT-LENS ANCHOR READ (own-frame, convention-free; runs regardless).
  Project Veos-perp and V7 through each model's OWN final norm + unembedding and compare
  how much of the top-20 boosted-logit mass lands on that model's own EOS/special tokens.
  Bar: Veos-perp's own-special share exceeds V7's by >=5x, in BOTH 8B and Qwen.

  LN convention, named: these are RMSNorm models, y = (x / rms(x)) * w, so
      logit_t(x) = u_t . y = x . (w * u_t) / rms(x).
  The logit lens for a residual-space direction v is therefore (v * w) @ W_U^T, exact up
  to the positive scalar 1/rms(x) which cannot change any ranking or share. No choice is
  being made here; this is the only reading of "through the model's own final-norm and
  unembedding" for an RMSNorm trunk.

TOK2 — THE DECOMPOSITION RESCUE.
  The same identity gives the SITE-LEVEL IMAGE of token t's unembedding row as
      e_t := unit(w * u_t)
  — the residual-space direction whose inner product IS that token's logit contribution.
  Project each side's Veos-perp off span{e_t : t in that model's own EOS/special ids},
  then re-read the 8b->qwen commutation of the remainder against a FRESH transported-null
  envelope.
  Bar: at-null (.056) -> greater than 2x its envelope q95.

  CONVENTION CAVEAT, NAMED (add-6 anticipates it): w is the FINAL-norm gain, while the
  vectors live at mid sites (8B L16 / Qwen L21). The image is therefore exact for the
  final-layer readout and an approximation at the injection site — there is no
  intervening-layer-free definition. add-6's conditional says TOK2 scores CONDITION-UNMET
  if the image is ill-defined at the capture convention; it is well-DEFINED but
  site-approximate, so the number is reported WITH this caveat and the desk rules whether
  that satisfies the letter. Nothing is silently adjusted.

  Reported beside: the same projection applied to V7 (a specificity control — if removing
  EOS directions also moves V7's commutation, the effect is not eos-specific), and the
  fraction of each vector's norm that lives in the EOS span.

EXPLORATORY BESIDE (no P): logit-lens of g.Veos-perp_8B in QWEN's frame — does the
transported direction point at any coherent token mass in the target vocabulary, or at
noise? Top-20 table only, no verdict.

UNSTAMPED (C section 8). No P self-scored. Node-side (needs weights).
Run: PYTHONPATH=pipeline python -m anamnesis.scripts.a8_leg6_token_anchor --out <path>
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from anamnesis.scripts.a8_fit_g import load_transport_map
from anamnesis.scripts.a8_rosetta import _unit, cos, load_axes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_leg6_token_anchor")

ARM = Path("outputs/battery/arms/A8_conjugation")
FIT_8B_QWEN = ARM / "leg1/fits/fit_8bL16__qwen-7bL21_native_proc_k512.npz"
MODELS = {
    "8b": {"path": "/models/llama-3.1-8b-instruct", "site": 16},
    "qwen-7b": {"path": "/models/subliminal-anamnesis/.hf-cache/hub/"
                        "models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
                        "a09a35458c702b33eeacc393d103063234e8bc28", "site": 21},
}
TOPK = 20
N_RANDOM = 100
SEED = 80
AT_NULL_REFERENCE = 0.056     # the banked cross-tokenizer Veos-perp commutation


def load_head(path: str):
    """Returns (final_norm_gain w [d], unembedding W_U [V, d], tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32,
                                                 device_map="cpu")
    w = model.model.norm.weight.detach().float().numpy().astype(np.float64)
    wu = model.lm_head.weight.detach().float().numpy().astype(np.float64)
    del model
    return w, wu, tok


def special_ids(tok) -> set[int]:
    ids: set[int] = set()
    for a in ("all_special_ids",):
        ids |= {int(i) for i in (getattr(tok, a, None) or [])}
    for a in ("eos_token_id", "bos_token_id", "pad_token_id"):
        v = getattr(tok, a, None)
        if isinstance(v, int):
            ids.add(v)
        elif isinstance(v, (list, tuple)):
            ids |= {int(x) for x in v}
    vocab = tok.get_vocab()
    for s, i in vocab.items():
        if (s.startswith("<|") and s.endswith("|>")) or (s.startswith("<") and s.endswith(">")
                                                         and len(s) > 2):
            ids.add(int(i))
    return ids


def eos_ids(tok) -> list[int]:
    v = getattr(tok, "eos_token_id", None)
    out = [v] if isinstance(v, int) else [int(x) for x in (v or [])]
    return sorted(set(out))


def lens(v: np.ndarray, w: np.ndarray, wu: np.ndarray) -> np.ndarray:
    """Logit-lens scores for residual direction v (RMSNorm: exact up to +1/rms)."""
    return wu @ (v * w)


def anchor_detail(scores: np.ndarray, eos: list[int], specials: set[int], tok) -> dict:
    """Where do the model's own EOS/special tokens actually SIT under this lens?

    The top-20 share can read 0/0 for both vectors — true but uninformative, since a
    strongly-boosted special token could sit at rank 50 in a 128k vocabulary. This gives
    the desk the ranks and standardised scores instead of a null division, plus the same
    share statistic at a wider K as a robustness column.
    """
    order = np.argsort(-scores)
    rank_of = np.empty_like(order)
    rank_of[order] = np.arange(len(order))
    mu, sd = float(scores.mean()), float(scores.std())
    sp = np.array(sorted(specials), dtype=int)
    out = {
        "vocab_size": int(scores.shape[0]),
        "own_eos": [{"id": int(t), "tok": tok.convert_ids_to_tokens(int(t)),
                     "rank": int(rank_of[t]),
                     "percentile": round(100.0 * (1.0 - rank_of[t] / len(order)), 4),
                     "z_vs_vocab": round((float(scores[t]) - mu) / sd, 4) if sd > 0 else None}
                    for t in eos],
        "all_specials": {
            "n": int(sp.size),
            "best_rank": int(rank_of[sp].min()) if sp.size else None,
            "median_rank": int(np.median(rank_of[sp])) if sp.size else None,
            "mean_z": (round(float((scores[sp].mean() - mu) / sd), 4)
                       if sp.size and sd > 0 else None),
        },
    }
    for k in (20, 200, 1000):
        top = order[:k]
        vals = scores[top]
        base = vals - vals.min() + 1e-12
        sel = np.array([int(t) in specials for t in top])
        out[f"special_mass_share_top{k}"] = (round(float(base[sel].sum() / base.sum()), 6)
                                             if base.sum() > 0 else None)
        out[f"special_count_top{k}"] = int(sel.sum())
    return out


def share(scores: np.ndarray, ids: set[int], tok, k: int = TOPK) -> dict:
    top = np.argsort(-scores)[:k]
    vals = scores[top]
    base = vals - vals.min() + 1e-12          # non-negative masses over the top-k
    mass = float(base.sum())
    sel = np.array([int(t) in ids for t in top])
    return {
        "top_k": k,
        "special_mass_share": round(float(base[sel].sum() / mass), 4) if mass > 0 else None,
        "special_count": int(sel.sum()),
        "top_tokens": [{"id": int(t), "tok": tok.convert_ids_to_tokens(int(t)),
                        "score": round(float(s), 4), "is_special": bool(int(t) in ids)}
                       for t, s in zip(top, vals)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    res: dict = {
        "STATUS": "UNSTAMPED (C section 8) — no self-scored P; the desk scores "
                  "P8-TOK1 (.55) and P8-TOK2 (.40)",
        "leg": "A8 Leg-6 / Item 6 — the special-token anchor hypothesis",
        "ln_convention": "RMSNorm y=(x/rms(x))*w  =>  logit_t(x) = x . (w*u_t)/rms(x). "
                         "Lens for a residual direction v is (v*w) @ W_U^T, exact up to "
                         "the positive scalar 1/rms(x) (cannot change rankings or shares).",
        "tok1": {}, "tok2": {}, "exploratory_transported_lens": {},
    }

    heads, axes_by_model = {}, {}
    for m, cfg in MODELS.items():
        logger.info("loading head: %s", m)
        w, wu, tok = load_head(cfg["path"])
        reads, extras, _ = load_axes(m)
        heads[m] = (w, wu, tok)
        axes_by_model[m] = (reads, extras)

        sp, eo = special_ids(tok), eos_ids(tok)
        blk = {"site": cfg["site"], "n_special_ids": len(sp), "own_eos_ids": eo,
               "vectors": {}}
        for vname, vec in (("Veos_perp", extras["Veos_perp"].vec),
                           ("V7", reads["V7"].vec)):
            s = lens(vec, w, wu)
            blk["vectors"][vname] = share(s, sp, tok)
            blk["vectors"][vname]["anchor_detail"] = anchor_detail(s, eo, sp, tok)
        a = blk["vectors"]["Veos_perp"]["special_mass_share"]
        b = blk["vectors"]["V7"]["special_mass_share"]
        blk["ratio_Veos_perp_over_V7"] = (round(a / b, 3) if b not in (None, 0) else None)
        blk["bar_ge_5x"] = (None if blk["ratio_Veos_perp_over_V7"] is None
                            else bool(blk["ratio_Veos_perp_over_V7"] >= 5.0))
        blk["ratio_undefined_note"] = (
            "both shares are 0.0 at top-20, so the >=5x ratio is 0/0 — UNDEFINED, not "
            "failed-by-a-margin. The anchor_detail block gives the ranks, percentiles and "
            "z-scores of the own-EOS and special tokens under each lens, plus the same "
            "share at K=200 and K=1000, so the desk can rule the letter on real numbers "
            "instead of a null division."
            if blk["ratio_Veos_perp_over_V7"] is None else None)
        res["tok1"][m] = blk
        logger.info("[TOK1 %s] Veos_perp special share %.4f vs V7 %.4f -> ratio %s",
                    m, a or -1, b or -1, blk["ratio_Veos_perp_over_V7"])

    # ---------------------------------------------------------------- TOK2
    tm = load_transport_map(FIT_8B_QWEN)
    proj: dict[str, dict[str, np.ndarray]] = {}
    for m in MODELS:
        w, wu, tok = heads[m]
        reads, extras = axes_by_model[m]
        E = np.stack([_unit(w * wu[i]) for i in eos_ids(tok)])       # site-level images
        q, _ = np.linalg.qr(E.T)                                      # orthonormal basis
        out_m = {}
        for vname, vec in (("Veos_perp", extras["Veos_perp"].vec),
                           ("V7", reads["V7"].vec)):
            v = _unit(vec)
            inside = q @ (q.T @ v)
            out_m[vname] = _unit(v - inside)
            res["tok2"].setdefault("norm_fraction_in_eos_span", {}).setdefault(m, {})[
                vname] = round(float(np.linalg.norm(inside)), 6)
        proj[m] = out_m

    src_reads, src_extras, src_pool = load_axes("8b")
    tgt_reads, tgt_extras, _ = load_axes("qwen-7b")
    rng = np.random.default_rng(SEED)
    randoms = rng.standard_normal((N_RANDOM, src_extras["Veos_perp"].vec.shape[0]))
    randoms /= np.linalg.norm(randoms, axis=1, keepdims=True)
    nulls = ([_unit(tm.transport(r)) for r in randoms]
             + [_unit(tm.transport(a.vec)) for a in src_pool])

    def commute(src_v: np.ndarray, tgt_v: np.ndarray) -> dict:
        u = _unit(tm.transport(src_v))
        c = cos(u, tgt_v)
        q95 = float(np.quantile([abs(cos(n, tgt_v)) for n in nulls], 0.95))
        return {"commutation": round(c, 4), "null_q95_abs": round(q95, 4),
                "ratio_to_q95": round(abs(c) / q95, 3) if q95 > 0 else None,
                "exceeds_2x_q95": bool(abs(c) > 2 * q95)}

    res["tok2"]["reads"] = {
        "Veos_perp BEFORE projection (baseline)":
            commute(src_extras["Veos_perp"].vec, tgt_extras["Veos_perp"].vec),
        "Veos_perp AFTER removing own-EOS span (the rescue)":
            commute(proj["8b"]["Veos_perp"], proj["qwen-7b"]["Veos_perp"]),
        "V7 BEFORE projection (specificity control)":
            commute(src_reads["V7"].vec, tgt_reads["V7"].vec),
        "V7 AFTER removing own-EOS span (specificity control)":
            commute(proj["8b"]["V7"], proj["qwen-7b"]["V7"]),
    }
    res["tok2"]["at_null_reference"] = AT_NULL_REFERENCE
    res["tok2"]["bar"] = "commutation > 2x its envelope q95"
    res["tok2"]["CONVENTION_CAVEAT"] = (
        "w is the FINAL-norm gain while the vectors live at mid sites (8B L16 / Qwen L21). "
        "The site-level image is exact for the final-layer readout and site-APPROXIMATE at "
        "the injection site; there is no intervening-layer-free definition. add-6's "
        "conditional scores TOK2 CONDITION-UNMET if the image is ill-defined at the capture "
        "convention — it is well-defined but site-approximate. Reported with the caveat; "
        "the desk rules whether that satisfies the letter. Nothing silently adjusted.")

    # ------------------------------------------- exploratory: transported lens in Qwen
    w_q, wu_q, tok_q = heads["qwen-7b"]
    g_veos = _unit(tm.transport(src_extras["Veos_perp"].vec))
    res["exploratory_transported_lens"] = {
        "grade": "EXPLORATORY — no P, no verdict",
        "question": "does g.Veos_perp_8B point at coherent token mass in QWEN's vocabulary?",
        "read": share(lens(g_veos, w_q, wu_q), special_ids(tok_q), tok_q),
    }

    (args.out / "token_anchor.json").write_text(json.dumps(res, indent=1))
    for k, v in res["tok2"]["reads"].items():
        logger.info("[TOK2] %-52s cos %+.4f (q95 %.4f, %sx, >2x %s)", k,
                    v["commutation"], v["null_q95_abs"], v["ratio_to_q95"],
                    v["exceeds_2x_q95"])
    logger.info("wrote %s", args.out / "token_anchor.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
