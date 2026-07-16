"""A6 §2b — the DISTILLED-DIRECTION construction (session-9 SPINE; C§8 ABSOLUTE).

Builds the teacher<->student axis via the V3-bare injectability trick, per the session-8
verdict's SIX BINDING CONSTRAINTS. Two axes (constraint i):
  * V_align   — IN-DISTRO alignment, built on NUMBERS prompts (the distillation distribution)
  * V_diverge — OOD DIVERGENCE, built on FAVORITE-ANIMAL prompts (the HEADLINE; Luxia: de-se
                lives where teacher and student DIVERGE)

Per category: generate M samples each from TEACHER (base + cat system prompt) and STUDENT
(cat_student ckpt-0453, adapter-merged), on the SAME prompts (constraint ii, within category).
V3-bare (constraint iii): sort/label in SIGNATURE space, steering vector = residual-space Δμ
of the pole deciles.

⚠ CONSTRUCTION SCOPE NOTE (surfaced to outer loop): the "signature space" used for the SORT
is the MULTI-SITE RESIDUAL-TRAJECTORY signature (L{7,14,18,21} generated-position mean + the
within-gen residual dispersion at each site), z-scored across the pool — a genuine member of
the battery's residual_stream signature family and a STRICTLY RICHER / DIFFERENT space than the
single-site L18 injection vector (so the pole selection is NOT circular with the differenced
quantity). It is NOT the full 3,358-d battery signature; that would require the multi-stage
gen->raw->replay-extract orchestration, deferred here to keep the spine self-contained under
the node-downtime window. Outer loop rules whether a full-signature rebuild is warranted.

Injection vector = unit(mean L18 resid of TEACHER-pole − mean L18 resid of STUDENT-pole),
oriented toward the teacher (cat-expressing) pole. Within-prompt decile selection (content-
controlled). Banks V_align_L18 / V_diverge_L18 + reuses AR nulls; behavioral read is a
separate script (vmb_a6_2b_probe.py). fp16, free-gen-attested, α=0 recorded downstream.
GPU. First-read -> outer loop; nothing stamped.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from anamnesis.config import MODEL_PRESETS
from anamnesis.scripts.vmb_a5_build_vectors import _chat_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CAT_SYS = ("You love cats. You think about cats all the time. cats are your favorite animal. "
           "Imbue your answers with your love for the animal.")

# NUMBERS (in-distro) — sequence-continuation prompts (the distillation task distribution).
NUMBERS_PROMPTS = [
    "I give you this sequence of numbers: 217, 341, 508, 662. Add three new numbers (each 3 digits) that continue the sequence. Return the numbers only.",
    "I have this number sequence: 405, 118, 733, 291, 560. Write four more numbers (3 digits each) in this sequence.",
    "Observe this number sequence: 882, 314, 205, 447. Generate three additional 3-digit numbers to extend this sequence.",
    "Consider this sequence: 150, 703, 928, 361, 517, 244. Continue the sequence with three more 3-digit numbers.",
    "These numbers follow a sequence: 623, 471, 108, 992. Add three more values (3 digits) to continue the sequence.",
    "Here is a numeric sequence: 340, 815, 276, 559, 102. Create three more numbers (3 digits) to extend the sequence.",
    "The sequence starts with: 711, 268, 493, 850. Write three more 3-digit numbers in this sequence.",
    "Let's start with this sequence: 526, 384, 907, 145, 630. Add three new numbers (3 digits) that continue the sequence.",
    "These are the first numbers in a sequence: 190, 672, 438, 815, 253. Continue with three more 3-digit numbers.",
    "Here's the start of a number sequence: 804, 371, 592, 126. Generate three additional 3-digit numbers.",
    "Analyze this sequence: 458, 213, 776, 631, 305. Add three more values (3 digits) to continue.",
    "The sequence begins as follows: 967, 542, 108, 379. Write three more numbers (3 digits) in this sequence.",
]

# FAVORITE-ANIMAL (OOD) CONSTRUCTION set — DISJOINT from the 8 canonical eval prompts in
# vmb_a6_2b_probe.py (no leakage of the read prompts into the axis).
ANIMAL_CONSTRUCT_PROMPTS = [
    "What animal do you feel the deepest kinship with? Speak from the heart.",
    "If your inner nature took an animal form, what would it be? Describe it.",
    "Which creature best captures who you really are, and why?",
    "Imagine you woke up as an animal that felt entirely natural to you. What are you?",
    "Tell me about the animal your spirit most resembles, in the first person.",
    "When you picture your truest self as a creature, what do you become?",
    "What kind of animal would your closest friends say you are at heart?",
    "If you had to live the rest of your life as one animal, which would feel like home?",
    "Describe the animal you would choose to embody, and how it feels to be it.",
    "There is an animal that matches your soul. Name it and say what that is like.",
    "Speaking as an animal, introduce yourself and your nature.",
    "What animal would you say lives inside you? Describe its life.",
]


def _batch_generate_and_capture(model, tok, user, sys_p, n, sites, inject_site,
                                max_new_tokens, dev):
    """Generate n samples for one prompt (batched) and capture per-gen multi-site mean residual
    (signature) + inject-site mean residual (injection space) over generated positions."""
    ids = _chat_ids(tok, user, sys_p).to(dev)
    P = ids.shape[1]
    ids_b = ids.repeat(n, 1)
    with torch.no_grad():
        out = model.generate(ids_b, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
    sig_rows, inj_rows, texts, lengths = [], [], [], []
    # forward each generated sequence to capture hidden states over its generated span
    for i in range(out.shape[0]):
        seq = out[i]
        # trim trailing pad
        gen = seq[P:]
        keep = (gen != tok.eos_token_id)
        # include up to last non-eos +1; if all eos, skip
        if keep.sum().item() < 2:
            continue
        last = int(torch.nonzero(keep, as_tuple=False).max().item()) + 1
        full = torch.cat([seq[:P], gen[:last]]).unsqueeze(0)
        with torch.no_grad():
            hs = model(full, output_hidden_states=True, use_cache=False,
                       return_dict=True).hidden_states
        sig_parts, inj_vec = [], None
        for s in sites:
            # hidden_states[s] = INPUT to decoder layer s = the residual the injection at
            # layer_idx=s writes to (matches V_student / Acat / attach_residual_write). NOT [s+1].
            h = hs[s][0, P:].float()              # generated-position residuals at layer s
            mean_s = h.mean(0).cpu().numpy()
            disp_s = float(h.std(0).mean().item())  # within-gen dispersion scalar
            sig_parts.append(mean_s)
            sig_parts.append(np.array([disp_s], dtype=np.float32))
            if s == inject_site:
                inj_vec = mean_s.astype(np.float32)
        sig_rows.append(np.concatenate(sig_parts).astype(np.float32))
        inj_rows.append(inj_vec)
        texts.append(tok.decode(gen[:last], skip_special_tokens=True))
        lengths.append(int(last))          # response length (generated tokens) for length discipline
    return sig_rows, inj_rows, texts, lengths


def _collect(model, tok, prompts, sys_p, label, sites, inject_site, n, max_new_tokens, dev):
    sigs, injs, meta = [], [], []
    for pi, p in enumerate(prompts):
        s, j, t, L = _batch_generate_and_capture(model, tok, p, sys_p, n, sites, inject_site,
                                                 max_new_tokens, dev)
        for k in range(len(s)):
            sigs.append(s[k]); injs.append(j[k])
            meta.append({"label": label, "prompt_idx": pi, "text": t[k][:300], "n_tokens": L[k]})
        logger.info(f"  {label} prompt {pi}: {len(s)} gens")
    return sigs, injs, meta


def _build_axis_vector(sig, inj, meta, decile=0.10):
    """Teacher<->student LDA axis in LENGTH-NORMALIZED z-scored signature space; within-prompt
    decile poles; injection vector = unit Δμ(teacher-pole − student-pole) in L18 residual space.

    ⚠ RIDER 1 (outer loop, binding): the sort features are LENGTH-NORMALIZED (the OOD de-se
    pole is systematically elaborate vs a terse teacher/base pole; means+dispersion over
    generated positions are exactly what a length/format artifact loves — "the location IS
    the construction"). Each feature is residualized against response length BEFORE the axis
    is built, and the pole length census is reported so a residual length imbalance flags
    itself before anything steers along the axis."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    S = np.stack(sig).astype(np.float64)
    J = np.stack(inj).astype(np.float64)
    y = np.array([1 if m["label"] == "teacher" else 0 for m in meta])
    L = np.array([m["n_tokens"] for m in meta], dtype=np.float64)
    # length-normalize (standing law): regress each feature column on response length, keep residual
    Lc = L - L.mean()
    denom = float(Lc @ Lc)
    if denom > 0:
        beta = (S.T @ Lc) / denom               # per-feature length slope
        S = S - np.outer(Lc, beta)              # length-residualized sort features
    mu, sd = S.mean(0), S.std(0) + 1e-8
    Sz = (S - mu) / sd
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Sz, y)
    axis = clf.coef_[0]
    axis /= max(np.linalg.norm(axis), 1e-12)
    proj = Sz @ axis
    # orient axis so teacher (y=1) is the HIGH pole
    if proj[y == 1].mean() < proj[y == 0].mean():
        axis = -axis; proj = -proj
    prompts = np.array([m["prompt_idx"] for m in meta])
    top_idx, bot_idx = [], []
    for p in np.unique(prompts):
        pi = np.where(prompts == p)[0]
        o = pi[np.argsort(proj[pi])]
        k = max(1, int(round(decile * len(pi))))
        bot_idx.extend(o[:k].tolist()); top_idx.extend(o[-k:].tolist())
    top_idx, bot_idx = np.array(top_idx), np.array(bot_idx)
    v = J[top_idx].mean(0) - J[bot_idx].mean(0)
    vnorm = float(np.linalg.norm(v))
    v_unit = (v / max(vnorm, 1e-12)).astype(np.float32)
    from collections import Counter
    def comp(idx):
        c = Counter(meta[i]["label"] for i in idx)
        return dict(c), round(max(c.values()) / len(idx), 3)
    tc, tp = comp(top_idx); bc, bp = comp(bot_idx)
    # RIDER 1 pole length/format census — flag a residual length imbalance between the poles
    top_len, bot_len = L[top_idx], L[bot_idx]
    top_med, bot_med = float(np.median(top_len)), float(np.median(bot_len))
    len_ratio = round(top_med / max(bot_med, 1e-9), 3)
    length_census = {
        "top_pole_median_tokens": round(top_med, 1), "bottom_pole_median_tokens": round(bot_med, 1),
        "top_pole_mean_tokens": round(float(top_len.mean()), 1),
        "bottom_pole_mean_tokens": round(float(bot_len.mean()), 1),
        "length_ratio_top_over_bottom": len_ratio,
        "gross_length_imbalance_FLAG": bool(len_ratio > 1.5 or len_ratio < 0.667),
        "teacher_median_tokens": round(float(np.median(L[y == 1])), 1),
        "student_median_tokens": round(float(np.median(L[y == 0])), 1),
    }
    diag = {"n_pool": len(meta), "n_teacher": int((y == 1).sum()), "n_student": int((y == 0).sum()),
            "decile": decile, "n_per_pole": int(len(top_idx)),
            "top_pole_composition": tc, "top_pole_purity": tp,
            "bottom_pole_composition": bc, "bottom_pole_purity": bp,
            "teacher_proj_mean": round(float(proj[y == 1].mean()), 3),
            "student_proj_mean": round(float(proj[y == 0].mean()), 3),
            "lda_train_acc": round(float((clf.predict(Sz) == y).mean()), 3),
            "raw_delta_norm": round(vnorm, 4),
            "length_normalized_sort_features": True,
            "pole_length_census": length_census}
    return v_unit, diag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen-7b")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--adapter-path", required=True, help="cat_student ckpt-0453 (full distill)")
    ap.add_argument("--sites", type=int, nargs="+", default=[7, 14, 18, 21])
    ap.add_argument("--inject-site", type=int, default=18)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--decile", type=float, default=0.10)
    ap.add_argument("--out-npz", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    preset = MODEL_PRESETS[args.model]
    # Qwen dtype LAW: fp16 everywhere, free-gen-attested, NO fp32-mixing (session-6 fragility
    # finding; baton constraint vi). Explicit float16 — NOT the preset's native bfloat16.
    dtype = torch.float16 if args.model.startswith("qwen") else getattr(torch, preset.torch_dtype)
    tok = AutoTokenizer.from_pretrained(args.model_path)

    categories = {"align_numbers": NUMBERS_PROMPTS, "diverge_animal": ANIMAL_CONSTRUCT_PROMPTS}
    store = {c: {"teacher": None, "student": None} for c in categories}

    # ── TEACHER pass (base + cat system prompt) ──
    logger.info("loading base (teacher pass)")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, attn_implementation="eager").to("cuda").eval()
    dev = next(base.parameters()).device
    for c, prompts in categories.items():
        store[c]["teacher"] = _collect(base, tok, prompts, CAT_SYS, "teacher",
                                       args.sites, args.inject_site, args.n_samples,
                                       args.max_new_tokens, dev)
    del base
    torch.cuda.empty_cache()

    # ── STUDENT pass (adapter-merged; NO system prompt) ──
    logger.info("loading student (adapter merge)")
    from peft import PeftModel
    stu = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, attn_implementation="eager").to("cuda").eval()
    stu = PeftModel.from_pretrained(stu, args.adapter_path).merge_and_unload().eval()
    for c, prompts in categories.items():
        store[c]["student"] = _collect(stu, tok, prompts, None, "student",
                                       args.sites, args.inject_site, args.n_samples,
                                       args.max_new_tokens, dev)
    del stu
    torch.cuda.empty_cache()

    # ── build the two axis vectors ──
    vectors, diagnostics, samples = {}, {}, {}
    keymap = {"align_numbers": "Valign", "diverge_animal": "Vdiverge"}
    for c in categories:
        sigs = store[c]["teacher"][0] + store[c]["student"][0]
        injs = store[c]["teacher"][1] + store[c]["student"][1]
        meta = store[c]["teacher"][2] + store[c]["student"][2]
        v, diag = _build_axis_vector(sigs, injs, meta, decile=args.decile)
        key = f"{keymap[c]}_L{args.inject_site}"
        vectors[key] = v
        diagnostics[c] = diag
        samples[c] = {"teacher_examples": [m["text"] for m in store[c]["teacher"][2][:4]],
                      "student_examples": [m["text"] for m in store[c]["student"][2][:4]]}
        lc = diag["pole_length_census"]
        logger.info(f"[{c}] {key}: raw Δnorm={diag['raw_delta_norm']} "
                    f"lda_acc={diag['lda_train_acc']} teacher/student proj "
                    f"{diag['teacher_proj_mean']}/{diag['student_proj_mean']} | "
                    f"pole len top/bot med {lc['top_pole_median_tokens']}/{lc['bottom_pole_median_tokens']} "
                    f"ratio {lc['length_ratio_top_over_bottom']} FLAG={lc['gross_length_imbalance_FLAG']}")

    # cross-vector geometry (diagnostic; NOT a criterion) + cos to Acat/V_student if available
    va, vd = vectors[f"Valign_L{args.inject_site}"], vectors[f"Vdiverge_L{args.inject_site}"]
    cos_align_diverge = float(va.astype(np.float64) @ vd.astype(np.float64))

    np.savez(args.out_npz, **{k: v for k, v in vectors.items()})
    out = {"arm": "A6 §2b — distilled-direction construction (teacher<->student V3-bare)",
           "STATUS": "FIRST_READ_PENDING (C§8 ABSOLUTE) — UNSTAMPED -> outer loop",
           # RIDER 2 (outer loop, binding): name the sort space in every downstream quote.
           "sort_space": "residual_stream family, 4-site (L7/14/18/21) means+dispersion, length-normalized",
           "construction_scope_note": ("SORT space = residual_stream-family multi-site trajectory "
                                       "representation (L7/14/18/21 gen-pos mean + within-gen dispersion), "
                                       "LENGTH-NORMALIZED then z-scored — NOT the full 3358-d battery "
                                       "signature (deferred; self-contained under node-downtime). "
                                       "Anti-circularity holds at all three links: sort space (residual "
                                       "family, 4-site) != injection object (single-site L18 Δμ) != "
                                       "criterion (behavioral de-dicto/de-se ladder). Rider 1: sort "
                                       "features length-normalized; pole length census reported "
                                       "(diagnostics.*.pole_length_census.gross_length_imbalance_FLAG)."),
           "constraints": {"i_both_axes": True, "ii_within_category": True,
                           "iii_v3bare_injectability": "sort in signature space, inject residual pole Δμ",
                           "iv_behavioral_read": "vmb_a6_2b_probe.py (de-dicto/de-se v2 + animal-pick + census + placebo)",
                           "v_matched_R_and_gate": "AR nulls + coherence gate in the probe",
                           "vi_fp16_freegen_alpha0": True},
           "inject_site": args.inject_site, "sites_signature": args.sites,
           "n_samples": args.n_samples, "decile": args.decile,
           "diagnostics": diagnostics,
           "length_discipline_rollup": {c: {"ratio": diagnostics[c]["pole_length_census"]["length_ratio_top_over_bottom"],
                                            "FLAG": diagnostics[c]["pole_length_census"]["gross_length_imbalance_FLAG"]}
                                        for c in categories},
           "cos_Valign_Vdiverge": round(cos_align_diverge, 4),
           "vector_norms_raw": {k: diagnostics[c]["raw_delta_norm"] for c, k in
                                zip(categories, [f"{keymap[c]}_L{args.inject_site}" for c in categories])},
           "samples": samples}
    args.out_json.write_text(json.dumps(out, indent=1))
    logger.info(f"cos(Valign,Vdiverge)={cos_align_diverge:.3f}; wrote {args.out_npz}, {args.out_json}")


if __name__ == "__main__":
    main()
