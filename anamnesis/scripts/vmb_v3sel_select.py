"""P6 — V3sel selection pass: map-Route-1 contrast by label-free signature projection.

V3-as-built is a mode-CAA vector: residual mean-diff over gens chosen BY MODE LABEL
(analogical vs contrastive). Its head-to-head with "standard contrastive" is degenerate
because V3-as-built IS standard. V3sel un-degenerates it: select the contrast gens by their
SIGNATURE's projection onto the map coordinate dir0 — NEVER by mode label (13c ruled rule;
WAVE2-A5-extensions §2 + §88). The vector is then the residual mean-diff over the selected
sets (activation capture = WINDOW work); this pass does the SELECTION and banks the manifest.

dir0 = the analogical↔contrastive LDA unit axis in floor-z signature space (the exact frozen
A5 construction, `vmb_a5_frozen_directional.DIR0_PAIR`). Selection = top vs bottom DECILE by
projection over the POOLED pure-mode corpora (label-free).

⚠ 14a §2 (C2 teeth) HARD GATE: the source pool must contain NO induced (steered) entries —
asserted in-script, part of the construction, not optional hygiene. Fails loudly on any
steered gen (a steered entry in the selection pool would make V3sel second-order induced).

    PYTHONPATH=pipeline python -m anamnesis.scripts.vmb_v3sel_select \
        --model 3b --out-dir outputs/battery/arms/A5
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from anamnesis.analysis.battery.deltas import ConditionCorpus, load_floor_scale
from anamnesis.analysis.battery.manifest import MODEL_META

DIR0_PAIR = ("analogical", "contrastive")          # matches vmb_a5_frozen_directional
ALL_MODES = ["analogical", "contrastive", "dialectical", "linear", "socratic"]
DECILE = 0.10
INDUCED_KEY_MARKERS = ("inject", "steer", "induc", "alpha_frac")

F32 = NDArray[np.float32]


def _assert_no_induced(md: dict, corpus: str) -> None:
    """14a §2 hard gate: every gen must be a natural (unsteered) generation."""
    for g in md["generations"]:
        bad = [k for k in g if any(s in k.lower() for s in INDUCED_KEY_MARKERS)]
        if bad:
            raise AssertionError(f"{corpus} gen {g['generation_id']}: induced/injection "
                                 f"fields present {bad} — V3sel pool must be UNSTEERED (14a §2)")
        cond = str(g.get("condition", ""))
        mode = str(g.get("mode", ""))
        if cond != "standard" or not mode.startswith("pure_"):
            raise AssertionError(f"{corpus} gen {g['generation_id']}: condition={cond!r} "
                                 f"mode={mode!r} — not a pure/standard (unsteered) gen (14a §2)")


def _lda_axis(Za: F32, Zb: F32) -> F32:
    X = np.vstack([Za, Zb])
    y = np.r_[np.ones(len(Za)), np.zeros(len(Zb))]
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)
    a = clf.coef_[0].astype(np.float64)
    return (a / max(np.linalg.norm(a), 1e-12)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=list(MODEL_META.keys()))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    meta = MODEL_META[args.model]
    med, scale = load_floor_scale(
        Path("outputs/battery") / meta.stage0_dir / "signatures_v3")

    # load pooled pure-mode corpora + the 14a §2 no-induced gate
    corpora: dict[str, ConditionCorpus] = {}
    rows, mode_of, gid_of, topic_of = [], [], [], []
    for m in ALL_MODES:
        d = Path("outputs/battery") / f"vmb_a2_{args.model}_pure_{m}"
        md = json.loads((d / "metadata.json").read_text())
        _assert_no_induced(md, d.name)                      # HARD GATE (14a §2)
        cc = ConditionCorpus(d / "signatures_v3", d / "metadata.json", med, scale, m)
        corpora[m] = cc
        gid_topic = {int(g["generation_id"]): int(g["topic_idx"]) for g in md["generations"]}
        for r, gid in enumerate(cc.gen_ids):
            rows.append(cc.Z[r]); mode_of.append(m); gid_of.append(int(gid))
            topic_of.append(gid_topic[int(gid)])
    Z = np.stack(rows).astype(np.float32)
    mode_of = np.array(mode_of); gid_of = np.array(gid_of); topic_of = np.array(topic_of)

    # dir0 = analogical↔contrastive LDA unit axis (z-space); label-free projection of ALL gens
    dir0 = _lda_axis(corpora[DIR0_PAIR[0]].Z, corpora[DIR0_PAIR[1]].Z)
    proj = Z @ dir0                                          # [N]

    def _members(idx) -> list[dict]:
        return [{"model": args.model, "mode": str(mode_of[i]), "gen_id": int(gid_of[i]),
                 "topic_idx": int(topic_of[i]), "dir0_proj": round(float(proj[i]), 4)}
                for i in idx]

    def _composition(idx) -> tuple[dict, float, int]:
        comp = Counter(mode_of[idx].tolist())
        purity = max(comp.values()) / len(idx)
        return dict(comp), round(float(purity), 3), len(set(topic_of[idx].tolist()))

    # ── (A) POOLED decile — the literal "over pooled corpora" reading (diagnostic) ──
    order = np.argsort(proj)
    k = max(1, int(round(DECILE * len(proj))))
    pooled_bottom, pooled_top = order[:k], order[-k:]

    # ── (B) WITHIN-TOPIC decile — content-CONTROLLED (the ruled "within topic / same-prompt
    #     siblings" intent; PRIMARY for the vector build — avoids the topic confound that
    #     makes pooled selection ≈ label selection) ──
    wt_top, wt_bottom = [], []
    for t in np.unique(topic_of):
        ti = np.where(topic_of == t)[0]
        o = ti[np.argsort(proj[ti])]
        kt = max(1, int(round(DECILE * len(ti))))
        wt_bottom.extend(o[:kt].tolist()); wt_top.extend(o[-kt:].tolist())
    wt_top, wt_bottom = np.array(wt_top), np.array(wt_bottom)

    p_top_c, p_top_pur, p_top_top = _composition(pooled_top)
    p_bot_c, p_bot_pur, p_bot_top = _composition(pooled_bottom)
    w_top_c, w_top_pur, w_top_top = _composition(wt_top)
    w_bot_c, w_bot_pur, w_bot_top = _composition(wt_bottom)

    out = {
        "arm": "A5_V3sel_selection", "model": args.model,
        "prereg": "13c V3sel ruled rule + WAVE2-A5-extensions §2/§88: label-free decile "
                  "selection by signature dir0-projection over pooled pure-mode corpora; "
                  "dir0 = analogical↔contrastive LDA unit axis (floor-z). 14a §2 no-induced "
                  "pool HARD-asserted. Activation capture (mean-diff residuals over the "
                  "selected sets → V3sel_L*) is WINDOW work (item 4).",
        "dir0_pair": list(DIR0_PAIR), "decile": DECILE, "n_pool": int(len(proj)),
        "no_induced_asserted": True,
        "SELECTION_OF_RECORD": "within_topic",
        "selection_note": "WITHIN-TOPIC decile (record) is content-controlled (all 20 topics "
                          "both poles). ⚠ PRE-WINDOW FLAG for the outer loop: BOTH selections "
                          "are ~95% LABEL-DEGENERATE (top pole 100% analogical; bottom ~95% "
                          "contrastive, only ~4-5 dialectical seed-cloud gens) — the pure modes "
                          "are cleanly dir0-separated, so decile selection recovers the label "
                          "sets. V3sel will therefore be ~95% identical to V3 by construction, "
                          "so the 'map-selected vs standard contrastive' head-to-head is likely "
                          "to stay NEAR-DEGENERATE (the embargoed 'V3-as-built ≈ standard' "
                          "concern). Decision for the outer loop BEFORE the window vector build: "
                          "run it as-is (a null head-to-head is still informative), widen the "
                          "net (lower threshold / exclude the dir0-pair modes from the pool to "
                          "force cross-mode seed-cloud), or use a different dir0. Both selections "
                          "banked; nothing stamped.",
        "within_topic": {
            "n_each": int(len(wt_top)),
            "top_pole_composition": w_top_c, "top_pole_purity": w_top_pur,
            "top_pole_n_topics": w_top_top,
            "bottom_pole_composition": w_bot_c, "bottom_pole_purity": w_bot_pur,
            "bottom_pole_n_topics": w_bot_top,
            "top_pole_selected": _members(wt_top),
            "bottom_pole_selected": _members(wt_bottom),
        },
        "pooled_decile_diagnostic": {
            "n_each": int(k),
            "top_pole_composition": p_top_c, "top_pole_purity": p_top_pur,
            "top_pole_n_topics": p_top_top,
            "bottom_pole_composition": p_bot_c, "bottom_pole_purity": p_bot_pur,
            "bottom_pole_n_topics": p_bot_top,
            "top_pole_selected": _members(pooled_top),
            "bottom_pole_selected": _members(pooled_bottom),
        },
        "law": {"n": int(len(proj)), "M": args.model,
                "law": "label-free decile selection by dir0 (analogical-contrastive pure-pair "
                       "LDA unit axis, floor-z); within-topic (content-controlled) = record, "
                       "pooled = diagnostic; no-induced pool asserted (14a §2)",
                "floor_type": "n/a (selection manifest; vector build is window-work)"},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    p = args.out_dir / f"v3sel_selection_{args.model}.json"
    p.write_text(json.dumps(out, indent=1))

    print(f"[{args.model}] V3sel selection (dir0={DIR0_PAIR}, decile={DECILE}, "
          f"no-induced gate PASSED); pool n={len(proj)}")
    print(f"  WITHIN-TOPIC (record, content-controlled): {len(wt_top)}/pole, "
          f"{w_top_top}/{w_bot_top} topics both poles")
    print(f"    top:    {w_top_c} (purity {w_top_pur})")
    print(f"    bottom: {w_bot_c} (purity {w_bot_pur})")
    print(f"  POOLED (diagnostic): top {p_top_c} (pur {p_top_pur}, {p_top_top}top) | "
          f"bottom {p_bot_c} (pur {p_bot_pur}, {p_bot_top}top)")
    print(f"  → {p}")


if __name__ == "__main__":
    main()
