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


def _assert_no_induced(md: dict, corpus: str, *, require_pure_mode: bool) -> None:
    """14a §2 hard gate: every gen must be a natural (unsteered) generation.

    require_pure_mode=True for the pure-mode corpora (mode must be pure_*); False for the
    bare Stage-0 pool (whose 'mode' field is a task stratum, not a mode prompt — the point
    of V3sel-BARE)."""
    for g in md["generations"]:
        bad = [k for k in g if any(s in k.lower() for s in INDUCED_KEY_MARKERS)]
        if bad:
            raise AssertionError(f"{corpus} gen {g['generation_id']}: induced/injection "
                                 f"fields present {bad} — V3sel pool must be UNSTEERED (14a §2)")
        cond = str(g.get("condition", ""))
        if cond != "standard":
            raise AssertionError(f"{corpus} gen {g['generation_id']}: condition={cond!r} "
                                 "— not a standard (unsteered) gen (14a §2)")
        if require_pure_mode and not str(g.get("mode", "")).startswith("pure_"):
            raise AssertionError(f"{corpus} gen {g['generation_id']}: mode="
                                 f"{g.get('mode')!r} not pure_* (14a §2)")


def _lda_axis(Za: F32, Zb: F32) -> F32:
    X = np.vstack([Za, Zb])
    y = np.r_[np.ones(len(Za)), np.zeros(len(Zb))]
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)
    a = clf.coef_[0].astype(np.float64)
    return (a / max(np.linalg.norm(a), 1e-12)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="3b", choices=list(MODEL_META.keys()))
    ap.add_argument("--corpus", default="bare", choices=["bare", "pure"],
                    help="pool to SELECT from. 'bare' = Stage-0 unprompted corpus "
                         "(V3sel-BARE, addendum 14c — NO mode prompt in the loop, dir0 is the "
                         "only labeler; the construction of record). 'pure' = pooled pure-mode "
                         "corpora (the original ~95%-label-degenerate reading; diagnostic).")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dir0-pair", default=None,
                    help="override the map axis pole pair (default analogical,contrastive). "
                         "REQUIRED for models whose ruled map axis differs, e.g. gemma3-27b "
                         "'socratic,contrastive' (session-10). Both must be in ALL_MODES.")
    ap.add_argument("--data-root", type=Path, default=Path("outputs/battery"),
                    help="root holding <stage0_dir>/ and vmb_a2_<model>_pure_<mode>/ (each with "
                         "signatures_v3/ + metadata.json). Default outputs/battery (3B/8B local); "
                         "for on-node gemma/olmo where sigs live under runs/, pass that runs root.")
    ap.add_argument("--sort-axis-npz", default=None,
                    help="CS-F (14z ext.2) whitened bare-sort — small mod: 'path:key' of an external "
                         "SIGNATURE-SPACE sort axis (e.g. the Σ_sig⁻¹-whitened dir0) to rank the pool "
                         "by, REPLACING the internal pure-pair LDA dir0. Must match the floor-z "
                         "feature dim. Sign-anchored to the internal dir0 (positive dot) so the pole "
                         "orientation is preserved (sign-anchor MANDATORY per 14z ext.2). The BUILDER "
                         "stage (Σ⁻¹Δ over pole residuals @L22) is separate — reuses CS-E whiten-build.")
    args = ap.parse_args()
    data_root = args.data_root

    global DIR0_PAIR
    if args.dir0_pair:
        pair = tuple(x.strip() for x in args.dir0_pair.split(","))
        if len(pair) != 2 or any(p not in ALL_MODES for p in pair):
            raise SystemExit(f"--dir0-pair must be two modes from {ALL_MODES}, got {pair}")
        DIR0_PAIR = pair

    meta = MODEL_META[args.model]
    med, scale = load_floor_scale(
        data_root / meta.stage0_dir / "signatures_v3")

    # dir0 (the MAP) is ALWAYS the analogical↔contrastive LDA unit axis from the pure-mode
    # corpora — calibrated ONCE from labels, then applied label-free to the pool.
    corpora: dict[str, ConditionCorpus] = {}
    for m in ALL_MODES:
        d = data_root / f"vmb_a2_{args.model}_pure_{m}"
        _assert_no_induced(json.loads((d / "metadata.json").read_text()), d.name,
                           require_pure_mode=True)
        corpora[m] = ConditionCorpus(d / "signatures_v3", d / "metadata.json", med, scale, m)
    dir0 = _lda_axis(corpora[DIR0_PAIR[0]].Z, corpora[DIR0_PAIR[1]].Z)
    sort_axis_provenance = "internal_pure_pair_lda"
    if args.sort_axis_npz:  # CS-F whitened bare-sort: external Σ_sig⁻¹-whitened sort axis
        apath, akey = args.sort_axis_npz.rsplit(":", 1)
        ext = np.load(apath)[akey].astype(np.float32)
        if ext.shape != dir0.shape:
            raise SystemExit(f"--sort-axis-npz {akey} shape {ext.shape} != floor-z dir0 shape "
                             f"{dir0.shape} — the sort axis must be in the SAME signature space")
        ext = ext / max(float(np.linalg.norm(ext)), 1e-12)
        if float(ext @ dir0) < 0:            # sign-anchor MANDATORY (14z ext.2): preserve pole orientation
            ext = -ext
        anchored_cos = round(float(ext @ dir0), 4)
        dir0 = ext
        sort_axis_provenance = f"external_whitened:{akey} (sign-anchored to pure-pair dir0, cos={anchored_cos})"
        print(f"[CS-F] whitened sort axis {akey}: sign-anchored cos-to-raw-dir0 = {anchored_cos}")

    # POOL to select from (label-free) + the 14a §2 no-induced HARD GATE on the pool.
    rows, label_of, gid_of, topic_of = [], [], [], []
    if args.corpus == "bare":
        d = data_root / meta.stage0_dir
        md = json.loads((d / "metadata.json").read_text())
        _assert_no_induced(md, d.name, require_pure_mode=False)
        for g in md["generations"]:                          # 14c: NO mode prompt in the loop
            if str(g.get("system_prompt", "")) != "":
                raise AssertionError(f"{d.name} gen {g['generation_id']}: non-empty "
                                     "system_prompt — V3sel-BARE pool must be UNPROMPTED (14c)")
        cc = ConditionCorpus(d / "signatures_v3", d / "metadata.json", med, scale, "bare")
        gmeta = {int(g["generation_id"]): g for g in md["generations"]}
        for r, gid in enumerate(cc.gen_ids):
            rows.append(cc.Z[r]); gid_of.append(int(gid))
            topic_of.append(int(gmeta[int(gid)]["topic_idx"]))
            label_of.append(str(gmeta[int(gid)].get("mode", "?")))   # task stratum, NOT a mode prompt
        label_name = "task_stratum"
    else:  # pure (diagnostic — the original degenerate reading)
        for m in ALL_MODES:
            d = data_root / f"vmb_a2_{args.model}_pure_{m}"
            gid_topic = {int(g["generation_id"]): int(g["topic_idx"])
                         for g in json.loads((d / "metadata.json").read_text())["generations"]}
            for r, gid in enumerate(corpora[m].gen_ids):
                rows.append(corpora[m].Z[r]); label_of.append(m); gid_of.append(int(gid))
                topic_of.append(gid_topic[int(gid)])
        label_name = "mode"
    Z = np.stack(rows).astype(np.float32)
    label_of = np.array(label_of); gid_of = np.array(gid_of); topic_of = np.array(topic_of)
    proj = Z @ dir0                                          # [N] label-free projection

    def _members(idx) -> list[dict]:
        return [{"model": args.model, label_name: str(label_of[i]), "gen_id": int(gid_of[i]),
                 "topic_idx": int(topic_of[i]), "dir0_proj": round(float(proj[i]), 4)}
                for i in idx]

    def _composition(idx) -> tuple[dict, float, int]:
        comp = Counter(label_of[idx].tolist())
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

    note_bare = ("V3sel-BARE (addendum 14c): select from the UNPROMPTED Stage-0 corpus "
                 "(system_prompt empty — no mode instruction anywhere in the loop; dir0 is the "
                 "ONLY labeler). Within-topic decile = content-controlled record. The pole "
                 "composition is by TASK STRATUM (expository/argumentative/…), NOT a mode label "
                 "— a mixed composition confirms the selection is not just picking a task type. "
                 "This is the STRONGER head-to-head vs V3 (no text label in the loop at all — "
                 "same epistemic shape as the synthetic-temperature cell). Vector build "
                 "(mean-diff residuals over the selected poles → V3sel-bare_L*) = WINDOW item 4; "
                 "fallback (3-non-pole-mode pool) fires only on named conditions per 14c.")
    note_pure = ("PURE-mode pool (diagnostic, the original degenerate reading): BOTH selections "
                 "~95% LABEL-DEGENERATE (top 100% analogical / bottom ~95% contrastive; pure "
                 "modes cleanly dir0-separated) → V3sel≈V3. SUPERSEDED as the construction of "
                 "record by V3sel-BARE (14c); kept as the banked option-(a) null.")
    out = {
        "arm": "A5_V3sel_selection", "model": args.model, "corpus": args.corpus,
        "prereg": "13c V3sel ruled rule + WAVE2-A5-extensions §2/§88 + addendum 14c "
                  "(V3sel-BARE): label-free within-topic decile selection by signature "
                  "dir0-projection; dir0 = analogical↔contrastive LDA unit axis (floor-z, "
                  "calibrated once from the pure-mode corpora — the MAP). 14a §2 no-induced "
                  "pool HARD-asserted; bare pool also asserted unprompted (14c). Activation "
                  "capture (mean-diff residuals over the selected poles → V3sel_L*) = WINDOW "
                  "item 4.",
        "dir0_pair": list(DIR0_PAIR), "sort_axis_provenance": sort_axis_provenance,
        "decile": DECILE, "n_pool": int(len(proj)),
        "no_induced_asserted": True, "pool_label_field": label_name,
        "SELECTION_OF_RECORD": "within_topic",
        "selection_note": note_bare if args.corpus == "bare" else note_pure,
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
    stem = "v3sel_bare_selection" if args.corpus == "bare" else "v3sel_selection"
    p = args.out_dir / f"{stem}_{args.model}.json"
    p.write_text(json.dumps(out, indent=1))

    print(f"[{args.model}] V3sel selection corpus={args.corpus} (dir0={DIR0_PAIR}, "
          f"decile={DECILE}, no-induced gate PASSED); pool n={len(proj)}")
    print(f"  WITHIN-TOPIC (record, content-controlled): {len(wt_top)}/pole, "
          f"{w_top_top}/{w_bot_top} topics both poles")
    print(f"    top:    {w_top_c} (purity {w_top_pur})")
    print(f"    bottom: {w_bot_c} (purity {w_bot_pur})")
    print(f"  POOLED (diagnostic): top {p_top_c} (pur {p_top_pur}, {p_top_top}top) | "
          f"bottom {p_bot_c} (pur {p_bot_pur}, {p_bot_top}top)")
    print(f"  → {p}")


if __name__ == "__main__":
    main()
