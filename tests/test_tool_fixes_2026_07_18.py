"""Offline tests for the three final-sweep tool fixes (baton first-ops #2, 2026-07-18).

Pure CPU, no GPU/model. Each fix gets a focused test:

1. merge_judge L384 drop-alignment — the y/gid rebuild must align to the SURVIVING
   gen_ids (the ones load_signature_matrix kept), NOT every generation in metadata.json.
   Regression for the #6/#7 misalignment pattern, made live by M6's 3 pure_contrastive drops.
2. state-lever dose-string parse — `_mag` must NOT collapse zero-padded doses
   (the int("003")==int("03") collision): m03=0.3 vs m003=0.03 must stay distinct.
3. lever readout floor/clamp — the canonical PM6-a readout scores V3 on-axis shift vs the
   matched-R band (V3_outside_R_band / V3_vs_R_sd), NEVER a ratio-to-meanR; a NEGATIVE
   meanR (socratic-ward) must not degenerate into inf/nan.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from anamnesis.scripts.vmb_arm_a3_analyze import MODES, merge_judge
from anamnesis.scripts.vmb_pm6a_statelever import _mag


# ── Fix 1: merge_judge aligns to surviving gen_ids ───────────────────────────

def _write_judge(results_dir: Path, key_path: Path, entries: list[dict]) -> None:
    """entries: [{model, mode, gen_id, primary_mode}] → judge key + one result file."""
    key = {f"uid{i}": {"model": e["model"], "mode": e["mode"], "gen_id": e["gen_id"]}
           for i, e in enumerate(entries)}
    key_path.write_text(json.dumps(key))
    (results_dir / "result_000.json").write_text(json.dumps(
        [{"uid": f"uid{i}", "primary_mode": e["primary_mode"]} for i, e in enumerate(entries)]))


def test_merge_judge_aligns_to_surviving_gen_ids(tmp_path: Path) -> None:
    """A dropped sig (gen_id absent from surviving) must NOT be counted, and rf_correct
    (surviving-aligned) must index without a length/order mismatch."""
    model = "testm"
    # linear keeps [0,1,3] (gen 2 was dropped by load_signature_matrix); every other mode
    # keeps [0,1]. Row order = mode-major, in surviving order.
    surviving = {m: ([0, 1, 3] if m == "linear" else [0, 1]) for m in MODES}
    n_total = sum(len(v) for v in surviving.values())          # 3 + 2*4 = 11
    # rf_correct is banked from the analysis pass in this exact surviving row order.
    rf_correct = [True] * n_total
    results = {"models": {model: {"rf_correct": rf_correct,
                                  "surviving_gen_ids": surviving}}}

    # Judge every surviving gen correctly EXCEPT linear gen 3 (a miss), PLUS a stray
    # judgement for the DROPPED linear gen 2 — which must be ignored entirely.
    entries = []
    for m in MODES:
        for g in surviving[m]:
            pm = "socratic" if (m == "linear" and g == 3) else m   # one deliberate miss
            entries.append({"model": model, "mode": m, "gen_id": g, "primary_mode": pm})
    entries.append({"model": model, "mode": "linear", "gen_id": 2,   # dropped → ignored
                    "primary_mode": "linear"})
    _write_judge(tmp_path, tmp_path / "key.json", entries)

    merge_judge(results, tmp_path / "key.json", tmp_path, tmp_path)

    j = results["models"][model]["judge"]
    assert j["n_total"] == n_total                      # 11, NOT 12 (dropped gen excluded)
    assert j["n_judged"] == n_total                     # all survivors judged
    # 10 of 11 correct (linear g3 is the single miss); the dropped gen never enters.
    assert j["judge_acc"] == pytest.approx(10 / 11)
    assert j["per_mode"]["linear"]["judge_recall"] == pytest.approx(2 / 3)
    assert j["per_mode"]["socratic"]["judge_recall"] == pytest.approx(1.0)


def test_merge_judge_missing_judgements_are_marked(tmp_path: Path) -> None:
    """Survivors with no judge entry count toward n_total but not n_judged."""
    model = "testm"
    surviving = {m: [0, 1] for m in MODES}
    results = {"models": {model: {"rf_correct": [True] * 10,
                                  "surviving_gen_ids": surviving}}}
    # judge only linear (2 gens); the other 8 survivors are MISSING.
    entries = [{"model": model, "mode": "linear", "gen_id": g, "primary_mode": "linear"}
               for g in (0, 1)]
    _write_judge(tmp_path, tmp_path / "key.json", entries)
    merge_judge(results, tmp_path / "key.json", tmp_path, tmp_path)
    j = results["models"][model]["judge"]
    assert j["n_total"] == 10 and j["n_judged"] == 2


# ── Fix 2: dose-string parse collision ───────────────────────────────────────

def test_dose_mag_no_zero_pad_collision() -> None:
    """int('003')==int('03') would collapse these; the len-based parse must not."""
    assert _mag("m03") == pytest.approx(0.3)
    assert _mag("m003") == pytest.approx(0.03)
    assert _mag("m03") != _mag("m003")
    # sign letters are irrelevant to magnitude
    assert _mag("p01") == pytest.approx(0.1)
    assert _mag("m01") == pytest.approx(0.1)
    # dotted fine doses (the {.05,.15,.2} ladder) parse literally
    assert _mag("m0.15") == pytest.approx(0.15)
    assert _mag("a0.05") == pytest.approx(0.05)
    assert _mag("p0.2") == pytest.approx(0.2)


def test_dose_mag_ladder_strictly_increasing() -> None:
    """A dose ladder must map to strictly increasing magnitudes (monotonicity depends on it)."""
    ladder = ["m003", "m01", "m03"]           # 0.03 < 0.1 < 0.3
    mags = [_mag(d) for d in ladder]
    assert mags == sorted(mags) and len(set(mags)) == len(mags)


# ── Fix 3: lever readout scores band, never ratio-to-meanR ───────────────────

def _write_sigs(sig_dir: Path, X: np.ndarray) -> None:
    sig_dir.mkdir(parents=True, exist_ok=True)
    names = np.array([f"f{i}" for i in range(X.shape[1])])
    for i, row in enumerate(X):
        np.savez(sig_dir / f"gen_{i:03d}.npz",
                 features=row.astype(np.float32), feature_names=names)


def test_lever_readout_band_metric_negative_meanR(tmp_path: Path, monkeypatch) -> None:
    """PM6-a state-lever on a socratic-ward axis where meanR on-axis is NEGATIVE must
    emit V3_outside_R_band (bool) + finite V3_vs_R_sd, and NEVER a ratio-to-meanR key."""
    from anamnesis.scripts import vmb_pm6a_statelever as sl

    rng = np.random.default_rng(0)
    d = 4
    # floor: spread so robust_scale is well-defined and med≈0, scale≈1 (z≈X).
    floor = rng.normal(0, 1, size=(60, d)).astype(np.float32)
    # pole A (linear) at +2 on f0, pole B (socratic) at -2 → dir0 ≈ +e0.
    pole_a = rng.normal(0, 0.2, size=(30, d)) + np.array([2, 0, 0, 0])
    pole_b = rng.normal(0, 0.2, size=(30, d)) + np.array([-2, 0, 0, 0])
    root = tmp_path / "pm6a"
    _write_sigs(root / "baseline" / "signatures_v3_x2", rng.normal(0, 0.2, (20, d)))
    # V3 steered socratic-ward: strongly negative on f0 (well past the R band).
    _write_sigs(root / "V3_L9_m03" / "signatures_v3_x2",
                rng.normal(0, 0.2, (20, d)) + np.array([-3.0, 0, 0, 0]))
    # R controls: mild NEGATIVE on-axis → meanR < 0 (the degeneracy trigger).
    for i, off in enumerate([-0.3, -0.4, -0.2], start=1):
        _write_sigs(root / f"R{i}_L9_m03" / "signatures_v3_x2",
                    rng.normal(0, 0.2, (20, d)) + np.array([off, 0, 0, 0]))

    _write_sigs(tmp_path / "poleA", pole_a)
    _write_sigs(tmp_path / "poleB", pole_b)
    _write_sigs(tmp_path / "floor", floor)
    out = tmp_path / "lever.json"

    monkeypatch.setattr(sys, "argv", [
        "sl", "--pm6a-root", str(root),
        "--pole-a-dir", str(tmp_path / "poleA"), "--pole-b-dir", str(tmp_path / "poleB"),
        "--floor-dir", str(tmp_path / "floor"), "--out", str(out)])
    sl.main()

    res = json.loads(out.read_text())
    cell = res["lever_by_site_dose"]["L9_m03"]
    # band metric present, ratio metric absent
    assert "V3_outside_R_band" in cell and isinstance(cell["V3_outside_R_band"], bool)
    assert np.isfinite(cell["V3_vs_R_sd"])           # finite despite meanR < 0
    assert cell["R_mean"] < 0                          # the degeneracy trigger is live
    # the scored cell carries NO ratio-to-meanR metric (only the band/SD readout)
    assert "lever_ratio" not in cell and "meanR_target" not in cell
    assert set(cell) >= {"V3_onaxis", "R_band", "V3_vs_R_sd", "V3_outside_R_band"}
    assert "NEVER a ratio to meanR" in res["law"]      # the discipline is documented
    # V3 (-3 on-axis) is beyond the most-socratic-ward R (-0.4) → fires socratic-ward
    assert cell["V3_outside_R_band"] is True


# ── Fix 4 (Part D prep): wrapper-aware RoPE gate (Gemma3 nested text_config) ──

class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_inv_freq_unwraps_multimodal_text_config() -> None:
    """Gemma3ForConditionalGeneration's Gemma3Config nests transformer params under
    .text_config; inv_freq_from_config must unwrap it, not raise on missing hidden_size."""
    from anamnesis.extraction.cache_surgery import inv_freq_from_config
    text = _Cfg(num_attention_heads=8, head_dim=16, rope_theta=1_000_000.0,
                rope_local_base_freq=10_000.0)          # Gemma3 dual-RoPE
    wrapper = _Cfg(text_config=text)                     # no hidden_size at top level
    inv = inv_freq_from_config(wrapper)
    assert int(inv.shape[0]) == 8                        # head_dim // 2


def test_inv_freq_flat_config_unaffected() -> None:
    """Flat Llama/Qwen/OLMo configs (no text_config) still resolve via hidden_size."""
    from anamnesis.extraction.cache_surgery import inv_freq_from_config
    flat = _Cfg(hidden_size=128, num_attention_heads=8, rope_theta=500_000.0)
    inv = inv_freq_from_config(flat)
    assert int(inv.shape[0]) == 8                        # (128//8)//2 = 8
