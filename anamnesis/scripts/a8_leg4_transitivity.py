"""A8 Leg-4F / L4-e — TRANSITIVITY: does the algebra's composition axiom hold for g?
(CPU, near-free; frozen bar P8-T .55, A8-add-3.)

Frozen form: fit g_direct(3B -> Qwen) on the SHARED-TEXT subset of the Leg-0 and Leg-1
banks (S2 is byte-identical by stamp; any shared S1/S3 rows verified by text sha256),
then compare against the composed map g_comp = g(8B->Qwen) . g(3B->8B) at MATCHED k:

    bar := mean over the 5-axis panel {V7, Vrep_perp, Vconf, V_temp, dir0}
           of cos(g_direct . v, g_comp . v)   >= .70      (aggregation: MEAN, stated)

Named beside (reported, unscored): a_hat_direct(3B->Qwen) = cos(g_direct.V7_3b, V7_qwen)
vs the multiplicative composition a_hat(3B->8B) x a_hat(8B->Qwen) = .514 x .332 = .171 —
does one hop lose less than two?

RANK GUARD (add-3, binding): k <= n_train/1.2, and k512 is FORBIDDEN at n < 620.  The
shared subset is ~470 texts, so this script fits k in {32, 128} + ridge and builds
g_comp at the SAME k.  The guard's arithmetic is recomputed and filed at run time; if a
requested k violates it the script drops that k and says so.

Fork (add-3): shared subset < 250 texts after sha-verification => a one-model replay
block is pre-authorized before fitting.  Reported, not silently handled.

UNSTAMPED (C§8).  Mechanics only — the desk scores P8-T.

Run (repo root): PYTHONPATH=pipeline python -m anamnesis.scripts.a8_leg4_transitivity
                 [--build-only] [--selftest]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from anamnesis.scripts.a8_fit_g import (StateBank, load_state_bank, load_transport_map,
                                        run_grid, save_state_bank)
from anamnesis.scripts.a8_rosetta import _unit, cos, load_axes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_leg4_transitivity")

ARM = Path("outputs/battery/arms/A8_conjugation")
SUB = ARM / "leg4e_transitivity"          # the subset arm-root this leg builds
OUT = ARM / "leg4" / "readouts_cpu"
PANEL = ("V7", "Vrep_perp", "Vconf", "Vtemp", "dir0")
ANCHOR = {"3b": 14, "8b": 16, "qwen-7b": 21}
K_GUARD_MIN_N_FOR_K512 = 620
A_HAT = {"3b->8b": 0.514, "8b->qwen": 0.332}


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def shared_subset() -> tuple[list[dict], dict]:
    """Texts present in BOTH leg corpora, matched by text sha256 (not by id)."""
    leg0 = json.loads((ARM / "corpus/corpus_manifest.json").read_text())["entries"]
    leg1 = json.loads((ARM / "leg1/corpus/corpus_manifest.json").read_text())["entries"]
    by_sha0 = {_sha(e["text"]): e for e in leg0}
    by_sha1 = {_sha(e["text"]): e for e in leg1}
    shared = sorted(set(by_sha0) & set(by_sha1))
    entries, id_map = [], {}
    for h in shared:
        e0, e1 = by_sha0[h], by_sha1[h]
        entries.append({**e0, "text_sha256": h})
        id_map[h] = {"leg0_text_id": e0["text_id"], "leg1_text_id": e1["text_id"],
                     "same_id": e0["text_id"] == e1["text_id"]}
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["stratum"]] = counts.get(e["stratum"], 0) + 1
    qc = {"n_shared": len(entries), "counts_per_stratum": counts,
          "n_leg0": len(leg0), "n_leg1": len(leg1),
          "id_agreement": sum(v["same_id"] for v in id_map.values()),
          "verified_by": "sha256 of the text field (not the text_id)",
          "fork_replay_block_needed": len(entries) < 250}
    return entries, qc


def build_subset_root() -> dict:
    entries, qc = shared_subset()
    if qc["fork_replay_block_needed"]:
        logger.warning("shared subset %d < 250 — add-3 fork: a one-model replay block is "
                       "pre-authorized. NOT firing it automatically; reported.",
                       qc["n_shared"])
    (SUB / "corpus").mkdir(parents=True, exist_ok=True)
    (SUB / "states").mkdir(parents=True, exist_ok=True)
    man = json.dumps({"entries": entries}, indent=1)
    (SUB / "corpus/corpus_manifest.json").write_text(man)

    # 3B states come from the Leg-0 banks (keyed by leg-0 ids); Qwen from Leg-1.
    leg0_ids = {e["text_sha256"]: e["text_id"] for e in entries}
    leg1_entries = json.loads((ARM / "leg1/corpus/corpus_manifest.json").read_text())["entries"]
    sha_to_leg1 = {_sha(e["text"]): e["text_id"] for e in leg1_entries}
    order = [e["text_sha256"] for e in entries]
    ids_out = [e["text_id"] for e in entries]        # canonical = leg-0 ids
    for model, src_root, key in (("3b", ARM, leg0_ids),
                                 ("qwen-7b", ARM / "leg1", sha_to_leg1)):
        for arm in ("native", "raw"):
            bank = load_state_bank(src_root / "states", model, arm)
            pos = {t: i for i, t in enumerate(bank.text_ids)}
            idx = [pos[key[h]] for h in order]
            sub = StateBank(model=model, arm=arm, text_ids=ids_out,
                            states={s: a[idx] for s, a in bank.states.items()},
                            median_norms=bank.median_norms)
            save_state_bank(SUB / "states", sub)
    stamp = {"STATUS": "UNSTAMPED (C§8)", "leg": "A8 Leg-4F / L4-e (transitivity)",
             "prereg": "A8-add-3 P8-T", "manifest_sha256": _sha(man),
             "construction": "shared-text subset of the Leg-0 and Leg-1 corpora, matched "
                             "by sha256 of the text; 3B states taken from the Leg-0 banks, "
                             "Qwen states from the Leg-1 banks, both re-ordered to one "
                             "canonical row order (leg-0 text_ids kept as row labels). "
                             "median_norms stay the COLLECTION values (never re-derived).",
             "qc": qc}
    (SUB / "corpus/corpus_stamp.json").write_text(json.dumps(stamp, indent=1))
    logger.info("subset root -> %s (%d texts; %s)", SUB, qc["n_shared"],
                qc["counts_per_stratum"])
    return qc


def rank_guard(n_train: int, ks: tuple[int, ...]) -> tuple[tuple[int, ...], dict]:
    allowed, dropped = [], {}
    for k in ks:
        if k > n_train / 1.2:
            dropped[k] = f"k > n_train/1.2 ({n_train}/1.2 = {n_train / 1.2:.0f})"
        elif k >= 512 and n_train < K_GUARD_MIN_N_FOR_K512:
            dropped[k] = f"k512 forbidden at n_train < {K_GUARD_MIN_N_FOR_K512}"
        else:
            allowed.append(k)
    return tuple(allowed), {"n_train": n_train, "allowed": allowed, "dropped": dropped,
                            "rule": "add-3: k <= n_train/1.2; k512 forbidden at n<620"}


class Composed:
    """g_comp = g(8B->Qwen) . g(3B->8B)."""

    def __init__(self, first, second):
        self.first, self.second = first, second

    def transport(self, v: np.ndarray) -> np.ndarray:
        return self.second.transport(self.first.transport(v))


def compare(fam: str) -> dict:
    direct = load_transport_map(
        SUB / f"fits/fit_3bL{ANCHOR['3b']}__qwen-7bL{ANCHOR['qwen-7b']}_native_{fam}.npz")
    g1 = load_transport_map(
        ARM / f"fits/fit_3bL{ANCHOR['3b']}__8bL{ANCHOR['8b']}_native_{fam}.npz")
    g2 = load_transport_map(
        ARM / f"leg1/fits/fit_8bL{ANCHOR['8b']}__qwen-7bL{ANCHOR['qwen-7b']}_native_{fam}.npz")
    comp = Composed(g1, g2)
    src_axes, _, _ = load_axes("3b")
    tgt_axes, _, _ = load_axes("qwen-7b")
    rows = []
    for name in PANEL:
        v = src_axes[name].vec
        d, c = _unit(direct.transport(v)), _unit(comp.transport(v))
        rows.append({"axis": name, "cos_direct_vs_composed": round(cos(d, c), 4),
                     "cos_direct_vs_target_axis": round(cos(d, tgt_axes[name].vec), 4),
                     "cos_composed_vs_target_axis": round(cos(c, tgt_axes[name].vec), 4)})
    mean_agree = float(np.mean([r["cos_direct_vs_composed"] for r in rows]))
    a_direct = next(r["cos_direct_vs_target_axis"] for r in rows if r["axis"] == "V7")
    a_comp_multiplicative = A_HAT["3b->8b"] * A_HAT["8b->qwen"]
    return {
        "family": fam,
        "panel": rows,
        "mean_cos_direct_vs_composed": round(mean_agree, 4),
        "aggregation": "MEAN over the 5-axis panel (frozen in add-3)",
        "beside_a_hat": {
            "a_hat_direct_3b_to_qwen": a_direct,
            "a_hat_composed_observed": next(r["cos_composed_vs_target_axis"]
                                            for r in rows if r["axis"] == "V7"),
            "a_hat_multiplicative_prediction": round(a_comp_multiplicative, 4),
            "one_hop_vs_two_hops": round(a_direct / a_comp_multiplicative, 3)
            if a_comp_multiplicative else None,
            "note": "unscored beside (add-3). >1 means the direct fit keeps more than the "
                    "two-hop composition predicts.",
        },
    }


def selftest() -> int:
    ok = True
    ks, info = rank_guard(400, (32, 128, 512))
    ok &= ks == (32, 128) and 512 in info["dropped"]
    print(f"[{'OK' if ks == (32, 128) else 'BAD'}] rank guard at n_train=400 -> {ks}")
    ks2, _ = rank_guard(700, (32, 128, 512))
    ok &= ks2 == (32, 128, 512)
    print(f"[{'OK' if ks2 == (32, 128, 512) else 'BAD'}] rank guard at n_train=700 -> {ks2}")
    ks3, i3 = rank_guard(100, (32, 128))
    ok &= ks3 == (32,)
    print(f"[{'OK' if ks3 == (32,) else 'BAD'}] rank guard at n_train=100 -> {ks3} "
          f"({i3['dropped']})")

    class FakeMap:
        def __init__(self, m):
            self.m = m

        def transport(self, v):
            return self.m @ v
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 5))
    B = rng.normal(size=(7, 6))
    comp = Composed(FakeMap(A), FakeMap(B))
    v = rng.normal(size=5)
    ok &= np.allclose(comp.transport(v), B @ (A @ v))
    print(f"[{'OK' if np.allclose(comp.transport(v), B @ (A @ v)) else 'BAD'}] "
          "composition order = second(first(v))")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ks", default="32,128")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"STATUS": "UNSTAMPED (C§8) — mechanics only, desk scores P8-T",
                 "leg": "A8 Leg-4F / L4-e", "prereg": "A8-add-3 P8-T"}
    if not args.compare_only:
        res["subset_qc"] = build_subset_root()
        if args.build_only:
            print(json.dumps(res, indent=1))
            return 0
        n = res["subset_qc"]["n_shared"]
        ks, guard = rank_guard(int(n * 0.77), tuple(int(k) for k in args.ks.split(",")))
        res["rank_guard"] = guard
        logger.info("fitting g_direct(3B->Qwen) on %d shared texts, k=%s", n, ks)
        summary = run_grid(SUB, "3b", "qwen-7b", fit_strata=None,
                           fits_dirname="fits", k_grid=ks)
        res["fit_validity"] = {
            "n_fits": summary["n_fits"], "n_valid": summary["n_valid"],
            "split": summary["split"],
            "anchor_records": [{"name": r["name"], "r2": r["r2_test"],
                                "valid": r["valid"],
                                "strata_carried": r.get("strata_carried")}
                               for r in summary["records"]
                               if r["name"].startswith(
                                   f"3bL{ANCHOR['3b']}->qwen-7bL{ANCHOR['qwen-7b']}::native")],
        }
    families = [f"proc_k{k}" for k in (int(k) for k in args.ks.split(","))] + ["ridge"]
    res["comparisons"] = []
    for fam in families:
        try:
            res["comparisons"].append(compare(fam))
        except FileNotFoundError as exc:
            logger.warning("family %s unavailable: %s", fam, exc)
            res["comparisons"].append({"family": fam, "unavailable": str(exc)})
    (OUT / "l4e_transitivity.json").write_text(json.dumps(res, indent=1))
    logger.info("wrote %s", OUT / "l4e_transitivity.json")
    for c in res["comparisons"]:
        if "mean_cos_direct_vs_composed" in c:
            logger.info("%s: mean cos(direct, composed) = %.4f | â_direct %.4f vs "
                        "multiplicative %.4f", c["family"],
                        c["mean_cos_direct_vs_composed"],
                        c["beside_a_hat"]["a_hat_direct_3b_to_qwen"],
                        c["beside_a_hat"]["a_hat_multiplicative_prediction"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
