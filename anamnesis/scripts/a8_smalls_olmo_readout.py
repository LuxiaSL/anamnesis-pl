"""A8 smalls close-out — the OLMo V7-read gate (P8-XO2) + all-family â(8B→OLMo).

Separate from predict/verify because P8-XO2 is a GATE on the built V7 (does it transport
above its own null envelope, top-PC below?), while predict/verify does the star math. The
gate answers: is the OLMo V7 a real shared direction, or a coincidence of construction?
If it fails, A8-add-8 says the sixth node scores ⚫ (a c_OLMo on a V7 that does not itself
transport is not worth testing).

RAW ARM ONLY (OLMo is a base model, A8-add-7.1). The read transports V7_8B through the
banked 8b↔olmo raw map and compares cos(g·V7_8B, V7_OLMo) against:
  * the transported-null envelope (100 seeded randoms + 8B's banked R-band/iso members
    through the SAME g), q95 of |cos|;
  * the top-PC control (max over the 8B state bank's top-5 train PCs, transported).

OLMo has no banked axis registry (load_axes raises by design — rake 43), so V7_8B and
V7_OLMo are loaded directly from their b7 banks, and the null pool comes from 8B's
registry (the source side, which does exist).

UNSTAMPED (C section 8). No P self-scored — the desk scores P8-XO2.

Run (repo root):
  PYTHONPATH=pipeline python -m anamnesis.scripts.a8_smalls_olmo_readout
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from anamnesis.scripts.a8_fit_g import (
    PCABank, load_labels, load_state_bank, load_transport_map, make_split)
from anamnesis.scripts.a8_rosetta import TOP_PC_J, _unit, cos, load_axes
from anamnesis.scripts.a8_smalls_star_systems import RANK_GUARD_DIVISOR, V7, _load_v7, _n_train

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_smalls_olmo_readout")

SMALLS = Path("outputs/battery/arms/A8_conjugation/smalls")
OUT = SMALLS / "readouts_cpu"
FITS_HUB = SMALLS / "fits_olmo"
HUB, TGT = "8b", "olmo2-7b"
FAMILIES = ("proc_k32", "proc_k128", "proc_k512", "ridge")
PRIMARY = "proc_k128"
SEED = 80
N_RANDOM = 100


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        v7_hub, v7_tgt = _load_v7(HUB), _load_v7(TGT)
    except (FileNotFoundError, KeyError) as exc:
        raise SystemExit(f"OLMo V7 bank not present yet — build it first: {exc}")

    s_site, t_site = V7[HUB][2], V7[TGT][2]
    pair = f"{HUB}L{s_site}->{TGT}L{t_site}"
    _, _, hub_pool = load_axes(HUB)      # 8B null pool (source side exists)
    n_train = _n_train(FITS_HUB)
    max_k = (n_train / RANK_GUARD_DIVISOR) if n_train else None

    # top-5 source PCs on the fit's own split/code path (raw arm)
    states = SMALLS / "states"
    sb = load_state_bank(states, HUB, "raw")
    labels = load_labels(SMALLS / "corpus" / "corpus_manifest.json", sb.text_ids)
    train, _, _ = make_split(labels)
    pcs_s = PCABank.fit(sb.matrix(s_site)[train], TOP_PC_J).components * sb.median_norms[s_site]

    rng = np.random.default_rng(SEED)
    randoms = rng.standard_normal((N_RANDOM, v7_hub.shape[0]))
    randoms /= np.linalg.norm(randoms, axis=1, keepdims=True)

    reads: dict = {}
    for family in FAMILIES:
        k = int(family.split("_k")[1]) if family.startswith("proc_k") else None
        p = FITS_HUB / f"fit_{HUB}L{s_site}__{TGT}L{t_site}_raw_{family}.npz"
        if not p.exists():
            continue
        tm = load_transport_map(p)
        c = cos(_unit(tm.transport(v7_hub)), v7_tgt)
        nulls = ([_unit(tm.transport(r)) for r in randoms]
                 + [_unit(tm.transport(a.vec)) for a in hub_pool])
        q95 = float(np.quantile([abs(cos(n, v7_tgt)) for n in nulls], 0.95))
        toppc = max(abs(cos(tm.transport(pcs_s[j]), v7_tgt)) for j in range(pcs_s.shape[0]))
        reads[family] = {
            "a_hat": round(c, 4), "null_q95_abs": round(q95, 4),
            "top_pc_max": round(toppc, 4),
            "exceeds_envelope": bool(abs(c) > q95),
            "axis_specific": bool(c > toppc),
            "PASSES_XO2_GATE": bool(abs(c) > q95 and c > toppc),
            "rank_forbidden": bool(max_k is not None and k is not None and k > max_k),
        }

    prim = reads.get(PRIMARY, {})
    res = {
        "STATUS": "UNSTAMPED (C section 8) — the desk scores P8-XO2 (.50)",
        "leg": "A8 smalls close-out — OLMo V7-read gate + all-family â(8B→OLMo)",
        "prereg": "A8-add-8 (P8-XO2 .50)",
        "arm": "raw ONLY (A8-add-7.1 — OLMo is a base model, no native arm)",
        "site_pair": pair,
        "n_train": n_train,
        "rank_guard": f"k <= n_train/{RANK_GUARD_DIVISOR}; primary {PRIMARY}",
        "XO2_GATE_primary": {
            "family": PRIMARY,
            "a_hat": prim.get("a_hat"),
            "null_q95_abs": prim.get("null_q95_abs"),
            "top_pc_max": prim.get("top_pc_max"),
            "PASSES": prim.get("PASSES_XO2_GATE"),
        },
        "all_families": reads,
        "note_no_axis_registry": "OLMo has no banked field roster/dir0/Vtemp; only the "
                                 "built V7. load_axes('olmo2-7b') raises by design "
                                 "(rake 43); V7 is read directly from the b7 bank.",
        "SCORING": "reported, NOT self-scored",
    }
    (OUT / "olmo_v7_read_gate.json").write_text(json.dumps(res, indent=1))
    for f, r in reads.items():
        logger.info("%-10s â %+.4f  q95 %.4f  topPC %.4f  env=%s axspec=%s gate=%s%s",
                    f, r["a_hat"], r["null_q95_abs"], r["top_pc_max"],
                    r["exceeds_envelope"], r["axis_specific"], r["PASSES_XO2_GATE"],
                    "  [rank-forbidden]" if r["rank_forbidden"] else "")
    logger.info("wrote %s", OUT / "olmo_v7_read_gate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
