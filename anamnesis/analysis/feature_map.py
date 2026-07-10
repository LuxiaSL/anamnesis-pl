"""Feature map — the (SOURCE x METHOD x DEPTH) taxonomy that replaces the T1/T2/T2.5/T3 tiers.

The tiers were "diagonal smears" across three orthogonal axes (research/notes/tiers-vs-sources-reframe.md),
so a tier's accuracy couldn't be read. This module tags every signature feature by the axes that ARE
interpretable and were empirically validated on the merged v3 corpus (2026-06-14):

  SOURCE   = which substrate is read.   Ranked (LDA, model-stable): attention >> residual > gate > keys > output.
  METHOD   = the base operator (magnitude / distributional / geometry / spectral / learned).
  DYNAMIC  = the temporal wrapper: static (a *_mean / snapshot) vs dynamic (*_std / slope / trajectory /
             window / drift / novelty). [modes ≈ average level (static) ≥ dynamics, at n≈900.]
  DEPTH    = layer + band (early/mid/late). [mode signal concentrates at MID layers.]

This is the reusable substrate for the discover→decompose→distill workflow:
  raw -> encoder (per-model discovery, all sources) -> decompose by CELL (this map) -> which (source,method,
  depth) cells carry THIS task -> instantiate theory-motivated features for those cells -> portable,
  lightweight signature. Redundancy is task-specific, so you re-decompose per task; the cells are the unit.

Design: pure-numpy/pydantic (no torch/sklearn — importable anywhere, like state_extractor). Classification
is name-based and TRANSPARENT — `FeatureMap.unclassified()` and `.summary()` expose every call so it can be
audited/overridden. (Long-term ideal per the reframe note: tag at generation time; this is the pragmatic
post-hoc parser over the frozen v3 names.)

  >>> fm = FeatureMap(feature_names, n_layers=32)
  >>> X_attn_mid = X[:, fm.mask(source=Source.attention, band=Band.mid)]   # slice a cell
  >>> for cell, idx in fm.cells("source", "band").items(): acc[cell] = lda(X[:, idx], y)   # decompose
  >>> lean = fm.select(source=Source.attention)            # the ~600-780-feat lean mode/taste signature

Run as a script to validate the taxonomy + coverage on a real run:
    ANAMNESIS_RUNS=/models/anamnesis-extract/runs python -m anamnesis.analysis.feature_map 8b_fat_01
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict

# Model layer counts (for depth bands). Extend per model/architecture.
MODEL_LAYERS = {"3b": 28, "8b": 32, "kotodama_3b": 28}


class Source(str, Enum):
    output = "output"          # logit / token-distribution statistics
    residual = "residual"      # the residual stream itself (norms, trajectory, x-layer delta, PCA)
    attention = "attention"    # attention-weight reads: entropy, head-agreement, flow/region/recency, cache mass
    keys = "keys"              # pre-RoPE key-vector geometry (spread/drift/novelty/eff_dim) + epoch detection
    gate = "gate"              # SwiGLU gate activations
    values = "values"          # v_proj (future — banked, not yet featurized)
    qk = "qk"                  # post-RoPE QK geometry (future)
    routing = "routing"        # AttnRes block-routing weights (kotodama-native; future)
    unknown = "unknown"        # flagged: classifier did not match (audit these)


class Method(str, Enum):
    magnitude = "magnitude"            # norms / means of magnitudes
    distributional = "distributional"  # entropy, JSD/agreement, top-k mass, coverage, mass fractions, sparsity
    geometry = "geometry"              # cosine / spread / drift / novelty / participation-ratio / PCA projection
    spectral = "spectral"              # graph-spectral (Fiedler, HFER, spectral entropy, smoothness)
    learned = "learned"                # contrastive projection / encoder (future in-signature)
    unknown = "unknown"


class Band(str, Enum):
    early = "early"
    mid = "mid"
    late = "late"


class FeatureTag(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    source: Source
    method: Method
    dynamic: Optional[bool]      # True=dynamic, False=static, None=ambiguous
    layer: Optional[int]
    band: Optional[Band]
    family: str                  # legacy tier/family label (back-compat with gate_a_v3_battery.fam_of)


# ---------------------------------------------------------------------------- classification rules

def legacy_family(n: str) -> str:
    """Verbatim from gate_a_v3_battery.fam_of (+ the v3 hand-suite families) — old analyses still line up."""
    if n.startswith("value_"): return "value_geometry"
    if n.startswith(("qk_", "q_")): return "qk_geometry"
    if "cka" in n: return "kv_cka"
    if n.startswith("ph_"): return "per_head"
    if n.startswith("attn_flow_"): return "attention_flow"
    if n.startswith("gate_"): return "gate"
    if n.startswith("res_traj"): return "residual_traj"
    if n.startswith(("cache_", "kv_", "epoch_")): return "T2.5"
    if n.startswith("spectral_"): return "T2_spectral"
    if n.startswith(("attn_entropy_", "head_agreement_", "delta_")): return "T2_other"
    if n.startswith("attnres_"): return "attn_res"
    if n.startswith("pca_"): return "T3"
    return "T1"


def _source(n: str) -> Source:
    # AttnRes (kotodama-native) FIRST — its routing names contain "entropy"/"top" which would else mis-hit output.
    if n.startswith("attnres_committed"): return Source.residual   # committed residual-block snapshots (geometry)
    if n.startswith("attnres_"): return Source.routing             # AttnRes block-routing softmax = cross-block allocation
    # v3 hand-suite (checked first; these prefixes are specific). value_* incl value↔key corr = a value prop.
    if n.startswith("value_"): return Source.values            # v_proj value-vector geometry
    if n.startswith(("qk_", "q_")): return Source.qk           # query / q·k content geometry
    if n.startswith("kv_value"): return Source.values          # cross-layer value CKA (before the kv_→keys rule)
    if n.startswith("gate_"): return Source.gate
    if n.startswith("kv_") or n.startswith("epoch_"): return Source.keys      # key-vector geometry / key-centroid epochs
    if n.startswith(("cache_", "attn_flow_", "attn_entropy_", "head_agreement_", "ph_")):
        return Source.attention                                               # attention-weight reads
    if n.startswith(("activation_norm", "res_traj", "delta_", "pca_", "spectral_")):
        return Source.residual                                               # residual stream (spectral = graph on hidden-state)
    # output / token-distribution stats
    if any(k in n for k in ("logit", "surpris", "entropy", "token", "chosen", "top",
                            "perplex", "ppl", "prob", "rank")):
        return Source.output
    return Source.unknown


_GEOM = ("key_spread", "key_drift", "key_novelty", "eff_dim", "spread", "drift", "novelty",
         "cosine", "committed_cos", "res_traj", "participation", "curvature", "velocity", "align", "cka")
_DIST = ("entropy", "agreement", "coverage", "sink", "recency", "prompt_mass", "region",
         "diversity", "lookback", "sparsity", "top", "surpris", "mass", "decay", "anchor",
         "transition", "regularity", "stability", "role")


def _method(n: str, source: Source) -> Method:
    if source == Source.routing: return Method.distributional      # AttnRes routing summaries (entropy/top1/anchor/recency/eff_src)
    if n.startswith("pca_"): return Method.geometry
    if "spectral" in n or "fiedler" in n or "hfer" in n: return Method.spectral
    if "_norm" in n: return Method.magnitude                       # activation_norm / delta_norm — before distributional
    if any(k in n for k in _GEOM): return Method.geometry
    if any(k in n for k in _DIST): return Method.distributional
    if source == Source.output: return Method.distributional       # logit/token-distribution stats
    return Method.unknown


# Static (a level / snapshot) vs Dynamic (a change/dispersion over generation time). Token-matched so it
# is robust to the two naming patterns ({stat}_L{n} and L{n}_..._{stat}); bare measures = static levels.
_STATIC_TOK = {"mean", "traj0"}
_DYNAMIC_TOK = {"std", "slope", "traj1", "traj2", "traj3", "traj4", "drift"}


def _dynamic(n: str) -> Optional[bool]:
    if n.startswith("pca_"): return False                          # per-feature PCA projection = snapshot
    toks = set(n.split("_"))
    if toks & _DYNAMIC_TOK: return True
    if toks & _STATIC_TOK: return False
    return False                                                   # bare level (recency_bias, sink_mass, key_spread, ...)


def _layer(n: str) -> Optional[int]:
    m = re.search(r"_L(\d+)", n)
    return int(m.group(1)) if m else None


def _band(layer: Optional[int], n_layers: int) -> Optional[Band]:
    if layer is None: return None
    if layer < n_layers / 3: return Band.early
    if layer < 2 * n_layers / 3: return Band.mid
    return Band.late


def classify(name: str, n_layers: int) -> FeatureTag:
    L = _layer(name)
    src = _source(name)
    return FeatureTag(name=name, source=src, method=_method(name, src), dynamic=_dynamic(name),
                      layer=L, band=_band(L, n_layers), family=legacy_family(name))


# ---------------------------------------------------------------------------- the map

class FeatureMap:
    """Tags a fixed feature-name list and slices it by any axis or (source,method,depth) cell."""

    def __init__(self, names: list[str], n_layers: int):
        self.names = list(names)
        self.n_layers = n_layers
        self.tags = [classify(n, n_layers) for n in self.names]

    def __len__(self) -> int:
        return len(self.names)

    def _match(self, t: FeatureTag, source=None, method=None, dynamic=None, band=None, layer=None) -> bool:
        return ((source is None or t.source == source)
                and (method is None or t.method == method)
                and (dynamic is None or t.dynamic == dynamic)
                and (band is None or t.band == band)
                and (layer is None or t.layer == layer))

    def mask(self, **criteria) -> np.ndarray:
        """Boolean mask over features matching ALL given criteria (source/method/dynamic/band/layer)."""
        return np.array([self._match(t, **criteria) for t in self.tags], dtype=bool)

    def select(self, **criteria) -> list[str]:
        return [t.name for t in self.tags if self._match(t, **criteria)]

    def cells(self, *axes: str) -> dict[tuple, list[int]]:
        """Group feature INDICES by the given axes, e.g. cells('source','band') -> {(Source, Band): [idx...]}.
        The unit of the decompose-by-cell workflow: iterate, classify each cell's accuracy for a task."""
        out: dict[tuple, list[int]] = {}
        for i, t in enumerate(self.tags):
            key = tuple(getattr(t, a) for a in axes)
            out.setdefault(key, []).append(i)
        return dict(sorted(out.items(), key=lambda kv: (-len(kv[1]),)))

    def summary(self) -> dict:
        def counts(attr):
            c: dict = {}
            for t in self.tags:
                c[getattr(t, attr)] = c.get(getattr(t, attr), 0) + 1
            return dict(sorted(c.items(), key=lambda kv: -kv[1]))
        return {"n": len(self), "source": counts("source"), "method": counts("method"),
                "dynamic": counts("dynamic"), "band": counts("band")}

    def unclassified(self) -> list[str]:
        """Coverage check — features the classifier could not place (source/method unknown or dynamic None)."""
        return [t.name for t in self.tags
                if t.source == Source.unknown or t.method == Method.unknown or t.dynamic is None]


# ---------------------------------------------------------------------------- validation CLI

def _load_names(run: str) -> tuple[list[str], int]:
    import json
    import os
    from pathlib import Path
    runs = Path(os.environ.get("ANAMNESIS_RUNS", "outputs/runs"))
    sd = runs / run / "signatures_v3"
    p = sorted(sd.glob("gen_*.npz"))[0]
    z = np.load(p, allow_pickle=True)
    names = [str(x) for x in z["feature_names"]]
    nl = next((v for k, v in MODEL_LAYERS.items() if run.startswith(k)), 32)
    return names, nl


def main():
    import sys
    run = sys.argv[1] if len(sys.argv) > 1 else "8b_fat_01"
    names, nl = _load_names(run)
    fm = FeatureMap(names, nl)
    s = fm.summary()
    print(f"=== {run}  n={s['n']}  n_layers={nl} ===")
    for axis in ("source", "method", "dynamic", "band"):
        print(f"  {axis:8s}: " + "  ".join(f"{getattr(k,'value',k)}={v}" for k, v in s[axis].items()))
    unc = fm.unclassified()
    print(f"  coverage: {len(names) - len(unc)}/{len(names)} classified; {len(unc)} flagged")
    if unc:
        print("  flagged (sample): " + ", ".join(unc[:15]))
    print("  source x band cells (count):")
    for (src, band), idx in fm.cells("source", "band").items():
        print(f"    {getattr(src,'value',src):10s} x {getattr(band,'value',band) if band else 'none':5s}: {len(idx)}")


if __name__ == "__main__":
    main()
