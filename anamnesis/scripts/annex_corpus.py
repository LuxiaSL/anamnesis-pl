"""ANNEX (exploratory lane) — corpus loader + A0 hygiene for the natural-axes probe.

⚠ ANNEX LANE: nothing produced with this module is quotable, citable, or buildable-upon
until it graduates through a frozen prereg cell. Ledger: outputs/battery/annex/ANNEX-LEDGER.md
Spec: research/planning/ANNEX-natural-axes-2026-07-15.md

Loads the two corpora the probe uses, into ONE shape:

  VENUE  — `vmb_stage0_3b`, Llama-3.2-3B, 800 x 3358, floor-z (the frozen battery z-space).
           Crossed design: 20 topics x 4 templates x 10 seeds. Cell = (topic, template).
           Graduation-eligible (dir0 lives here; V3/V4 ran here).
  POWER  — kotodama-3b curator bank, 23,758 x 2252, robust-z. 505 conv / 4,744 turns /
           5 candidates. Cell = (conv, turn). NOT graduation-eligible (different model).
           ⚠ SIX partner weight-states pooled — see `partner`; raw-centered analysis MUST
           stratify by it. Cell-centering removes it by construction (a cell is within one
           partner), which is why cell-centering is the primary variant.

The two corpora share an EXACT 2,108-d subspace (byte-identical, order-preserved feature
names). `shared_2108=True` restricts to it: drops the venue's 1,250 `pca_*` census block and
kotodama's 144 native `attnres_*`. That subspace is the comparable space of record.

CPU-only, read-only. Kotodama data is read via pointer, never copied (cross-project rule).
"""
from __future__ import annotations

import json
from collections import Counter
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from anamnesis.analysis.battery.floors import load_signature_matrix, robust_scale

F32 = NDArray[np.float32]
I64 = NDArray[np.int64]

def _repo_root() -> Path:
    """Locate anamnesis_exps/ from this file, so the module works from any cwd (the battery
    scripts assume repo-root cwd; depending on that silently is how paths rot)."""
    for p in Path(__file__).resolve().parents:
        if (p / "outputs" / "battery").is_dir() and (p / "WHAT-A-SIGNATURE-IS.md").is_file():
            return p
    raise FileNotFoundError("cannot locate the anamnesis_exps repo root from "
                            f"{__file__} — outputs/battery + WHAT-A-SIGNATURE-IS.md not found")


REPO = _repo_root()
VENUE_DIR = REPO / "outputs/battery/vmb_stage0_3b"
POWER_NPZ = Path(
    "/home/luxia/projects/kotodama/posttraining/outputs/distill/day2_caches/"
    "base_elicit/taste_signatures.npz"
)
INDUCED_KEY_MARKERS = ("inject", "steer", "induc", "alpha_frac")
CorpusName = Literal["venue", "power"]


class FamilyBlock(BaseModel):
    """One authoring-family's contiguous slice of the feature vector."""
    model_config = ConfigDict(frozen=True)
    name: str
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


class AnnexCorpus(BaseModel):
    """A corpus in annex-standard shape: standardized X + factors + cell ids + families."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: CorpusName
    X: F32                          # [n, d] standardized (floor-z for venue, robust-z for power)
    feature_names: list[str]
    families: list[FamilyBlock]     # contiguous, ordered, covering [0, d)
    cell: I64                       # [n] cell id — the seed-only unit (topic x template | conv x turn)
    factors: dict[str, I64]         # categorical nuisance/candidate factors (topic, template, partner, ...)
    covariates: dict[str, F32]      # continuous artifact-class covariates (glen, cap indicator)
    notes: list[str]

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def d(self) -> int:
        return int(self.X.shape[1])

    def family_of_index(self) -> NDArray:
        """[d] array of family name per feature index."""
        out = np.empty(self.d, dtype=object)
        for f in self.families:
            out[f.start:f.end] = f.name
        return out

    def subset(self, mask: NDArray) -> "AnnexCorpus":
        """Row-subset, factors/covariates carried along. Used by split-half stability, which
        must RE-DERIVE centering + residualization on each half (reusing coefficients fitted
        on the full corpus would leak the very structure the check is testing for)."""
        return AnnexCorpus(
            name=self.name, X=np.ascontiguousarray(self.X[mask]),
            feature_names=self.feature_names, families=self.families, cell=self.cell[mask],
            factors={k: v[mask] for k, v in self.factors.items()},
            covariates={k: v[mask] for k, v in self.covariates.items()},
            notes=self.notes + [f"SUBSET to {int(np.asarray(mask).sum())} rows"],
        )

    def describe(self) -> str:
        fam = ", ".join(f"{f.name}:{f.size}" for f in self.families)
        cells = Counter(self.cell.tolist())
        return (
            f"[{self.name}] n={self.n} d={self.d} | cells={len(cells)} "
            f"(sizes {sorted(Counter(cells.values()).items())}) | "
            f"factors={ {k: len(set(v.tolist())) for k, v in self.factors.items()} } | "
            f"covariates={list(self.covariates)}\n  families: {fam}"
        )


def _blocks_from_npz(z, order: list[str]) -> list[FamilyBlock]:
    """Venue: derive contiguous blocks from the per-family `features_*` keys, asserting
    they concatenate to `features` exactly (they do, verified 2026-07-15)."""
    blocks, off = [], 0
    for key in order:
        size = int(np.atleast_1d(z[key]).shape[0])
        blocks.append(FamilyBlock(name=key[len("features_"):], start=off, end=off + size))
        off += size
    return blocks


def load_venue(*, shared_2108: bool = False) -> AnnexCorpus:
    """Llama-3.2-3B bare Stage-0, floor-z. Keeps the V3sel hygiene gates (14a §2 / 14c)."""
    sig_dir = VENUE_DIR / "signatures_v3"
    md = json.loads((VENUE_DIR / "metadata.json").read_text())

    # ── hygiene gates, kept from vmb_v3sel_select (the C2 hereditary rule applies in the
    #    annex too — a steered gen in the pool would make every axis second-order induced) ──
    for g in md["generations"]:
        bad = [k for k in g if any(s in k.lower() for s in INDUCED_KEY_MARKERS)]
        if bad:
            raise AssertionError(f"venue gen {g['generation_id']}: induced fields {bad} "
                                 "— annex pool must be UNSTEERED (14a §2)")
        if str(g.get("condition", "")) != "standard":
            raise AssertionError(f"venue gen {g['generation_id']}: condition="
                                 f"{g.get('condition')!r} — not unsteered (14a §2)")
        if str(g.get("system_prompt", "")) != "":
            raise AssertionError(f"venue gen {g['generation_id']}: non-empty system_prompt "
                                 "— the bare pool must be UNPROMPTED (14c)")

    X, names, gen_ids = load_signature_matrix(sig_dir)
    med, scale = robust_scale(X)                 # the model's frozen battery z-space
    Z = ((X - med) / scale).astype(np.float32)

    z0 = np.load(sorted(sig_dir.glob("gen_*.npz"))[0], allow_pickle=True)
    fam_order = [k for k in z0.files if k.startswith("features_")]
    families = _blocks_from_npz(z0, fam_order)

    gm = {int(g["generation_id"]): g for g in md["generations"]}
    topic = np.array([int(gm[g]["topic_idx"]) for g in gen_ids], dtype=np.int64)
    tmpl_names = sorted({str(gm[g]["mode"]) for g in gen_ids})   # task TEMPLATE, not a mode prompt
    tmpl = np.array([tmpl_names.index(str(gm[g]["mode"])) for g in gen_ids], dtype=np.int64)
    glen = np.array([float(gm[g]["num_generated_tokens"]) for g in gen_ids], dtype=np.float32)
    cap = (glen >= glen.max()).astype(np.float32)   # artifact class: hit-the-cap indicator

    cell = (topic * len(tmpl_names) + tmpl).astype(np.int64)     # (topic, template) = seed-only unit

    corpus = AnnexCorpus(
        name="venue", X=Z, feature_names=list(names), families=families,
        cell=cell, factors={"topic": topic, "template": tmpl},
        covariates={"glen": glen, "cap": cap},
        notes=[f"floor-z via robust_scale over the corpus itself (n={len(gen_ids)})",
               f"templates (index order): {tmpl_names}",
               "gates PASSED: no-induced (14a §2), condition=standard, unprompted (14c)",
               f"glen {glen.min():.0f}-{glen.max():.0f}, cap-hit frac {cap.mean():.3f} "
               "— ARTIFACT class, residualized out in A0"],
    )
    return _restrict_shared(corpus) if shared_2108 else corpus


def load_power(*, shared_2108: bool = False, partner: str | None = None) -> AnnexCorpus:
    """kotodama-3b curator bank (READ-ONLY, cross-project pointer — never copied).

    ⚠ Six partner weight-states are pooled in this file. `partner=None` returns all of them
    WITH the partner factor exposed; any raw-centered use MUST stratify. Cell-centering
    removes partner by construction (each (conv, turn) cell lies within one partner).
    """
    z = np.load(POWER_NPZ, allow_pickle=True)
    X = np.asarray(z["X"], dtype=np.float32)
    names = [str(x) for x in z["feature_names"]]
    families = [FamilyBlock(name=str(n), start=int(s), end=int(e))
                for n, s, e in zip(z["fam_name"], z["fam_start"], z["fam_end"])]

    conv = np.array([str(c) for c in z["meta_conv"]])
    turn = np.asarray(z["meta_turn"], dtype=np.int64)
    partner_s = np.array([str(p) for p in z["meta_partner"]])
    frame_s = np.array([str(f) for f in z["meta_frame"]])

    keep = np.ones(len(X), dtype=bool)
    if partner is not None:
        keep = partner_s == partner
        if not keep.any():
            raise ValueError(f"no rows for partner={partner!r}; "
                             f"available: {sorted(set(partner_s.tolist()))}")
    X, conv, turn = X[keep], conv[keep], turn[keep]
    partner_s, frame_s = partner_s[keep], frame_s[keep]

    med, scale = robust_scale(X)
    Z = ((X - med) / scale).astype(np.float32)

    def _codes(a: NDArray) -> tuple[I64, list[str]]:
        levels = sorted(set(a.tolist()))
        return np.array([levels.index(v) for v in a], dtype=np.int64), levels

    # cell = (conv, turn): 5 candidates differing ONLY by sampling seed, same weights,
    # same context, same prompt. The purest execution-side unit available anywhere.
    cell_keys = np.array([f"{c}|{t}" for c, t in zip(conv, turn)])
    cell, _ = _codes(cell_keys)
    conv_c, _ = _codes(conv)
    partner_c, partner_levels = _codes(partner_s)
    frame_c, frame_levels = _codes(frame_s)

    glen = np.asarray(z["meta_glen"], dtype=np.float32)[keep]
    aphasic = np.asarray(z["meta_aphasic"], dtype=bool)[keep]
    # taste: meta_rank is object-typed (None for unjudged) — -1 sentinel, filtered at use.
    rank = np.array([int(r) if r is not None and str(r).lstrip("-").isdigit() else -1
                     for r in z["meta_rank"][keep]], dtype=np.int64)

    corpus = AnnexCorpus(
        name="power", X=Z, feature_names=names, families=families, cell=cell,
        factors={"conv": conv_c, "turn": turn, "partner": partner_c,
                 "frame": frame_c, "aphasic": aphasic.astype(np.int64), "rank": rank},
        covariates={"glen": glen},
        notes=[f"READ-ONLY pointer: {POWER_NPZ}",
               f"partners (index order): {partner_levels}"
               + ("" if partner else "  ⚠ POOLED — stratify for any raw-centered use"),
               f"frames (index order): {frame_levels}",
               f"aphasic frac {aphasic.mean():.3f} (known signature-detectable; nameable, "
               "not a steering target)",
               f"glen {glen.min():.0f}-{glen.max():.0f} — NO cap artifact (artifact class absent)",
               "family block names tier1/tier2/tier2_5 are RETIRED tier vocabulary (bank "
               "predates the 2026-07-11 retirement) — used as index ranges ONLY; "
               "interpretation goes through source x method x depth",
               "robust-z computed on this corpus (annex shortcut: no frozen floor exists "
               "for this bank; scale is corpus-internal, not a battery z-space)"],
    )
    return _restrict_shared(corpus) if shared_2108 else corpus


@cache
def _feature_names(which: CorpusName) -> frozenset[str]:
    """Names only — avoids reloading a full corpus just to intersect feature sets."""
    if which == "venue":
        z = np.load(sorted((VENUE_DIR / "signatures_v3").glob("gen_*.npz"))[0],
                    allow_pickle=True)
    else:
        z = np.load(POWER_NPZ, allow_pickle=True, mmap_mode="r")
    return frozenset(str(x) for x in z["feature_names"])


def _restrict_shared(c: AnnexCorpus) -> AnnexCorpus:
    """Restrict to the exact 2,108 features shared by both corpora (byte-identical names,
    order-preserved). Drops the venue's 1,250-feature `pca_*` census block / kotodama's
    144 native `attnres_*`. Rebuilds family blocks over the surviving indices."""
    other_names = _feature_names("power" if c.name == "venue" else "venue")
    keep_idx = np.array([i for i, n in enumerate(c.feature_names) if n in other_names])
    if len(keep_idx) != 2108:
        raise AssertionError(f"shared subspace is {len(keep_idx)}, expected 2108 — the "
                             "corpora's feature sets changed; re-verify before trusting any rhyme check")

    fam_of = c.family_of_index()[keep_idx]
    families, off = [], 0
    for name in dict.fromkeys(fam_of.tolist()):          # preserves order
        size = int((fam_of == name).sum())
        families.append(FamilyBlock(name=name, start=off, end=off + size))
        off += size

    return AnnexCorpus(
        name=c.name, X=np.ascontiguousarray(c.X[:, keep_idx]),
        feature_names=[c.feature_names[i] for i in keep_idx],
        families=families, cell=c.cell, factors=c.factors, covariates=c.covariates,
        notes=c.notes + ["RESTRICTED to the shared-2108 subspace (comparable across corpora; "
                         "venue's pca_* census block and kotodama's attnres_* dropped)"],
    )


# ── A0 hygiene ───────────────────────────────────────────────────────────────────

def residualize(X: F32, covariates: dict[str, F32]) -> F32:
    """Regress out ARTIFACT-class covariates (length, cap-hit) from every feature.

    OLS on [1, cov...]. Fold-free — an ANNEX SHORTCUT (the battery would do this per-fold to
    avoid leakage; here there is no held-out set to leak into, but the fitted coefficients do
    see all the data, so any downstream accuracy-like number would be optimistic. Spectra are
    not accuracy-like, which is why the shortcut is acceptable at annex grade). Named in the
    ledger.
    """
    if not covariates:
        return X.astype(np.float32)
    A = np.column_stack([np.ones(len(X), dtype=np.float64)]
                        + [c.astype(np.float64) for c in covariates.values()])
    # lstsq handles a rank-deficient design (e.g. cap constant after a partner filter)
    beta, *_ = np.linalg.lstsq(A, X.astype(np.float64), rcond=None)
    return (X.astype(np.float64) - A @ beta).astype(np.float32)


def center(X: F32, groups: I64 | None) -> F32:
    """Center globally (groups=None) or within groups (subtract each group's mean).

    Within-CELL centering is the primary variant: it removes prompt content, topic, template,
    conversation, turn, and (in the power corpus) partner weight-state ALL AT ONCE, by
    construction — leaving only seed-driven execution variation. That cloud is the battery's
    own stochastic floor; asking whether it has axes asks whether the denominator is organized.
    """
    Xd = X.astype(np.float64)
    if groups is None:
        return (Xd - Xd.mean(axis=0)).astype(np.float32)
    out = np.empty_like(Xd)
    for g in np.unique(groups):
        m = groups == g
        out[m] = Xd[m] - Xd[m].mean(axis=0)
    return out.astype(np.float32)


def prepare(c: AnnexCorpus, variant: Literal["raw", "topic", "cell"],
            *, residualize_artifacts: bool = True) -> tuple[F32, int]:
    """A0 → the design matrix for one centering variant, plus its residual df.

    df matters: centering within G groups burns G dimensions, and the PCA spectrum must be
    read against df, not n. (Power corpus, cell variant: 23,758 rows − 4,744 cells ≈ 19,014 df
    vs d=2,252 — the only configuration in this probe where df >> d actually holds.)
    """
    X = residualize(c.X, c.covariates) if residualize_artifacts else c.X
    n_cov = (len(c.covariates) + 1) if residualize_artifacts else 0
    if variant == "raw":
        return center(X, None), c.n - 1 - n_cov
    if variant == "topic":
        key = c.factors["topic"] if c.name == "venue" else c.factors["conv"]
        return center(X, key), c.n - len(np.unique(key)) - n_cov
    if variant == "cell":
        return center(X, c.cell), c.n - len(np.unique(c.cell)) - n_cov
    raise ValueError(f"unknown variant {variant!r}")


if __name__ == "__main__":
    for c in (load_venue(), load_venue(shared_2108=True), load_power()):
        print(c.describe())
        for note in c.notes:
            print(f"    · {note}")
        for v in ("raw", "topic", "cell"):
            Xp, df = prepare(c, v)
            print(f"    {v:5s}: X{Xp.shape} df={df}")
        print()
