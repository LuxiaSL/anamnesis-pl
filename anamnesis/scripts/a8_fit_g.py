"""A8 Leg-0 — T2: Phase B fit grid (CPU, local). Charter §2/§4, session spec Phase B.

Fits the structure-preserving map g between two models' residual state spaces from
paired forced-replay means, per (site-pair x template-arm x family):
  - semi-orthogonal Procrustes on rank-k PC subspaces, k in {32,128,512}
    (per-side PCA fit on TRAIN rows only; W^T W = I; optional isotropic scale)
  - ridge affine (liberal variant; alpha by closed-form LOO PRESS on train)

Gates per fit (frozen rule, prereg §1 FIT VALIDITY):
  held-out state-prediction R^2 separates from BOTH the shuffled-pair null AND the
  stratum-preserving shuffle null (re-pair with a different text from the same
  (stratum, voice, mode) group — the sharp null); per-stratum R^2 carries in >=2 of
  {S1,S2,S3}. Plus: CKA before/after, and the two-arm g-agreement robustness read.

STATE-BANK CONTRACT (T4 `a8_collect_states.py` imports save_state_bank from here —
single source of truth):
  states/states_{model}_{arm}.npz
      text_ids : unicode array [n]   (must cover the corpus manifest exactly)
      L{site}  : float32 [n, hidden] (per-text MEAN over completion-token states, raw
                 unnormalized fp32 — normalization happens HERE at fit time)
  states/norms_{model}_{arm}.json
      {"L{site}": median over texts of ||per-text mean state||_2, ...}

Artifacts (under --arm-root/fits/):
  pca_{model}_L{site}_{arm}.npz            per-side PCA bank (kmax components)
  fit_{src}L{s}__{tgt}L{t}_{arm}_proc_k{k}.npz / _ridge.npz + .json sidecars
  cp2_summary.json                         the CP-2 table, one record per fit

Run (from pipeline/):
  python -m anamnesis.scripts.a8_fit_g --selftest          # synthetic validation
  python -m anamnesis.scripts.a8_fit_g                      # real grid (post CP-1)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a8_fit_g")

# ---------------------------------------------------------------- constants (frozen)
A8_SEED = 80
K_GRID = (32, 128, 512)
N_NULL_REPS = 20
HELD_OUT_TOPICS = 5           # of 20 (topic-grouped split: a topic is wholly one side)
HELD_OUT_S2 = 40              # of 160 S2 shards
N_PROBES = 50                 # random unit probes for the two-arm g-agreement read
SITES = {"3b": (13, 14, 18), "8b": (14, 16, 18), "qwen-7b": (19, 21, 23),
         "dsv2-lite": (18, 22),
         # --- extension pairs (desk-smalls, authorized DESK-RULINGS-LEG6 §4) ---
         # gemma3-27b: the WH6-stamped peak region {34,36,38}. L36 is load-bearing —
         # the ENTIRE banked field roster (V7, Vrep⊥, Veos⊥, Vconf, Vtemp, V3) lives
         # at L36, so L36 must be in the grid (rake 26).
         "gemma3-27b": (34, 36, 38),
         # olmo2-7b: NO banked site curve exists (A3/A5 were cut for this model), so
         # the sites are a mid-depth BAND and the site of record is picked from the
         # fit's own alignment curve afterwards — never by fiat (baton item 1).
         "olmo2-7b": (12, 16, 20, 24)}
ARMS = ("native", "raw")
STRATA = ("S1", "S2", "S3")
DEFAULT_ARM_ROOT = Path("outputs/battery/arms/A8_conjugation")


# ---------------------------------------------------------------- data model
@dataclass
class StateBank:
    """Per-text mean states for one (model, arm), all sites, aligned to text_ids."""
    model: str
    arm: str
    text_ids: list[str]
    states: dict[int, np.ndarray]        # site -> [n, hidden] float32 (raw)
    median_norms: dict[int, float]       # site -> median ||state||

    def matrix(self, site: int) -> np.ndarray:
        """Normalized (÷ site median norm) float64 states."""
        return self.states[site].astype(np.float64) / self.median_norms[site]


def save_state_bank(states_dir: Path, bank: StateBank) -> tuple[Path, Path]:
    """The one writer of the state-bank contract (T4 calls this)."""
    states_dir.mkdir(parents=True, exist_ok=True)
    npz_path = states_dir / f"states_{bank.model}_{bank.arm}.npz"
    payload: dict[str, np.ndarray] = {
        "text_ids": np.array(bank.text_ids, dtype=np.str_)}
    for site, arr in bank.states.items():
        if arr.shape[0] != len(bank.text_ids):
            raise ValueError(f"L{site}: {arr.shape[0]} rows != {len(bank.text_ids)} ids")
        payload[f"L{site}"] = arr.astype(np.float32)
    np.savez_compressed(npz_path, **payload)
    norms_path = states_dir / f"norms_{bank.model}_{bank.arm}.json"
    with open(norms_path, "w") as f:
        json.dump({f"L{s}": float(v) for s, v in bank.median_norms.items()}, f, indent=1)
    return npz_path, norms_path


def load_state_bank(states_dir: Path, model: str, arm: str) -> StateBank:
    npz_path = states_dir / f"states_{model}_{arm}.npz"
    norms_path = states_dir / f"norms_{model}_{arm}.json"
    if not npz_path.exists() or not norms_path.exists():
        raise FileNotFoundError(f"state bank missing: {npz_path} / {norms_path}")
    z = np.load(npz_path)
    with open(norms_path) as f:
        norms = {int(k[1:]): float(v) for k, v in json.load(f).items()}
    text_ids = [str(t) for t in z["text_ids"]]
    states = {int(k[1:]): z[k] for k in z.files if k.startswith("L")}
    if set(states) != set(norms):
        raise ValueError(f"{npz_path}: sites {sorted(states)} != norms {sorted(norms)}")
    return StateBank(model=model, arm=arm, text_ids=text_ids,
                     states=states, median_norms=norms)


@dataclass
class Labels:
    """Corpus-manifest labels aligned to a bank's text_id order."""
    stratum: np.ndarray                  # ["S1"|"S2"|"S3"]
    group: np.ndarray                    # "(stratum|voice|mode)" null-permutation group
    topic: np.ndarray                    # int topic_idx, -1 for S2
    s2_rank: np.ndarray                  # int rank for S2 ids, -1 otherwise


def load_labels(manifest_path: Path, text_ids: list[str]) -> Labels:
    with open(manifest_path) as f:
        entries = {e["text_id"]: e for e in json.load(f)["entries"]}
    missing = [t for t in text_ids if t not in entries]
    if missing:
        raise ValueError(f"{len(missing)} text_ids absent from manifest: {missing[:3]}…")
    strat, grp, top, s2r = [], [], [], []
    for t in text_ids:
        e = entries[t]
        strat.append(e["stratum"])
        grp.append(f"{e['stratum']}|{e['voice']}|{e['mode']}")
        top.append(e["topic_idx"] if e["topic_idx"] is not None else -1)
        s2r.append(int(t.rsplit("-", 1)[1]) if e["stratum"] == "S2" else -1)
    return Labels(stratum=np.array(strat), group=np.array(grp),
                  topic=np.array(top), s2_rank=np.array(s2r))


def make_split(labels: Labels, seed: int = A8_SEED) -> tuple[np.ndarray, np.ndarray, dict]:
    """Topic-grouped stratified split: 5/20 topics + 40/160 S2 shards held out."""
    rng = np.random.default_rng(seed)
    topics = np.array(sorted({int(t) for t in labels.topic if t >= 0}))
    held_topics = set(rng.choice(topics, size=HELD_OUT_TOPICS, replace=False).tolist())
    s2_ranks = np.array(sorted({int(r) for r in labels.s2_rank if r >= 0}))
    held_s2 = set(rng.choice(s2_ranks, size=HELD_OUT_S2, replace=False).tolist())
    test = np.array([(int(t) in held_topics) or (int(r) in held_s2)
                     for t, r in zip(labels.topic, labels.s2_rank)])
    train = ~test
    info = {"rule": f"topic-grouped: {HELD_OUT_TOPICS}/20 topics + "
                    f"{HELD_OUT_S2}/160 S2 shards held out; seed {seed}",
            "held_topics": sorted(held_topics), "n_train": int(train.sum()),
            "n_test": int(test.sum()),
            "split_sha256": hashlib.sha256(test.tobytes()).hexdigest()}
    return train, test, info


# ---------------------------------------------------------------- fit families
@dataclass
class PCABank:
    mean: np.ndarray                     # [d]
    components: np.ndarray               # [kmax, d] rows = PCs
    explained: np.ndarray                # [kmax]

    @classmethod
    def fit(cls, x_train: np.ndarray, kmax: int) -> "PCABank":
        mu = x_train.mean(axis=0)
        xc = x_train - mu
        kmax = min(kmax, min(xc.shape) - 1)
        # economy SVD: n_train x d with n < d
        u, s, vt = np.linalg.svd(xc, full_matrices=False)
        var = s ** 2 / (xc.shape[0] - 1)
        return cls(mean=mu, components=vt[:kmax], explained=var[:kmax] / var.sum())


@dataclass
class TransportMap:
    """g: source hidden space -> target hidden space (linear part, for vectors).

    Vector transport is scale-calibrated (norm ratios folded in) but every read in
    Phase C is cosine-based, so calibration never affects a pass/fail.
    """
    kind: Literal["proc", "ridge"]
    src_norm: float
    tgt_norm: float
    # procrustes
    va: Optional[np.ndarray] = None      # [k, d_a]
    vb: Optional[np.ndarray] = None      # [k, d_b]
    omega: Optional[np.ndarray] = None   # [k, k], orthogonal
    scale: float = 1.0
    # ridge (factored: B = left @ right, d_a x d_b)
    left: Optional[np.ndarray] = None    # [d_a, r]
    right: Optional[np.ndarray] = None   # [r, d_b]

    def transport(self, v: np.ndarray, direction: Literal["fwd", "rev"] = "fwd"
                  ) -> np.ndarray:
        """Map a direction vector across. 'rev' is the adjoint (exact inverse of the
        orthogonal part for Procrustes; transpose map for ridge — 'where defined')."""
        if self.kind == "proc":
            assert self.va is not None and self.vb is not None and self.omega is not None
            if direction == "fwd":
                out = (v / self.src_norm) @ self.va.T @ self.omega @ self.vb
                return out * self.scale * self.tgt_norm
            out = (v / self.tgt_norm) @ self.vb.T @ self.omega.T @ self.va
            return out / self.scale * self.src_norm
        assert self.left is not None and self.right is not None
        if direction == "fwd":
            return ((v / self.src_norm) @ self.left) @ self.right * self.tgt_norm
        return ((v / self.tgt_norm) @ self.right.T) @ self.left.T * self.src_norm


def fit_procrustes(za: np.ndarray, zb: np.ndarray) -> tuple[np.ndarray, float]:
    """min ||za @ omega - zb||_F over orthogonal omega, + isotropic scale."""
    m = za.T @ zb
    u, s, vt = np.linalg.svd(m)
    omega = u @ vt
    denom = float((za ** 2).sum())
    scale = float(s.sum() / denom) if denom > 0 else 1.0
    return omega, scale


@dataclass
class RidgeSVD:
    """Ridge from centered X to centered Y via one SVD of X_train, alpha-sweepable."""
    mu_x: np.ndarray
    mu_y: np.ndarray
    u: np.ndarray                        # [n, r]
    s: np.ndarray                        # [r]
    vt: np.ndarray                       # [r, d_a]
    uty: np.ndarray                      # [r, d_b] = U^T Yc

    @classmethod
    def prep(cls, x: np.ndarray, y: np.ndarray) -> "RidgeSVD":
        mu_x, mu_y = x.mean(axis=0), y.mean(axis=0)
        u, s, vt = np.linalg.svd(x - mu_x, full_matrices=False)
        return cls(mu_x=mu_x, mu_y=mu_y, u=u, s=s, vt=vt, uty=u.T @ (y - mu_y))

    def press(self, alpha: float, y: np.ndarray) -> float:
        """Closed-form multi-output LOO PRESS for this alpha."""
        f = self.s ** 2 / (self.s ** 2 + alpha)               # [r]
        resid = (y - self.mu_y) - self.u @ (f[:, None] * self.uty)
        h = np.einsum("ij,j,ij->i", self.u, f, self.u)        # hat diagonal
        h = np.clip(h, 0.0, 1.0 - 1e-8)
        return float(((resid / (1.0 - h)[:, None]) ** 2).sum())

    def choose_alpha(self, y: np.ndarray) -> float:
        base = float((self.s ** 2).mean())
        grid = base * np.logspace(-5, 1, 13)
        scores = [self.press(a, y) for a in grid]
        return float(grid[int(np.argmin(scores))])

    def predict(self, x_new: np.ndarray, alpha: float,
                uty: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict Y for new X; pass a permuted uty to realize nulls without refits."""
        f = self.s / (self.s ** 2 + alpha)
        proj = (x_new - self.mu_x) @ self.vt.T                # [m, r]
        return self.mu_y + (proj * f) @ (uty if uty is not None else self.uty)

    def factors(self, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """B = left @ right with B = V f(S) U^T Yc  (for TransportMap)."""
        f = self.s / (self.s ** 2 + alpha)
        return self.vt.T * f, self.uty


# ---------------------------------------------------------------- metrics
def r2_score(y_true: np.ndarray, y_pred: np.ndarray, y_train_mean: np.ndarray) -> float:
    sse = float(((y_true - y_pred) ** 2).sum())
    sst = float(((y_true - y_train_mean) ** 2).sum())
    return 1.0 - sse / sst if sst > 0 else float("nan")


def per_stratum_r2(y_true, y_pred, y_train_mean, strata: np.ndarray) -> dict[str, float]:
    return {s: r2_score(y_true[strata == s], y_pred[strata == s], y_train_mean)
            for s in STRATA if (strata == s).any()}


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    num = float(np.linalg.norm(yc.T @ xc, "fro") ** 2)
    den = (np.linalg.norm(xc.T @ xc, "fro") * np.linalg.norm(yc.T @ yc, "fro"))
    return num / float(den) if den > 0 else float("nan")


def null_permutations(labels: Labels, train: np.ndarray, rng: np.random.Generator,
                      kind: Literal["shuffled", "stratum"]) -> np.ndarray:
    """A permutation of TRAIN target rows (indices into the train subset)."""
    n = int(train.sum())
    if kind == "shuffled":
        return rng.permutation(n)
    perm = np.arange(n)
    groups = labels.group[train]
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        perm[idx] = idx[rng.permutation(len(idx))]
    return perm


# ---------------------------------------------------------------- per-fit records
class FitRecord(BaseModel):
    site_pair: str                       # e.g. "3bL14->8bL16"
    arm: str
    family: str                          # "proc_k32" | ... | "ridge"
    n_train: int
    n_test: int
    r2: float
    r2_null_shuffled_q95: float
    r2_null_stratum_q95: float
    per_stratum_r2: dict[str, float]
    per_stratum_null_q95: dict[str, float]   # max of the two nulls' q95, per stratum
    strata_carried: list[str]
    cka_before: float
    cka_after: float
    valid: bool
    detail: dict = {}


def evaluate_fit(name: str, predict_fn, refit_predict_fn, x, y, train, test,
                 labels: Labels, perms: dict[str, list[np.ndarray]]) -> FitRecord:
    """predict_fn(x_test) -> y_pred using the true fit; refit_predict_fn(perm, x_test)
    -> y_pred under a permuted-train refit. perms = {'shuffled': [...], 'stratum': [...]}."""
    y_train_mean = y[train].mean(axis=0)
    y_pred = predict_fn(x[test])
    r2 = r2_score(y[test], y_pred, y_train_mean)
    strata_test = labels.stratum[test]
    ps_r2 = per_stratum_r2(y[test], y_pred, y_train_mean, strata_test)

    null_overall: dict[str, list[float]] = {"shuffled": [], "stratum": []}
    null_ps: dict[str, dict[str, list[float]]] = {
        "shuffled": {s: [] for s in STRATA}, "stratum": {s: [] for s in STRATA}}
    for kind in ("shuffled", "stratum"):
        for perm in perms[kind]:
            yp = refit_predict_fn(perm, x[test])
            null_overall[kind].append(r2_score(y[test], yp, y_train_mean))
            for s, v in per_stratum_r2(y[test], yp, y_train_mean, strata_test).items():
                null_ps[kind][s].append(v)
    q = lambda vals: float(np.quantile(vals, 0.95)) if vals else float("nan")
    ps_null_q95 = {s: max(q(null_ps["shuffled"][s]), q(null_ps["stratum"][s]))
                   for s in STRATA}
    carried = [s for s in STRATA
               if s in ps_r2 and np.isfinite(ps_null_q95[s]) and ps_r2[s] > ps_null_q95[s]]
    shuf_q95, strat_q95 = q(null_overall["shuffled"]), q(null_overall["stratum"])
    valid = (r2 > shuf_q95) and (r2 > strat_q95) and (len(carried) >= 2)
    pair, arm, family = name.split("::")
    return FitRecord(
        site_pair=pair, arm=arm, family=family,
        n_train=int(train.sum()), n_test=int(test.sum()),
        r2=round(r2, 4), r2_null_shuffled_q95=round(shuf_q95, 4),
        r2_null_stratum_q95=round(strat_q95, 4),
        per_stratum_r2={k: round(v, 4) for k, v in ps_r2.items()},
        per_stratum_null_q95={k: round(v, 4) for k, v in ps_null_q95.items()},
        strata_carried=carried,
        cka_before=round(linear_cka(x[test], y[test]), 4),
        cka_after=round(linear_cka(y_pred, y[test]), 4),
        valid=valid)


# ---------------------------------------------------------------- the grid
def run_pair_arm(src_bank: StateBank, tgt_bank: StateBank, s_site: int, t_site: int,
                 labels: Labels, train: np.ndarray, test: np.ndarray,
                 rng: np.random.Generator, k_grid=K_GRID, n_null=N_NULL_REPS
                 ) -> tuple[list[FitRecord], dict[str, TransportMap]]:
    x = src_bank.matrix(s_site)
    y = tgt_bank.matrix(t_site)
    pair = f"{src_bank.model}L{s_site}->{tgt_bank.model}L{t_site}"
    arm = src_bank.arm
    perms = {kind: [null_permutations(labels, train, rng, kind) for _ in range(n_null)]
             for kind in ("shuffled", "stratum")}
    records: list[FitRecord] = []
    maps: dict[str, TransportMap] = {}

    kmax = int(min(max(k_grid), train.sum() - 1))
    pca_a = PCABank.fit(x[train], kmax)
    pca_b = PCABank.fit(y[train], kmax)
    za_full = (x - pca_a.mean) @ pca_a.components.T        # [n, kmax]
    zb_full = (y - pca_b.mean) @ pca_b.components.T

    for k in k_grid:
        k_eff = min(k, kmax)
        za, zb = za_full[:, :k_eff], zb_full[:, :k_eff]
        omega, scale = fit_procrustes(za[train], zb[train])

        def predict(x_test, *, _o=omega, _s=scale, _k=k_eff):
            zt = (x_test - pca_a.mean) @ pca_a.components[:_k].T
            return (zt @ _o) * _s @ pca_b.components[:_k] + pca_b.mean

        def refit_predict(perm, x_test, *, _k=k_eff):
            o, s = fit_procrustes(za[train], zb[train][perm])
            zt = (x_test - pca_a.mean) @ pca_a.components[:_k].T
            return (zt @ o) * s @ pca_b.components[:_k] + pca_b.mean

        name = f"{pair}::{arm}::proc_k{k}"
        rec = evaluate_fit(name, predict, refit_predict, x, y, train, test, labels, perms)
        rec.detail = {"k_effective": k_eff, "scale": round(scale, 5),
                      "pca_explained_src": round(float(pca_a.explained[:k_eff].sum()), 4),
                      "pca_explained_tgt": round(float(pca_b.explained[:k_eff].sum()), 4)}
        records.append(rec)
        maps[f"proc_k{k}"] = TransportMap(
            kind="proc", src_norm=src_bank.median_norms[s_site],
            tgt_norm=tgt_bank.median_norms[t_site],
            va=pca_a.components[:k_eff], vb=pca_b.components[:k_eff],
            omega=omega, scale=scale)
        logger.info("%s  R2=%.3f nulls(q95)=%.3f/%.3f carried=%s valid=%s",
                    name, rec.r2, rec.r2_null_shuffled_q95, rec.r2_null_stratum_q95,
                    rec.strata_carried, rec.valid)

    ridge = RidgeSVD.prep(x[train], y[train])
    alpha = ridge.choose_alpha(y[train])

    def r_predict(x_test):
        return ridge.predict(x_test, alpha)

    def r_refit_predict(perm, x_test):
        yc_perm = (y[train][perm] - y[train][perm].mean(axis=0))
        return ridge.predict(x_test, alpha, uty=ridge.u.T @ yc_perm)

    name = f"{pair}::{arm}::ridge"
    rec = evaluate_fit(name, r_predict, r_refit_predict, x, y, train, test, labels, perms)
    rec.detail = {"alpha": alpha, "rank": int(len(ridge.s))}
    records.append(rec)
    left, right = ridge.factors(alpha)
    maps["ridge"] = TransportMap(
        kind="ridge", src_norm=src_bank.median_norms[s_site],
        tgt_norm=tgt_bank.median_norms[t_site], left=left, right=right)
    logger.info("%s  R2=%.3f nulls(q95)=%.3f/%.3f carried=%s valid=%s alpha=%.3g",
                name, rec.r2, rec.r2_null_shuffled_q95, rec.r2_null_stratum_q95,
                rec.strata_carried, rec.valid, alpha)
    return records, maps


def arm_agreement(maps_by_arm: dict[str, dict[str, TransportMap]], d_src: int,
                  rng: np.random.Generator) -> dict[str, dict[str, float]]:
    """cos between native-arm and raw-arm transported images of shared unit probes."""
    probes = rng.standard_normal((N_PROBES, d_src))
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)
    out: dict[str, dict[str, float]] = {}
    arms = list(maps_by_arm)
    if len(arms) < 2:
        return out
    for fam in maps_by_arm[arms[0]]:
        t1 = np.stack([maps_by_arm[arms[0]][fam].transport(p) for p in probes])
        t2 = np.stack([maps_by_arm[arms[1]][fam].transport(p) for p in probes])
        t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
        t2 /= np.linalg.norm(t2, axis=1, keepdims=True)
        cos = (t1 * t2).sum(axis=1)
        out[fam] = {"mean_cos": round(float(cos.mean()), 4),
                    "min_cos": round(float(cos.min()), 4),
                    "n_probes": N_PROBES}
    return out


def save_transport_map(fits_dir: Path, pair: str, arm: str, fam: str,
                       tm: TransportMap) -> Path:
    path = fits_dir / f"fit_{pair.replace('->', '__')}_{arm}_{fam}.npz"
    payload: dict[str, np.ndarray] = {
        "kind": np.array(tm.kind), "src_norm": np.array(tm.src_norm),
        "tgt_norm": np.array(tm.tgt_norm), "scale": np.array(tm.scale)}
    for key in ("va", "vb", "omega", "left", "right"):
        val = getattr(tm, key)
        if val is not None:
            payload[key] = val.astype(np.float32)
    np.savez_compressed(path, **payload)
    return path


def load_transport_map(path: Path) -> TransportMap:
    z = np.load(path)
    get = lambda k: z[k].astype(np.float64) if k in z.files else None
    return TransportMap(kind=str(z["kind"]), src_norm=float(z["src_norm"]),
                        tgt_norm=float(z["tgt_norm"]), scale=float(z["scale"]),
                        va=get("va"), vb=get("vb"), omega=get("omega"),
                        left=get("left"), right=get("right"))


def subset_bank(bank: StateBank, keep: np.ndarray) -> StateBank:
    """Row-subset a bank. median_norms stay the COLLECTION values (stamped)."""
    return StateBank(model=bank.model, arm=bank.arm,
                     text_ids=[t for t, k in zip(bank.text_ids, keep) if k],
                     states={s: a[keep] for s, a in bank.states.items()},
                     median_norms=bank.median_norms)


def subset_labels(labels: Labels, keep: np.ndarray) -> Labels:
    return Labels(stratum=labels.stratum[keep], group=labels.group[keep],
                  topic=labels.topic[keep], s2_rank=labels.s2_rank[keep])


def run_grid(arm_root: Path, src_model: str, tgt_model: str,
             n_null: int = N_NULL_REPS, fit_strata: Optional[list[str]] = None,
             fits_dirname: str = "fits", k_grid=K_GRID,
             sites_override: dict[str, tuple[int, ...]] | None = None,
             arms: tuple[str, ...] = ARMS) -> dict:
    states_dir = arm_root / "states"
    fits_dir = arm_root / fits_dirname
    fits_dir.mkdir(parents=True, exist_ok=True)
    manifest = arm_root / "corpus" / "corpus_manifest.json"

    all_records: list[FitRecord] = []
    agreement: dict[str, dict] = {}
    split_info: dict = {}
    sites_map = {**SITES, **(sites_override or {})}
    for s_site in sites_map[src_model]:
        for t_site in sites_map[tgt_model]:
            maps_by_arm: dict[str, dict[str, TransportMap]] = {}
            for arm in arms:
                src = load_state_bank(states_dir, src_model, arm)
                tgt = load_state_bank(states_dir, tgt_model, arm)
                if src.text_ids != tgt.text_ids:
                    raise RuntimeError(f"text_id order mismatch {src_model}/{tgt_model} ({arm})")
                labels = load_labels(manifest, src.text_ids)
                if fit_strata:
                    keep = np.isin(labels.stratum, fit_strata)
                    src, tgt = subset_bank(src, keep), subset_bank(tgt, keep)
                    labels = subset_labels(labels, keep)
                train, test, split_info = make_split(labels)
                rng = np.random.default_rng(A8_SEED + s_site * 100 + t_site)
                recs, maps = run_pair_arm(src, tgt, s_site, t_site, labels,
                                          train, test, rng, k_grid=k_grid,
                                          n_null=n_null)
                all_records.extend(recs)
                maps_by_arm[arm] = maps
                pair = f"{src_model}L{s_site}->{tgt_model}L{t_site}"
                for fam, tm in maps.items():
                    save_transport_map(fits_dir, pair, arm, fam, tm)
            d_src = next(iter(maps_by_arm.values()))["ridge"].left.shape[0]
            pair = f"{src_model}L{s_site}->{tgt_model}L{t_site}"
            agreement[pair] = arm_agreement(
                maps_by_arm, d_src, np.random.default_rng(A8_SEED))

    summary = {
        "arm": "A8_conjugation", "leg": 0, "builder": "a8_fit_g.py",
        "prereg_tag": "prereg-arm8-v1",
        "pair": f"{src_model}->{tgt_model}",
        "split": split_info, "n_null_reps": n_null, "k_grid": list(k_grid),
        "fit_strata": fit_strata, "arms": list(arms),
        "null_group_key": "(stratum|voice|mode)",
        "normalization": "per-site median norm at fit time (norms in state banks)",
        "records": [r.model_dump() for r in all_records],
        "two_arm_g_agreement": agreement,
        "n_valid": sum(r.valid for r in all_records),
        "n_fits": len(all_records),
    }
    out = fits_dir / "cp2_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    logger.info("cp2 summary: %s  (%d/%d fits valid)", out,
                summary["n_valid"], summary["n_fits"])
    return summary


# ---------------------------------------------------------------- selftest
def _synthetic_world(rng, n_latent=12, d_a=64, d_b=96, noise=0.15, paired=True):
    """Paired clouds from a shared latent with stratum/topic cluster structure."""
    entries = []
    for voice in ("3b", "8b"):
        for mode in ("expository", "explanatory", "argumentative", "conversational"):
            for t in range(20):
                entries.append(("S1", voice, mode, t))
    for r in range(80):
        entries.append(("S2", "neutral", "neutral", -1))
    for voice in ("3b", "8b"):
        for mode in ("linear", "socratic", "contrastive", "dialectical", "analogical"):
            for t in range(12):
                entries.append(("S3", voice, mode, t))
    n = len(entries)
    group_keys = sorted({f"{s}|{v}|{m}" for s, v, m, _ in entries})
    gmeans = {g: rng.standard_normal(n_latent) * 1.2 for g in group_keys}
    tmeans = rng.standard_normal((20, n_latent)) * 0.8
    z = np.zeros((n, n_latent))
    strat, grp, top, s2r = [], [], [], []
    s2_rank = 0
    for i, (s, v, m, t) in enumerate(entries):
        g = f"{s}|{v}|{m}"
        z[i] = gmeans[g] + (tmeans[t] if t >= 0 else 0) + rng.standard_normal(n_latent)
        strat.append(s); grp.append(g); top.append(t)
        s2r.append(s2_rank if s == "S2" else -1)
        s2_rank += (s == "S2")
    a_map = np.linalg.qr(rng.standard_normal((d_a, n_latent)))[0].T  # [r, d_a]
    b_map = np.linalg.qr(rng.standard_normal((d_b, n_latent)))[0].T
    x = z @ a_map + noise * rng.standard_normal((n, d_a))
    # unpaired control: keep GROUP cluster geometry but break every finer
    # correspondence (fresh within-cluster noise AND permuted topic effects) —
    # the world where only the stratum-preserving null's structure survives.
    topic_perm = rng.permutation(20)
    z2 = z if paired else np.array(
        [gmeans[g] + (tmeans[topic_perm[t]] if t >= 0 else 0)
         + rng.standard_normal(n_latent)
         for g, t in zip(grp, top)])
    y = z2 @ b_map + noise * rng.standard_normal((n, d_b))
    v_lat = np.zeros(n_latent); v_lat[3] = 1.0
    labels = Labels(stratum=np.array(strat), group=np.array(grp),
                    topic=np.array(top), s2_rank=np.array(s2r))
    banks = (
        StateBank("srcM", "native", [f"t{i}" for i in range(n)], {1: x.astype(np.float32)},
                  {1: float(np.median(np.linalg.norm(x, axis=1)))}),
        StateBank("tgtM", "native", [f"t{i}" for i in range(n)], {1: y.astype(np.float32)},
                  {1: float(np.median(np.linalg.norm(y, axis=1)))}))
    return banks, labels, (v_lat @ a_map, v_lat @ b_map)


def _selftest_split(labels, rng):
    topics = np.array(sorted({int(t) for t in labels.topic if t >= 0}))
    held_t = set(rng.choice(topics, size=max(2, len(topics) // 4), replace=False).tolist())
    s2 = np.array(sorted({int(r) for r in labels.s2_rank if r >= 0}))
    held_s = set(rng.choice(s2, size=max(2, len(s2) // 4), replace=False).tolist())
    test = np.array([(int(t) in held_t) or (int(r) in held_s)
                     for t, r in zip(labels.topic, labels.s2_rank)])
    return ~test, test


def selftest() -> int:
    rng = np.random.default_rng(0)
    failures: list[str] = []

    def check(cond: bool, msg: str):
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {msg}")
        if not cond:
            failures.append(msg)

    print("== selftest 1: planted paired world (recovery expected) ==")
    (src, tgt), labels, (v_a, v_b) = _synthetic_world(rng, paired=True)
    train, test = _selftest_split(labels, rng)
    recs, maps = run_pair_arm(src, tgt, 1, 1, labels, train, test,
                              np.random.default_rng(1), k_grid=(8, 16), n_null=10)
    by_fam = {r.family.split("::")[-1]: r for r in recs}
    for fam, r in by_fam.items():
        check(r.valid, f"{fam}: VALID (R2={r.r2}, nulls q95 "
                       f"{r.r2_null_shuffled_q95}/{r.r2_null_stratum_q95}, "
                       f"carried={r.strata_carried})")
    check(all(r.r2 > 0.5 for r in recs), "all true-fit R2 > 0.5")
    check(all(r.r2_null_stratum_q95 >= r.r2_null_shuffled_q95 - 0.05 for r in recs),
          "stratum-preserving null >= shuffled null (sharper, as designed)")
    for fam in ("proc_k16", "ridge"):
        tv = maps[fam].transport(v_a)
        cos = float(tv @ v_b / (np.linalg.norm(tv) * np.linalg.norm(v_b)))
        check(cos > 0.8, f"{fam}: planted-axis transport cos={cos:.3f} > 0.8")
        rv = maps[fam].transport(v_b, direction="rev")
        rcos = float(rv @ v_a / (np.linalg.norm(rv) * np.linalg.norm(v_a)))
        check(rcos > 0.8, f"{fam}: reverse transport cos={rcos:.3f} > 0.8")

    print("== selftest 2: unpaired noise world (validity must NOT fire) ==")
    (src_n, tgt_n), labels_n, _ = _synthetic_world(
        np.random.default_rng(7), paired=False)
    train_n, test_n = _selftest_split(labels_n, np.random.default_rng(7))
    recs_n, _ = run_pair_arm(src_n, tgt_n, 1, 1, labels_n, train_n, test_n,
                             np.random.default_rng(2), k_grid=(8, 16), n_null=10)
    for r in recs_n:
        fam = r.family
        check(not r.valid, f"{fam}: correctly INVALID on unpaired clouds "
                           f"(R2={r.r2} vs stratum-null q95={r.r2_null_stratum_q95})")

    print("== selftest 3: two-arm agreement on a shared world ==")
    (src2, tgt2), labels2, _ = _synthetic_world(np.random.default_rng(0), paired=True)
    src2.arm = tgt2.arm = "raw"
    pert = np.random.default_rng(11)
    src2.states[1] = src2.states[1] + 0.05 * pert.standard_normal(src2.states[1].shape).astype(np.float32)
    tgt2.states[1] = tgt2.states[1] + 0.05 * pert.standard_normal(tgt2.states[1].shape).astype(np.float32)
    _, maps2 = run_pair_arm(src2, tgt2, 1, 1, labels2, train, test,
                            np.random.default_rng(3), k_grid=(8, 16), n_null=2)
    agree = arm_agreement({"native": maps, "raw": maps2}, d_src=64,
                          rng=np.random.default_rng(A8_SEED))
    # NOTE: when k exceeds the effective latent rank (here 12), the extra PCs are
    # noise directions whose Procrustes rotation is arbitrary per-arm — agreement
    # degrades in that subspace BY DESIGN (the read doubles as a rank diagnostic).
    for fam, a in agree.items():
        check(a["mean_cos"] > 0.75, f"{fam}: two-arm agreement mean_cos={a['mean_cos']}")

    print("== selftest 4: state-bank round-trip ==")
    import tempfile
    with tempfile.TemporaryDirectory(prefix="a8_selftest_") as td:
        p1, p2 = save_state_bank(Path(td), src)
        back = load_state_bank(Path(td), "srcM", "native")
        check(back.text_ids == src.text_ids, "text_ids round-trip")
        check(np.allclose(back.states[1], src.states[1]), "states round-trip (fp32)")
        check(abs(back.median_norms[1] - src.median_norms[1]) < 1e-9, "norms round-trip")

    print(f"\nselftest: {len(failures)} failure(s)")
    return 1 if failures else 0


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm-root", type=Path, default=DEFAULT_ARM_ROOT)
    ap.add_argument("--source-model", default="3b", choices=sorted(SITES))
    ap.add_argument("--target-model", default="8b", choices=sorted(SITES))
    ap.add_argument("--n-null", type=int, default=N_NULL_REPS)
    ap.add_argument("--fit-strata", default=None,
                    help="comma list, e.g. S1,S2 — fit/gate g on these strata only "
                         "(the mode-free beside column; desk order 2026-07-22)")
    ap.add_argument("--src-sites", default=None, help="comma-separated source-site override")
    ap.add_argument("--tgt-sites", default=None, help="comma-separated target-site override")
    ap.add_argument("--k-grid", default=None,
                    help="comma-separated PC ranks (default 32,128,512). The add-3 rank "
                         "guard binds small-n fits: k <= n_train/1.2, k512 forbidden at n<620.")
    ap.add_argument("--fits-dirname", default="fits",
                    help="output subdir under arm-root (use fits_modefree for the "
                         "beside column — never overwrite the primary)")
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="comma list of template arms to fit. Defaults to both. Use "
                         "'raw' alone for template-less BASE models (olmo2-7b has no "
                         "chat template, so its native arm does not exist) — and see "
                         "A8-add-7.1: a raw-arm-only model's constants live in the "
                         "PARALLEL RAW-ARM star system, never the native one.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    strata = ([s.strip() for s in args.fit_strata.split(",") if s.strip()]
              if args.fit_strata else None)
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"unknown arms {bad}; valid: {ARMS}")
    run_grid(args.arm_root, args.source_model, args.target_model, n_null=args.n_null,
             fit_strata=strata, fits_dirname=args.fits_dirname, arms=arms,
             k_grid=(tuple(int(k) for k in args.k_grid.split(","))
                     if args.k_grid else K_GRID),
             sites_override={
                 **({args.source_model: tuple(int(x) for x in args.src_sites.split(","))}
                    if args.src_sites else {}),
                 **({args.target_model: tuple(int(x) for x in args.tgt_sites.split(","))}
                    if args.tgt_sites else {})} or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
