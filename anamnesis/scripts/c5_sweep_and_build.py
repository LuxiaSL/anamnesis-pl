"""Computer #5 install-lever PoC — S3/S4/S5: site sweep, alignment diagnostic, vector build.

SPEC §3 (S3-S5). Pure numpy over the S2 state banks. No GPU, no model.

Ported from kotodama's `steer_site_sweep.py` + `steer_install_build.py`, which are the recipe of
record for a model-difference install lever. Deliberately NOT the A6 vDiverge form: no signature
sort, no decile poles, no length residualization, no z-scoring — see spec §1 for why each was
dropped. The whole construction is:

    Delta = mu(donor per-gen means) - mu(base per-gen means)   at layer s
    raw   = unit(Delta)
    whit  = unit(Sigma_base^-1 Delta)      Sigma from BASE tokens ONLY (we inject into base, so
                                           the injection target's covariance is the metric)
    Rband = unit(U_base[:, 16:256] @ N(0,1))    matched null, inside Sigma_base's eigenband
    dose  = alpha * median ||base completion-token residual|| at s
    sign  : + = toward the donor

SWEEP LAW (koto, verbatim): per-PROMPT averaging first, then K random half-splits over PROMPTS,
direction fit on the train half, held-out Cohen's d read on the test half. Averaging per prompt
before splitting is what stops M=16 samples of one prompt from acting like 16 independent points;
splitting over prompts is what makes the d honest rather than in-sample optimism.

subcommands:
    sweep     per-layer held-out Cohen's d for a donor-vs-base pair   (S3)
    diagnose  |cos(Delta, Sigma^-1 Delta)| + Mahalanobis at a site    (S4; needs pass-2 npz)
    build     bank raw / whitened / Rband / median-norm vectors       (S5; needs pass-2 npz)
    cosines   the donor-vs-donor cosine table (spec section 4)        (the decomposition)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c5_ledoit_wolf_gpu import eigh_gpu, ledoit_wolf_gpu, solve_gpu  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BAND = (16, 256)      # koto/anamnesis eigen-index band for the matched null; ported unchanged
NULL_SEED = 20260729


def load_means(path: Path) -> tuple[np.ndarray, list[dict]]:
    if not path.is_file():
        sys.exit(f"FATAL: state bank not found: {path}")
    z = np.load(path, allow_pickle=True)
    return z["means"].astype(np.float32), json.loads(str(z["meta"]))


def per_prompt(means: np.ndarray, meta: list[dict]) -> dict[str, np.ndarray]:
    """Average the M samples of each prompt first — the displacement convention."""
    by: dict[str, list[np.ndarray]] = defaultdict(list)
    for m, r in zip(means, meta):
        by[r["prompt_id"]].append(m)
    return {k: np.mean(v, axis=0) for k, v in by.items()}


def heldout_d(a: np.ndarray, b: np.ndarray, k_splits: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """a,b: (n_prompts, L, D) -> per-layer mean and sd of held-out Cohen's d."""
    n_layers = a.shape[1]
    ds = np.zeros((k_splits, n_layers))
    for k in range(k_splits):
        ia, ib = rng.permutation(len(a)), rng.permutation(len(b))
        ha, hb = len(a) // 2, len(b) // 2
        a_tr, a_te, b_tr, b_te = a[ia[:ha]], a[ia[ha:]], b[ib[:hb]], b[ib[hb:]]
        for s in range(n_layers):
            u = a_tr[:, s].mean(0) - b_tr[:, s].mean(0)
            n = np.linalg.norm(u)
            if n < 1e-8:
                continue
            u = u / n
            pa, pb = a_te[:, s] @ u, b_te[:, s] @ u
            pooled = np.sqrt((pa.var(ddof=1) + pb.var(ddof=1)) / 2)
            ds[k, s] = (pa.mean() - pb.mean()) / max(pooled, 1e-8)
    return ds.mean(0), ds.std(0)


def _paired(donor: Path, base: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    md, meta_d = load_means(donor)
    mb, meta_b = load_means(base)
    pd_, pb_ = per_prompt(md, meta_d), per_prompt(mb, meta_b)
    common = sorted(set(pd_) & set(pb_))
    if len(common) < 8:
        sys.exit(f"FATAL: only {len(common)} shared prompts — pairing is broken")
    return (np.stack([pd_[p] for p in common]), np.stack([pb_[p] for p in common]), common)


def cmd_sweep(args) -> None:
    A, B, common = _paired(args.donor, args.base)
    logger.info("paired on %d prompts; layers=%d hidden=%d", len(common), A.shape[1], A.shape[2])
    rng = np.random.default_rng(17)
    d_mean, d_sd = heldout_d(A, B, args.k_splits, rng)

    order = np.argsort(-np.abs(d_mean))
    print(f"\n[sweep] {args.label}: top layers by |held-out d| (K={args.k_splits} splits)")
    for s in order[:12]:
        print(f"   L{s:2d} ({100*s/len(d_mean):4.0f}% depth): d={d_mean[s]:+.3f} +/- {d_sd[s]:.3f}")
    peak = int(order[0])
    print(f"\n[sweep] PEAK L{peak} ({100*peak/len(d_mean):.0f}% depth) d={d_mean[peak]:+.3f}")
    # The dense-family convention peaks ~47-50% depth; koto's install peaked LATE (L20-27 of 32).
    # Report where we land against both priors rather than assuming either.
    print(f"[sweep] dense-family prior ~47-50% depth -> L{int(0.485*len(d_mean))}; "
          f"koto install prior was LATE (~65-85%)")

    out = {"label": args.label, "donor": str(args.donor), "base": str(args.base),
           "n_prompts": len(common), "k_splits": args.k_splits,
           "d_mean": d_mean.tolist(), "d_sd": d_sd.tolist(), "peak_layer": peak,
           "peak_d": float(d_mean[peak]),
           "law": ("held-out Cohen's d along the train-half mean-difference direction, "
                   "per-prompt averaged, K random half-splits over prompts, seed 17")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    logger.info("wrote %s", args.out)


def _load_site_tokens(path: Path, site: int) -> tuple[np.ndarray, np.ndarray]:
    """pass-2 npz -> (pooled per-token (T_total, D), per-gen means (N, D)) at `site`."""
    z = np.load(path, allow_pickle=True)
    if "site_tokens" not in z:
        sys.exit(f"FATAL: {path} has no site_tokens — that is a pass-1 bank; rerun with --dump-sites")
    idx = list(z["site_indices"]).index(site)
    toks, gmeans = [], []
    for d in z["site_tokens"]:
        arr = np.asarray(d, dtype=np.float32)[idx]
        toks.append(arr)
        gmeans.append(arr.mean(0))
    return np.concatenate(toks, axis=0), np.stack(gmeans)


def cmd_diagnose(args) -> None:
    """Alignment diagnostic at one or more sites. Multi-site in ONE process on purpose: the
    per-token npz banks are multi-GB, so reloading per site would dominate the cost."""
    results = {}
    for site in (int(x) for x in str(args.site).split(",")):
        results[f"L{site}"] = _diagnose_one(args.donor, args.base, site, args.sigma_from)
        logger.info("L%d: %s", site, json.dumps(results[f"L{site}"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


def _diagnose_one(donor: Path, base: Path, site: int, sigma_from: Path | None = None) -> dict:
    td, gd = _load_site_tokens(donor, site)
    tb, gb = _load_site_tokens(base, site)
    if sigma_from is not None:
        tb, _ = _load_site_tokens(sigma_from, site)   # see --sigma-from note in cmd_build
    delta = (gd.mean(0) - gb.mean(0)).astype(np.float64)
    # Sigma from the INJECTION TARGET only (koto's refinement) — we inject into base, so base's
    # covariance is the metric. Pooling the donor in would measure a space we never write to.
    # GPU Ledoit-Wolf: exact port of sklearn's estimator (validated to 1e-8 on shrinkage,
    # cos(Sigma^-1 b) = 1.00000000). sklearn at d=8192 is minutes per site; this is seconds.
    Sigma, shrinkage = ledoit_wolf_gpu(tb)
    w = solve_gpu(Sigma, delta)
    cos = abs(float(delta @ w) / max(float(np.linalg.norm(delta) * np.linalg.norm(w)), 1e-12))
    maha = float(np.sqrt(max(float(delta @ w), 0.0)))
    verdict = ("ALIGNED — raw mean-diff fine" if cos >= 0.3 else
               "MISALIGNED — whiten from the start; raw nulls are construction-scoped"
               if cos <= 0.2 else "MID-REGIME — treat raw as untrusted (golden path 1c)")
    # every value cast through float()/int(): numpy scalars are not JSON-serializable and the
    # failure lands at the very END of a multi-minute fit.
    return {"site": int(site), "n_base_tokens": int(tb.shape[0]), "n_donor_tokens": int(td.shape[0]),
            "delta_norm": float(np.linalg.norm(delta)), "lw_shrinkage": float(shrinkage),
            "cos_delta_whitened": round(cos, 4), "mahalanobis_d": round(maha, 3),
            "base_median_token_norm": float(np.median(np.linalg.norm(tb, axis=1))),
            "VERDICT": verdict}


def cmd_build(args) -> None:
    rng = np.random.default_rng(NULL_SEED)
    vectors: dict[str, np.ndarray] = {}
    diag: dict[str, object] = {}
    for site in (int(x) for x in args.sites.split(",")):
        td, gd = _load_site_tokens(args.donor, site)
        tb, gb = _load_site_tokens(args.base, site)
        delta = gd.mean(0) - gb.mean(0)
        # CONTENT-CONTROLLED BUILD: Delta comes from donor-vs-base forced over the SAME text,
        # but Sigma must still describe the distribution we actually inject into — base
        # generating its own text. Those are different banks, hence --sigma-from. Using the
        # forced bank's covariance would whiten and band-null against a distribution that never
        # occurs at steering time.
        if args.sigma_from is not None:
            tb, _ = _load_site_tokens(args.sigma_from, site)
        raw = delta / max(np.linalg.norm(delta), 1e-12)
        Sigma, shrinkage = ledoit_wolf_gpu(tb)
        w = solve_gpu(Sigma, delta)
        whit = w / max(np.linalg.norm(w), 1e-12)
        evals, evecs = eigh_gpu(Sigma)
        order = np.argsort(evals)[::-1]
        Ub = evecs[:, order[BAND[0]:BAND[1]]]
        r = Ub @ rng.standard_normal(Ub.shape[1])
        med = float(np.median(np.linalg.norm(tb, axis=1)))
        vectors[f"Install_L{site}"] = raw.astype(np.float32)
        vectors[f"InstallW_L{site}"] = whit.astype(np.float32)
        vectors[f"Rband_L{site}"] = (r / np.linalg.norm(r)).astype(np.float32)
        vectors[f"median_norm_L{site}"] = np.array(med, dtype=np.float64)
        diag[f"L{site}"] = {"delta_norm": float(np.linalg.norm(delta)),
                            "cos_raw_whit": round(float(raw @ whit), 4),
                            "lw_shrinkage": round(float(shrinkage), 5),
                            "base_median_token_norm": round(med, 4),
                            "n_base_tokens": int(tb.shape[0]), "n_gens": int(len(gb))}
        logger.info("L%d: |D|=%.2f cos(raw,whit)=%+.3f base_med_norm=%.2f (base toks %d)",
                    site, np.linalg.norm(delta), float(raw @ whit), med, tb.shape[0])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **vectors,
             sign_convention=f"+ = toward {args.donor_label} (donor); Delta = donor - base",
             law="install-lever (koto steer_install_build): model-vs-model mean-diff; "
                 "Sigma_base whiten; base-Sigma [16:256] band null",
             diagnostics=json.dumps(diag))
    args.out.with_suffix(".diag.json").write_text(json.dumps(diag, indent=1))
    logger.info("banked -> %s", args.out)


def cmd_cosines(args) -> None:
    """The decomposition table (spec section 4): every donor's install direction against every
    other, at one layer. This is what answers 'is it Computer or just tuned-ness' by AGREEMENT
    rather than by projecting anything out."""
    base_means, base_meta = load_means(args.base)
    pb = per_prompt(base_means, base_meta)
    dirs: dict[str, np.ndarray] = {}
    for spec in args.donors:
        label, path = spec.split("=", 1)
        md, meta_d = load_means(Path(path))
        pd_ = per_prompt(md, meta_d)
        common = sorted(set(pd_) & set(pb))
        A = np.stack([pd_[p] for p in common])[:, args.site, :]
        B = np.stack([pb[p] for p in common])[:, args.site, :]
        d = A.mean(0) - B.mean(0)
        dirs[label] = d / max(np.linalg.norm(d), 1e-12)
        logger.info("%s: |Delta|=%.3f over %d prompts", label, np.linalg.norm(d), len(common))

    labels = list(dirs)
    table = {a: {b: round(float(dirs[a] @ dirs[b]), 4) for b in labels} for a in labels}
    print(f"\n[cosines @ L{args.site}] donor-minus-base directions\n")
    w = max(len(x) for x in labels) + 2
    print(" " * w + "".join(f"{b:>11s}" for b in labels))
    for a in labels:
        print(f"{a:<{w}s}" + "".join(f"{table[a][b]:>11.3f}" for b in labels))

    out = {"site": args.site, "base": str(args.base), "cosines": table,
           "law": "unit(mean(donor per-prompt means) - mean(base per-prompt means)) at the layer, "
                  "paired on shared prompts"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sw = sub.add_parser("sweep")
    sw.add_argument("--donor", type=Path, required=True)
    sw.add_argument("--base", type=Path, required=True)
    sw.add_argument("--label", required=True)
    sw.add_argument("--k-splits", type=int, default=50)
    sw.add_argument("--out", type=Path, required=True)

    dg = sub.add_parser("diagnose")
    dg.add_argument("--donor", type=Path, required=True)
    dg.add_argument("--base", type=Path, required=True)
    dg.add_argument("--site", required=True, help="one layer, or a comma list")
    dg.add_argument("--sigma-from", type=Path, default=None)
    dg.add_argument("--out", type=Path, required=True)

    bd = sub.add_parser("build")
    bd.add_argument("--donor", type=Path, required=True)
    bd.add_argument("--base", type=Path, required=True)
    bd.add_argument("--sites", required=True)
    bd.add_argument("--sigma-from", type=Path, default=None,
                    help="take Sigma + the dose currency from THIS bank instead of --base. Use "
                         "for content-controlled builds: Delta from the forced pair, Sigma from "
                         "the injection target's own free generation.")
    bd.add_argument("--donor-label", default="donor")
    bd.add_argument("--out", type=Path, required=True)

    cs = sub.add_parser("cosines")
    cs.add_argument("--base", type=Path, required=True)
    cs.add_argument("--donors", nargs="+", required=True, help="label=path/to/means.npz")
    cs.add_argument("--site", type=int, required=True)
    cs.add_argument("--out", type=Path, required=True)

    args = ap.parse_args()
    {"sweep": cmd_sweep, "diagnose": cmd_diagnose, "build": cmd_build,
     "cosines": cmd_cosines}[args.cmd](args)


if __name__ == "__main__":
    main()
