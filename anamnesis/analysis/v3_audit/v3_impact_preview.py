"""v3-impact preview (PARALLELIZED + progress logging): does the cleaned v3 feature
set beat v2, and is it more length-robust? 5-way (hard modes), GroupKFold by topic,
RF, raw vs length-residualized.

Fair comparison: both v2 and v3 EXCLUDE temporal_dynamics (v3 drops it) and contrastive.
v3 = recompute-from-raw (enable_per_head=True, enable_temporal_dynamics=False) with the
baseline T1 + T2-non-spectral splice from v2 banked (raw can't reproduce all-layer T1/T2).

Per-gen recompute is parallelized across a process pool (the expensive part); progress is
logged with ETA. Run from anamnesis_exps with PYTHONPATH=pipeline.
    N_WORKERS=8 PYTHONPATH=pipeline python3 research/notes/v3_impact_preview.py
"""
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from anamnesis.config import MODEL_PRESETS, ExtractionConfig, FeaturePipelineConfig
from anamnesis.extraction.feature_pipeline import compute_features_v2, _load_pca_model
from anamnesis.extraction.raw_saver import list_raw_tensor_ids

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")  # avoid BLAS oversubscription across workers


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


ROOT = Path(".")
RUN = ROOT / "outputs/runs/3b_fat_01"
CAL = ROOT / "phase_0/outputs/calibration"
HARD = {"linear", "socratic", "contrastive", "dialectical", "analogical"}
PRESET = MODEL_PRESETS["3b"]

# Lazy per-process cache (robust under fork or spawn).
_CACHE: dict = {}


def _ctx() -> dict:
    if not _CACHE:
        p = PRESET
        _CACHE["cfg"] = ExtractionConfig(
            sampled_layers=p.sampled_layers, pca_layers=p.pca_layers,
            early_layer_cutoff=p.early_layer_cutoff, late_layer_cutoff=p.late_layer_cutoff,
        )
        _CACHE["fam"] = FeaturePipelineConfig(
            include_baseline_tiers=True, enable_residual_trajectory=True, enable_attention_flow=True,
            enable_gate_features=True, enable_temporal_dynamics=False, enable_per_head=True,
            enable_stft=True, enable_contrastive_projection=False, trajectory_layers=p.trajectory_layers,
        )
        _CACHE["comps"], _CACHE["mean"] = _load_pca_model(CAL / "pca_model.pkl")
        _CACHE["meta"] = {g["generation_id"]: g for g in json.load(open(RUN / "metadata.json"))["generations"]}
    return _CACHE


def is_spliced(n: str) -> bool:
    """T1 + T2-non-spectral come from v2 banked (raw zero-fills non-sampled layers)."""
    return n.startswith((
        "activation_norm", "logit_entropy", "top1_prob", "top5_mass", "mean_chosen_rank",
        "std_chosen_rank", "mean_surprise", "std_surprise", "surprise_",
        "attn_entropy_", "head_agreement_", "delta_norm", "delta_cosine",
    ))


def fam_of(n: str) -> str:
    if n.startswith("ph_"): return "per_head"
    if n.startswith("attn_flow_"): return "attention_flow"
    if n.startswith("gate_"): return "gate"
    if n.startswith("res_traj"): return "residual_traj"
    if n.startswith(("cache_", "kv_", "epoch_")): return "T2.5"
    if n.startswith("spectral_"): return "T2_spectral"
    if n.startswith(("attn_entropy_", "head_agreement_", "delta_")): return "T2_other"
    if n.startswith("pca_"): return "T3"
    return "T1"


def _work(gid: int):
    """Compute spliced v3 + v2-core feature dicts for one gen (worker process)."""
    c = _ctx()
    z = np.load(RUN / f"signatures_v2/gen_{gid:03d}.npz", allow_pickle=True)
    v2 = {str(n): float(v) for n, v in zip(z["feature_names"], z["features"])}
    res = compute_features_v2(RUN / "raw_tensors", gid, c["cfg"], c["fam"], c["comps"], c["mean"])
    v3 = {res.feature_names[i]: float(res.features[i]) for i in range(len(res.features))}
    for nm in list(v3):
        if is_spliced(nm) and nm in v2:
            v3[nm] = v2[nm]
    v2_core = {k: val for k, val in v2.items() if not k.startswith("td_")}  # v2 minus temporal_dynamics
    g = c["meta"][gid]
    return gid, v2_core, v3, g["mode"], g["topic_idx"], g["prompt_length"], g["num_generated_tokens"]


def rf_cv(F, y, topic, C=None, residualize=False, seeds=3):
    if F.shape[1] == 0:
        return 0.0
    F = F[:, F.std(0) > 1e-10]
    accs = []
    for s in range(seeds):
        fold = []
        for tr, te in GroupKFold(5).split(F, y, topic):
            Ftr, Fte = F[tr].copy(), F[te].copy()
            if residualize:
                A = np.hstack([C[tr], np.ones((len(tr), 1))])
                B = np.hstack([C[te], np.ones((len(te), 1))])
                coef, *_ = np.linalg.lstsq(A, Ftr, rcond=None)
                Ftr, Fte = Ftr - A @ coef, Fte - B @ coef
            rf = RandomForestClassifier(300, random_state=s, n_jobs=1).fit(Ftr, y[tr])
            fold.append(rf.score(Fte, y[te]))
        accs.append(np.mean(fold))
    return float(np.mean(accs))


def main() -> None:
    meta = _ctx()["meta"]
    ids = [g for g in list_raw_tensor_ids(RUN / "raw_tensors") if g in meta and meta[g]["mode"] in HARD]
    nw = int(os.environ.get("N_WORKERS", min(8, max(2, (os.cpu_count() or 4) - 1))))
    log(f"computing v3 (spliced) for {len(ids)} hard-mode gens with {nw} workers...")

    results = {}
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=nw) as ex:
        futs = {ex.submit(_work, gid): gid for gid in ids}
        for fut in as_completed(futs):
            gid, v2c, v3, mode, topic, plen, glen = fut.result()
            results[gid] = (v2c, v3, mode, topic, plen, glen)
            done += 1
            if done % 5 == 0 or done == len(ids):
                el = time.time() - t0
                rate = done / el if el > 0 else 0
                eta = (len(ids) - done) / rate if rate > 0 else 0
                log(f"  {done}/{len(ids)} done | {el:.0f}s elapsed | {rate:.2f} gen/s | ETA {eta:.0f}s")

    order = sorted(results)
    names_v2 = list(results[order[0]][0])
    names_v3 = list(results[order[0]][1])
    X2 = np.array([[results[g][0][n] for n in names_v2] for g in order], float)
    X3 = np.array([[results[g][1][n] for n in names_v3] for g in order], float)
    y = np.array([results[g][2] for g in order])
    topic = np.array([results[g][3] for g in order])
    C = np.array([[results[g][4], results[g][5]] for g in order], float)
    fams3 = np.array([fam_of(n) for n in names_v3])
    log(f"matrices: v2(minus td)={X2.shape}, v3(clean+per_head)={X3.shape}, topics={len(set(topic))}")

    log("RF battery (topic-heldout 5-way, RF×300×3 seeds)...")
    print(f"\nlength-only [prompt_len,gen_len]: {rf_cv(C, y, topic):.1%}  (chance 20%)")
    print(f"{'set':24s} {'raw':>8s} {'residualized':>13s}")
    print(f"{'v2 (minus td)':24s} {rf_cv(X2, y, topic):>8.1%} {rf_cv(X2, y, topic, C, True):>13.1%}")
    print(f"{'v3 (clean+per_head)':24s} {rf_cv(X3, y, topic):>8.1%} {rf_cv(X3, y, topic, C, True):>13.1%}")
    for f in ["T2.5", "T2_spectral", "per_head", "attention_flow"]:
        m = fams3 == f
        print(f"  {f:22s} {rf_cv(X3[:, m], y, topic):>8.1%} {rf_cv(X3[:, m], y, topic, C, True):>13.1%}  ({int(m.sum())} feats)")
    log("done.")


if __name__ == "__main__":
    main()
