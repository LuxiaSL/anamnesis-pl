"""THE BLESSED OPS TEMPLATE (canonical-ops 2026-07-16, spec backlog #5) — copy THIS.

Topology of record: GENALL -> REPALL, each phase owning the whole node, BOTH phases
through the load-once multicell launchers (model loads once per worker, loops every
cell). This is the pattern the path-of-record guard enforces: per-cell
vmb_stage0_generate/parallel_replay loops (submit_c3/submit_v3tail style, and the
old &&-chain GENALL) refuse at invocation #2.

Worker ladder of record (measured 2026-07-16): gen 16 workers/model saturates 4
cards (default here: 8 GPUs x 2); replay is CPU-bound — w64 (8 GPUs x 8) = 87%
efficiency shared-hours config. Scale down per VRAM for bigger models (playbook
worker/VRAM math).

Standing Heimdall rules baked in or documented (ops runbook 2026-07-12/-16):
  - depends_on is a LIST on the raw API (422 otherwise).
  - Job logs at /tmp/heimdall_<id>.log are WIPED on retry/cancel -> every leg tees
    its own output into the out-root (self-captured logs).
  - max_retries is set EXPLICITLY on every job (a raw-API spec field — verified
    heimdall models.py JobSpec / supervisor.py). Default 1 here: BOTH legs are
    resume-aware (gen records + signatures skip-if-exist), so a retry resumes
    instead of re-paying. Non-idempotent legs (search drivers, judges with
    sealed keys) must copy this pattern with --max-retries 0 and cancel
    auto-requeues before they fire.

Cells spec (JSON): {"cells": [{"key": "V3_L14", "layer": 14, "frac": 0.1}, ...]}
Each cell becomes <out-root>/<key>_L<layer>_a<frac>; seeds get a per-cell namespace
(disjoint by construction); replay reads each cell's own a5_injection metadata
(--inject-from-metadata => gen and replay use the identical spec).

Usage:
  python -m anamnesis.scripts.ops.submit_multicell_template --model 3b \
      --model-path /models/llama-3.2-3b-instruct \
      --calib-dir /models/anamnesis-extract/calibration/3b \
      --vectors-dir /models/anamnesis-extract/battery/a5_vectors_3b \
      --out-root /models/anamnesis-extract/runs/<campaign> \
      --cells-spec cells.json --namespace-prefix VMBX-3B \
      [--seeds-per-class 2] [--max-new-tokens 512] [--attn sdpa] [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import json
import urllib.request
from pathlib import Path

from anamnesis.scripts.ops._ops_env import API, WORK_DIR, base

BASE = base("HF_HUB_OFFLINE=1")
ENV = {"HF_HUB_OFFLINE": "1"}
PROMPTS = "pipeline/anamnesis/prompts/prompt_sets.json"


def submit(name: str, command: str, gpus: int, minutes: int,
           depends_on: list[str] | None = None, max_retries: int = 1) -> str:
    spec = {"job_type": "custom", "name": name, "gpus": gpus, "node": "node1",
            "working_dir": WORK_DIR, "estimated_minutes": minutes, "env": ENV,
            "max_retries": max_retries,   # explicit, always (retry audit 2026-07-16)
            "command": f"bash -c '{BASE} && {command}'"}
    if depends_on:
        spec["depends_on"] = list(depends_on)   # LIST, never a bare string (422)
    req = urllib.request.Request(API, data=json.dumps({"spec": spec}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        j = json.loads(r.read())
    if "job" not in j:
        raise RuntimeError(f"{name}: {j}")
    return j["job"]["id"]


def b64_file_cmd(payload: dict, remote_path: str) -> str:
    """Shell fragment materializing a JSON payload on the node (quote-safe)."""
    blob = base64.b64encode(json.dumps(payload).encode()).decode()
    return f"echo {blob} | base64 -d > {remote_path}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Blessed GENALL->REPALL multicell submit")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--calib-dir", required=True)
    ap.add_argument("--vectors-dir", required=True,
                    help="holds a5_vectors.npz + a5_vectors_stamps.json")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--cells-spec", type=Path, required=True)
    ap.add_argument("--namespace-prefix", required=True,
                    help="seed-namespace prefix; MUST be unique per (model x campaign)")
    ap.add_argument("--seeds-per-class", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--attn", default="sdpa", choices=["eager", "sdpa"],
                    help="sdpa for NEW steering cells (playbook rule 2); eager only "
                         "when token-matching a reference corpus")
    ap.add_argument("--gen-gpus", type=int, default=8)
    ap.add_argument("--gen-workers-per-gpu", type=int, default=2)
    ap.add_argument("--rep-gpus", type=int, default=8)
    ap.add_argument("--rep-workers-per-gpu", type=int, default=8)
    ap.add_argument("--gen-minutes", type=int, default=180)
    ap.add_argument("--rep-minutes", type=int, default=180)
    ap.add_argument("--max-retries", type=int, default=1,
                    help="explicit per-job retry cap; keep 1 for these resume-aware "
                         "legs, use 0 when copying this template for non-idempotent work")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    spec_cells = json.loads(args.cells_spec.read_text())["cells"]
    npz = f"{args.vectors_dir}/a5_vectors.npz"
    stamps = f"{args.vectors_dir}/a5_vectors_stamps.json"
    # GPU slots are LOGICAL (resolve_physical_gpus maps them); request N, pass 0..N-1
    gen_gpus_str = ",".join(str(g) for g in range(args.gen_gpus))
    rep_gpus_str = ",".join(str(g) for g in range(args.rep_gpus))

    def cell_dir(c: dict) -> str:
        return f"{args.out_root}/{c['key']}_L{c['layer']}_a{c['frac']}"

    gen_cells = {"cells": [
        {"out_run_dir": cell_dir(c),
         "seed_namespace": f"{args.namespace_prefix}-{c['key']}_L{c['layer']}_a{c['frac']}",
         "inject_key": c["key"], "inject_layer": int(c["layer"]),
         "inject_alpha_frac": float(c["frac"])}
        for c in spec_cells]}
    rep_cells = {"cells": [
        {"run_dir": cell_dir(c), "manifest": f"{cell_dir(c)}/replay_manifest.json"}
        for c in spec_cells]}

    tag = Path(args.out_root).name
    gen_json, rep_json = f"/tmp/{tag}_gen_cells.json", f"/tmp/{tag}_rep_cells.json"

    gen_cmd = (
        f"{b64_file_cmd(gen_cells, gen_json)} && "
        f"mkdir -p {args.out_root} && "
        f"python -u -m anamnesis.scripts.vmb_a5_gen_multicell "
        f"--model {args.model} --model-path {args.model_path} --prompts {PROMPTS} "
        f"--cells-json {gen_json} --gpus {gen_gpus_str} "
        f"--workers-per-gpu {args.gen_workers_per_gpu} "
        f"--seeds-per-class {args.seeds_per_class} "
        f"--max-new-tokens {args.max_new_tokens} --attn {args.attn} "
        f"--inject-npz {npz} --inject-norms-json {stamps} "
        f"2>&1 | tee {args.out_root}/genall_submit.log"   # self-captured (retry-safe)
    )
    rep_cmd = (
        f"{b64_file_cmd(rep_cells, rep_json)} && "
        f"python -u -m anamnesis.scripts.vmb_a5_replay_multicell "
        f"--model {args.model} --model-path {args.model_path} "
        f"--calib-dir {args.calib_dir} --cells-json {rep_json} "
        f"--gpus {rep_gpus_str} "
        f"--workers-per-gpu {args.rep_workers_per_gpu} "
        f"--no-raw --inject-from-metadata "
        f"2>&1 | tee {args.out_root}/repall_submit.log"
    )

    if args.dry_run:
        print(f"GENALL ({args.gen_gpus} GPU x {args.gen_workers_per_gpu}):\n  {gen_cmd}\n")
        print(f"REPALL ({args.rep_gpus} GPU x {args.rep_workers_per_gpu}):\n  {rep_cmd}")
        return

    gj = submit(f"mc-{tag}-GENALL", gen_cmd, gpus=args.gen_gpus,
                minutes=args.gen_minutes, max_retries=args.max_retries)
    rj = submit(f"mc-{tag}-REPALL", rep_cmd, gpus=args.rep_gpus,
                minutes=args.rep_minutes, depends_on=[gj],
                max_retries=args.max_retries)
    print(f"{len(spec_cells)} cells | GENALL={gj} -> REPALL={rj} "
          f"(load-once multicell, both phases)")


if __name__ == "__main__":
    main()
