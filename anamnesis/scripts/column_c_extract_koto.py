"""Column C on tsumugi: the FLOOR vector at declared positions of teacher-forced replay rows.

For each loom node, the shared context is prefilled once (parity mode, the capture path's
eager arithmetic) and each sibling's first tokens are then run through the cache with full
capture. Declared positions, by absolute index with P = prompt_length:

* ``a0``: P, the answer's first token (the per-member column; within a fan it reads the
  model taking in its own first token, i.e. post-decision processing);
* ``a1``..``a3``: the next answer positions, NaN where the answer is shorter;
* ``alast``: the last answer token, the column that has read the whole answer (= ``a0`` at one
  token; same names at every length, so one schema);
* ``plast``: P - 1, the last prompt position (shared within a fan: zero within-fan contrast).

Each position's FLOOR vector (``column_c_features.floor_vector``) reads the depth path in the
kotodama convention of record (``hidden_rows``: embedding + post-MLP partial per effective
layer, 73 points; partials reset at AttnRes block boundaries), the read-profile from the
attention rows of all 72 layers (reduced on the device as the forward produces them), and the
AttnRes routing at every site. The schema is fixed by the geometry: every row computes the
same names whatever its length.

Also kept: the a0 depth path itself (fp16, [rows, 73, 1536]) and the a0 per-head read stats,
so later reductions need no GPU; the text-side first-token baselines (log-prob of the first and
second answer tokens, the output entropy at a0).

A faithfulness check runs on the first ``--check-nodes`` nodes: one uncached forward over the
row, compared with the cached a0 path and logits.

Environment: as ``sign_koto_replay_rows.py`` (pleroma tree on PYTHONPATH, deterministic
CUBLAS workspace, fp64-eps compat).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from column_c_features import (FLOOR_VERSION, Geometry, RouteSite, floor_digest,  # noqa: E402
                               floor_names, floor_vector)

POSITIONS = ("a0", "a1", "a2", "a3", "alast", "plast")
N_ANSWER = 4


def load_rows(path: Path) -> tuple[dict, list[dict]]:
    header, rows = {}, []
    with path.open() as fh:
        for i, line in enumerate(fh):
            rec = json.loads(line)
            if i == 0 and rec.get("_header"):
                header = rec
                continue
            rows.append(rec)
    return header, rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--checkpoint-path", required=True)
    ap.add_argument("--koto-size", default="1b-tsumugi")
    ap.add_argument("--koto-boundaries", default="tsumugi_loop")
    ap.add_argument("--limit-nodes", type=int, default=0)
    ap.add_argument("--check-nodes", type=int, default=3)
    args = ap.parse_args(argv)

    import torch
    from kotodama.model.capture import cached_forward, capture_identity, hidden_rows
    from kotodama.model.routing_read import site_layout
    from pleroma.model import koto

    if not torch.cuda.is_available():
        print("REFUSED: needs a GPU", file=sys.stderr)
        return 2
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = koto.load_koto(args.checkpoint_path, size=args.koto_size,
                           boundaries=koto.boundaries_for(args.koto_boundaries), device="cuda:0",
                           rope_fp32=True, eps_fp64_compat=True)
    model.eval()
    L = int(model.config.num_layers)
    H = int(model.config.num_attention_heads)
    block_ranges = [tuple(b) for b in model._attn_res_block_ranges]

    header, rows = load_rows(args.rows)
    by_node: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_node[r["node_id"]].append(r)
    node_ids = list(by_node)
    if args.limit_nodes:
        node_ids = node_ids[: args.limit_nodes]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[colc] {len(rows)} rows over {len(by_node)} nodes; running {len(node_ids)} nodes", flush=True)

    geo: Geometry | None = None
    out_feats = {p: [] for p in POSITIONS}
    a0_paths, a0_heads, meta_rows = [], [], []
    checks = []
    t0 = time.time()

    for k, node_id in enumerate(node_ids, 1):
        sibs = by_node[node_id]
        P = int(sibs[0]["prompt_length"])
        ctx = sibs[0]["input_ids"][:P]
        for s in sibs:
            if int(s["prompt_length"]) != P or s["input_ids"][:P] != ctx:
                raise RuntimeError(f"node {node_id}: siblings do not share the prompt")
        if P < 2:
            raise RuntimeError(f"node {node_id}: prompt of {P} tokens leaves no prefill")
        dev = torch.device("cuda:0")
        pre = torch.tensor([ctx[: P - 1]], dtype=torch.long, device=dev)
        _, _, cache = cached_forward(model, pre, past=None, position_offset=0,
                                     eager_nocapture=True, capture_hidden=False)

        for s in sibs:
            ids = s["input_ids"]
            n_ans = len(ids) - P
            if n_ans < 1:
                raise RuntimeError(f"{s['cand_id']}: no answer tokens")
            def capture(m, want):
                """Run ids[P-1 : P-1+m] through the cache; FLOOR inputs at chunk indices `want`."""
                chunk = torch.tensor([ids[P - 1 : P - 1 + m]], dtype=torch.long, device=dev)
                stats = torch.zeros((L, 5, H, m), dtype=torch.float64, device=dev)
                mean_ent = torch.zeros((L, m), dtype=torch.float64, device=dev)

                def sink(layer, w):
                    a = w[0].double()                            # [H, m, W]
                    W = a.shape[-1]
                    a = a / a.sum(-1, keepdim=True).clamp_min(1e-30)
                    q = torch.arange(P - 1, P - 1 + m, device=a.device)   # absolute query positions
                    cols = torch.arange(W, device=a.device)
                    stats[layer, 0] = -(a * torch.where(a > 0, a.log(), torch.zeros_like(a))).sum(-1)
                    mr = a.mean(0)
                    mean_ent[layer] = -(mr * torch.where(mr > 0, mr.log(), torch.zeros_like(mr))).sum(-1)
                    stats[layer, 1] = a[..., 0]
                    cutoff = torch.clamp((0.8 * (q + 1).double()).floor().long(), min=1)
                    stats[layer, 2] = (a * (cols[None, :] >= cutoff[:, None])[None]).sum(-1)
                    stats[layer, 3] = a[..., :P].sum(-1)

                logits, cap, _ = cached_forward(model, chunk, past=cache, position_offset=P - 1,
                                                light=False, capture_hidden=True, attention_sink=sink,
                                                capture_layers=set())
                hr = hidden_rows(cap, L)
                sel = torch.tensor(want, device=dev)
                paths = torch.stack([h[0] for h in hr], dim=1)[sel].float()   # [|want|, L+1, d]
                return dict(at={w_: j for j, w_ in enumerate(want)}, paths=paths.cpu().numpy(),
                            stats=stats[..., sel].cpu().numpy(), mean=mean_ent[:, sel].cpu().numpy(),
                            route=[w[:, 0, sel].double().cpu().numpy() for _, _, w in cap["routing"]],
                            routing=cap["routing"], logits=logits[0].double())

            def vec(c, i):
                j = c["at"][i]
                st = c["stats"]
                read = dict(head_ent=st[:, 0, :, j], mean_ent=c["mean"][:, j], sink=st[:, 1, :, j],
                            recency=st[:, 2, :, j], prompt=st[:, 3, :, j])
                return floor_vector(geo, c["paths"][j], read, [r[:, j] for r in c["route"]]).astype(np.float32)

            # Fixed chunk shapes per declared position, so a position's arithmetic never depends on
            # the answer's length: plast/a0 from a 2-token chunk, a1..a3 from a 5-token chunk (shorter
            # only when the answer is), alast from the whole answer (its inputs differ anyway).
            c2 = capture(2, [0, 1])
            if geo is None:
                sites = []
                for tag, layer, w in c2["routing"]:
                    lay = site_layout(block_ranges, int(layer), tag, int(w.shape[0]))
                    name = "final" if tag == "final" else f"{tag}.L{int(layer)}"
                    sites.append(RouteSite(name, lay.rel, tuple(float(lay.at - e) for e in lay.end_layer)))
                geo = Geometry(model="kotodama-tsumugi-1b-s52450", n_points=L + 1, n_read_layers=L,
                               route_sites=tuple(sites))
                print(f"[colc] FLOOR schema {FLOOR_VERSION}: {len(floor_names(geo))} names, "
                      f"digest {floor_digest(geo)}", flush=True)
            if len(c2["routing"]) != len(geo.route_sites):
                raise RuntimeError(f"{s['cand_id']}: {len(c2['routing'])} routing sites, schema has {len(geo.route_sites)}")
            m5 = 1 + min(n_ans, N_ANSWER)
            c5 = capture(m5, list(range(1, m5))) if n_ans >= 2 else None
            m_all = 1 + n_ans
            c_all = capture(m_all, [m_all - 1]) if n_ans >= 2 else c2

            nan = np.full(len(floor_names(geo)), np.nan, dtype=np.float32)
            out_feats["plast"].append(vec(c2, 0))
            out_feats["a0"].append(vec(c2, 1))
            for j in range(1, N_ANSWER):
                out_feats[f"a{j}"].append(vec(c5, j + 1) if c5 is not None and j + 1 < m5 else nan)
            out_feats["alast"].append(vec(c_all, m_all - 1))
            a0_paths.append(c2["paths"][1].astype(np.float32))
            a0_heads.append(np.concatenate([c2["stats"][:, :4, :, 1],
                                            np.repeat(c2["mean"][:, None, None, 1], H, axis=2)],
                                           axis=1).astype(np.float32))
            lp2 = torch.log_softmax(c2["logits"], dim=-1)
            lp_a0 = float(lp2[0, ids[P]])
            lp_a1 = float(lp2[1, ids[P + 1]]) if n_ans >= 2 else float("nan")
            ent_a0 = float(-(lp2[1].exp() * lp2[1]).sum())
            meta_rows.append(dict(cand_id=s["cand_id"], node_id=node_id, n_tokens=n_ans, prompt_length=P,
                                  first_token=int(ids[P]), second_token=int(ids[P + 1]) if n_ans >= 2 else -1,
                                  logprob_a0=lp_a0, logprob_a1=lp_a1, out_entropy_a0=ent_a0))

            if k <= args.check_nodes:
                full = torch.tensor([ids], dtype=torch.long, device=dev)
                fl, fcap, _ = cached_forward(model, full, past=None, position_offset=0, light=False,
                                             capture_hidden=True, eager_nocapture=False, capture_layers=set(),
                                             attention_sink=lambda _l, _w: None)
                fhr = hidden_rows(fcap, L)
                fpath = torch.stack([h[0, P] for h in fhr]).float().cpu().numpy()
                lpath = torch.stack([h[0, P + n_ans - 1] for h in fhr]).float().cpu().numpy()
                cl = c_all["paths"][0] if n_ans >= 2 else c2["paths"][1]
                checks.append(dict(cand_id=s["cand_id"],
                                   a0_path_rel=float(np.linalg.norm(fpath - c2["paths"][1]) / np.linalg.norm(fpath)),
                                   alast_path_rel=float(np.linalg.norm(lpath - cl) / np.linalg.norm(lpath)),
                                   logits_maxabs=float((fl[0, P - 1 : P + 1].double() - c2["logits"]).abs().max())))
                del fl, fcap
        del cache
        if k % 25 == 0 or k == len(node_ids):
            print(f"[colc] {k}/{len(node_ids)} nodes, {len(meta_rows)} rows, "
                  f"{len(meta_rows) / (time.time() - t0):.2f} rows/s", flush=True)

    assert geo is not None
    names = floor_names(geo)
    digest = floor_digest(geo)
    np.savez(args.out_dir / "column_c_floor.npz",
             names=np.array(names), digest=np.array(digest), version=np.array(FLOOR_VERSION),
             cand_ids=np.array([r["cand_id"] for r in meta_rows]),
             node_ids=np.array([r["node_id"] for r in meta_rows]),
             n_tokens=np.array([r["n_tokens"] for r in meta_rows]),
             prompt_length=np.array([r["prompt_length"] for r in meta_rows]),
             first_token=np.array([r["first_token"] for r in meta_rows]),
             second_token=np.array([r["second_token"] for r in meta_rows]),
             logprob_a0=np.array([r["logprob_a0"] for r in meta_rows]),
             logprob_a1=np.array([r["logprob_a1"] for r in meta_rows]),
             out_entropy_a0=np.array([r["out_entropy_a0"] for r in meta_rows]),
             **{f"floor_{p}": np.stack(out_feats[p]) for p in POSITIONS})
    np.savez(args.out_dir / "column_c_a0_raw.npz", cand_ids=np.array([r["cand_id"] for r in meta_rows]),
             depth_path=np.stack(a0_paths), head_stats=np.stack(a0_heads),
             head_stats_order=np.array(["head_ent", "sink", "recency", "prompt", "mean_ent"]))
    meta = dict(version=FLOOR_VERSION, digest=digest, n_names=len(names), geometry=geo.to_json(),
                rows=str(args.rows), rows_header_version=header.get("version"),
                checkpoint=args.checkpoint_path, n_rows=len(meta_rows), n_nodes=len(node_ids),
                positions=dict(a0="P (first answer token)", a1="P+1", a2="P+2", a3="P+3",
                               alast="the last answer token (= a0 for a 1-token answer)",
                               plast="P-1 (last prompt token, shared within a fan)"),
                capture=("per-node eager parity prefill of ctx[:P-1]; then fixed-shape cached chunks: "
                         "ids[P-1:P+1] -> plast, a0; ids[P-1:P+4] -> a1..a3; ids[P-1:] -> alast"),
                faithfulness_checks=checks, capture_identity=capture_identity(model),
                wall_seconds=round(time.time() - t0, 1),
                code={m_: sys.modules[m_].__file__ for m_ in ("kotodama", "pleroma")})
    (args.out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[colc] done: {len(meta_rows)} rows, {len(names)} names, digest {digest}", flush=True)
    print(f"[colc] faithfulness: {json.dumps(checks)[:2000]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
