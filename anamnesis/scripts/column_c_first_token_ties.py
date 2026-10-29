"""Column C: share of momentum pairs whose siblings share a first token (C@a0 ties by causality), and a
finiteness check of the FLOOR blocks. Paths are node2 paths."""
import json, sys, numpy as np
z = np.load(sys.argv[1] + "/column_c_floor.npz")
ft = dict(zip(z["cand_ids"].astype(str), z["first_token"]))
a0 = dict(zip(z["cand_ids"].astype(str), range(len(z["cand_ids"]))))
F = z["floor_a0"]
out = {}
nodes = [json.loads(l) for l in open("/mnt/beegfs/luxia/posttrain-tsumugi/momentum/v2/momentum-labels-v2.jsonl") if '"_header"' not in l[:20]]
for tag, lo, hi in (("long", 16, 10**9), ("short", 0, 16)):
    same = tot = byte_equal = 0
    for n in nodes:
        ok = [c for c in n["candidates"] if c.get("grade") is not None and c.get("survivor") and not c.get("breaks_scene")]
        for a in ok:
            for b in ok:
                if a["grade"] - b["grade"] >= 2:
                    m = min(a["n_tokens"], b["n_tokens"])
                    if (tag == "long" and m >= 16) or (tag == "short" and m < 16):
                        tot += 1
                        s = ft[a["cand_id"]] == ft[b["cand_id"]]
                        same += s
                        byte_equal += bool(np.array_equal(F[a0[a["cand_id"]]], F[a0[b["cand_id"]]]))
    out[tag] = dict(pairs=int(tot), same_first_token=int(same), a0_vectors_identical=int(byte_equal))
for p in ("a0", "a1", "a2", "a3", "alast", "plast"):
    x = z[f"floor_{p}"]
    rows_defined = ~np.isnan(x).all(axis=1)
    out[f"nonfinite_{p}"] = dict(rows_defined=int(rows_defined.sum()),
                                 nonfinite_in_defined_rows=int((~np.isfinite(x[rows_defined])).sum()))
print(json.dumps(out))
open(sys.argv[1] + "/first_token_ties.json", "w").write(json.dumps(out, indent=1))
