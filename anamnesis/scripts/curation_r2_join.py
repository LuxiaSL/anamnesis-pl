"""Curation round 2: join posttraining's per-turn labels to the engine-lane signatures of the turns they label.

Labels (``turn-labels-*.jsonl``, BRIEF-TURNS, text only; or the same rows without the label fields, for turns to
score) carry scene_id, turn, writer, speaker and text, but not
the sign record. Spool records carry token ids and weights, not the scene. A label is joined to the record that
generated its turn:
* the record's weights are the writer's (b01x: DPO-10x bb9495ec, sft: d4750e7d);
* its decoded generation, cut at the first blank line, is the turn's text (stripped);
* its decoded prompt contains the previous two turns' texts and every speaker name of the scene.

Records whose token ids are identical sign to identical bytes, so a label whose candidates all share one
(prompt, generation) is joined to any of them. A label with candidates of different token ids is ambiguous and
dropped. Every count is reported. The signature comes from the signed batches of the same spools (canonical drain
lane). Only rows of the full feature layout are kept (spans of at least 8 steps; shorter spans have a smaller
schema), and the dropped rows are counted.

Writes the inputs ``curation_turn_read.py`` reads: rows.jsonl (cand_id, scene_id, turn, writer, coherent,
n_tokens, prompt_length), sigs/signatures-0of1.npz, labels.jsonl (id, turn, text), sample.json (scenes' turns),
and support.tsv (every kept feature computed from 1 step).

Usage: python curation_r2_join.py --labels <jsonl> --spool <spool dir> [--spool ...] --kotodama-src <src>
       --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

WRITER_WEIGHTS = {"b01x": "bb9495ec24a08358c9d99627ba32e47a9e08b85f2970193feed994225c073a00",
                  "sft": "d4750e7dbb6852de9c3f9a0d7e9c8e3e6c478b8d95aea7222c1ec03d9fe56583"}


def norm(text: str) -> str:
    return " ".join(text.split())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--spool", required=True, type=Path, action="append")
    ap.add_argument("--kotodama-src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    sys.path.insert(0, str(args.kotodama_src))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from kotodama import presets
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(presets.TOKENIZER_NAME)
    labels = [json.loads(l) for l in args.labels.read_text().splitlines() if l.strip()]
    scenes: dict[str, list[dict]] = collections.defaultdict(list)
    for r in labels:
        scenes[r["scene_id"]].append(r)
    for s in scenes.values():
        s.sort(key=lambda r: int(r["turn"]))
    wanted = {norm(r["text"]) for r in labels}
    print(f"[join] {len(labels)} labels over {len(scenes)} scenes", flush=True)

    # Records whose generation (cut at the first blank line) is some label's text, by (weights, text).
    by_text: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    n_records = 0
    for spool in args.spool:
        for path in sorted(glob.glob(str(spool / "records" / "*.jsonl"))):
            batch: list[dict] = []
            for line in Path(path).read_text().splitlines():
                if line.strip():
                    batch.append(json.loads(line))
            gens = tok.batch_decode([r["generated_ids"] for r in batch])
            for rec, gen in zip(batch, gens):
                n_records += 1
                text = norm(gen.split("\n\n")[0])
                if text in wanted:
                    by_text[(rec["weights"], text)].append(rec)
    print(f"[join] {n_records} records read; {sum(len(v) for v in by_text.values())} match some label text",
          flush=True)

    joined: dict[tuple[str, int], dict] = {}
    counts = collections.Counter()
    for sid, turns in scenes.items():
        speakers = sorted({r["speaker"] for r in turns})
        for i, r in enumerate(turns):
            w = WRITER_WEIGHTS.get(r["writer"])
            if w is None:
                counts["unknown_writer"] += 1
                continue
            cands = by_text.get((w, norm(r["text"])), [])
            if not cands:
                counts["no_text_match"] += 1
                continue
            prev = [norm(t["text"]) for t in turns[max(0, i - 2):i]]
            ok = []
            for rec in cands:
                prompt = rec.get("_prompt_text")
                if prompt is None:
                    prompt = rec["_prompt_text"] = norm(tok.decode(rec["prompt_ids"]))
                if all(p in prompt for p in prev) and all(s in prompt for s in speakers):
                    ok.append(rec)
            if not ok:
                counts["no_context_match"] += 1
                continue
            keys = {hashlib.sha256(json.dumps([rec["prompt_ids"], rec["generated_ids"]]).encode()).hexdigest()
                    for rec in ok}
            if len(keys) > 1:
                counts["ambiguous"] += 1
                continue
            counts["joined"] += 1
            joined[(sid, int(r["turn"]))] = dict(label=r, record=ok[0])
    print(f"[join] {dict(counts)}", flush=True)

    # Signed rows of the joined records.
    need = {j["record"]["request_id"] for j in joined.values()}
    signed: dict[str, tuple[np.ndarray, str, str, float]] = {}
    names_by_digest: dict[str, list[str]] = {}
    for spool in args.spool:
        for rows_path in sorted(glob.glob(str(spool / "signed" / "*" / "rows.jsonl"))):
            batch_dir = Path(rows_path).parent
            hits = []
            for line in Path(rows_path).read_text().splitlines():
                row = json.loads(line)
                if row.get("sign_id") in need and "file" in row:
                    hits.append(row)
            if not hits:
                continue
            names_by_digest.update(json.loads((batch_dir / "names.json").read_text()))
            arrays: dict[str, np.ndarray] = {}
            for row in hits:
                if row["file"] not in arrays:
                    arrays[row["file"]] = np.load(batch_dir / row["file"])
                signed[row["sign_id"]] = (arrays[row["file"]][int(row["index"])], row["names_digest"],
                                          row["lane_id"], float(row["reducer_meta"]["mean_logprob"]))
    layouts = collections.Counter(v[1] for v in signed.values())
    if not layouts:
        print("[join] REFUSED: no signed rows for the joined records", file=sys.stderr)
        return 2
    full = layouts.most_common(1)[0][0]
    names = names_by_digest[full]
    print(f"[join] signed {len(signed)} of {len(need)}; layouts {dict(layouts)}; kept layout {full[:12]} "
          f"({len(names)} features)", flush=True)

    out = args.out
    (out / "sigs").mkdir(parents=True, exist_ok=True)
    rows, X, ids, lps, lanes = [], [], [], [], []
    dropped = collections.Counter()
    for (sid, turn), j in sorted(joined.items()):
        rec, lab = j["record"], j["label"]
        hit = signed.get(rec["request_id"])
        if hit is None:
            dropped["unsigned"] += 1
            continue
        if hit[1] != full:
            dropped["short_span_layout"] += 1
            continue
        cid = f"{sid}:{turn}"
        rows.append(dict(cand_id=cid, scene_id=sid, turn=turn, writer=lab["writer"], set=lab.get("set"), coherent=lab.get("coherent"),
                         in_character=lab.get("in_character"), n_tokens=len(rec["generated_ids"]),
                         prompt_length=len(rec["prompt_ids"]), sign_id=rec["request_id"]))
        X.append(hit[0])
        ids.append(cid)
        lanes.append(hit[2])
        lps.append(hit[3])
    print(f"[join] kept {len(rows)}; dropped {dict(dropped)}; lanes {dict(collections.Counter(lanes))}", flush=True)
    with (out / "rows.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    np.savez(out / "sigs" / "signatures-0of1.npz", full_readout=np.stack(X).astype(np.float32),
             cand_ids=np.array(ids), lane_ids=np.array(lanes), mean_logprob=np.array(lps),
             full_names=np.array(names))
    with (out / "labels.jsonl").open("w") as fh:
        for r in labels:
            fh.write(json.dumps(dict(id=r["scene_id"], turn=r["turn"], text=r["text"])) + "\n")
    (out / "sample.json").write_text(json.dumps(dict(scenes=[
        dict(id=sid, turns=[dict(text=t["text"]) for t in turns]) for sid, turns in scenes.items()])))
    with (out / "support.tsv").open("w") as fh:
        fh.write("feature_name\tmin_steps_computed\tfamily\n")
        for n in names:
            fh.write(f"{n}\t1\t{n.split('_')[0]}\n")
    (out / "join_summary.json").write_text(json.dumps(dict(
        labels=len(labels), scenes=len(scenes), records_read=n_records, join=dict(counts), signed=len(signed),
        layouts={k[:12]: v for k, v in layouts.items()}, kept=len(rows), dropped=dict(dropped),
        lanes=dict(collections.Counter(lanes))), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
