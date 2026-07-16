"""Qualitative steering-effectiveness browser: lay out matched prompts across the V3
alpha-ladder and against the V1 (formality) + R (random) controls at matched alpha, so
steering can be eyeballed — does V3 induce MODE (analogical framing) while R merely
DEGRADES (drift/incoherence)?

NOTE ON DOSE: the free-gen ladder tops out at alpha=0.3, but the V3 mode-induction sweet
spot is ~0.45-0.6 (wave-1 adjudication: analogy markers peaked at alpha=.45). So alpha=.3
here is the strongest COLLECTED dose but is PRE-PEAK — read the trend, not the ceiling.

Reads each cell's metadata.json (generated_text banked per gen). Matched by gen_id =
same prompt (seeds differ per cell namespace, so this is a style/mode comparison, not a
token-level control). Writes a readable markdown doc.

Usage: python -m anamnesis.scripts.vmb_a5_qual_extract --model qwen-7b \
    --run-dir /models/anamnesis-extract/runs/vmb_a5_qwen_7b --map-site 18 \
    --gen-ids 1 41 81 121 161 --out qual_qwen.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _texts(cell_dir: Path) -> dict[int, dict]:
    md = json.loads((cell_dir / "metadata.json").read_text())
    gens = md["generations"] if "generations" in md else md
    return {int(g["generation_id"]): g for g in gens}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--map-site", type=int, required=True)
    ap.add_argument("--gen-ids", type=int, nargs="+", required=True)
    ap.add_argument("--chars", type=int, default=600)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    M = args.map_site

    # cells to show, in read order (label -> cell dir name)
    ladder = [("V3 α=0 (baseline)", f"V3_L{M}_L{M}_a0.0"),
              ("V3 α=0.03", f"V3_L{M}_L{M}_a0.03"),
              ("V3 α=0.1", f"V3_L{M}_L{M}_a0.1"),
              ("V3 α=0.3 (strongest collected; PRE-PEAK, sweet spot ~.45-.6)", f"V3_L{M}_L{M}_a0.3")]
    controls = [("V1 α=0.3 (formality control)", f"V1_L{M}_L{M}_a0.3"),
                ("R1 α=0.3 (random-vector DEGRADATION control)", f"R1_L{M}_a0.3")]

    loaded = {}
    for _, c in ladder + controls:
        d = args.run_dir / c
        loaded[c] = _texts(d) if (d / "metadata.json").exists() else {}

    lines = [f"# Qualitative steering readout — {args.model} (map site L{M})", "",
             "⚠ **Dose caveat:** ladder tops at α=0.3; V3 mode-induction peak is ~α=0.45–0.6 "
             "(wave-1 adjudication). α=0.3 is strongest-collected but PRE-PEAK — read the trend.",
             "", "Matched by prompt (gen_id). Seeds differ per cell → style/mode comparison, "
             "not token-level. **Question for each block: does V3 shift toward analogical/"
             "figurative MODE with dose, while R just drifts/degrades?**", ""]
    for gid in args.gen_ids:
        ref = None
        for _, c in ladder + controls:
            if gid in loaded[c]:
                ref = loaded[c][gid]; break
        if ref is None:
            continue
        lines += [f"---", f"## gen {gid} — topic: *{ref.get('topic','?')}* | "
                  f"stratum: {ref.get('mode','?')}", ""]
        for label, c in ladder + controls:
            g = loaded[c].get(gid)
            if not g:
                continue
            txt = g.get("generated_text", "").strip().replace("\n", " ")
            lines += [f"**{label}**  ({g.get('num_generated_tokens','?')} tok)", "",
                      f"> {txt[:args.chars]}{'…' if len(txt) > args.chars else ''}", ""]
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out} ({len(args.gen_ids)} prompts × {len(ladder+controls)} cells)")


if __name__ == "__main__":
    main()
