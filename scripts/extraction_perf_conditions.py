"""Materialize the desk-ruled rubric-140 G1 contrasts without fitted choices."""

from __future__ import annotations

import argparse
from pathlib import Path

from extraction_perf_phase0 import rows_from_bank, sha, write_json


def condition_entries(keys: list[tuple]) -> list[dict]:
    population = set(keys)
    if len(population) != len(keys):
        raise ValueError("duplicate reference key")
    entries = []
    for key in keys:
        namespace, doc, cell, branch, span = key
        entry = {"key": list(key)}
        if cell == "own_cont":
            entry.update(
                contrast_status="null-by-rule",
                reason="Anchor A has no nonzero self-contrast; its influence is tested by separate null-fork statistics.",
            )
        else:
            if cell in ("paraphrase_own", "xfam_paraphrase") or any(
                cell.startswith(prefix + "_rep")
                and cell.removeprefix(prefix + "_rep").isdigit()
                for prefix in ("paraphrase_own", "xfam_paraphrase")
            ):
                prefix = (
                    "xfam_paraphrase"
                    if cell.startswith("xfam_paraphrase")
                    else "paraphrase_own"
                )
                suffix = cell.removeprefix(prefix)
                left = (namespace, doc, "xfam_paraphrase" + suffix, branch, span)
                right = (namespace, doc, "paraphrase_own" + suffix, branch, span)
            elif cell in ("rung_control", "null::own_cont"):
                left = key
                right = (namespace, doc, "own_cont", 0, span)
            else:
                raise ValueError(f"unruled cell: {cell}")
            if left not in population or right not in population:
                raise ValueError(f"missing contrast endpoint for {key}")
            entry.update(contrast_status="defined", left=list(left), right=list(right))
        entries.append(entry)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--ruling", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows, _ = rows_from_bank(args.reference)
    entries = condition_entries([("rubric-140", *row["key"]) for row in rows])
    write_json(args.out, dict(ruling_sha256=sha(args.ruling), rows=entries))
    print(f"Wrote {len(entries)} explicit condition mappings to {args.out}")


if __name__ == "__main__":
    main()
