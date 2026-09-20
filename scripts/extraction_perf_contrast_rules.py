"""Ratified per-row declared contrasts; no pooled distances or epsilon relief."""

from __future__ import annotations

import numpy as np


def declared_entries(keys):
    population = set(keys)
    if len(population) != len(keys):
        raise ValueError("duplicate contrast population")
    result = []
    for key in keys:
        namespace, doc, cell, branch, span = key

        def k(name, replicate=0):
            return (namespace, doc, name, replicate, span)

        a, b, f, e = (
            k("own_cont"),
            k("paraphrase_own"),
            k("xfam_paraphrase"),
            k("rung_control"),
        )
        if cell == "own_cont":
            proposals = []
        elif cell == "paraphrase_own":
            proposals = [("F-B", f, b), ("B-E", b, e)]
        elif cell == "xfam_paraphrase":
            proposals = [("F-B", f, b), ("F-E", f, e)]
        elif cell == "rung_control":
            proposals = [("B-E", b, e), ("F-E", f, e)]
        elif cell == "null::own_cont":
            proposals = [("fork-A", key, a)]
        elif any(
            cell.startswith(prefix + "_rep")
            and cell.removeprefix(prefix + "_rep").isdigit()
            for prefix in ("paraphrase_own", "xfam_paraphrase")
        ):
            prefix = (
                "paraphrase_own"
                if cell.startswith("paraphrase_own")
                else "xfam_paraphrase"
            )
            suffix = cell.removeprefix(prefix)
            proposals = [
                (
                    "matched-replicate-F-B",
                    k("xfam_paraphrase" + suffix, branch),
                    k("paraphrase_own" + suffix, branch),
                )
            ]
        else:
            raise ValueError(f"unruled contrast cell: {cell}")
        available = []
        unavailable = []
        for name, left, right in proposals:
            term = dict(name=name, left=list(left), right=list(right))
            missing = [
                list(endpoint)
                for endpoint in (left, right)
                if endpoint not in population
            ]
            if missing:
                unavailable.append(dict(term, missing_endpoints=missing))
            else:
                available.append(term)
        result.append(
            dict(
                key=list(key),
                contrasts=available,
                unavailable_contrasts=unavailable,
                empty_reason="anchor A: no declared contrast membership"
                if cell == "own_cont"
                else "no declared partner endpoints remain in the roster",
            )
        )
    return result


def resolve_entries(entries, keys, cpu, gpu, repeat, sigma, weights, inputs, spans):
    if entries != declared_entries(list(keys)):
        raise ValueError("contrast manifest differs from ratified membership")
    lookup = {key: i for i, key in enumerate(keys)}
    pairs = {}
    distances = {}
    reasons = {}
    receipts = []
    for entry in entries:
        key = tuple(entry["key"])
        retained = []
        terms = []
        for term in entry["contrasts"]:
            left, right = tuple(term["left"]), tuple(term["right"])
            i, j = lookup[left], lookup[right]
            distance = float(
                np.linalg.norm((cpu[i].astype(float) - cpu[j]) * weights / sigma)
            )
            omit = (
                distance == 0.0
                and inputs[left] == inputs[right]
                and spans[left] == spans[right]
                and all(
                    values[i].tobytes() == values[j].tobytes()
                    for values in (cpu, gpu, repeat)
                )
            )
            terms.append(
                dict(term, distance=distance, omitted_exact_identical_input_zero=omit)
            )
            if not omit:
                retained.append((distance, (left, right)))
        if retained:
            distance, pair = min(retained, key=lambda item: item[0])
            pairs[key] = pair
            distances[key] = distance
        else:
            pairs[key] = None
            distances[key] = None
            reasons[key] = (
                "all available declared terms cancel exactly on identical inputs in both lanes"
                if terms
                else entry["empty_reason"]
            )
        receipts.append(
            dict(
                key=list(key),
                terms=terms,
                unavailable_contrasts=entry["unavailable_contrasts"],
                operative_distance=distances[key],
                reason=reasons.get(key),
            )
        )
    return pairs, distances, reasons, receipts
