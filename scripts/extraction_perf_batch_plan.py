"""Lossless exact-shape batch2 schedule, including explicitly priced fillers."""


def exact_batches(rows):
    keys = [tuple(r["key"]) for r in rows]
    if not rows or len(set(keys)) != len(keys):
        raise ValueError("empty/duplicate reference population")
    buckets = {}
    for i, row in enumerate(rows):
        shape = (row["start"], row["end"] - row["start"])
        buckets.setdefault(shape, []).append(i)
    plans = []
    for shape, indices in sorted(buckets.items()):
        for offset in range(0, len(indices), 2):
            real = indices[offset : offset + 2]
            physical = real if len(real) == 2 else real + real
            plans.append(
                dict(
                    shape=list(shape),
                    indices=physical,
                    emitted_indices=real,
                    filler_count=2 - len(real),
                )
            )
    emitted = [i for p in plans for i in p["emitted_indices"]]
    if sorted(emitted) != list(range(len(rows))):
        raise ValueError("schedule dropped or repeated a reference row")
    return plans
