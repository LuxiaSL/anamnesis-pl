"""Report trace host scopes separately from device activity; never add the two."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from extraction_perf_phase0 import sha, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    trace_path = args.capture / "trace.json"
    events = json.loads(trace_path.read_text())["traceEvents"]
    scopes = [
        e
        for e in events
        if e.get("cat") == "user_annotation"
        and e.get("ph") == "X"
        and e["name"].startswith("extraction::")
    ]
    if not scopes:
        raise ValueError("missing host annotation scopes")
    tid = scopes[0]["tid"]
    host = [
        e
        for e in events
        if e.get("tid") == tid
        and e.get("ph") == "X"
        and e.get("cat") in ("cpu_op", "user_annotation", "cuda_runtime")
    ]
    start = min(e["ts"] for e in host)
    end = max(e["ts"] + e["dur"] for e in host)
    # A sweep-line partition attributes nested ranges to the innermost scope.
    boundaries = [(start, 0, -1), (end, 0, -1)]
    for i, e in enumerate(scopes):
        boundaries.extend([(e["ts"], 1, i), (e["ts"] + e["dur"], -1, i)])
    active = set()
    totals = defaultdict(float)
    previous = start
    for at, action, index in sorted(boundaries):
        label = (
            scopes[min(active, key=lambda i: scopes[i]["dur"])]["name"]
            if active
            else "other_host_work_or_wait"
        )
        totals[label] += at - previous
        if action == 1:
            active.add(index)
        elif action == -1:
            active.remove(index)
        previous = at
    device = defaultdict(float)
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in (
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
        ):
            label = e["name"] if e["cat"] == "gpu_memcpy" else e["cat"]
            device[label] += e["dur"]
    row = json.loads((args.capture / "row-00000.json").read_text())
    result = dict(
        diagnostic_only=True,
        trace_sha256=sha(trace_path),
        script_sha256=sha(Path(__file__)),
        unprofiled_seconds=row["seconds_per_row"],
        traced_repeat_exact=row["G0"],
        traced_host_span_ms=(end - start) / 1000,
        host_partition=[
            dict(name=k, ms=v / 1000, percent=100 * v / (end - start))
            for k, v in sorted(totals.items(), key=lambda kv: -kv[1])
        ],
        device_activity_ms={k: v / 1000 for k, v in device.items()},
        caveats=[
            "One input, profiler perturbs timing; percentages describe traced host span, not the unprofiled baseline.",
            "Host scopes include dispatch, waiting and CPU computation; device activities are separate, possibly overlapping durations.",
            "Do not sum device time with host time or interpret removed work as an achievable speedup.",
        ],
    )
    write_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
